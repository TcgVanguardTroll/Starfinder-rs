use crate::models::{Performer, SearchFilters};
use anyhow::{Context, Result};
use rusqlite::{params, Connection};

mod corpus;
pub use corpus::{aggregate_views, AggregatedViews, BodyIndexEntry, ImageRow};

/// Decodes an embedding column value, accepting both the new f32 BLOB format
/// and legacy JSON text rows transparently.
fn decode_embedding(val: rusqlite::types::Value) -> Option<Vec<f32>> {
    use rusqlite::types::Value;
    match val {
        Value::Blob(b) => crate::embedder::blob_to_embedding(&b),
        Value::Text(t) => crate::embedder::blob_to_embedding(t.as_bytes()),
        _ => None,
    }
}

/// Decodes a stored performer JSON blob. One corrupt row used to panic every
/// command that lists performers; instead skip it with a warning so the rest
/// of the library stays usable.
fn decode_performer(data: &str) -> Option<Performer> {
    match serde_json::from_str(data) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("warning: skipping corrupt performer row: {e}");
            None
        }
    }
}

/// Database manager for storing performers
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Creates a new database connection
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path).context("Failed to open database")?;
        // Wait rather than error if another process holds the write lock — keeps
        // searches working while a long `index` build runs in the background.
        conn.busy_timeout(std::time::Duration::from_secs(15))?;
        let db = Database { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Initializes the database schema
    fn init_schema(&self) -> Result<()> {
        // Add embedding column if missing (safe to run on existing DBs)
        let _ = self
            .conn
            .execute("ALTER TABLE performers ADD COLUMN embedding TEXT", []);

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS aliases (
                alias      TEXT PRIMARY KEY,
                canonical  TEXT NOT NULL
            )",
            [],
        )?;

        // Local face corpus: every candidate we've ever embedded, searchable by
        // face. Grows as you run searches / `warm`, enabling whole-library
        // face-similarity lookups without re-hitting the API or re-embedding.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS candidates (
                name       TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                embedding  TEXT NOT NULL
            )",
            [],
        )?;

        // Cached body-vector index: precomputed pose (frame) and seg (shape/
        // volume) centroids for a roster of performers, so body-search/find can
        // rank against a rich candidate pool instantly instead of re-fetching and
        // re-embedding images every run. Built by `luminary index`.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS body_index (
                name      TEXT PRIMARY KEY,
                data      TEXT NOT NULL,
                pose_vec  BLOB,
                seg_vec   BLOB,
                proj_vec  BLOB,
                bust_vec  BLOB,
                n_frames  INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )?;
        // Add proj_vec (side-view projection centroid) to a pre-existing index.
        let _ = self
            .conn
            .execute("ALTER TABLE body_index ADD COLUMN proj_vec BLOB", []);
        // Add bust_vec (chest shape/projection centroid) to a pre-existing index.
        let _ = self
            .conn
            .execute("ALTER TABLE body_index ADD COLUMN bust_vec BLOB", []);

        // Per-image corpus: one row per gathered image, tagged with its source,
        // view (front/rear/side/…) and a 0–1 quality score, plus the per-image
        // vectors. body_index centroids are derived from this (quality-weighted,
        // view-filtered). Stores vectors + metadata only — never image bytes —
        // so "keep every good image" stays cheap. Grows as sources are added.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS images (
                performer TEXT NOT NULL,
                url       TEXT NOT NULL,
                source    TEXT NOT NULL,
                view      TEXT NOT NULL DEFAULT 'unknown',
                quality   REAL NOT NULL DEFAULT 0,
                pose_vec  BLOB,
                seg_vec   BLOB,
                face_vec  BLOB,
                proj_vec  BLOB,
                bust_vec  BLOB,
                PRIMARY KEY (performer, url)
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_view ON images(performer, view)",
            [],
        )?;
        // Add proj_vec (side-view projection vector) to a pre-existing corpus.
        let _ = self
            .conn
            .execute("ALTER TABLE images ADD COLUMN proj_vec BLOB", []);
        // Add bust_vec (chest shape/projection vector) to a pre-existing corpus.
        let _ = self
            .conn
            .execute("ALTER TABLE images ADD COLUMN bust_vec BLOB", []);

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS performers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                body_type TEXT,
                measurements TEXT,
                height TEXT,
                weight TEXT,
                ethnicity TEXT,
                hair_color TEXT,
                eye_color TEXT,
                age INTEGER,
                birthdate TEXT,
                categories TEXT,
                active_years TEXT,
                profile_image_url TEXT,
                gallery_urls TEXT,
                source TEXT,
                source_url TEXT,
                last_updated TEXT,
                data TEXT
            )",
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_body_type ON performers(body_type)",
            [],
        )?;
        self.conn
            .execute("CREATE INDEX IF NOT EXISTS idx_age ON performers(age)", [])?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ethnicity ON performers(ethnicity)",
            [],
        )?;

        Ok(())
    }

    /// Adds a performer to the database
    pub fn add_performer(&self, performer: &Performer) -> Result<i64> {
        let categories_json = serde_json::to_string(&performer.categories)?;
        let gallery_urls_json = serde_json::to_string(&performer.gallery_urls)?;
        let data_json = serde_json::to_string(performer)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO performers (
                name, body_type, measurements, height, weight,
                ethnicity, hair_color, eye_color, age, birthdate,
                categories, active_years, profile_image_url, gallery_urls,
                source, source_url, last_updated, data
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                performer.name,
                performer.body_type,
                performer.measurements,
                performer.height,
                performer.weight,
                performer.ethnicity,
                performer.hair_color,
                performer.eye_color,
                performer.age,
                performer.birthdate,
                categories_json,
                performer.active_years,
                performer.profile_image_url,
                gallery_urls_json,
                performer.source,
                performer.source_url,
                performer.last_updated,
                data_json,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Upserts a candidate (with its face embedding) into the local corpus.
    pub fn save_candidate(&self, p: &crate::models::Performer, embedding: &[f32]) -> Result<()> {
        let data = serde_json::to_string(p)?;
        let emb = crate::embedder::embedding_to_blob(embedding);
        self.conn.execute(
            "INSERT OR REPLACE INTO candidates (name, data, embedding) VALUES (?1, ?2, ?3)",
            rusqlite::params![p.name, data, emb],
        )?;
        Ok(())
    }

    /// Looks up a cached embedding in either the performers table or the
    /// candidates corpus (performers first).
    pub fn get_embedding_any(&self, name: &str) -> Result<Option<Vec<f32>>> {
        if let Some(e) = self.get_embedding(name)? {
            return Ok(Some(e));
        }
        let mut stmt = self
            .conn
            .prepare("SELECT embedding FROM candidates WHERE name = ?1")?;
        let mut rows = stmt.query(rusqlite::params![name])?;
        if let Some(row) = rows.next()? {
            let val: rusqlite::types::Value = row.get(0)?;
            return Ok(decode_embedding(val));
        }
        Ok(None)
    }

    /// Number of candidates in the local face corpus.
    pub fn candidate_count(&self) -> Result<usize> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM candidates", [], |r| r.get(0))?)
    }

    /// Loads the whole face corpus as (embedding, performer) pairs.
    pub fn load_candidates(&self) -> Result<Vec<(Vec<f32>, crate::models::Performer)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT data, embedding FROM candidates")?;
        let rows = stmt
            .query_map([], |row| {
                let data: String = row.get(0)?;
                let val: rusqlite::types::Value = row.get(1)?;
                Ok((data, val))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut out = Vec::with_capacity(rows.len());
        for (data, val) in rows {
            if let (Ok(p), Some(e)) = (
                serde_json::from_str::<crate::models::Performer>(&data),
                decode_embedding(val),
            ) {
                out.push((e, p));
            }
        }
        Ok(out)
    }

    /// Saves a name alias pointing to a canonical performer name
    pub fn save_alias(&self, alias: &str, canonical: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO aliases (alias, canonical) VALUES (?1, ?2)",
            rusqlite::params![alias.to_lowercase(), canonical],
        )?;
        Ok(())
    }

    /// Resolves an alias to its canonical name, if one exists
    pub fn resolve_alias(&self, name: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT canonical FROM aliases WHERE alias = ?1")?;
        let mut rows = stmt.query(rusqlite::params![name.to_lowercase()])?;
        if let Some(row) = rows.next()? {
            let canonical: String = row.get(0)?;
            return Ok(Some(canonical));
        }
        Ok(None)
    }

    /// Lists all stored aliases
    pub fn list_aliases(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT alias, canonical FROM aliases ORDER BY alias")?;
        let pairs = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(pairs)
    }

    /// Removes an alias
    pub fn remove_alias(&self, alias: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM aliases WHERE alias = ?1",
            rusqlite::params![alias.to_lowercase()],
        )?;
        Ok(())
    }

    /// Gets a performer by name, falling back to alias → word-order match
    pub fn get_performer(&self, name: &str) -> Result<Option<Performer>> {
        // Check alias table first
        if let Some(canonical) = self.resolve_alias(name)? {
            return self.get_performer(&canonical);
        }

        // Exact match first
        let mut stmt = self
            .conn
            .prepare("SELECT data FROM performers WHERE name = ?1")?;
        let mut rows = stmt.query(params![name])?;
        if let Some(row) = rows.next()? {
            let data: String = row.get(0)?;
            return Ok(Some(serde_json::from_str(&data)?));
        }

        // Fallback: all words present in name (any order, case-insensitive)
        let words: Vec<String> = name
            .split_whitespace()
            .map(|w| format!("%{}%", w.to_lowercase()))
            .collect();

        if words.is_empty() {
            return Ok(None);
        }

        let conditions = words
            .iter()
            .enumerate()
            .map(|(i, _)| format!("LOWER(name) LIKE ?{}", i + 1))
            .collect::<Vec<_>>()
            .join(" AND ");

        let query = format!("SELECT data FROM performers WHERE {}", conditions);
        let mut stmt = self.conn.prepare(&query)?;

        let param_refs: Vec<&dyn rusqlite::ToSql> =
            words.iter().map(|w| w as &dyn rusqlite::ToSql).collect();

        let mut rows = stmt.query(param_refs.as_slice())?;
        if let Some(row) = rows.next()? {
            let data: String = row.get(0)?;
            return Ok(Some(serde_json::from_str(&data)?));
        }

        Ok(None)
    }

    /// Best stored Performer metadata for a name *outside* the user's library:
    /// the candidates corpus (from `warm`) or the cached `body_index` entry,
    /// preferring whichever actually carries measurements (an over-aggregated
    /// index row can be bare). Lets `aggregate` keep a roster performer's
    /// attributes instead of blanking them with a name-only record.
    pub fn get_known_performer(&self, name: &str) -> Result<Option<Performer>> {
        let load = |sql: &str| -> Option<Performer> {
            self.conn
                .query_row(sql, params![name], |r| r.get::<_, String>(0))
                .ok()
                .and_then(|d| serde_json::from_str::<Performer>(&d).ok())
        };
        let has_meas = |p: &Performer| p.measurements.as_deref().is_some_and(|m| !m.is_empty());
        let cand = load("SELECT data FROM candidates WHERE name = ?1");
        let idx = load("SELECT data FROM body_index WHERE name = ?1");
        // Primary record: prefer the one carrying measurements (candidates first,
        // then the index); else whichever exists.
        let mut primary = if cand.as_ref().is_some_and(&has_meas) {
            cand.clone()
        } else if idx.as_ref().is_some_and(&has_meas) {
            idx.clone()
        } else {
            cand.clone().or_else(|| idx.clone())
        };
        // Enrich missing stature/mass from the other source. height/weight are
        // backfilled into body_index only (StashDB has no weight; the bulk loader
        // skipped height), so without this a re-`aggregate` — which rebuilds the
        // index row from this record — would drop them whenever the candidates
        // copy (which lacks them) wins as primary.
        if let Some(p) = primary.as_mut() {
            for src in [&cand, &idx].into_iter().flatten() {
                if p.height.is_none() {
                    p.height = src.height.clone();
                }
                if p.weight.is_none() {
                    p.weight = src.weight.clone();
                }
            }
        }
        Ok(primary)
    }

    /// Gets all performers
    pub fn get_all_performers(&self) -> Result<Vec<Performer>> {
        let mut stmt = self.conn.prepare("SELECT data FROM performers")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows.iter().filter_map(|d| decode_performer(d)).collect())
    }

    /// Searches for performers matching filters
    pub fn search(&self, filters: &SearchFilters) -> Result<Vec<Performer>> {
        let mut query = String::from("SELECT data FROM performers WHERE 1=1");
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(body_type) = &filters.body_type {
            query.push_str(" AND body_type = ?");
            params.push(Box::new(body_type.clone()));
        }
        if let Some(ethnicity) = &filters.ethnicity {
            query.push_str(" AND ethnicity = ?");
            params.push(Box::new(ethnicity.clone()));
        }
        if let Some(age_min) = filters.age_min {
            query.push_str(" AND age >= ?");
            params.push(Box::new(age_min));
        }
        if let Some(age_max) = filters.age_max {
            query.push_str(" AND age <= ?");
            params.push(Box::new(age_max));
        }
        if let Some(hair_color) = &filters.hair_color {
            query.push_str(" AND hair_color = ?");
            params.push(Box::new(hair_color.clone()));
        }

        let mut stmt = self.conn.prepare(&query)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = params
            .iter()
            .map(|p| p.as_ref() as &dyn rusqlite::ToSql)
            .collect();

        let performers = stmt
            .query_map(param_refs.as_slice(), |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .filter_map(|d| decode_performer(d))
            .collect();

        Ok(performers)
    }

    /// Saves a face embedding for a performer as a compact f32 BLOB.
    pub fn save_embedding(&self, name: &str, embedding: &[f32]) -> Result<()> {
        let blob = crate::embedder::embedding_to_blob(embedding);
        self.conn.execute(
            "UPDATE performers SET embedding = ?1 WHERE name = ?2",
            rusqlite::params![blob, name],
        )?;
        Ok(())
    }

    /// Retrieves the stored face embedding for a performer (BLOB or legacy JSON).
    pub fn get_embedding(&self, name: &str) -> Result<Option<Vec<f32>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT embedding FROM performers WHERE name = ?1")?;
        let mut rows = stmt.query(rusqlite::params![name])?;
        if let Some(row) = rows.next()? {
            let val: rusqlite::types::Value = row.get(0)?;
            return Ok(decode_embedding(val));
        }
        Ok(None)
    }

    /// Gets all performers that have a face_url stored (needed for embedding generation)
    pub fn get_performers_without_embedding(&self) -> Result<Vec<crate::models::Performer>> {
        let mut stmt = self
            .conn
            .prepare("SELECT data FROM performers WHERE embedding IS NULL")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows.iter().filter_map(|d| decode_performer(d)).collect())
    }

    /// Removes a performer by name
    pub fn remove_performer(&self, name: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM performers WHERE name = ?1", params![name])?;
        Ok(())
    }

    /// Gets the count of performers
    pub fn count(&self) -> Result<usize> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM performers", [], |row| row.get(0))?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Performer;

    #[test]
    fn body_index_roundtrip_and_count() {
        let db = Database::new(":memory:").unwrap();

        let mut p = Performer::new("Test Star".to_string());
        p.measurements = Some("34D-30-42".to_string());
        db.save_body_index(
            &p,
            Some(&[1.0, 2.0, 3.0]),
            Some(&[0.5, 0.6]),
            Some(&[0.9]),
            Some(&[0.7]),
            5,
        )
        .unwrap();
        // Second performer: a pose vector but no clean shape/projection/bust frame.
        db.save_body_index(
            &Performer::new("No Shape".to_string()),
            Some(&[9.0]),
            None,
            None,
            None,
            1,
        )
        .unwrap();

        assert_eq!(db.body_index_count().unwrap(), 2);

        let names = db.body_indexed_names().unwrap();
        assert!(names.contains("test star")); // lowercased for resumability
        assert!(names.contains("no shape"));

        let entries = db.load_body_index().unwrap();
        let star = entries
            .iter()
            .find(|e| e.performer.name == "Test Star")
            .unwrap();
        assert_eq!(star.pose, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(star.seg, Some(vec![0.5, 0.6]));
        assert_eq!(star.proj, Some(vec![0.9]));
        assert_eq!(star.bust, Some(vec![0.7]));
        let no_shape = entries
            .iter()
            .find(|e| e.performer.name == "No Shape")
            .unwrap();
        assert_eq!(no_shape.pose, Some(vec![9.0]));
        assert_eq!(no_shape.seg, None);
        assert_eq!(no_shape.proj, None);
        assert_eq!(no_shape.bust, None);
    }

    #[test]
    fn body_index_upsert_replaces() {
        let db = Database::new(":memory:").unwrap();
        let p = Performer::new("Dup".to_string());
        db.save_body_index(&p, Some(&[1.0]), None, None, None, 1)
            .unwrap();
        db.save_body_index(&p, Some(&[2.0]), None, None, None, 1)
            .unwrap();
        assert_eq!(db.body_index_count().unwrap(), 1);
        assert_eq!(db.load_body_index().unwrap()[0].pose, Some(vec![2.0]));
    }

    #[test]
    fn known_performer_prefers_record_with_measurements() {
        let db = Database::new(":memory:").unwrap();
        // A bare body_index row (what an over-aggregation leaves) plus a candidate
        // record that still carries measurements — prefer the latter.
        db.save_body_index(
            &Performer::new("Roster Star".to_string()),
            Some(&[1.0]),
            None,
            None,
            None,
            1,
        )
        .unwrap();
        let mut full = Performer::new("Roster Star".to_string());
        full.measurements = Some("34D-26-38".to_string());
        db.save_candidate(&full, &[0.1, 0.2]).unwrap();

        let got = db.get_known_performer("Roster Star").unwrap().unwrap();
        assert_eq!(got.measurements.as_deref(), Some("34D-26-38"));
        assert!(db.get_known_performer("Nobody").unwrap().is_none());
    }

    #[test]
    fn known_performer_enriches_height_weight_from_index() {
        let db = Database::new(":memory:").unwrap();
        // body_index carries the backfilled stature/mass; candidates is the
        // measurements-bearing primary but lacks height/weight. A re-aggregate
        // must keep the stature/mass rather than drop it.
        let mut idx = Performer::new("Backfilled Star".to_string());
        idx.height = Some("168cm".to_string());
        idx.weight = Some("60kg".to_string());
        db.save_body_index(&idx, Some(&[1.0]), None, None, None, 1)
            .unwrap();
        let mut cand = Performer::new("Backfilled Star".to_string());
        cand.measurements = Some("34D-26-38".to_string());
        db.save_candidate(&cand, &[0.1, 0.2]).unwrap();

        let got = db.get_known_performer("Backfilled Star").unwrap().unwrap();
        assert_eq!(got.measurements.as_deref(), Some("34D-26-38")); // from candidates
        assert_eq!(got.height.as_deref(), Some("168cm")); // enriched from index
        assert_eq!(got.weight.as_deref(), Some("60kg"));
    }

    fn img(url: &str, view: &str, face: Option<Vec<f32>>) -> ImageRow {
        ImageRow {
            performer: "Star".to_string(),
            url: url.to_string(),
            source: "pornpics".to_string(),
            view: view.to_string(),
            quality: 0.8,
            pose: None,
            seg: None,
            face,
            proj: None,
            bust: None,
        }
    }

    #[test]
    fn images_roundtrip_view_filter_and_upsert() {
        let db = Database::new(":memory:").unwrap();
        db.save_image(&img("u1", "front", Some(vec![3.0]))).unwrap();
        db.save_image(&img("u2", "rear", None)).unwrap();

        assert_eq!(db.images_count().unwrap(), 2);
        assert_eq!(db.load_images("Star", None).unwrap().len(), 2);

        let front = db.load_images("Star", Some("front")).unwrap();
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].face, Some(vec![3.0]));

        let urls = db.existing_image_urls("Star").unwrap();
        assert!(urls.contains("u1") && urls.contains("u2"));

        // Re-saving the same (performer, url) replaces rather than duplicates.
        db.save_image(&img("u1", "side", None)).unwrap();
        assert_eq!(db.images_count().unwrap(), 2);
        assert_eq!(db.load_images("Star", Some("side")).unwrap().len(), 1);
    }

    fn vrow(
        view: &str,
        quality: f32,
        pose: Option<Vec<f32>>,
        seg: Option<Vec<f32>>,
        proj: Option<Vec<f32>>,
    ) -> ImageRow {
        ImageRow {
            performer: "Star".to_string(),
            url: format!("u-{}-{}", view, quality),
            source: "pornpics".to_string(),
            view: view.to_string(),
            quality,
            pose,
            seg,
            face: None,
            proj,
            bust: None,
        }
    }

    #[test]
    fn aggregate_weights_by_quality_and_splits_views() {
        let images = vec![
            vrow("front", 3.0, Some(vec![4.0]), Some(vec![2.0]), None),
            vrow("rear", 1.0, Some(vec![0.0]), None, None),
            // A side frame feeds the projection centroid; its (bogus) pose/seg
            // must NOT reach the frontal centroids.
            vrow(
                "side",
                2.0,
                Some(vec![100.0]),
                Some(vec![100.0]),
                Some(vec![7.0]),
            ),
        ];
        let (pose, seg, proj, _bust, n) = aggregate_views(&images);
        assert_eq!(pose, Some(vec![3.0])); // (4*3 + 0*1)/4, side excluded
        assert_eq!(seg, Some(vec![2.0])); // only the front frame carries seg
        assert_eq!(proj, Some(vec![7.0])); // from the side frame only
        assert_eq!(n, 2); // two frontal pose frames

        // Only side frames: no frontal pose/seg, but projection still aggregates.
        let only_side = vec![vrow("side", 1.0, None, None, Some(vec![5.0]))];
        let (p, s, pr, _bust, n) = aggregate_views(&only_side);
        assert!(p.is_none() && s.is_none());
        assert_eq!(pr, Some(vec![5.0]));
        assert_eq!(n, 1); // n spans all views, so a side-only performer still counts
    }
}
