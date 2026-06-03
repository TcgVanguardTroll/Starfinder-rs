use crate::models::{Performer, SearchFilters};
use anyhow::{Context, Result};
use rusqlite::{params, Connection};

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

/// One cached entry in the body-vector index: a performer plus their (optional)
/// pose/frame and seg/shape centroid vectors.
pub struct BodyIndexEntry {
    pub performer: Performer,
    pub pose: Option<Vec<f32>>,
    pub seg: Option<Vec<f32>>,
    /// Posterior-projection centroid, aggregated from `side` frames (None until
    /// the performer has been ingested with profile shots).
    pub proj: Option<Vec<f32>>,
}

/// One row of the per-image corpus: a single gathered image, its provenance
/// (`source`), classified `view`, 0–1 `quality`, and the per-image vectors.
pub struct ImageRow {
    pub performer: String,
    pub url: String,
    pub source: String,
    pub view: String,
    pub quality: f32,
    pub pose: Option<Vec<f32>>,
    pub seg: Option<Vec<f32>>,
    pub face: Option<Vec<f32>>,
    /// Side-view posterior-projection vector (set only for `side` frames).
    pub proj: Option<Vec<f32>>,
}

/// The pose, seg, and proj centroids plus the contributing frame count returned
/// by [`aggregate_views`].
pub type AggregatedViews = (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, usize);

/// Aggregates a performer's per-image corpus into quality-weighted body
/// centroids for the cached index. Frontal views (front/rear) feed the pose and
/// seg centroids — a side/profile frame collapses shoulder width and corrupts
/// both ratio vectors — while `side` frames feed the posterior-projection
/// centroid that only a profile can reveal. Each image contributes in proportion
/// to its `quality`. Returns `(pose, seg, proj, n_frames)`, where `n_frames` is
/// the largest number of frames feeding any one centroid.
pub fn aggregate_views(images: &[ImageRow]) -> AggregatedViews {
    let is_frontal = |v: &str| v == "front" || v == "rear";
    let pose: Vec<(Vec<f32>, f32)> = images
        .iter()
        .filter(|im| is_frontal(&im.view))
        .filter_map(|im| im.pose.clone().map(|p| (p, im.quality)))
        .collect();
    let seg: Vec<(Vec<f32>, f32)> = images
        .iter()
        .filter(|im| is_frontal(&im.view))
        .filter_map(|im| im.seg.clone().map(|s| (s, im.quality)))
        .collect();
    let proj: Vec<(Vec<f32>, f32)> = images
        .iter()
        .filter(|im| im.view == "side")
        .filter_map(|im| im.proj.clone().map(|p| (p, im.quality)))
        .collect();
    let n = pose.len().max(seg.len()).max(proj.len());
    (
        crate::embedder::weighted_body_centroid(&pose),
        crate::embedder::weighted_body_centroid(&seg),
        crate::embedder::weighted_body_centroid(&proj),
        n,
    )
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
                n_frames  INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )?;
        // Add proj_vec (side-view projection centroid) to a pre-existing index.
        let _ = self
            .conn
            .execute("ALTER TABLE body_index ADD COLUMN proj_vec BLOB", []);

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

    /// Upserts a performer's cached body vectors into the index. Either vector
    /// may be None (no clean frame of that kind); the row is still written so a
    /// resumed `index` run skips this performer rather than re-fetching them.
    pub fn save_body_index(
        &self,
        p: &Performer,
        pose: Option<&[f32]>,
        seg: Option<&[f32]>,
        proj: Option<&[f32]>,
        n_frames: usize,
    ) -> Result<()> {
        let data = serde_json::to_string(p)?;
        let pose_blob = pose.map(crate::embedder::embedding_to_blob);
        let seg_blob = seg.map(crate::embedder::embedding_to_blob);
        let proj_blob = proj.map(crate::embedder::embedding_to_blob);
        self.conn.execute(
            "INSERT OR REPLACE INTO body_index (name, data, pose_vec, seg_vec, proj_vec, n_frames)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                p.name,
                data,
                pose_blob,
                seg_blob,
                proj_blob,
                n_frames as i64
            ],
        )?;
        Ok(())
    }

    /// Lowercased names already present in the body index (for resumable builds).
    pub fn body_indexed_names(&self) -> Result<std::collections::HashSet<String>> {
        let mut stmt = self.conn.prepare("SELECT name FROM body_index")?;
        let names = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(names.into_iter().map(|n| n.to_lowercase()).collect())
    }

    /// Number of performers in the body index.
    pub fn body_index_count(&self) -> Result<usize> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM body_index", [], |r| r.get(0))?)
    }

    /// Loads the whole body index (performer + optional pose/seg/proj vectors).
    pub fn load_body_index(&self) -> Result<Vec<BodyIndexEntry>> {
        let mut stmt = self
            .conn
            .prepare("SELECT data, pose_vec, seg_vec, proj_vec FROM body_index")?;
        let rows = stmt
            .query_map([], |row| {
                let data: String = row.get(0)?;
                let pose: rusqlite::types::Value = row.get(1)?;
                let seg: rusqlite::types::Value = row.get(2)?;
                let proj: rusqlite::types::Value = row.get(3)?;
                Ok((data, pose, seg, proj))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut out = Vec::with_capacity(rows.len());
        for (data, pose, seg, proj) in rows {
            if let Ok(p) = serde_json::from_str::<Performer>(&data) {
                out.push(BodyIndexEntry {
                    performer: p,
                    pose: decode_embedding(pose),
                    seg: decode_embedding(seg),
                    proj: decode_embedding(proj),
                });
            }
        }
        Ok(out)
    }

    /// Upserts one image row (keyed by performer + url) into the corpus.
    pub fn save_image(&self, img: &ImageRow) -> Result<()> {
        let pose = img.pose.as_deref().map(crate::embedder::embedding_to_blob);
        let seg = img.seg.as_deref().map(crate::embedder::embedding_to_blob);
        let face = img.face.as_deref().map(crate::embedder::embedding_to_blob);
        let proj = img.proj.as_deref().map(crate::embedder::embedding_to_blob);
        self.conn.execute(
            "INSERT OR REPLACE INTO images
                (performer, url, source, view, quality, pose_vec, seg_vec, face_vec, proj_vec)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                img.performer,
                img.url,
                img.source,
                img.view,
                img.quality,
                pose,
                seg,
                face,
                proj,
            ],
        )?;
        Ok(())
    }

    /// URLs already stored for a performer — so ingest can skip re-embedding
    /// images it has already seen (the corpus grows incrementally).
    pub fn existing_image_urls(
        &self,
        performer: &str,
    ) -> Result<std::collections::HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT url FROM images WHERE performer = ?1")?;
        let urls = stmt
            .query_map(rusqlite::params![performer], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(urls.into_iter().collect())
    }

    /// Loads a performer's images, optionally filtered to one `view`
    /// (front/rear/side/…). Pass `None` for all views.
    pub fn load_images(&self, performer: &str, view: Option<&str>) -> Result<Vec<ImageRow>> {
        let (sql, has_view) = match view {
            Some(_) => (
                "SELECT performer, url, source, view, quality, pose_vec, seg_vec, face_vec, proj_vec
                 FROM images WHERE performer = ?1 AND view = ?2",
                true,
            ),
            None => (
                "SELECT performer, url, source, view, quality, pose_vec, seg_vec, face_vec, proj_vec
                 FROM images WHERE performer = ?1",
                false,
            ),
        };
        let mut stmt = self.conn.prepare(sql)?;
        let map_row = |row: &rusqlite::Row| {
            Ok(ImageRow {
                performer: row.get(0)?,
                url: row.get(1)?,
                source: row.get(2)?,
                view: row.get(3)?,
                quality: row.get(4)?,
                pose: decode_embedding(row.get(5)?),
                seg: decode_embedding(row.get(6)?),
                face: decode_embedding(row.get(7)?),
                proj: decode_embedding(row.get(8)?),
            })
        };
        let rows = if has_view {
            stmt.query_map(rusqlite::params![performer, view.unwrap()], map_row)?
                .collect::<Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(rusqlite::params![performer], map_row)?
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(rows)
    }

    /// Total number of images in the corpus.
    pub fn images_count(&self) -> Result<usize> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))?)
    }

    /// Distinct performer names with at least one image in the corpus — the set
    /// `aggregate` rebuilds when no explicit names are given.
    pub fn images_performers(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT performer FROM images ORDER BY performer")?;
        let names = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(names)
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

    /// Gets all performers
    pub fn get_all_performers(&self) -> Result<Vec<Performer>> {
        let mut stmt = self.conn.prepare("SELECT data FROM performers")?;
        let performers = stmt
            .query_map([], |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str(&data).unwrap())
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(performers)
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
            .query_map(param_refs.as_slice(), |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str(&data).unwrap())
            })?
            .collect::<Result<Vec<_>, _>>()?;

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
        let performers = stmt
            .query_map([], |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str(&data).unwrap())
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(performers)
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
            5,
        )
        .unwrap();
        // Second performer: a pose vector but no clean shape or projection frame.
        db.save_body_index(
            &Performer::new("No Shape".to_string()),
            Some(&[9.0]),
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
        let no_shape = entries
            .iter()
            .find(|e| e.performer.name == "No Shape")
            .unwrap();
        assert_eq!(no_shape.pose, Some(vec![9.0]));
        assert_eq!(no_shape.seg, None);
        assert_eq!(no_shape.proj, None);
    }

    #[test]
    fn body_index_upsert_replaces() {
        let db = Database::new(":memory:").unwrap();
        let p = Performer::new("Dup".to_string());
        db.save_body_index(&p, Some(&[1.0]), None, None, 1).unwrap();
        db.save_body_index(&p, Some(&[2.0]), None, None, 1).unwrap();
        assert_eq!(db.body_index_count().unwrap(), 1);
        assert_eq!(db.load_body_index().unwrap()[0].pose, Some(vec![2.0]));
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
        let (pose, seg, proj, n) = aggregate_views(&images);
        assert_eq!(pose, Some(vec![3.0])); // (4*3 + 0*1)/4, side excluded
        assert_eq!(seg, Some(vec![2.0])); // only the front frame carries seg
        assert_eq!(proj, Some(vec![7.0])); // from the side frame only
        assert_eq!(n, 2); // two frontal pose frames

        // Only side frames: no frontal pose/seg, but projection still aggregates.
        let only_side = vec![vrow("side", 1.0, None, None, Some(vec![5.0]))];
        let (p, s, pr, n) = aggregate_views(&only_side);
        assert!(p.is_none() && s.is_none());
        assert_eq!(pr, Some(vec![5.0]));
        assert_eq!(n, 1); // n spans all views, so a side-only performer still counts
    }
}
