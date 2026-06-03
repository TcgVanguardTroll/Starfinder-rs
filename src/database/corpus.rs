//! Per-image corpus (`images`) and cached body-vector index (`body_index`)
//! storage, split out of the core Database store.
use super::{decode_embedding, Database};
use crate::models::Performer;
use anyhow::Result;

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

impl Database {
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
}
