use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

/// Generates ArcFace embeddings for many image URLs in ONE sidecar call.
/// The Python model loads once and processes all URLs, so this is far cheaper
/// than calling `generate_embedding` per image. Returns one slot per input URL,
/// in order: `Some(vec)` on success, `None` if no face was detected/decoded.
pub fn generate_embeddings(image_urls: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
    run_sidecar("face_embed.py", "embedding", &[], image_urls)
}

/// Generates body-shape (pose) vectors for many image URLs via body_embed.py,
/// in one batched call. `Some(vec)` per URL where a pose was detected, else None.
pub fn generate_body_embeddings(image_urls: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
    run_sidecar("body_embed.py", "body", &[], image_urls)
}

/// Generates silhouette *volume* vectors (waist/hip/thigh widths from the body
/// outline) via body_embed.py `--seg`. Captures glute & thigh fullness that the
/// skeletal pose vector and measurements both miss. `Some(vec)` per URL with a
/// clean full-body standing silhouette, else None.
pub fn generate_seg_embeddings(image_urls: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
    run_sidecar("body_embed.py", "seg", &["--seg"], image_urls)
}

/// One image's dual result: `(pose/frame, seg/shape)`, each present only if that
/// vector passed its gates.
pub type DualVec = (Option<Vec<f32>>, Option<Vec<f32>>);

/// Generates BOTH the pose/frame and seg/shape vectors in a single `--seg` pass
/// (the seg sidecar computes landmarks anyway, so the pose vector is free).
/// Returns `(pose, seg)` per URL — used by the index builder so one embedding
/// pass populates both search modes. Halves the cost vs running them separately.
pub fn generate_dual_embeddings(image_urls: &[String]) -> Result<Vec<DualVec>> {
    let arr = run_sidecar_raw("body_embed.py", &["--seg"], image_urls)?;
    Ok(arr
        .into_iter()
        .map(|entry| (extract_field(&entry, "body"), extract_field(&entry, "seg")))
        .collect())
}

/// Pulls a `[floats]` field out of one sidecar JSON entry, or None if absent/empty.
fn extract_field(entry: &serde_json::Value, field: &str) -> Option<Vec<f32>> {
    let v: Vec<f32> = entry
        .get(field)?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();
    (!v.is_empty()).then_some(v)
}

/// Runs a Python embedding sidecar and parses its stdout into the per-URL JSON
/// array (`[{<field>: [floats]} | {error: ...}, ...]`), in input order.
///
/// Retries once on failure: a single slow or failed image download can make the
/// whole batch produce no parseable output, which callers would otherwise
/// surface as a misleading "no face/pose detected". A clean retry almost always
/// succeeds, so transient hiccups don't look like empty results.
fn run_sidecar_raw(
    script_name: &str,
    extra_args: &[&str],
    image_urls: &[String],
) -> Result<Vec<serde_json::Value>> {
    if image_urls.is_empty() {
        return Ok(Vec::new());
    }
    let script = find_script(script_name)?;
    let python = find_python()?;

    let mut last_err = None;
    for attempt in 1..=2 {
        log::info!(
            "{}: {} image(s), attempt {}",
            script_name,
            image_urls.len(),
            attempt
        );
        match run_sidecar_attempt(&python, &script, extra_args, image_urls) {
            Ok(values) => return Ok(values),
            Err(e) => {
                log::warn!("{} attempt {} failed: {}", script_name, attempt, e);
                last_err = Some(e);
            }
        }
    }
    Err(last_err.expect("the retry loop always runs at least once"))
}

/// A single invocation of an embedding sidecar.
fn run_sidecar_attempt(
    python: &str,
    script: &std::path::Path,
    extra_args: &[&str],
    image_urls: &[String],
) -> Result<Vec<serde_json::Value>> {
    let output = Command::new(python)
        .arg(script)
        .args(extra_args)
        .args(image_urls)
        .output()
        .with_context(|| format!("Failed to run {} {}", python, script.display()))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("produced no output. stderr: {}", stderr.trim());
    }

    serde_json::from_str(stdout.trim())
        .with_context(|| format!("could not parse sidecar output: {}", stdout))
}

/// Runs a sidecar and extracts a single named float-array field per URL.
fn run_sidecar(
    script_name: &str,
    field: &str,
    extra_args: &[&str],
    image_urls: &[String],
) -> Result<Vec<Option<Vec<f32>>>> {
    let arr = run_sidecar_raw(script_name, extra_args, image_urls)?;
    Ok(arr
        .iter()
        .map(|entry| extract_field(entry, field))
        .collect())
}

/// Averages body-proportion vectors (plain mean — these are ratios, not unit
/// embeddings). Stabilises the pose-dependent noise of single 2D images.
pub fn body_centroid(vecs: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vecs.first()?;
    let dims = first.len();
    let mut sum = vec![0.0_f32; dims];
    let mut count = 0usize;
    for v in vecs {
        if v.len() != dims {
            continue;
        }
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += *x;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    for s in sum.iter_mut() {
        *s /= count as f32;
    }
    Some(sum)
}

/// Body-proportion similarity as a 0–100%. Per-dimension weighted Euclidean
/// distance mapped to a percentage (closer frame ⇒ higher).
pub fn body_similarity_pct(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let d: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    // Empirically, distances run ~0 (identical) to ~1.2 (very different builds).
    let sim = 1.0 - (d as f64 / 1.2).clamp(0.0, 1.0);
    (sim * 100.0).round()
}

/// Silhouette *volume* similarity as a 0–100%. Same weighted-Euclidean→percent
/// idea as `body_similarity_pct`, but calibrated for the seg vector, whose
/// components (width ratios ~1.0–1.6) spread wider than the pose ratios — so the
/// "very different" distance is larger. Divisor is a rough calibration; ranking
/// order (what we use) is robust to its exact value.
pub fn seg_similarity_pct(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let d: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    let sim = 1.0 - (d as f64 / 2.0).clamp(0.0, 1.0);
    (sim * 100.0).round()
}

/// Generates a single embedding (convenience wrapper over the batch API).
pub fn generate_embedding(image_url: &str) -> Result<Vec<f32>> {
    let urls = [image_url.to_string()];
    generate_embeddings(&urls)?
        .into_iter()
        .next()
        .flatten()
        .context("No face detected in image")
}

/// L2-normalises a vector in place (unit length).
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Generates a centroid embedding by averaging the face embeddings of several
/// images of the same person. Each embedding is L2-normalised before averaging
/// so no single photo dominates. More angles/lighting ⇒ a more robust face
/// signature.
///
/// Distinguishes the two "no embedding" outcomes so callers can report honestly
/// instead of always blaming the face:
/// - `Err(_)`   — the sidecar call itself failed (both attempts errored, e.g. a
///   transient model-load or image-download hiccup); retrying usually succeeds.
/// - `Ok(None)` — the sidecar ran fine but no image yielded a usable face.
pub fn generate_centroid_embedding(image_urls: &[String]) -> Result<Option<Vec<f32>>> {
    // One batched sidecar call for all images (model loads once).
    let embeddings = generate_embeddings(image_urls)?;

    let mut sum: Vec<f32> = Vec::new();
    let mut count = 0usize;
    for mut emb in embeddings.into_iter().flatten() {
        normalize(&mut emb);
        if sum.is_empty() {
            sum = emb;
        } else if sum.len() == emb.len() {
            for (s, e) in sum.iter_mut().zip(emb.iter()) {
                *s += *e;
            }
        } else {
            continue; // dimension mismatch, skip
        }
        count += 1;
    }

    if count == 0 {
        return Ok(None);
    }
    for x in sum.iter_mut() {
        *x /= count as f32;
    }
    normalize(&mut sum);
    log::info!("Centroid embedding from {} image(s)", count);
    Ok(Some(sum))
}

/// Averages several stored embeddings into one L2-normalised centroid, so a
/// search can match a *blend* of multiple reference faces. Returns None if empty.
pub fn average_embeddings(embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
    let mut sum: Vec<f32> = Vec::new();
    let mut count = 0usize;
    for emb in embeddings {
        let mut e = emb.clone();
        normalize(&mut e);
        if sum.is_empty() {
            sum = e;
        } else if sum.len() == e.len() {
            for (s, x) in sum.iter_mut().zip(e.iter()) {
                *s += *x;
            }
        } else {
            continue;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    for x in sum.iter_mut() {
        *x /= count as f32;
    }
    normalize(&mut sum);
    Some(sum)
}

/// Cosine similarity between two embedding vectors (range −1..1, higher = more similar).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Converts similarity (−1..1) to a 0–100% display score.
pub fn similarity_pct(sim: f32) -> f32 {
    ((sim + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0)
}

/// Serialises an embedding to a compact little-endian f32 BLOB (~4x smaller
/// and faster to parse than JSON text).
pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(embedding.len() * 4);
    for f in embedding {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decodes an embedding BLOB. Backward-compatible: if the bytes are actually
/// legacy JSON text (start with '['), parse that instead.
pub fn blob_to_embedding(blob: &[u8]) -> Option<Vec<f32>> {
    if blob.first() == Some(&b'[') {
        return serde_json::from_slice(blob).ok();
    }
    if blob.is_empty() || !blob.len().is_multiple_of(4) {
        return None;
    }
    Some(
        blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn find_python() -> Result<String> {
    for candidate in &["python3", "python", "py"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Ok(candidate.to_string());
        }
    }
    anyhow::bail!(
        "Python not found. Install Python 3 and run:\n  pip install insightface onnxruntime"
    )
}

fn find_script(name: &str) -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = [
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join(name))),
        std::env::current_dir().ok().map(|d| d.join(name)),
        Some(PathBuf::from(name)),
    ]
    .into_iter()
    .flatten()
    .collect();

    candidates.into_iter().find(|p| p.exists()).ok_or_else(|| {
        anyhow::anyhow!(
            "{} not found. Place it alongside the luminary binary.",
            name
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_are_perfectly_similar() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
        assert!((similarity_pct(cosine_similarity(&v, &v)) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn opposite_vectors_are_least_similar() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6); // -1
        assert!(similarity_pct(-1.0) < 1.0); // ~0%
    }

    #[test]
    fn orthogonal_vectors_are_midpoint() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6); // 0
        assert!((similarity_pct(0.0) - 50.0).abs() < 1e-4); // 50%
    }

    #[test]
    fn mismatched_or_empty_vectors_are_zero() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn extract_field_reads_skips_and_handles_empty() {
        let arr: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"body":[1.0,2.0],"seg":[3.0]},{"error":"no pose"},{"seg":[]}]"#,
        )
        .unwrap();
        // dual entry: both fields present
        assert_eq!(extract_field(&arr[0], "body"), Some(vec![1.0, 2.0]));
        assert_eq!(extract_field(&arr[0], "seg"), Some(vec![3.0]));
        // error entry: field absent
        assert_eq!(extract_field(&arr[1], "body"), None);
        // present-but-empty array decodes to None, not Some(vec![])
        assert_eq!(extract_field(&arr[2], "seg"), None);
    }

    #[test]
    fn seg_similarity_orders_by_shape_distance() {
        let a = vec![1.0, 1.5, 1.4, 1.4, 1.0]; // curvy reference
        let near = vec![1.05, 1.45, 1.45, 1.35, 0.95]; // similar build
        let far = vec![0.9, 1.0, 0.9, 1.1, 0.9]; // straighter, slimmer
        assert_eq!(seg_similarity_pct(&a, &a), 100.0);
        assert!(seg_similarity_pct(&a, &near) > seg_similarity_pct(&a, &far));
        assert_eq!(seg_similarity_pct(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn blob_round_trip() {
        let v = vec![0.5_f32, -0.25, 1.0];
        let blob = embedding_to_blob(&v);
        // f32 LE bytes: 3 floats × 4 bytes
        assert_eq!(blob.len(), 12);
        assert_eq!(blob_to_embedding(&blob), Some(v));
    }

    #[test]
    fn blob_reads_legacy_json() {
        // Old rows stored embeddings as JSON text — must still decode.
        let legacy = b"[0.5,-0.25,1.0]";
        assert_eq!(blob_to_embedding(legacy), Some(vec![0.5_f32, -0.25, 1.0]));
    }
}
