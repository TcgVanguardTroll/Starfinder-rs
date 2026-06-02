use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

/// Generates ArcFace embeddings for many image URLs in ONE sidecar call.
/// The Python model loads once and processes all URLs, so this is far cheaper
/// than calling `generate_embedding` per image. Returns one slot per input URL,
/// in order: `Some(vec)` on success, `None` if no face was detected/decoded.
pub fn generate_embeddings(image_urls: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
    if image_urls.is_empty() {
        return Ok(Vec::new());
    }
    let script = find_script()?;
    let python = find_python()?;

    log::info!("Embedding {} image(s) in one batch", image_urls.len());

    let output = Command::new(&python)
        .arg(&script)
        .args(image_urls)
        .output()
        .with_context(|| format!("Failed to run {} {}", python, script.display()))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "face_embed.py produced no output.\nstderr: {}",
            stderr.trim()
        );
    }

    let arr: Vec<serde_json::Value> = serde_json::from_str(stdout.trim())
        .with_context(|| format!("Could not parse face_embed.py output: {}", stdout))?;

    Ok(arr
        .into_iter()
        .map(|entry| {
            entry.get("embedding").and_then(|e| e.as_array()).map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Vec<f32>>()
            })
        })
        .map(|opt| opt.filter(|v| !v.is_empty()))
        .collect())
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
/// signature. Returns None if no image yielded a detectable face.
pub fn generate_centroid_embedding(image_urls: &[String]) -> Option<Vec<f32>> {
    // One batched sidecar call for all images (model loads once).
    let embeddings = generate_embeddings(image_urls).ok()?;

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
        return None;
    }
    for x in sum.iter_mut() {
        *x /= count as f32;
    }
    normalize(&mut sum);
    log::info!("Centroid embedding from {} image(s)", count);
    Some(sum)
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

fn find_script() -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = [
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("face_embed.py"))),
        std::env::current_dir()
            .ok()
            .map(|d| d.join("face_embed.py")),
        Some(PathBuf::from("face_embed.py")),
    ]
    .into_iter()
    .flatten()
    .collect();

    candidates.into_iter().find(|p| p.exists()).ok_or_else(|| {
        anyhow::anyhow!("face_embed.py not found. Place it alongside the luminary binary.")
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
