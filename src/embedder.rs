use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

/// Calls face_embed.py to generate a 512-dim ArcFace embedding for an image URL.
pub fn generate_embedding(image_url: &str) -> Result<Vec<f32>> {
    let script = find_script()?;
    let python  = find_python()?;

    log::info!("Generating face embedding: {}", image_url);

    let output = Command::new(&python)
        .arg(&script)
        .arg(image_url)
        .output()
        .with_context(|| format!("Failed to run {} {}", python, script.display()))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("face_embed.py produced no output.\nstderr: {}", stderr.trim());
    }

    let result: serde_json::Value = serde_json::from_str(stdout.trim())
        .with_context(|| format!("Could not parse face_embed.py output: {}", stdout))?;

    if let Some(err) = result.get("error").and_then(|e| e.as_str()) {
        anyhow::bail!("{}", err);
    }

    let embedding: Vec<f32> = result["embedding"]
        .as_array()
        .context("No 'embedding' field in face_embed.py output")?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    if embedding.is_empty() {
        anyhow::bail!("Empty embedding returned from face_embed.py");
    }

    Ok(embedding)
}

/// Cosine similarity between two embedding vectors (range −1..1, higher = more similar).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot:    f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

/// Converts similarity (−1..1) to a 0–100% display score.
pub fn similarity_pct(sim: f32) -> f32 {
    ((sim + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0)
}

pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    serde_json::to_string(embedding)
        .unwrap_or_default()
        .into_bytes()
}

pub fn blob_to_embedding(blob: &[u8]) -> Option<Vec<f32>> {
    let s = std::str::from_utf8(blob).ok()?;
    serde_json::from_str(s).ok()
}

fn find_python() -> Result<String> {
    for candidate in &["python3", "python"] {
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
        "Python not found. Install Python 3 and run:\n  pip install deepface tf-keras"
    )
}

fn find_script() -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = [
        std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("face_embed.py"))),
        std::env::current_dir().ok().map(|d| d.join("face_embed.py")),
        Some(PathBuf::from("face_embed.py")),
    ]
    .into_iter()
    .flatten()
    .collect();

    candidates.into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!(
            "face_embed.py not found. Place it alongside the starfinder binary."
        ))
}
