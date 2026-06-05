//! `eval` — offline ranking-quality of the `overall` blend, measured against the
//! user's own liked set (leave-one-out, no network).
//!
//! For every liked performer that also has a body-index entry, we query the blend
//! with that performer and rank the rest of the index; the *other* liked
//! performers are the relevant set. Averaging precision@k / recall@k / MAP /
//! NDCG@k over those queries gives a single number to tune the blend weights,
//! proj calibration, etc. against (the objective the learned-weights work builds
//! on) instead of eyeballing one ranking.
//!
//! The per-candidate modality scoring mirrors `search/overall.rs`; if that blend
//! changes, keep this in step (a shared scorer is the eventual cleanup).
use super::*;
use std::collections::{HashMap, HashSet};

pub(crate) fn eval_quality(db: &Database) -> anyhow::Result<()> {
    let index = db.load_body_index()?;
    if index.is_empty() {
        anyhow::bail!("body index is empty — run 'luminary index' (or 'ingest' + 'aggregate').");
    }
    let liked: HashSet<String> = db
        .get_all_performers()?
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();

    // Queries: liked performers that are in the index (so the blend can rank them
    // and the other liked-in-index members form a non-trivial relevant set).
    let queries: Vec<&luminary::database::BodyIndexEntry> = index
        .iter()
        .filter(|e| liked.contains(&e.performer.name.to_lowercase()))
        .collect();
    if queries.len() < 2 {
        anyhow::bail!(
            "need at least 2 liked performers present in the body index to evaluate \
             (found {}). Like more indexed performers, or ingest some of your likes.",
            queries.len()
        );
    }
    let total_relevant = queries.len() - 1; // every other liked-in-index entry

    println!(
        "{}",
        format!(
            "Evaluating overall-blend ranking on {} liked performers in the index \
             (leave-one-out, {} relevant each)...",
            queries.len(),
            total_relevant
        )
        .bright_cyan()
        .bold()
    );
    println!();

    // Memoise face embeddings — the same ~1000 candidates are scored for every
    // query, so a per-name cache turns N*M lookups into M.
    let mut face_cache: HashMap<String, Option<Vec<f32>>> = HashMap::new();
    let mut face_of = |db: &Database, name: &str| -> Option<Vec<f32>> {
        face_cache
            .entry(name.to_string())
            .or_insert_with(|| db.get_embedding_any(name).ok().flatten())
            .clone()
    };

    let weights = blend::Weights::default();
    let mut per_query: Vec<(Vec<bool>, usize)> = Vec::new();
    let mut rows: Vec<(String, f64, usize)> = Vec::new(); // (name, ndcg@10, rank of first hit)

    for q in &queries {
        let ref_face = face_of(db, &q.performer.name);
        let ref_meas = recommender::feature_vector(&q.performer);
        let ref_height = recommender::performer_height_cm(&q.performer);
        let q_lc = q.performer.name.to_lowercase();

        let scored: Vec<(blend::ModalityScores, String)> = index
            .iter()
            .filter(|e| e.performer.name.to_lowercase() != q_lc)
            .map(|e| {
                let face = match (&ref_face, face_of(db, &e.performer.name)) {
                    (Some(rf), Some(cf)) => {
                        Some(embedder::similarity_pct(embedder::cosine_similarity(rf, &cf)) as f64)
                    }
                    _ => None,
                };
                let build = match (&q.pose, &e.pose) {
                    (Some(r), Some(c)) => Some(embedder::body_similarity_pct(r, c)),
                    _ => None,
                };
                let volume = match (&q.seg, &e.seg) {
                    (Some(r), Some(c)) => Some(embedder::seg_similarity_pct(r, c)),
                    _ => None,
                };
                let proj = match (&q.proj, &e.proj) {
                    (Some(r), Some(c))
                        if embedder::is_plausible_proj(r) && embedder::is_plausible_proj(c) =>
                    {
                        Some(embedder::proj_similarity_pct(r, c))
                    }
                    _ => None,
                };
                let bust = match (&q.bust, &e.bust) {
                    (Some(r), Some(c))
                        if embedder::is_plausible_proj(r) && embedder::is_plausible_proj(c) =>
                    {
                        Some(embedder::bust_similarity_pct(r, c))
                    }
                    _ => None,
                };
                let meas = match (&ref_meas, recommender::feature_vector(&e.performer)) {
                    (Some(r), Some(c)) => Some(r.similarity_pct(&c)),
                    _ => None,
                };
                let height = match (ref_height, recommender::performer_height_cm(&e.performer)) {
                    (Some(r), Some(c)) => Some((100.0 * (1.0 - (r - c).abs() / 40.0)).max(0.0)),
                    _ => None,
                };
                (
                    blend::ModalityScores {
                        face,
                        build,
                        volume,
                        proj,
                        bust,
                        meas,
                        height,
                    },
                    e.performer.name.clone(),
                )
            })
            .collect();

        let raw: Vec<blend::ModalityScores> = scored.iter().map(|(m, _)| m.clone()).collect();
        let blended = blend::blend_scores(&raw, &weights);
        let mut ranked: Vec<(f64, String)> = blended
            .into_iter()
            .zip(scored.into_iter().map(|(_, n)| n))
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let rels: Vec<bool> = ranked
            .iter()
            .map(|(_, n)| liked.contains(&n.to_lowercase()))
            .collect();
        let first_hit = rels.iter().position(|r| *r).map(|i| i + 1).unwrap_or(0);
        rows.push((
            q.performer.name.clone(),
            luminary::eval::ndcg_at_k(&rels, 10),
            first_hit,
        ));
        per_query.push((rels, total_relevant));
    }

    // Per-query transparency: where did each liked query's best match land?
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, ndcg, first) in &rows {
        let hit = if *first == 0 {
            "no liked match in list".to_string()
        } else {
            format!("first liked match @ rank {first}")
        };
        println!(
            "  {:<22} {}  {}",
            name.bright_white(),
            format!("nDCG@10 {:.2}", ndcg).bright_cyan(),
            hit.bright_black()
        );
    }

    let s = luminary::eval::aggregate(&per_query);
    println!();
    println!("{}", "Aggregate (mean over queries):".bright_cyan().bold());
    println!("  {:<14} {:.3}", "Precision@5", s.precision_at_5);
    println!("  {:<14} {:.3}", "Precision@10", s.precision_at_10);
    println!("  {:<14} {:.3}", "Recall@10", s.recall_at_10);
    println!("  {:<14} {:.3}", "MAP", s.map);
    println!("  {:<14} {:.3}", "nDCG@10", s.ndcg_at_10);
    println!();
    println!(
        "{}",
        "Higher = your liked performers cluster tighter under the blend. \
         Re-run after tuning weights / proj / bust to compare."
            .bright_black()
    );
    Ok(())
}
