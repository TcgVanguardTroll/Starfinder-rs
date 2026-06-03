//! `face-search`: rank the local face corpus by ArcFace similarity.
use super::super::*;

pub(crate) async fn face_search(
    db: &Database,
    name: &str,
    limit: usize,
    images: bool,
    source: Option<String>,
) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let reference = db
        .get_performer(name)?
        .ok_or_else(|| anyhow::anyhow!("'{}' not found in your library. Add them first.", name))?;
    let ref_emb = db.get_embedding(&reference.name)?.ok_or_else(|| {
        anyhow::anyhow!(
            "No face embedding for '{}'. Run 'luminary embed' first.",
            reference.name
        )
    })?;

    // Optionally pull a fresh candidate pool from StashDB (matching the
    // reference's attributes), embed it, and fold it into the corpus first.
    if source.as_deref() == Some("stashdb") {
        let key = cfg
            .stashdb_key
            .clone()
            .filter(|k| !k.is_empty())
            .context("No StashDB key. Set one with 'luminary config stashdb-key <key>'.")?;
        let client = StashdbClient::new(key);
        let gender = cfg.gender_filter.tpdb_value(); // "Female" etc.
        println!(
            "{}",
            format!(
                "Fetching StashDB candidates like {} ({}, {}, {})...",
                reference.name,
                gender.unwrap_or("any"),
                reference.ethnicity.as_deref().unwrap_or("?"),
                reference.hair_color.as_deref().unwrap_or("?"),
            )
            .bright_cyan()
        );
        let pool = client
            .query_similar(
                gender,
                reference.ethnicity.as_deref(),
                reference.hair_color.as_deref(),
                30,
            )
            .await?;

        // Gather new candidates + a face image, then embed in one batched call.
        let mut targets: Vec<(models::Performer, String)> = Vec::new();
        for p in pool {
            if p.name.to_lowercase() == reference.name.to_lowercase() {
                continue;
            }
            if db.get_embedding_any(&p.name)?.is_some() {
                continue; // already in corpus
            }
            if let Some(url) = p.profile_image_url.clone().or_else(|| p.face_url.clone()) {
                targets.push((p, url));
            }
        }

        let urls: Vec<String> = targets.iter().map(|(_, u)| u.clone()).collect();
        let embeddings = embedder::generate_embeddings(&urls).unwrap_or_default();
        let mut embedded = 0usize;
        for ((p, _), emb) in targets.iter().zip(embeddings) {
            if let Some(e) = emb {
                db.save_candidate(p, &e)?;
                embedded += 1;
            }
        }
        println!(
            "{}",
            format!("  +{} new StashDB faces", embedded).bright_black()
        );
        println!();
    }

    let corpus = db.load_candidates()?;
    if corpus.is_empty() {
        println!(
            "{}",
            "Face corpus is empty. Run 'luminary warm' or pass --source stashdb.".yellow()
        );
        return Ok(());
    }

    let mut scored: Vec<(f32, models::Performer)> = corpus
        .into_iter()
        .filter(|(_, p)| cfg.gender_filter.matches(p.gender.as_deref()))
        .filter(|(_, p)| p.name.to_lowercase() != reference.name.to_lowercase())
        // Hide placeholder entries (empty / purely-numeric names)
        .filter(|(_, p)| !p.name.trim().is_empty() && !p.name.chars().all(|c| c.is_ascii_digit()))
        .map(|(e, p)| (embedder::cosine_similarity(&ref_emb, &e), p))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    println!(
        "{}",
        format!(
            "Faces most like {} (from a corpus of {} candidates):",
            reference.name,
            scored.len()
        )
        .bright_cyan()
        .bold()
    );
    println!();

    let img_cache = if images { ImageCache::new().ok() } else { None };
    for (i, (sim, p)) in scored.iter().enumerate() {
        let age_str = p
            .age
            .map(|a| format!(", {}", recommender::age_bucket(a)))
            .unwrap_or_default();
        let body = if p.body_type.is_empty() {
            "?"
        } else {
            &p.body_type
        };
        println!(
            "{}. {} {}  {}",
            (i + 1).to_string().bright_black(),
            p.name.bright_white().bold(),
            format!(
                "({}, {}{}{})",
                body,
                p.ethnicity.as_deref().unwrap_or("?"),
                p.hair_color
                    .as_ref()
                    .map(|h| format!(", {}", h))
                    .unwrap_or_default(),
                age_str,
            )
            .bright_black(),
            format!("face {:.0}%", embedder::similarity_pct(*sim)).bright_cyan(),
        );
        if let Some(url) = &p.source_url {
            println!("   {} {}", "↳".bright_black(), url.blue().underline());
        }
        if let Some(cache) = &img_cache {
            if let Some(url) = p.face_url.as_deref().or(p.profile_image_url.as_deref()) {
                render_thumbnail(cache, url).await;
            }
        }
    }
    println!();
    println!(
        "{}",
        "Use 'luminary add <name>' to add any to your profile.".bright_black()
    );
    Ok(())
}
