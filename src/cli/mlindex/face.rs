//! Face corpus builders: embed library faces (`embed`) and warm a candidate pool (`warm`).
use super::super::*;

pub(crate) async fn embed_all(db: &Database, force: bool) -> anyhow::Result<()> {
    let pending = if force {
        db.get_all_performers()?
    } else {
        db.get_performers_without_embedding()?
    };

    if pending.is_empty() {
        println!(
            "{}",
            "All performers already have face embeddings. Use --force to re-embed.".green()
        );
        return Ok(());
    }

    let cfg = config::Config::load();
    let stash = cfg
        .stashdb_key
        .as_ref()
        .filter(|k| !k.is_empty())
        .map(|k| StashdbClient::new(k.clone()));

    println!(
        "{}",
        format!("Generating embeddings for {} performers...", pending.len())
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "  Note: first run downloads the ArcFace model (~100 MB)".bright_black()
    );
    println!();

    let mut ok = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for p in &pending {
        let urls = build_centroid_urls(p, stash.as_ref()).await;
        if urls.is_empty() {
            println!(
                "  {} {} — no image URL",
                "–".bright_black(),
                p.name.bright_black()
            );
            skipped += 1;
            continue;
        }

        print!("  {} {}... ", "→".bright_black(), p.name.bright_white());
        std::io::Write::flush(&mut std::io::stdout()).ok();

        match embedder::generate_centroid_embedding(&urls) {
            Ok(Some(emb)) => {
                db.save_embedding(&p.name, &emb)?;
                println!("{} ({} img)", "done".green(), urls.len());
                ok += 1;
            }
            Ok(None) => {
                println!("{}", "no face detected".red());
                skipped += 1;
            }
            Err(e) => {
                // The sidecar call failed (not a genuine no-face result), so
                // don't claim "no face" — this row can be retried.
                println!("{}", format!("embedding failed: {}", e).yellow());
                log::warn!("embedding sidecar failed for {}: {:#}", p.name, e);
                failed += 1;
            }
        }
    }

    println!();
    if failed > 0 {
        println!(
            "{}",
            format!(
                "Done: {} embedded, {} skipped, {} failed (retry with 'luminary embed')",
                ok, skipped, failed
            )
            .yellow()
        );
    } else {
        println!(
            "{}",
            format!("Done: {} embedded, {} skipped", ok, skipped).green()
        );
    }
    Ok(())
}

/// Pre-fetch a candidate pool and embed faces into the local corpus.
pub(crate) async fn warm(db: &Database, limit: usize) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let api_key = cfg.resolve_api_key()?;
    let performers = db.get_all_performers()?;
    if performers.is_empty() {
        println!(
            "{}",
            "Add some performers first with 'luminary add'.".yellow()
        );
        return Ok(());
    }

    let tree = recommender::build_preference_tree(&performers);
    let path = recommender::dominant_query_path(&tree);
    let liked_ids: Vec<i64> = performers.iter().filter_map(|p| p.tpdb_id).collect();
    let top_ethnicity = path.get(1).map(|s| s.as_str());
    let known: std::collections::HashSet<String> =
        performers.iter().map(|p| p.name.to_lowercase()).collect();

    println!(
        "{}",
        format!("Warming face corpus (up to {} candidates)...", limit)
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "  This embeds faces now so searches are instant later.".bright_black()
    );
    println!();

    let client = TpdbClient::new(api_key);
    let pool = client
        .get_recommendations(&liked_ids, top_ethnicity, None, &cfg.gender_filter)
        .await?;

    // Collect new candidates + their best image URL, then embed them all in ONE
    // batched sidecar call (model loads once instead of per-candidate).
    let targets: Vec<(models::Performer, String)> = pool
        .into_iter()
        .filter(|p| !known.contains(&p.name.to_lowercase()))
        .filter_map(|p| {
            p.face_url
                .clone()
                .or_else(|| p.profile_image_url.clone())
                .map(|u| (p, u))
        })
        .take(limit)
        .collect();

    let urls: Vec<String> = targets.iter().map(|(_, u)| u.clone()).collect();
    let embeddings = embedder::generate_embeddings(&urls).unwrap_or_default();

    let mut done = 0usize;
    for ((p, _), emb) in targets.iter().zip(embeddings) {
        if let Some(e) = emb {
            db.save_candidate(p, &e)?;
            done += 1;
        }
    }

    println!();
    println!(
        "{}",
        format!(
            "Embedded {} new faces. Corpus now holds {} total.",
            done,
            db.candidate_count()?
        )
        .green()
    );
    println!(
        "{}",
        "  Run 'luminary face-search <name>' to search it.".bright_black()
    );
    Ok(())
}
