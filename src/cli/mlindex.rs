//! Embedding and index-building handlers: `embed`, `warm`, and `index`.
use super::*;
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

/// Build (or extend) the cached body-vector index. For a roster of popular
/// performers it gathers full-body images from every source, runs ONE dual
/// embedding pass per chunk (pose + seg together), and stores gated centroids.
/// Resumable: already-indexed performers are skipped unless `force`.
pub(crate) async fn build_index(
    db: &Database,
    limit: usize,
    images_per: usize,
    force: bool,
) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let stash_key = cfg.stashdb_key.clone().filter(|k| !k.is_empty()).context(
        "index needs a StashDB key for the roster. Set 'luminary config stashdb-key <key>'.",
    )?;
    let stash = StashdbClient::new(stash_key);
    let tpdb = cfg.resolve_api_key().ok().map(TpdbClient::new);
    let pornpics = PornpicsClient::new();

    // 1. Build the roster: paginate StashDB's most-popular performers.
    println!(
        "{}",
        format!("Gathering a roster of {} popular performers...", limit)
            .bright_cyan()
            .bold()
    );
    let mut roster: Vec<models::Performer> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut page = 1usize;
    while roster.len() < limit && page <= 60 {
        let batch = stash
            .query_popular(cfg.gender_filter.tpdb_value(), page, 25)
            .await?;
        if batch.is_empty() {
            break;
        }
        for p in batch {
            if seen.insert(p.name.to_lowercase()) {
                roster.push(p);
            }
        }
        page += 1;
    }
    roster.truncate(limit);

    // 2. Skip performers already indexed (unless forced).
    let done = if force {
        std::collections::HashSet::new()
    } else {
        db.body_indexed_names()?
    };
    let todo: Vec<models::Performer> = roster
        .into_iter()
        .filter(|p| !done.contains(&p.name.to_lowercase()))
        .collect();

    if todo.is_empty() {
        println!(
            "{}",
            "Nothing to do — roster already indexed (use --force to rebuild).".green()
        );
        return Ok(());
    }
    println!(
        "{}",
        format!(
            "Indexing {} performers (pornpics + TPDB scenes + StashDB, gated)...",
            todo.len()
        )
        .bright_cyan()
    );
    println!();

    // 3. Process in chunks so one sidecar call (model loads once) covers many
    //    performers, while keeping the per-call URL count under the OS arg limit.
    let mut indexed = 0usize;
    for group in todo.chunks(12) {
        let mut flat: Vec<String> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for p in group {
            let mut imgs: Vec<String> = Vec::new();
            for u in pornpics.image_urls(&p.name, images_per).await {
                if !imgs.contains(&u) {
                    imgs.push(u);
                }
            }
            if let Some(t) = &tpdb {
                for u in t.scene_image_urls(&p.name, 8).await {
                    if imgs.len() >= images_per {
                        break;
                    }
                    if !imgs.contains(&u) {
                        imgs.push(u);
                    }
                }
            }
            for u in p.gallery_urls.iter().take(4) {
                if imgs.len() >= images_per {
                    break;
                }
                if !imgs.contains(u) {
                    imgs.push(u.clone());
                }
            }
            let start = flat.len();
            flat.extend(imgs);
            ranges.push((start, flat.len()));
        }

        let all = embedder::generate_dual_embeddings(&flat).unwrap_or_default();

        for (p, (s, e)) in group.iter().zip(ranges) {
            let slice = all.get(s..e).unwrap_or(&[]);
            let pose_vecs: Vec<Vec<f32>> =
                slice.iter().filter_map(|(pose, _)| pose.clone()).collect();
            let seg_vecs: Vec<Vec<f32>> = slice.iter().filter_map(|(_, seg)| seg.clone()).collect();
            let n = pose_vecs.len().max(seg_vecs.len());
            let pose_c = embedder::body_centroid(&pose_vecs);
            let seg_c = embedder::body_centroid(&seg_vecs);
            db.save_body_index(p, pose_c.as_deref(), seg_c.as_deref(), n)?;
            indexed += 1;
            println!(
                "  {} {} {}",
                format!("[{}/{}]", indexed, todo.len()).bright_black(),
                p.name.bright_white(),
                format!(
                    "{} frame / {} shape frames",
                    pose_vecs.len(),
                    seg_vecs.len()
                )
                .bright_black(),
            );
        }
    }

    println!();
    println!(
        "{}",
        format!(
            "Index built: {} added, {} total in index.",
            indexed,
            db.body_index_count()?
        )
        .green()
        .bold()
    );
    Ok(())
}
