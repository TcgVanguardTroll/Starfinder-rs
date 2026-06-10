//! Body corpus builders: the roster `index`, per-performer `ingest`, and the corpus->body_index `aggregate`.
use super::super::*;

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

        // Propagate sidecar failure instead of writing empty index rows for
        // the whole chunk (which would mark these performers "indexed" with
        // no vectors and silently exclude them from every future search).
        let all = embedder::generate_dual_embeddings(&flat).context(
            "Body-embedding sidecar failed for this chunk (transient — model \
             load or image download). Re-run 'index' to resume.",
        )?;

        for (p, (s, e)) in group.iter().zip(ranges) {
            let slice = all.get(s..e).unwrap_or(&[]);
            let pose_vecs: Vec<Vec<f32>> =
                slice.iter().filter_map(|(pose, _)| pose.clone()).collect();
            let seg_vecs: Vec<Vec<f32>> = slice.iter().filter_map(|(_, seg)| seg.clone()).collect();
            let n = pose_vecs.len().max(seg_vecs.len());
            let pose_c = embedder::body_centroid(&pose_vecs);
            let seg_c = embedder::body_centroid(&seg_vecs);
            db.save_body_index(p, pose_c.as_deref(), seg_c.as_deref(), None, None, n)?;
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

/// The identity anchor for ingest: a performer's cached face embedding, else a
/// fresh centroid built from their clean profile faces. `None` if neither is
/// available (a niche performer with no clean face) — ingest then trusts the
/// gallery instead of face-gating.
async fn resolve_seed_face(db: &Database, name: &str) -> Option<Vec<f32>> {
    if let Ok(Some(e)) = db.get_embedding_any(name) {
        return Some(e);
    }
    let p = db
        .get_performer(name)
        .ok()
        .flatten()
        .or_else(|| db.get_known_performer(name).ok().flatten())?;
    let cfg = config::Config::load();
    let stash = cfg
        .stashdb_key
        .as_ref()
        .filter(|k| !k.is_empty())
        .map(|k| StashdbClient::new(k.clone()));
    let urls = build_centroid_urls(&p, stash.as_ref()).await;
    if urls.is_empty() {
        return None;
    }
    match embedder::generate_centroid_embedding(&urls) {
        Ok(face) => face,
        Err(e) => {
            // A transient sidecar failure here silently disables identity-gating
            // for this performer (ingest falls back to trusting the gallery), so
            // surface it rather than treating it as a genuine "no clean face".
            log::warn!(
                "seed-face sidecar failed for {name} \
                 (identity-gating disabled this run): {e:#}"
            );
            None
        }
    }
}

/// Build (or extend) the per-image corpus for specific performers. For each one
/// it gathers multi-angle images from every source, runs a face pass (identity-
/// verify + presence) and a pose/seg pass (vectors + coarse view) over them,
/// classifies each image's view, quality-scores it, and writes one `images` row.
/// Incremental — images already stored are skipped unless `force`. This is the
/// corpus the view-aware body index is later aggregated from.
pub(crate) async fn ingest(
    db: &Database,
    names: Vec<String>,
    images_per: usize,
    manual_urls: Vec<String>,
    id_threshold: f32,
    force: bool,
    roster: bool,
    allow_unverified: bool,
) -> anyhow::Result<()> {
    use luminary::database::ImageRow;
    use luminary::source::{classify_view, quality_score, ImageSource, ManualSource};
    use std::collections::{BTreeMap, HashSet};

    // Min seed-matching faces to trust a gather. Below this the sources likely
    // returned the wrong/missing person (a bogus name) → skip the whole batch.
    const MIN_ID_MATCHES: usize = 2;

    // --roster ingests every performer in the cached body index (resumable).
    let names = if roster {
        db.load_body_index()?
            .into_iter()
            .map(|e| e.performer.name)
            .collect()
    } else {
        names
    };

    // The sanctioned image sources. pornpics + pichunter performer pages are
    // single-performer, so a face-less rear shot can be gallery-trusted. Manual
    // URLs, when supplied, ride along as their own source.
    let mut sources: Vec<Box<dyn ImageSource>> = vec![
        Box::new(PornpicsClient::new()),
        Box::new(PichunterClient::new()),
    ];
    if !manual_urls.is_empty() {
        sources.push(Box::new(ManualSource { urls: manual_urls }));
    }

    println!(
        "{}",
        format!("Ingesting images for {} performer(s)...", names.len())
            .bright_cyan()
            .bold()
    );
    println!();

    for name in &names {
        // Use the canonical performer name (when known) so corpus rows line up
        // with the rest of the system, which keys on `Performer::name`.
        let performer = db
            .get_performer(name)
            .ok()
            .flatten()
            .map(|p| p.name)
            .unwrap_or_else(|| name.clone());

        let seed = resolve_seed_face(db, &performer).await;
        if seed.is_none() {
            if !allow_unverified {
                println!(
                    "  {} {} — no seed face; skipping (--allow-unverified to override)",
                    "–".bright_black(),
                    performer.bright_black()
                );
                continue;
            }
            println!(
                "  {} no seed face for {} — identity unverified (gallery-trusted)",
                "!".yellow(),
                performer.bright_white()
            );
        }

        // Gather candidate URLs from every source, tagged with origin, skipping
        // ones already in the corpus (unless forced).
        let existing = if force {
            HashSet::new()
        } else {
            db.existing_image_urls(&performer)?
        };
        let mut targets: Vec<(String, &'static str)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for src in &sources {
            for u in src.gather(&performer, images_per).await {
                if !existing.contains(&u) && seen.insert(u.clone()) {
                    targets.push((u, src.name()));
                }
            }
        }
        if targets.is_empty() {
            println!(
                "  {} {} — no new images",
                "–".bright_black(),
                performer.bright_black()
            );
            continue;
        }

        // Two passes over the same URLs (each loads its model once): face for
        // identity + presence, body for pose/seg vectors + the coarse view.
        let urls: Vec<String> = targets.iter().map(|(u, _)| u.clone()).collect();
        let faces = embedder::generate_embeddings(&urls).unwrap_or_default();
        let bodies = embedder::generate_body_views(&urls).unwrap_or_default();

        // Bogus-name guard: with a seed, require enough gathered faces to match
        // it. A wrong/missing gallery matches few or none — skip the whole batch
        // rather than store another performer's (or generic) images under this
        // name. Per-image co-star rejection still happens in the loop below.
        if let Some(s) = &seed {
            let matched = faces
                .iter()
                .flatten()
                .filter(|f| embedder::cosine_similarity(f, s) >= id_threshold)
                .count();
            if matched < MIN_ID_MATCHES {
                println!(
                    "  {} {} — {} seed-matching face(s) (< {}); likely wrong/missing gallery, skipping",
                    "–".bright_black(),
                    performer.bright_black(),
                    matched,
                    MIN_ID_MATCHES
                );
                continue;
            }
        }

        let mut kept = 0usize;
        let mut rejected = 0usize;
        let mut by_view: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, (url, source)) in targets.iter().enumerate() {
            let face = faces.get(i).cloned().flatten();
            let body = bodies.get(i);
            let pose = body.and_then(|b| b.pose.clone());
            let seg = body.and_then(|b| b.seg.clone());
            let proj = body.and_then(|b| b.proj.clone());
            let pose_view = body.and_then(|b| b.view.as_deref());

            // Identity gate: a detected face that doesn't match the seed is a
            // different performer (mixed gallery / co-star) — drop it. A face-less
            // shot has no `id_sim` and so is never rejected here.
            let id_sim = match (&face, &seed) {
                (Some(f), Some(s)) => Some(embedder::cosine_similarity(f, s)),
                _ => None,
            };
            if id_sim.is_some_and(|sim| sim < id_threshold) {
                rejected += 1;
                continue;
            }

            // Nothing usable extracted at all — don't store an empty row.
            if face.is_none() && pose.is_none() && seg.is_none() && proj.is_none() {
                continue;
            }

            let view = classify_view(pose_view, face.is_some());
            let quality = quality_score(id_sim, pose.is_some(), seg.is_some());
            db.save_image(&ImageRow {
                performer: performer.clone(),
                url: url.clone(),
                source: source.to_string(),
                view: view.to_string(),
                quality,
                pose,
                seg,
                face,
                proj,
                bust: None, // set by the bust CV (#16); not computed at ingest yet
            })?;
            kept += 1;
            *by_view.entry(view).or_insert(0) += 1;
        }

        let breakdown = by_view
            .iter()
            .map(|(v, n)| format!("{} {}", n, v))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  {} {} — {} kept{}, {} rejected",
            "✓".green(),
            performer.bright_white(),
            kept,
            if breakdown.is_empty() {
                String::new()
            } else {
                format!(" ({})", breakdown)
            },
            rejected,
        );
    }

    println!();
    println!(
        "{}",
        format!("Corpus now holds {} image(s).", db.images_count()?)
            .green()
            .bold()
    );
    Ok(())
}

/// Rebuild the cached body-vector index for performers from their per-image
/// corpus (the rows `ingest` fills). Quality-weighted and view-aware — only
/// frontal (front/rear) frames feed the pose/seg centroids. Cheap and pure, so
/// re-run it to re-tune aggregation without re-embedding. With no names, rebuilds
/// every performer that has images.
pub(crate) fn aggregate(db: &Database, names: Vec<String>) -> anyhow::Result<()> {
    let targets = if names.is_empty() {
        db.images_performers()?
    } else {
        names
    };
    if targets.is_empty() {
        println!(
            "{}",
            "No ingested images yet — run 'luminary ingest <name>' first.".yellow()
        );
        return Ok(());
    }

    println!(
        "{}",
        format!(
            "Aggregating {} performer(s) from the image corpus...",
            targets.len()
        )
        .bright_cyan()
        .bold()
    );
    println!();

    let mut written = 0usize;
    for name in &targets {
        let images = db.load_images(name, None)?;
        let (pose, seg, proj, bust, n) = luminary::database::aggregate_views(&images);
        if pose.is_none() && seg.is_none() && proj.is_none() && bust.is_none() {
            println!(
                "  {} {} — no usable frames ({} image(s))",
                "–".bright_black(),
                name.bright_black(),
                images.len()
            );
            continue;
        }
        // Carry the performer's metadata into the index: the library record if we
        // have one, else the roster/candidate metadata (so a roster performer
        // keeps measurements/ethnicity instead of being blanked), else bare name.
        let performer = db
            .get_performer(name)
            .ok()
            .flatten()
            .or_else(|| db.get_known_performer(name).ok().flatten())
            .unwrap_or_else(|| models::Performer::new(name.clone()));
        db.save_body_index(
            &performer,
            pose.as_deref(),
            seg.as_deref(),
            proj.as_deref(),
            bust.as_deref(),
            n,
        )?;
        written += 1;
        println!(
            "  {} {} {}",
            "✓".green(),
            name.bright_white(),
            format!(
                "{} frame(s) → pose {} / shape {} / proj {} / bust {}",
                n,
                if pose.is_some() { "✓" } else { "—" },
                if seg.is_some() { "✓" } else { "—" },
                if proj.is_some() { "✓" } else { "—" },
                if bust.is_some() { "✓" } else { "—" },
            )
            .bright_black()
        );
    }

    println!();
    println!(
        "{}",
        format!(
            "Aggregated {} performer(s); index now holds {}.",
            written,
            db.body_index_count()?
        )
        .green()
        .bold()
    );
    Ok(())
}
