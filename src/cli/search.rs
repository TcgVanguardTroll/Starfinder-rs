//! Image-based search handlers: `find` (attributes + face), `body-search`
//! (`--by overall`/`frame`/`curves`/`stats`), and `face-search`.
use super::*;

mod face;
mod find;
mod measure;
mod overall;

pub(crate) use face::face_search;
pub(crate) use find::find;
use measure::search_by_measure;
use overall::search_blend;

/// Find performers with a similar body, ranked against the cached index.
///
/// `by` selects the lens:
///   - `overall` (default): multi-modal blend of every available signal
///     (handled by `search_blend`).
///   - `frame`: skeletal pose vector (shoulder/hip/leg proportions).
///   - `curves`: silhouette/segmentation vector (waist/hip/thigh fullness — the
///     butt & thigh shape the skeleton can't see). Built from a combined,
///     gated pool (pornpics + TPDB scenes + StashDB).
///   - `stats`: recorded WHR/hips/cup (handled by `search_by_measure`).
pub(crate) async fn body_search(
    db: &Database,
    name: &str,
    limit: usize,
    images: bool,
    by: &str,
) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let reference = db
        .get_performer(name)?
        .ok_or_else(|| anyhow::anyhow!("'{}' not found in your library. Add them first.", name))?;
    // Measurements lens: rank the cached index by recorded build (WHR/hips/cup/…).
    // Needs no reference *images*, so it works even for niche performers who have
    // no clean full-body photo (where the visual lenses can't build a vector).
    if by == "stats" {
        return search_by_measure(db, &reference, limit, images).await;
    }
    // The default: multi-modal blend of face + frame + curves + projection + stats.
    if by == "overall" {
        return search_blend(db, &reference, limit, images).await;
    }
    // `curves` = silhouette/segmentation lens; otherwise the `frame` (skeletal
    // pose) lens. The label is reused throughout the output.
    let volume = by == "curves";
    let kind = if volume { "curves" } else { "frame" };
    let key = cfg.stashdb_key.clone().filter(|k| !k.is_empty()).context(
        "body-search needs full-body images from StashDB. Set 'luminary config stashdb-key <key>'.",
    )?;
    let stash = StashdbClient::new(key);

    // Reference body vector: centroid over a combined image pool from every
    // source we have. The per-image pose gate (visibility + upright) discards
    // crops and non-standing frames, so volume here is good — more candidates
    // in means more *clean* full-body frames survive.
    println!(
        "{}",
        format!(
            "Building body {} for {} from pornpics + TPDB scenes + StashDB...",
            kind, reference.name
        )
        .bright_cyan()
    );
    let tpdb = cfg.resolve_api_key().ok().map(TpdbClient::new);
    let mut ref_imgs: Vec<String> = Vec::new();
    // pornpics gallery covers — the richest source of distinct full-body shoots.
    for u in PornpicsClient::new().image_urls(&reference.name, 20).await {
        if !ref_imgs.contains(&u) {
            ref_imgs.push(u);
        }
    }
    if let Some(t) = &tpdb {
        for u in t.scene_image_urls(&reference.name, 12).await {
            if !ref_imgs.contains(&u) {
                ref_imgs.push(u);
            }
        }
    }
    for u in stash.image_urls(&reference.name, 6).await {
        if !ref_imgs.contains(&u) {
            ref_imgs.push(u);
        }
    }
    let embed = |urls: &[String]| {
        if volume {
            embedder::generate_seg_embeddings(urls)
        } else {
            embedder::generate_body_embeddings(urls)
        }
    };
    let ref_vecs: Vec<Vec<f32>> = embed(&ref_imgs)
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect();
    let ref_body = embedder::body_centroid(&ref_vecs).context(
        "No usable full-body standing photo found for the reference. Their images \
         are headshots/crops or non-standing poses, which can't yield a reliable \
         body vector.",
    )?;
    println!(
        "{}",
        format!(
            "  {} from {}/{} images (cropped & non-standing poses rejected)",
            kind,
            ref_vecs.len(),
            ref_imgs.len()
        )
        .bright_black()
    );

    let known: std::collections::HashSet<String> = db
        .get_all_performers()?
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();
    let ref_lc = reference.name.to_lowercase();

    // Candidates: prefer the cached body index (instant, rich pool). Fall back to
    // a live StashDB fetch + embed when the index hasn't been built yet.
    let index = db.load_body_index()?;
    let mut scored: Vec<(f64, models::Performer)> = if !index.is_empty() {
        println!(
            "{}",
            format!("Ranking against {} indexed performers...", index.len()).bright_cyan()
        );
        index
            .into_iter()
            .filter(|e| {
                let n = e.performer.name.to_lowercase();
                n != ref_lc && !known.contains(&n)
            })
            .filter_map(|e| {
                let cv = if volume { e.seg } else { e.pose }?;
                let pct = if volume {
                    embedder::seg_similarity_pct(&ref_body, &cv)
                } else {
                    embedder::body_similarity_pct(&ref_body, &cv)
                };
                Some((pct, e.performer))
            })
            .collect()
    } else {
        // Live fallback: fetch a StashDB pool and embed it now.
        let pool: Vec<models::Performer> = stash
            .query_similar(cfg.gender_filter.tpdb_value(), None, None, 40)
            .await?
            .into_iter()
            .filter(|p| {
                p.name.to_lowercase() != ref_lc
                    && !known.contains(&p.name.to_lowercase())
                    && !p.gallery_urls.is_empty()
            })
            .collect();
        println!(
            "{}",
            format!(
                "Embedding {} candidates (no index yet — run 'index')...",
                pool.len()
            )
            .bright_cyan()
        );
        let mut flat: Vec<String> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for p in &pool {
            let imgs: Vec<String> = p.gallery_urls.iter().take(3).cloned().collect();
            let start = flat.len();
            flat.extend(imgs);
            ranges.push((start, flat.len()));
        }
        let all = embed(&flat).unwrap_or_default();
        pool.into_iter()
            .zip(ranges)
            .filter_map(|(p, (s, e))| {
                let vecs: Vec<Vec<f32>> = all.get(s..e)?.iter().flatten().cloned().collect();
                let cv = embedder::body_centroid(&vecs)?;
                let pct = if volume {
                    embedder::seg_similarity_pct(&ref_body, &cv)
                } else {
                    embedder::body_similarity_pct(&ref_body, &cv)
                };
                Some((pct, p))
            })
            .collect()
    };
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    if scored.is_empty() {
        println!(
            "{}",
            "No candidates with a detectable body pose found.".yellow()
        );
        return Ok(());
    }

    println!();
    println!(
        "{}",
        format!(
            "Top {} by body {} similarity to {}:",
            scored.len(),
            kind,
            reference.name
        )
        .bright_cyan()
        .bold()
    );
    println!();
    let img_cache = if images { ImageCache::new().ok() } else { None };
    for (i, (pct, p)) in scored.iter().enumerate() {
        let ht = recommender::performer_height_cm(p)
            .map(|h| format!(", {:.0}cm", h))
            .unwrap_or_default();
        println!(
            "{}. {} {}  {}",
            (i + 1).to_string().bright_black(),
            p.name.bright_white().bold(),
            format!(
                "({}{}{})",
                p.ethnicity.as_deref().unwrap_or("?"),
                p.measurements
                    .as_ref()
                    .map(|m| format!(", {}", m))
                    .unwrap_or_default(),
                ht,
            )
            .bright_black(),
            format!("{} {:.0}%", kind, pct).bright_cyan(),
        );
        if let Some(url) = &p.source_url {
            println!("   {} {}", "↳".bright_black(), url.blue().underline());
        }
        if let Some(cache) = &img_cache {
            if let Some(url) = p.profile_image_url.as_deref() {
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
