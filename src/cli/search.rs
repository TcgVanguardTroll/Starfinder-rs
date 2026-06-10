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

/// Height-band gate for "same stature, not just proportions". Returns
/// `Some((ref_cm, tol_cm))` when a tolerance is set AND the reference has a
/// recorded height; otherwise `None` (no filtering).
pub(super) fn height_band(reference: &models::Performer, tol: Option<f64>) -> Option<(f64, f64)> {
    match (tol, crate::recommender::performer_height_cm(reference)) {
        (Some(t), Some(h)) => Some((h, t)),
        _ => None,
    }
}

/// True when the candidate passes the height band (or no band is active). A
/// candidate with no recorded height is excluded while a band is active — we
/// can't confirm it shares the reference's stature.
pub(super) fn in_band(band: Option<(f64, f64)>, p: &models::Performer) -> bool {
    match band {
        None => true,
        Some((h, t)) => {
            crate::recommender::performer_height_cm(p).is_some_and(|c| (c - h).abs() <= t)
        }
    }
}

/// True when the candidate's recorded hair colour contains `want`
/// (case-insensitive substring, so `blond` matches `Blonde`/`Dark Blond`), or no
/// hair filter is active. A candidate with no recorded hair colour is excluded
/// while a filter is active — we can't confirm it matches.
pub(super) fn hair_match(want: Option<&str>, p: &models::Performer) -> bool {
    match want {
        None => true,
        Some(w) => p
            .hair_color
            .as_deref()
            .is_some_and(|h| h.to_lowercase().contains(&w.to_lowercase())),
    }
}

// ── Shared result-rendering helpers (used by every body/face lens) ───────────

/// A fixed-width colour bar for a 0–100 score, with partial-block glyphs for
/// sub-cell precision. `width` cells = 100%. Green ≥80, yellow ≥60, else red — so
/// a glance down the column reads the match strength without parsing numbers.
pub(super) fn score_bar(score: f64, width: usize) -> String {
    const EIGHTHS: [&str; 9] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
    let s = score.clamp(0.0, 100.0);
    let eighths = (s / 100.0 * width as f64 * 8.0).round() as usize;
    let full = (eighths / 8).min(width);
    let rem = eighths % 8;
    let mut bar = "█".repeat(full);
    let mut cells = full;
    if rem > 0 && cells < width {
        bar.push_str(EIGHTHS[rem]);
        cells += 1;
    }
    bar.push_str(&" ".repeat(width.saturating_sub(cells)));
    let c = if s >= 80.0 {
        bar.green()
    } else if s >= 60.0 {
        bar.yellow()
    } else {
        bar.red()
    };
    c.to_string()
}

/// Truncate (with …) or right-pad a name to a fixed display width so the columns
/// after it line up down the list. Returns a plain string — colour it afterwards
/// (padding a coloured string would count the ANSI codes and misalign).
pub(super) fn pad_name(name: &str, width: usize) -> String {
    let chars: Vec<char> = name.chars().collect();
    if chars.len() > width {
        let mut s: String = chars[..width.saturating_sub(1)].iter().collect();
        s.push('…');
        s
    } else {
        format!("{name:<width$}")
    }
}

/// One modality value as a fixed-width 3-char cell — the rounded score, or a dim
/// `·` when that modality is absent for this candidate. Keeps the breakdown row's
/// columns aligned regardless of which modalities are present.
pub(super) fn modality_cell(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:>3.0}"),
        None => format!("{:>3}", "·").bright_black().to_string(),
    }
}

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
    height_tol: Option<f64>,
    hair: Option<String>,
) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let reference = db
        .get_performer(name)?
        .ok_or_else(|| anyhow::anyhow!("'{}' not found in your library. Add them first.", name))?;
    let band = height_band(&reference, height_tol);
    let hair = hair.map(|h| h.to_lowercase());
    if let Some(h) = hair.as_deref() {
        println!("{}", format!("  hair filter: {} only", h).bright_black());
    }
    if height_tol.is_some() && band.is_none() {
        println!(
            "{}",
            format!(
                "  (--height-tol ignored: no recorded height for {})",
                reference.name
            )
            .yellow()
        );
    }
    if let Some((h, t)) = band {
        println!(
            "{}",
            format!("  height band: {:.0}±{:.0}cm (stature-matched)", h, t).bright_black()
        );
    }
    // Measurements lens: rank the cached index by recorded build (WHR/hips/cup/…).
    // Needs no reference *images*, so it works even for niche performers who have
    // no clean full-body photo (where the visual lenses can't build a vector).
    if by == "stats" {
        return search_by_measure(db, &reference, limit, images, band, hair.as_deref()).await;
    }
    // The default: multi-modal blend of face + frame + curves + projection + stats.
    // `body` excludes face (pure body type); `lookalike` lets face dominate
    // (closest-looking actress). All three share the blend path.
    if by == "overall" || by == "body" || by == "lookalike" {
        return search_blend(db, &reference, limit, images, by, band, hair.as_deref()).await;
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
    // Separate a sidecar *failure* (retryable — don't blame the images) from a
    // genuine *no usable frame* result (the images really are headshots/crops).
    let ref_vecs: Vec<Vec<f32>> = embed(&ref_imgs)
        .with_context(|| {
            format!(
                "Body-embedding sidecar failed for {} (transient — model load or \
                 image download). Re-run the search.",
                reference.name
            )
        })?
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
                n != ref_lc
                    && !known.contains(&n)
                    && in_band(band, &e.performer)
                    && hair_match(hair.as_deref(), &e.performer)
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
                    && hair_match(hair.as_deref(), p)
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
        // A sidecar failure here used to be swallowed (empty vec → every
        // candidate filtered out → misleading "no results"). Surface it.
        let all = embed(&flat).context(
            "Body-embedding sidecar failed while embedding the candidate pool \
             (transient — model load or image download). Re-run the search.",
        )?;
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

/// Plain-English search: parse a free-text query into `find`'s structured inputs
/// (see `luminary::query`) and run it. The interpretation is echoed so the user
/// can see how their sentence was read.
pub(crate) async fn nl_query(
    db: &Database,
    text: &str,
    images: bool,
    limit: usize,
) -> anyhow::Result<()> {
    let q = luminary::query::parse(text);

    let mut bits: Vec<String> = Vec::new();
    if !q.looks_like.is_empty() {
        bits.push(format!("face like {}", q.looks_like.join(" + ")));
    }
    if !q.body_like.is_empty() {
        bits.push(format!("body like {}", q.body_like.join(" + ")));
    }
    if let Some(e) = &q.eye {
        bits.push(format!("eyes {}", e));
    }
    if let Some(h) = &q.hair {
        bits.push(format!("hair {}", h));
    }
    if let Some(et) = &q.ethnicity {
        bits.push(format!("ethnicity {}", et));
    }
    if bits.is_empty() {
        anyhow::bail!(
            "Couldn't read any filters from that. Try e.g. 'blue-eyed blondes that \
             look like <name>' or '<attrs> with a butt like <name>'."
        );
    }
    println!(
        "{} {}",
        "Interpreted as:".bright_black(),
        bits.join(" · ").bright_white()
    );
    println!();

    find(
        db,
        None,
        None,
        q.looks_like,
        q.body_like,
        q.hair,
        q.eye,
        q.ethnicity,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
        images,
        None,
        None,
        limit,
    )
    .await
}
