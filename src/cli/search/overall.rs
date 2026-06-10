//! The `overall` lens: the multi-modal blend (see `luminary::blend`).
use super::super::*;

/// The `overall` lens: fuse face + frame + curves + projection + stats into one
/// rank (the default for body-search). The reference's modalities are read from
/// local data — its face embedding, its ingested image corpus (aggregated the
/// same way `aggregate` builds the index), and its recorded measurements — so the
/// blend needs no fresh gathering. Candidates are the cached body index, and each
/// modality is rank-normalised before blending so the scales stay comparable
/// (see `luminary::blend`).
pub(super) async fn search_blend(
    db: &Database,
    reference: &models::Performer,
    limit: usize,
    images: bool,
    by: &str,
    band: Option<(f64, f64)>,
    hair: Option<&str>,
) -> anyhow::Result<()> {
    // `by` selects the weight profile: "body" excludes face (pure body type),
    // "lookalike" lets face dominate (closest-looking), else the balanced default.
    let body_only = by == "body";
    let lookalike = by == "lookalike";
    let mode_label = if body_only {
        "BODY-TYPE"
    } else if lookalike {
        "look-alike (face-led)"
    } else {
        "overall"
    };
    // Reference modalities — all local, no gathering. `get_embedding_any` so a
    // roster/footage-seeded face (in `candidates`) resolves too, not just library.
    let ref_face = db.get_embedding_any(&reference.name).ok().flatten();
    let ref_meas = recommender::feature_vector(reference);
    let ref_height = recommender::performer_height_cm(reference);
    let ref_bmi = recommender::performer_bmi(reference);
    let ref_imgs = db.load_images(&reference.name, None)?;
    let (ref_pose, ref_seg, ref_proj, ref_bust) = if ref_imgs.is_empty() {
        (None, None, None, None)
    } else {
        let (p, s, pr, bu, _) = luminary::database::aggregate_views(&ref_imgs);
        (p, s, pr, bu)
    };

    let active = |label: &str, on: bool| {
        format!(
            "{} {}",
            label,
            if on {
                "✓".green()
            } else {
                "—".bright_black()
            }
        )
    };
    println!(
        "{}",
        format!(
            "Blending candidates by {} similarity to {}:",
            mode_label, reference.name
        )
        .bright_cyan()
        .bold()
    );
    println!(
        "  {}  {}  {}  {}  {}  {}",
        active("face", ref_face.is_some()),
        active("frame", ref_pose.is_some()),
        active("curves", ref_seg.is_some()),
        active("proj", ref_proj.is_some()),
        active("bust", ref_bust.is_some()),
        active("stats", ref_meas.is_some()),
    );
    if ref_face.is_none()
        && ref_pose.is_none()
        && ref_seg.is_none()
        && ref_proj.is_none()
        && ref_bust.is_none()
        && ref_meas.is_none()
    {
        anyhow::bail!(
            "No signal for '{}': no face embedding, no ingested images, no measurements. \
             Run 'luminary embed' and/or 'luminary ingest {}' first.",
            reference.name,
            reference.name
        );
    }
    if ref_pose.is_none() && ref_seg.is_none() && ref_proj.is_none() {
        println!(
            "{}",
            format!(
                "  (no ingested body images — blending face + stats only; \
                 run 'luminary ingest {}' to add frame/curves/projection)",
                reference.name
            )
            .bright_black()
        );
    }

    let index = db.load_body_index()?;
    if index.is_empty() {
        anyhow::bail!(
            "body index is empty — run 'luminary index' (or 'ingest' + 'aggregate') first."
        );
    }
    let known: std::collections::HashSet<String> = db
        .get_all_performers()?
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();
    let ref_lc = reference.name.to_lowercase();

    // Raw per-modality similarity scores for each candidate.
    let candidates: Vec<(blend::ModalityScores, models::Performer)> = index
        .into_iter()
        .filter(|e| {
            let n = e.performer.name.to_lowercase();
            n != ref_lc
                && !known.contains(&n)
                && super::in_band(band, &e.performer)
                && super::hair_match(hair, &e.performer)
        })
        .map(|e| {
            let face = match (
                &ref_face,
                db.get_embedding_any(&e.performer.name).ok().flatten(),
            ) {
                (Some(rf), Some(cf)) => {
                    Some(embedder::similarity_pct(embedder::cosine_similarity(rf, &cf)) as f64)
                }
                _ => None,
            };
            let build = match (&ref_pose, &e.pose) {
                (Some(r), Some(c)) => Some(embedder::body_similarity_pct(r, c)),
                _ => None,
            };
            let volume = match (&ref_seg, &e.seg) {
                (Some(r), Some(c)) => Some(embedder::seg_similarity_pct(r, c)),
                _ => None,
            };
            let proj = match (&ref_proj, &e.proj) {
                (Some(r), Some(c))
                    if embedder::is_plausible_proj(r) && embedder::is_plausible_proj(c) =>
                {
                    Some(embedder::proj_similarity_pct(r, c))
                }
                _ => None,
            };
            let bust = match (&ref_bust, &e.bust) {
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
            // Soft stature proximity: 100% at equal height, falling ~linearly to
            // 0 by ~40cm apart. Rank-normalised in the blend, so it nudges
            // similar-height candidates up without dropping anyone.
            let height = match (ref_height, recommender::performer_height_cm(&e.performer)) {
                (Some(r), Some(c)) => Some((100.0 * (1.0 - (r - c).abs() / 40.0)).max(0.0)),
                _ => None,
            };
            // Build size: BMI proximity (100% at equal BMI, ~0 by 8 apart) — the
            // fuller-vs-slimmer frame the scale-free vectors can't see.
            let size = match (ref_bmi, recommender::performer_bmi(&e.performer)) {
                (Some(r), Some(c)) => Some((100.0 * (1.0 - (r - c).abs() / 8.0)).max(0.0)),
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
                    size,
                },
                e.performer,
            )
        })
        .collect();

    let raw: Vec<blend::ModalityScores> = candidates.iter().map(|(m, _)| m.clone()).collect();
    let weights = if body_only {
        blend::Weights::body_only()
    } else if lookalike {
        blend::Weights::lookalike()
    } else {
        blend::Weights::default()
    };
    let scores = blend::blend_scores(&raw, &weights);
    let mut ranked: Vec<(f64, blend::ModalityScores, models::Performer)> = scores
        .into_iter()
        .zip(candidates)
        .map(|(sc, (m, p))| (sc, m, p))
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);

    if ranked.is_empty() {
        println!("{}", "No comparable candidates in the index.".yellow());
        return Ok(());
    }

    println!();
    println!(
        "{}",
        format!(
            "Top {} by {} to {}:",
            ranked.len(),
            if body_only {
                "body-type match"
            } else if lookalike {
                "look-alike (face-led) blend"
            } else {
                "multi-modal blend"
            },
            reference.name
        )
        .bright_cyan()
        .bold()
    );
    // Legend: the bar is the headline blend; the indented row below each result
    // breaks it into the eight modalities, grouped face │ body │ build-&-fit.
    println!(
        "{}",
        "  bar = blend 0–100      breakdown below: \
         face │ frame curves proj bust │ stats height build  (· = n/a)"
            .bright_black()
    );
    println!();
    let sep = "│".bright_black().to_string();
    let field = |label: &str, v: Option<f64>| {
        format!("{} {}", label.bright_black(), super::modality_cell(v))
    };
    let img_cache = if images { ImageCache::new().ok() } else { None };
    for (i, (score, m, p)) in ranked.iter().enumerate() {
        let ht = recommender::performer_height_cm(p)
            .map(|h| format!(" · {:.0}cm", h))
            .unwrap_or_default();
        let look = format!(
            "{}{}{}",
            p.ethnicity.as_deref().unwrap_or("?"),
            p.measurements
                .as_ref()
                .map(|m| format!(" · {}", m))
                .unwrap_or_default(),
            ht,
        );
        // Headline row: rank, name (aligned), blend bar + score, then the look.
        println!(
            "  {:>2}  {}  {} {}   {}",
            (i + 1).to_string().bright_black(),
            super::pad_name(&p.name, 20).bright_white().bold(),
            super::score_bar(*score, 10),
            format!("{:>3.0}", score).bright_cyan().bold(),
            look.bright_black(),
        );
        // Breakdown row: the eight modalities, grouped and column-aligned.
        println!(
            "      {}  {}  {} {} {} {}  {}  {} {} {}",
            field("face", m.face),
            sep,
            field("frame", m.build),
            field("curves", m.volume),
            field("proj", m.proj),
            field("bust", m.bust),
            sep,
            field("stats", m.meas),
            field("height", m.height),
            field("build", m.size),
        );
        if let Some(url) = &p.source_url {
            println!("      {} {}", "↳".bright_black(), url.blue().underline());
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
