//! The `stats` lens: rank the cached index by recorded measurements.
use super::super::*;

/// Rank the cached index by recorded build (WHR / hips / cup / height / …)
/// similarity to the reference. Uses no reference images, so it works for niche
/// performers with no clean full-body photo — and surfaces recognizable indexed
/// performers rather than obscure TPDB-wide stubs.
pub(super) async fn search_by_measure(
    db: &Database,
    reference: &models::Performer,
    limit: usize,
    images: bool,
    band: Option<(f64, f64)>,
    hair: Option<&str>,
) -> anyhow::Result<()> {
    let ref_vec = recommender::feature_vector(reference)
        .context("the reference has no usable measurements (need bust/waist/hips) to match on.")?;
    let index = db.load_body_index()?;
    if index.is_empty() {
        anyhow::bail!("body index is empty — run 'luminary index' first.");
    }
    let known: std::collections::HashSet<String> = db
        .get_all_performers()?
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();
    let ref_lc = reference.name.to_lowercase();

    println!(
        "{}",
        format!(
            "Ranking {} indexed performers by build/measurements to {}...",
            index.len(),
            reference.name
        )
        .bright_cyan()
    );

    let mut scored: Vec<(f64, models::Performer)> = index
        .into_iter()
        .filter(|e| {
            let n = e.performer.name.to_lowercase();
            n != ref_lc
                && !known.contains(&n)
                && super::in_band(band, &e.performer)
                && super::hair_match(hair, &e.performer)
        })
        .filter_map(|e| {
            recommender::feature_vector(&e.performer)
                .map(|cv| (ref_vec.similarity_pct(&cv), e.performer))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    if scored.is_empty() {
        println!(
            "{}",
            "No indexed performers with measurements to compare.".yellow()
        );
        return Ok(());
    }

    println!();
    println!(
        "{}",
        format!(
            "Top {} by build/measurements to {}:",
            scored.len(),
            reference.name
        )
        .bright_cyan()
        .bold()
    );
    println!();
    let img_cache = if images { ImageCache::new().ok() } else { None };
    for (i, (pct, p)) in scored.iter().enumerate() {
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
        println!(
            "  {:>2}  {}  {} {}   {}",
            (i + 1).to_string().bright_black(),
            super::pad_name(&p.name, 20).bright_white().bold(),
            super::score_bar(*pct, 10),
            format!("{:>3.0}", pct).bright_cyan().bold(),
            look.bright_black(),
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
