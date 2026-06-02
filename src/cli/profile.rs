//! Taste-profile handlers: the preference tree (`profile`), `recommend`, and
//! `clusters`/`similar`.
use super::*;
pub(crate) fn show_profile(db: &Database, mermaid: bool) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;

    if performers.is_empty() {
        println!(
            "{}",
            "No performers in database yet. Add some with 'luminary add'.".yellow()
        );
        return Ok(());
    }

    let total = performers.len();
    let tree = recommender::build_preference_tree(&performers);
    let path = recommender::dominant_query_path(&tree);

    // Mermaid export: just print the diagram source (paste into any renderer).
    if mermaid {
        print!("{}", recommender::to_mermaid(&tree, total));
        return Ok(());
    }

    println!("{}", "Your Taste Profile".bright_cyan().bold());
    println!("{}", "═".repeat(42).bright_black());
    println!(
        "{}",
        format!("  Based on {} liked performers", total).bright_black()
    );
    println!();

    recommender::print_tree(&tree, "  ", total);

    println!();
    if path.is_empty() {
        println!(
            "{}",
            "  Add more performers to refine your type.".bright_black()
        );
    } else {
        println!(
            "{} {}",
            "  Your type:".bright_black(),
            path.join(" → ").bright_white().bold()
        );
        println!(
            "{}",
            "  (The deeper the tree, the more specific your recommendations)".bright_black()
        );
    }
    println!();

    Ok(())
}

pub(crate) async fn recommend(
    db: &Database,
    limit: usize,
    images: bool,
    by_cluster: bool,
) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;
    if performers.is_empty() {
        println!(
            "{}",
            "No performers in database yet. Add some with 'luminary add'.".yellow()
        );
        return Ok(());
    }

    let cfg = config::Config::load();
    let api_key = cfg.resolve_api_key()?;
    let known_names: std::collections::HashSet<String> =
        performers.iter().map(|p| p.name.to_lowercase()).collect();
    let img_cache = if images { ImageCache::new().ok() } else { None };

    if by_cluster {
        let vecs: Vec<Vec<f32>> = performers.iter().map(recommender::cluster_vector).collect();
        let k = auto_k(performers.len());
        let assign = recommender::kmeans(&vecs, k);

        println!(
            "{}",
            format!("Recommendations across your {} taste clusters:", k)
                .bright_cyan()
                .bold()
        );
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for c in 0..k {
            let members: Vec<models::Performer> = performers
                .iter()
                .zip(&assign)
                .filter(|(_, &a)| a == c)
                .map(|(p, _)| p.clone())
                .collect();
            if members.is_empty() {
                continue;
            }
            println!();
            println!(
                "{}",
                format!("◆ {} ({} of yours)", cluster_label(&members), members.len())
                    .bright_white()
                    .bold()
            );
            let mut scored = recommend_pool(&members, &cfg, &api_key, &known_names).await?;
            scored.retain(|(_, p)| seen.insert(p.name.to_lowercase()));
            scored.truncate(5);
            if scored.is_empty() {
                println!("  {}", "(no new matches)".bright_black());
            } else {
                print_recs(&scored, &img_cache).await;
            }
        }
    } else {
        println!(
            "{}",
            "Finding performers you might like...".bright_cyan().bold()
        );
        let mut scored = recommend_pool(&performers, &cfg, &api_key, &known_names).await?;
        scored.truncate(limit);
        if scored.is_empty() {
            println!("{}", "No matching recommendations found.".yellow());
            return Ok(());
        }
        println!();
        print_recs(&scored, &img_cache).await;
    }

    println!();
    println!(
        "{}",
        "Use 'luminary add <name>' to add any to your profile.".bright_black()
    );
    Ok(())
}

/// Detect and print the taste clusters in the library.
pub(crate) fn show_clusters(db: &Database, k: Option<usize>) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;
    if performers.len() < 2 {
        println!("{}", "Add a few more performers to find clusters.".yellow());
        return Ok(());
    }
    let vecs: Vec<Vec<f32>> = performers.iter().map(recommender::cluster_vector).collect();
    let k = k.unwrap_or_else(|| auto_k(performers.len()));
    let assign = recommender::kmeans(&vecs, k);

    println!(
        "{}",
        format!(
            "Your taste clusters ({} from {} performers)",
            k,
            performers.len()
        )
        .bright_cyan()
        .bold()
    );

    for c in 0..k {
        let members: Vec<&models::Performer> = performers
            .iter()
            .zip(&assign)
            .filter(|(_, &a)| a == c)
            .map(|(p, _)| p)
            .collect();
        if members.is_empty() {
            continue;
        }
        let owned: Vec<models::Performer> = members.iter().map(|p| (*p).clone()).collect();
        println!();
        println!(
            "{}",
            format!("◆ {} ({})", cluster_label(&owned), members.len())
                .bright_white()
                .bold()
        );
        println!(
            "  {}",
            members
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
                .bright_black()
        );
    }
    println!();
    println!(
        "{}",
        "Tip: 'luminary recommend --by-cluster' recommends per cluster.".bright_black()
    );
    Ok(())
}

pub(crate) async fn similar(
    db: &Database,
    name: &str,
    limit: usize,
    images: bool,
) -> anyhow::Result<()> {
    let performer = db.get_performer(name)?.ok_or_else(|| {
        anyhow::anyhow!(
            "'{}' not found in your database. Add them first with 'luminary add'.",
            name
        )
    })?;

    let tpdb_uuid = performer
        .source_url
        .as_deref()
        .and_then(|url| url.split('/').next_back())
        .ok_or_else(|| anyhow::anyhow!("No TPDB ID stored for '{}'", name))?
        .to_string();

    let cfg = config::Config::load();
    let api_key = cfg.resolve_api_key()?;
    let client = TpdbClient::new(api_key);

    println!(
        "{} {}",
        "Finding performers similar to".bright_cyan(),
        performer.name.bright_white().bold()
    );
    println!(
        "{} {}  {}  {}",
        " ".bright_black(),
        performer.body_type.bright_black(),
        performer.ethnicity.as_deref().unwrap_or("?").bright_black(),
        performer
            .hair_color
            .as_deref()
            .unwrap_or("?")
            .bright_black(),
    );
    println!();

    let known_names: std::collections::HashSet<String> = db
        .get_all_performers()?
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();

    let mut results = client.similar_to(&tpdb_uuid, &cfg.gender_filter).await?;
    results.retain(|p| !known_names.contains(&p.name.to_lowercase()));

    // Score each result against the reference performer
    let ref_embedding = db.get_embedding(&performer.name).ok().flatten();
    let has_face_ml = ref_embedding.is_some();

    let mut scored: Vec<(f64, Option<f32>, models::Performer)> = results
        .into_iter()
        .map(|p| {
            let attr_score = recommender::score_against(&p, &performer);
            let face_sim = ref_embedding.as_ref().and_then(|ref_emb| {
                db.get_embedding(&p.name)
                    .ok()
                    .flatten()
                    .map(|e| embedder::cosine_similarity(ref_emb, &e))
            });
            // Combined score: attributes (70%) + face similarity (30%) if available
            let combined = match face_sim {
                Some(fs) => attr_score * 0.70 + embedder::similarity_pct(fs) as f64 * 0.30,
                None => attr_score,
            };
            (combined, face_sim, p)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    if scored.is_empty() {
        println!("{}", "No similar performers found.".yellow());
        return Ok(());
    }

    println!(
        "{}",
        format!(
            "Top {} similar to {}{}:",
            scored.len(),
            performer.name,
            if has_face_ml {
                " (attr + face)"
            } else {
                " (attributes)"
            }
        )
        .bright_cyan()
        .bold()
    );
    println!();

    let img_cache = if images { ImageCache::new().ok() } else { None };

    for (i, (score, face_sim, p)) in scored.iter().enumerate() {
        let age_str = p
            .age
            .map(|a| format!(", {}", recommender::age_bucket(a)))
            .unwrap_or_default();
        let face_str = face_sim
            .map(|s| format!("  face {:.0}%", embedder::similarity_pct(s)))
            .unwrap_or_default();
        println!(
            "{}. {} {}  {}{}",
            (i + 1).to_string().bright_black(),
            p.name.bright_white().bold(),
            format!(
                "({}, {}{}{})",
                p.body_type,
                p.ethnicity.as_deref().unwrap_or("?"),
                p.hair_color
                    .as_ref()
                    .map(|h| format!(", {}", h))
                    .unwrap_or_default(),
                age_str,
            )
            .bright_black(),
            format!("match {:.0}%", score).bright_cyan(),
            face_str.bright_black(),
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
    if !has_face_ml {
        println!(
            "{}",
            "  Tip: run 'luminary embed' to add face similarity scoring".bright_black()
        );
    }
    println!(
        "{}",
        "Use 'luminary add <name>' to add any to your profile.".bright_black()
    );
    Ok(())
}
