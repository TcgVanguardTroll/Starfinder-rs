//! `find`: attribute + face search over a fresh TPDB candidate pool.
use super::super::*;

pub(crate) async fn find(
    db: &Database,
    like: Option<String>,
    match_mode: Option<String>,
    looks_like: Vec<String>,
    body_like: Vec<String>,
    hair_arg: Option<String>,
    eye_arg: Option<String>,
    eth_arg: Option<String>,
    cup_arg: Option<String>,
    hips_arg: Option<u32>,
    waist_arg: Option<u32>,
    whr_arg: Option<f64>,
    tattoo_arg: Option<String>,
    region: Option<String>,
    face_only: bool,
    images: bool,
    age_min: Option<u32>,
    age_max: Option<u32>,
    limit: usize,
) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let api_key = cfg.resolve_api_key()?;

    // `--like X --match <mode>` is the simple single-reference form; it maps onto
    // the looks_like / body_like / face_only / blend machinery below.
    let (mut looks_like, mut body_like, mut face_only) = (looks_like, body_like, face_only);
    let mut blend = false;
    if let Some(name) = like {
        match match_mode.as_deref().unwrap_or("both") {
            "face" => {
                looks_like.push(name);
                face_only = true;
            }
            "body" => {
                body_like.push(name);
            }
            _ => {
                // both: face ranking + body filters, scored as a 50/50 blend
                looks_like.push(name.clone());
                body_like.push(name);
                blend = true;
            }
        }
    }

    // Validate region up front so a typo fails fast with the valid options.
    if let Some(ref r) = region {
        if !luminary::region::known_regions().contains(&r.to_lowercase().as_str()) {
            anyhow::bail!(
                "Unknown region '{}'. Options: {}",
                r,
                luminary::region::known_regions().join(", ")
            );
        }
    }

    // ── Pull attributes from named performers ─────────────────────────────
    let mut ethnicity = eth_arg;
    let mut hair = hair_arg;
    let mut eye = eye_arg;
    let cup = cup_arg;
    let hips = hips_arg;
    let mut waist = waist_arg;
    let mut whr = whr_arg;

    // Resolve every reference performer up front.
    let resolve = |names: &[String]| -> anyhow::Result<Vec<models::Performer>> {
        names
            .iter()
            .map(|n| {
                db.get_performer(n)?
                    .ok_or_else(|| anyhow::anyhow!("'{}' not in database", n))
            })
            .collect()
    };
    let looks_refs = resolve(&looks_like)?;
    let body_refs_explicit = resolve(&body_like)?;

    let face_ranking = looks_refs
        .iter()
        .any(|p| db.get_embedding(&p.name).ok().flatten().is_some());

    // Coloring filters only when there's a single face reference; blending
    // multiple faces spans colorings, so the face embedding handles that.
    if looks_refs.len() == 1 {
        let p = &looks_refs[0];
        if ethnicity.is_none() {
            ethnicity = p.ethnicity.clone();
        }
        if hair.is_none() {
            hair = p.hair_color.clone();
        }
        if eye.is_none() && !face_ranking {
            eye = p.eye_color.clone();
        }
    }

    // Body references: explicit --body-like, else fall back to --looks-like.
    // Skipped entirely in --face-only mode.
    let body_refs: Vec<models::Performer> = if face_only {
        Vec::new()
    } else if !body_refs_explicit.is_empty() {
        body_refs_explicit
    } else {
        looks_refs.clone()
    };

    // Only WHR (the defining build shape) becomes a derived hard filter, and
    // only with a single body reference; cup/hips/height/weight are captured by
    // the k-NN ranking. (Explicit --cup/--hips still hard-filter.)
    if whr.is_none() && body_refs.len() == 1 {
        whr = recommender::performer_whr(&body_refs[0]);
    }
    if whr.is_some() {
        waist = None;
    } // redundant with WHR

    // ── Tattoo: SOFT bonus only — from the first body reference ───────────
    let tattoo_pref: Option<String> = if face_only {
        None
    } else {
        tattoo_arg.clone().or_else(|| {
            body_refs.first().and_then(|p| {
                let locs = recommender::parse_tattoos(p.tattoos.as_deref());
                locs.iter()
                    .find(|t| t.contains("lower back"))
                    .cloned()
                    .or_else(|| locs.iter().find(|t| t.contains("back")).cloned())
            })
        })
    };

    // ── Display criteria ──────────────────────────────────────────────────
    println!("{}", "Searching for performers with:".bright_cyan().bold());
    let mut criteria: Vec<String> = vec![];
    if !looks_like.is_empty() {
        criteria.push(format!(
            "face like {}",
            looks_like.join(" + ").bright_white()
        ));
    }
    if !body_like.is_empty() {
        criteria.push(format!(
            "body like {}",
            body_like.join(" + ").bright_white()
        ));
    }
    if let Some(ref v) = ethnicity {
        criteria.push(format!("ethnicity: {}", v.bright_white()));
    }
    if let Some(ref v) = hair {
        criteria.push(format!("hair: {}", v.bright_white()));
    }
    if let Some(ref v) = eye {
        criteria.push(format!("eyes: {}", v.bright_white()));
    }
    if let Some(ref v) = cup {
        criteria.push(format!("cup: {}", v.bright_white()));
    }
    if let Some(v) = hips {
        criteria.push(format!("hips: {}\" ±4", v.to_string().bright_white()));
    }
    if let Some(v) = waist {
        criteria.push(format!("waist: {}\" ±4", v.to_string().bright_white()));
    }
    if let Some(v) = whr {
        criteria.push(
            format!("waist-to-hip ratio: {:.3} ±0.05 (butt shape)", v)
                .bright_white()
                .to_string(),
        );
    }
    if let Some(ref v) = tattoo_pref {
        criteria.push(format!(
            "tattoo bonus: {} (preferred, not required)",
            v.bright_white()
        ));
    }
    if let Some(ref v) = region {
        criteria.push(format!(
            "region: {} (any nationality in the group)",
            v.bright_white()
        ));
    }
    if let Some(v) = age_min {
        criteria.push(format!("age ≥ {}", v.to_string().bright_white()));
    }
    if let Some(v) = age_max {
        criteria.push(format!("age ≤ {}", v.to_string().bright_white()));
    }
    for c in &criteria {
        println!("  · {}", c);
    }
    println!();

    let known_names: std::collections::HashSet<String> = db
        .get_all_performers()?
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();

    let client = TpdbClient::new(api_key);
    let mut results = client
        .search_by_attributes(
            ethnicity.as_deref(),
            hair.as_deref(),
            eye.as_deref(),
            cup.as_deref(),
            hips,
            waist,
            whr,
            age_min,
            age_max,
            &cfg.gender_filter,
            // Region filtering keeps only a fraction of the pool, so fetch more.
            if region.is_some() { 100 } else { limit * 8 },
        )
        .await?;
    results.retain(|p| !known_names.contains(&p.name.to_lowercase()));

    // Region / nationality group filter (e.g. Slavic = Russian, Polish, …)
    if let Some(ref r) = region {
        results.retain(|p| {
            luminary::region::in_region(p.nationality.as_deref(), p.birthplace_code.as_deref(), r)
        });
    }

    // ── Ranking references (averaged across multiple references) ──────────
    let looks_embeddings: Vec<Vec<f32>> = looks_refs
        .iter()
        .filter_map(|p| db.get_embedding(&p.name).ok().flatten())
        .collect();
    let ref_embedding = embedder::average_embeddings(&looks_embeddings);

    let body_vecs: Vec<_> = body_refs
        .iter()
        .filter_map(recommender::feature_vector)
        .collect();
    let ref_body_vec = recommender::FeatureVec::average(&body_vecs);

    // Cap on-the-fly embeddings; pre-rank by body distance so we spend them well.
    const MAX_ONTHEFLY_EMBEDS: usize = 16;
    if ref_embedding.is_some() {
        if let Some(rv) = &ref_body_vec {
            results.sort_by(|a, b| {
                let da = recommender::feature_vector(a)
                    .map(|v| rv.distance(&v))
                    .unwrap_or(f64::MAX);
                let db_ = recommender::feature_vector(b)
                    .map(|v| rv.distance(&v))
                    .unwrap_or(f64::MAX);
                da.partial_cmp(&db_).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    let mut embeds_done = 0usize;

    // Score each candidate. (sort_key, face_sim, body_sim, has_stamp, performer)
    //
    // When looks-like provides a face embedding, FACIAL similarity drives the
    // ranking — a candidate with a real face match must outrank one scored only
    // by body. We can't compare a face cosine (~50–65%) against a body % (~90%)
    // on the same scale, so face-bearing candidates are lifted into a higher
    // band (+1000) and ordered by face among themselves; body-only candidates
    // fall below them, ordered by build. Without a face embedding we rank by body.
    let face_ranks = ref_embedding.is_some();

    let mut scored: Vec<(f64, Option<f32>, Option<f64>, bool, models::Performer)> = results
        .into_iter()
        .map(|p| {
            let face = ref_embedding.as_ref().and_then(|ref_emb| {
                let emb = db.get_embedding_any(&p.name).ok().flatten().or_else(|| {
                    if embeds_done >= MAX_ONTHEFLY_EMBEDS {
                        return None;
                    }
                    embeds_done += 1;
                    p.face_url
                        .as_deref()
                        .or(p.profile_image_url.as_deref())
                        .and_then(|url| {
                            // Persist to the local face corpus so it's free next time.
                            embedder::generate_embedding(url).ok().inspect(|e| {
                                let _ = db.save_candidate(&p, e);
                            })
                        })
                });
                emb.map(|e| embedder::cosine_similarity(ref_emb, &e))
            });
            let body = ref_body_vec
                .as_ref()
                .and_then(|rv| recommender::feature_vector(&p).map(|cv| rv.similarity_pct(&cv)));
            let has_stamp = tattoo_pref
                .as_ref()
                .is_some_and(|kw| recommender::has_tattoo(&p, kw));
            let stamp_bonus = if has_stamp { 5.0 } else { 0.0 };

            let sort_key = if blend {
                // "both": 50/50 blend of face and build similarity
                match (face, body) {
                    (Some(f), Some(b)) => {
                        0.5 * embedder::similarity_pct(f) as f64 + 0.5 * b + stamp_bonus
                    }
                    (Some(f), None) => embedder::similarity_pct(f) as f64 + stamp_bonus,
                    (None, Some(b)) => b + stamp_bonus,
                    (None, None) => stamp_bonus,
                }
            } else if face_ranks {
                match face {
                    // Face match: high band, ordered by facial similarity
                    Some(f) => 1000.0 + embedder::similarity_pct(f) as f64 + stamp_bonus,
                    // No usable face image: ranked below all facial matches, by build
                    None => body.unwrap_or(0.0) + stamp_bonus,
                }
            } else {
                body.unwrap_or(0.0) + stamp_bonus
            };
            (sort_key, face, body, has_stamp, p)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    if scored.is_empty() {
        println!(
            "{}",
            "No results found. Try relaxing some filters.".yellow()
        );
        return Ok(());
    }

    let rank_by = if ref_embedding.is_some() {
        "face similarity"
    } else {
        "body/build similarity"
    };
    println!(
        "{}",
        format!(
            "Top {} matches (ranked by {}, +tattoo bonus):",
            scored.len(),
            rank_by
        )
        .bright_cyan()
        .bold()
    );
    println!();

    let img_cache = if images { ImageCache::new().ok() } else { None };

    for (i, (_score, face, body, has_stamp, p)) in scored.iter().enumerate() {
        let age_str = p
            .age
            .map(|a| format!(", {}", recommender::age_bucket(a)))
            .unwrap_or_default();
        let meas_str = p
            .measurements
            .as_deref()
            .map(|m| {
                let parts: Vec<&str> = m.split('-').collect();
                if parts.len() >= 3 {
                    let whr_str = recommender::performer_whr(p)
                        .map(|r| format!(" whr {:.2}", r))
                        .unwrap_or_default();
                    let ht_str = recommender::performer_height_cm(p)
                        .map(|h| format!(" {:.0}cm", h))
                        .unwrap_or_default();
                    format!(
                        ", {}w {}h{}{}",
                        parts[1].trim(),
                        parts[2]
                            .trim()
                            .trim_end_matches(|c: char| !c.is_ascii_digit()),
                        whr_str,
                        ht_str
                    )
                } else {
                    String::new()
                }
            })
            .unwrap_or_default();

        let mut tags = String::new();
        if let Some(s) = face {
            tags.push_str(&format!("  face {:.0}%", embedder::similarity_pct(*s)));
        }
        if let Some(b) = body {
            tags.push_str(&format!("  body {:.0}%", b));
        }
        let stamp = if *has_stamp {
            "  +stamp".to_string()
        } else {
            String::new()
        };

        println!(
            "{}. {} {}{}{}",
            (i + 1).to_string().bright_black(),
            p.name.bright_white().bold(),
            format!(
                "({}, {}{}{}{}{}{})",
                p.body_type,
                p.ethnicity.as_deref().unwrap_or("?"),
                p.nationality
                    .as_ref()
                    .filter(|n| !n.is_empty())
                    .map(|n| format!(", {}", n))
                    .unwrap_or_default(),
                p.hair_color
                    .as_ref()
                    .map(|h| format!(", {}", h))
                    .unwrap_or_default(),
                p.eye_color
                    .as_ref()
                    .map(|e| format!(", {} eyes", e))
                    .unwrap_or_default(),
                meas_str,
                age_str,
            )
            .bright_black(),
            tags.bright_cyan(),
            stamp.bright_green(),
        );
        // Clickable TPDB profile link disambiguates generic names
        if let Some(url) = &p.source_url {
            println!("   {} {}", "↳".bright_black(), url.blue().underline());
        }
        // Optional inline thumbnail
        if let Some(cache) = &img_cache {
            if let Some(url) = p.face_url.as_deref().or(p.profile_image_url.as_deref()) {
                render_thumbnail(cache, url).await;
            }
        }
    }
    println!();
    if ref_embedding.is_none() && !looks_like.is_empty() {
        println!(
            "{}",
            "  Tip: run 'luminary embed' first for ML face-similarity ranking".bright_black()
        );
    }
    println!(
        "{}",
        "Use 'luminary add <name>' to add any to your profile.".bright_black()
    );
    Ok(())
}
