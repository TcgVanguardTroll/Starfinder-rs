//! Image-based search handlers: `find` (attributes + face), `body-search`
//! (`--by build`/`volume`/`measurements`), and `face-search`.
use super::*;
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

/// Find performers with a similar body, ranked against the cached index.
///
/// `by` selects the lens:
///   - `build` (default): skeletal pose vector (shoulder/hip/leg proportions).
///   - `volume`: silhouette/segmentation vector (waist/hip/thigh fullness — the
///     butt & thigh shape the skeleton can't see). Built from a combined,
///     gated pool (pornpics + TPDB scenes + StashDB).
///   - `measurements`: recorded WHR/hips/cup (handled by `search_by_measure`).
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
    if by == "measurements" {
        return search_by_measure(db, &reference, limit, images).await;
    }
    // Multi-modal blend: fuse face + build + volume + projection + measurements.
    if by == "blend" {
        return search_blend(db, &reference, limit, images).await;
    }
    // `volume` = silhouette/segmentation lens; otherwise the default `build`
    // (skeletal pose) lens. Noun reused throughout the output.
    let volume = by == "volume";
    let kind = if volume { "volume" } else { "build" };
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

/// Rank the cached index by recorded build (WHR / hips / cup / height / …)
/// similarity to the reference. Uses no reference images, so it works for niche
/// performers with no clean full-body photo — and surfaces recognizable indexed
/// performers rather than obscure TPDB-wide stubs.
async fn search_by_measure(
    db: &Database,
    reference: &models::Performer,
    limit: usize,
    images: bool,
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
            n != ref_lc && !known.contains(&n)
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
            format!("build {:.0}%", pct).bright_cyan(),
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

/// The multi-modal `blend` lens: fuse face + build + volume + projection +
/// measurements into one rank. The reference's modalities are read entirely from
/// local data — its face embedding, its ingested image corpus (aggregated the
/// same way `aggregate` builds the index), and its recorded measurements — so the
/// blend needs no fresh gathering. Candidates are the cached body index, and each
/// modality is rank-normalised before blending so the scales stay comparable
/// (see `luminary::blend`).
async fn search_blend(
    db: &Database,
    reference: &models::Performer,
    limit: usize,
    images: bool,
) -> anyhow::Result<()> {
    // Reference modalities — all local, no gathering.
    let ref_face = db.get_embedding(&reference.name).ok().flatten();
    let ref_meas = recommender::feature_vector(reference);
    let ref_imgs = db.load_images(&reference.name, None)?;
    let (ref_pose, ref_seg, ref_proj) = if ref_imgs.is_empty() {
        (None, None, None)
    } else {
        let (p, s, pr, _) = luminary::database::aggregate_views(&ref_imgs);
        (p, s, pr)
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
        format!("Blending candidates by similarity to {}:", reference.name)
            .bright_cyan()
            .bold()
    );
    println!(
        "  {}  {}  {}  {}  {}",
        active("face", ref_face.is_some()),
        active("build", ref_pose.is_some()),
        active("volume", ref_seg.is_some()),
        active("proj", ref_proj.is_some()),
        active("measurements", ref_meas.is_some()),
    );
    if ref_face.is_none()
        && ref_pose.is_none()
        && ref_seg.is_none()
        && ref_proj.is_none()
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
                "  (no ingested body images — blending face + measurements only; \
                 run 'luminary ingest {}' to add build/volume/projection)",
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
            n != ref_lc && !known.contains(&n)
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
                (Some(r), Some(c)) => Some(embedder::seg_similarity_pct(r, c)),
                _ => None,
            };
            let meas = match (&ref_meas, recommender::feature_vector(&e.performer)) {
                (Some(r), Some(c)) => Some(r.similarity_pct(&c)),
                _ => None,
            };
            (
                blend::ModalityScores {
                    face,
                    build,
                    volume,
                    proj,
                    meas,
                },
                e.performer,
            )
        })
        .collect();

    let raw: Vec<blend::ModalityScores> = candidates.iter().map(|(m, _)| m.clone()).collect();
    let scores = blend::blend_scores(&raw, &blend::Weights::default());
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
            "Top {} by multi-modal blend to {}:",
            ranked.len(),
            reference.name
        )
        .bright_cyan()
        .bold()
    );
    println!();
    let img_cache = if images { ImageCache::new().ok() } else { None };
    for (i, (score, m, p)) in ranked.iter().enumerate() {
        let mut tags = String::new();
        for (label, v) in [
            ("face", m.face),
            ("build", m.build),
            ("vol", m.volume),
            ("proj", m.proj),
            ("meas", m.meas),
        ] {
            if let Some(v) = v {
                tags.push_str(&format!("  {} {:.0}%", label, v));
            }
        }
        let ht = recommender::performer_height_cm(p)
            .map(|h| format!(", {:.0}cm", h))
            .unwrap_or_default();
        println!(
            "{}. {} {}  {}{}",
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
            format!("blend {:.0}", score).bright_cyan().bold(),
            tags.bright_black(),
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
