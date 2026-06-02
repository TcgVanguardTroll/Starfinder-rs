//! Library management handlers: add, list, search, view, remove, stats,
//! clear-cache, aliases, and JSON import.
use super::*;
pub(crate) async fn add_performers(db: &Database, names: Vec<String>) -> anyhow::Result<()> {
    println!(
        "{}",
        "Adding performers to your profile...".bright_cyan().bold()
    );
    println!();

    let cfg = config::Config::load();
    let api_key = cfg.resolve_api_key().ok();
    let tpdb_client = api_key.as_ref().map(|key| TpdbClient::new(key.clone()));
    let stash_client = cfg
        .stashdb_key
        .as_ref()
        .filter(|k| !k.is_empty())
        .map(|k| StashdbClient::new(k.clone()));
    let scraper = Scraper::new();

    if tpdb_client.is_some() {
        println!("{}", "Using ThePornDB API".bright_green());
    } else {
        println!(
            "{}",
            "No API key found, using web scraping (may fail)".yellow()
        );
        println!(
            "{}",
            "   Set one with 'luminary config api-key <key>' for better results".bright_black()
        );
    }
    println!();

    for name in names {
        print!(
            "{} ",
            format!("Fetching data for {}...", name).bright_black()
        );
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut performer_data = None;

        if let Some(ref client) = tpdb_client {
            match client.search_performer(&name).await {
                Ok(Some(performer)) => {
                    performer_data = Some(performer);
                }
                Ok(None) => {
                    log::warn!("ThePornDB: No results for {}", name);
                }
                Err(e) => {
                    log::warn!("ThePornDB error for {}: {}", name, e);
                }
            }
        }

        if performer_data.is_none() {
            match scraper.scrape_performer(&name).await {
                Ok(performer) => {
                    performer_data = Some(performer);
                }
                Err(e) => {
                    log::warn!("Scraper error for {}: {}", name, e);
                }
            }
        }

        match performer_data {
            Some(performer) => {
                let source = performer.source.as_deref().unwrap_or("Unknown");
                match db.add_performer(&performer) {
                    Ok(_) => {
                        // Auto-save alias if user typed a different name than TPDB returned
                        if name.to_lowercase() != performer.name.to_lowercase() {
                            let _ = db.save_alias(&name, &performer.name);
                        }

                        println!(
                            "\r{} {} {}{}",
                            "[OK]".green(),
                            "Added:".white(),
                            performer.name.bright_white().bold(),
                            if name.to_lowercase() != performer.name.to_lowercase() {
                                format!(" (alias: {})", name).bright_black().to_string()
                            } else {
                                String::new()
                            }
                        );
                        println!(
                            "     {}",
                            format!("({}, {})", performer.body_type, source).bright_black()
                        );
                        // Build a robust centroid embedding from several images
                        // (StashDB multi-image + TPDB face/profile/gallery)
                        let urls = build_centroid_urls(&performer, stash_client.as_ref()).await;
                        if !urls.is_empty() {
                            match embedder::generate_centroid_embedding(&urls) {
                                Ok(Some(emb)) => {
                                    let _ = db.save_embedding(&performer.name, &emb);
                                    println!(
                                        "     {} face embedding stored ({} image{})",
                                        "↳".bright_black(),
                                        urls.len(),
                                        if urls.len() == 1 { "" } else { "s" }
                                    );
                                }
                                Ok(None) => {
                                    println!(
                                        "     {} {}",
                                        "↳".bright_black(),
                                        "no face detected — skipping embedding".bright_black()
                                    );
                                    log::debug!("No face detected for {}", name);
                                }
                                Err(e) => {
                                    // A sidecar hiccup, not a real no-face result —
                                    // say so (and that it's retryable) instead of
                                    // silently dropping the embedding.
                                    println!(
                                        "     {} {}",
                                        "↳".bright_black(),
                                        "embedding failed — retry with 'luminary embed'".yellow()
                                    );
                                    log::warn!("embedding sidecar failed for {}: {:#}", name, e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!(
                            "\r{} {} {}: {}",
                            "[X]".red(),
                            "Failed to save:".white(),
                            name,
                            e
                        );
                    }
                }
            }
            None => {
                println!("\r{} {} {}", "[X]".red(), "Not found:".white(), name);
            }
        }
    }

    println!();
    println!(
        "{}",
        "Tip: Use 'luminary search' to find similar performers".bright_black()
    );
    Ok(())
}

pub(crate) fn list_performers(db: &Database) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;

    if performers.is_empty() {
        println!("{}", "No performers in database yet.".yellow());
        println!(
            "{}",
            "Use 'luminary add <names...>' to add some!".bright_black()
        );
        return Ok(());
    }

    println!(
        "{}",
        format!("Your Performers ({} total)", performers.len())
            .bright_cyan()
            .bold()
    );
    println!();

    for performer in performers {
        println!(
            "{} {} {}",
            "*".bright_black(),
            performer.name.bright_white().bold(),
            format!("({})", performer.body_type).bright_black()
        );
    }

    Ok(())
}

pub(crate) async fn search_performers(
    db: &Database,
    body_type: Option<String>,
    age_min: Option<u32>,
    age_max: Option<u32>,
    ethnicity: Option<String>,
    hair_color: Option<String>,
    _show_images: bool,
    limit: usize,
) -> anyhow::Result<()> {
    let filters = SearchFilters {
        body_type,
        age_min,
        age_max,
        ethnicity,
        hair_color,
        categories: Vec::new(),
        min_score: None,
    };

    let results = db.search(&filters)?;

    if results.is_empty() {
        println!("{}", "No matches found.".yellow());
        return Ok(());
    }

    println!(
        "{}",
        format!("Search Results ({} matches)", results.len())
            .bright_cyan()
            .bold()
    );
    println!();

    for (i, performer) in results.iter().take(limit).enumerate() {
        println!(
            "{}. {} {}",
            (i + 1).to_string().bright_black(),
            performer.name.bright_white().bold(),
            format!(
                "(Age: {}, Body: {})",
                performer
                    .age
                    .map(|a| a.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                performer.body_type
            )
            .bright_black()
        );
    }

    Ok(())
}

pub(crate) async fn view_performer(
    db: &Database,
    name: &str,
    _gallery: bool,
) -> anyhow::Result<()> {
    match db.get_performer(name)? {
        Some(performer) => {
            println!("{}", "=".repeat(50).bright_black());
            println!("{}", performer.name.bright_white().bold());
            println!("{}", "=".repeat(50).bright_black());
            println!();

            println!("  {} {}", "Body Type:".bright_black(), performer.body_type);
            if let Some(age) = performer.age {
                println!("  {} {}", "Age:".bright_black(), age);
            }
            if let Some(eth) = &performer.ethnicity {
                println!("  {} {}", "Ethnicity:".bright_black(), eth);
            }
            if let Some(hair) = &performer.hair_color {
                println!("  {} {}", "Hair:".bright_black(), hair);
            }
            if let Some(eye) = &performer.eye_color {
                println!("  {} {}", "Eyes:".bright_black(), eye);
            }
            if let Some(meas) = &performer.measurements {
                let boob_tag = match performer.fake_boobs {
                    Some(true) => "  (enhanced)",
                    Some(false) => "  (natural)",
                    None => "",
                };
                println!(
                    "  {} {}{}",
                    "Measurements:".bright_black(),
                    meas,
                    boob_tag.bright_black()
                );
            }
            if let Some(h) = &performer.height {
                println!("  {} {}", "Height:".bright_black(), h);
            }
            if let Some(t) = &performer.tattoos {
                println!("  {} {}", "Tattoos:".bright_black(), t);
            }
            if let Some(p) = &performer.piercings {
                println!("  {} {}", "Piercings:".bright_black(), p);
            }
            println!();
        }
        None => {
            println!(
                "{}",
                format!("Performer '{}' not found in database.", name).red()
            );
        }
    }
    Ok(())
}

pub(crate) fn remove_performer(db: &Database, name: &str) -> anyhow::Result<()> {
    db.remove_performer(name)?;
    println!("{} {}", "Removed:".green(), name);
    Ok(())
}

pub(crate) fn show_stats(db: &Database) -> anyhow::Result<()> {
    let count = db.count()?;
    let cache = image_cache::ImageCache::new()?;
    let cache_size = cache.cache_size()?;
    let cache_count = cache.cache_count()?;

    println!("{}", "Luminary Statistics".bright_cyan().bold());
    println!();
    println!(
        "  {} {}",
        "Performers:".bright_black(),
        count.to_string().bright_white()
    );
    println!(
        "  {} {}",
        "Cached Images:".bright_black(),
        cache_count.to_string().bright_white()
    );
    println!(
        "  {} {:.2} MB",
        "Cache Size:".bright_black(),
        cache_size as f64 / 1024.0 / 1024.0
    );
    println!();

    Ok(())
}

pub(crate) fn clear_cache() -> anyhow::Result<()> {
    let cache = image_cache::ImageCache::new()?;
    cache.clear()?;
    println!("{}", "Image cache cleared".green());
    Ok(())
}

pub(crate) fn manage_alias(
    db: &Database,
    alias: Option<String>,
    canonical: Option<String>,
    remove: bool,
) -> anyhow::Result<()> {
    match (alias, canonical, remove) {
        // List all aliases
        (None, _, _) => {
            let aliases = db.list_aliases()?;
            if aliases.is_empty() {
                println!("{}", "No aliases stored.".bright_black());
            } else {
                println!("{}", "Name Aliases".bright_cyan().bold());
                println!("{}", "═".repeat(35).bright_black());
                for (alias, canonical) in &aliases {
                    println!(
                        "  {} {} {}",
                        alias.bright_white(),
                        "→".bright_black(),
                        canonical.bright_white().bold()
                    );
                }
            }
        }
        // Remove an alias
        (Some(alias), _, true) => {
            db.remove_alias(&alias)?;
            println!("{} alias '{}'", "Removed".green(), alias);
        }
        // Add alias → canonical
        (Some(alias), Some(canonical), false) => {
            // Verify canonical exists
            if db.get_performer(&canonical)?.is_none() {
                println!(
                    "{} '{}' not found in database.",
                    "Warning:".yellow(),
                    canonical
                );
            }
            db.save_alias(&alias, &canonical)?;
            println!(
                "{} {} {} {}",
                "Saved:".green(),
                alias.bright_white(),
                "→".bright_black(),
                canonical.bright_white().bold()
            );
        }
        // Look up what an alias resolves to
        (Some(alias), None, false) => match db.resolve_alias(&alias)? {
            Some(canonical) => println!(
                "{} → {}",
                alias.bright_white(),
                canonical.bright_white().bold()
            ),
            None => println!("'{}' has no alias stored.", alias),
        },
    }
    Ok(())
}

pub(crate) fn import_from_json(db: &Database, file_path: &str) -> anyhow::Result<()> {
    use models::Performer;
    use std::fs;

    println!(
        "{}",
        format!("Importing performers from {}...", file_path)
            .bright_cyan()
            .bold()
    );
    println!();

    let json_data =
        fs::read_to_string(file_path).with_context(|| format!("Failed to read {}", file_path))?;

    let performers: Vec<Performer> = serde_json::from_str(&json_data)
        .with_context(|| format!("Failed to parse JSON from {}", file_path))?;

    let mut success_count = 0;
    let mut fail_count = 0;

    for performer in performers {
        match db.add_performer(&performer) {
            Ok(_) => {
                println!(
                    "{} {} {}",
                    "[OK]".green(),
                    "Imported:".white(),
                    performer.name.bright_white().bold()
                );
                success_count += 1;
            }
            Err(e) => {
                println!(
                    "{} {} {}: {}",
                    "[X]".red(),
                    "Failed:".white(),
                    performer.name,
                    e
                );
                fail_count += 1;
            }
        }
    }

    println!();
    println!(
        "{}",
        format!(
            "Imported {} performers ({} failed)",
            success_count, fail_count
        )
        .green()
    );
    Ok(())
}
