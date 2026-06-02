use clap::{Parser, Subcommand};
use colored::*;
use anyhow::Context;

mod models;
mod database;
mod image_cache;
mod scraper;
mod tpdb;
mod recommender;
mod config;
mod embedder;

use database::Database;
use models::SearchFilters;
use scraper::Scraper;
use tpdb::TpdbClient;

#[derive(Parser)]
#[command(name = "starfinder")]
#[command(about = "Find your stars - A Rust-powered recommendation engine", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add performers you like to build your profile
    Add {
        /// Names of performers to add
        #[arg(required = true)]
        names: Vec<String>,
    },
    /// List all performers in your database
    List,
    /// Search for performers by attributes
    Search {
        #[arg(long)]
        body_type: Option<String>,
        #[arg(long)]
        age_min: Option<u32>,
        #[arg(long)]
        age_max: Option<u32>,
        #[arg(long)]
        ethnicity: Option<String>,
        #[arg(long)]
        hair_color: Option<String>,
        #[arg(long, default_value_t = false)]
        show_images: bool,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// View details of a specific performer
    View {
        name: String,
        #[arg(long, default_value_t = false)]
        gallery: bool,
    },
    /// Remove a performer from the database
    Remove {
        name: String,
    },
    /// Show statistics about your database
    Stats,
    /// Clear image cache
    ClearCache,
    /// Import performers from JSON file
    Import {
        #[arg(default_value = "performers_data.json")]
        file: String,
    },
    /// Show your taste profile based on liked performers
    Profile,
    /// Get performer recommendations based on your profile
    Recommend {
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Search by mixing attributes from stored performers or manual values
    Find {
        /// Copy face attributes (ethnicity, hair, eye) from this performer
        #[arg(long)]
        looks_like: Option<String>,
        /// Copy body shape (hips, waist, cup) from this performer
        #[arg(long)]
        body_like: Option<String>,
        /// Hair color (Blonde, Brunette, Black, Red, Auburn)
        #[arg(long)]
        hair: Option<String>,
        /// Eye color (Blue, Green, Brown, Hazel, Grey)
        #[arg(long)]
        eye: Option<String>,
        /// Ethnicity (Caucasian, Latin, Black, Asian, Indian)
        #[arg(long)]
        ethnicity: Option<String>,
        /// Cup size (A, B, C, D, DD, DDD)
        #[arg(long)]
        cup: Option<String>,
        /// Target hip measurement in inches (searches ±4 inches)
        #[arg(long)]
        hips: Option<u32>,
        /// Target waist measurement in inches (searches ±4 inches)
        #[arg(long)]
        waist: Option<u32>,
        /// Minimum age
        #[arg(long)]
        age_min: Option<u32>,
        /// Maximum age
        #[arg(long)]
        age_max: Option<u32>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Find performers similar to a specific one
    Similar {
        /// Name of the performer to base results on
        name: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Generate face embeddings for all performers missing one
    Embed,
    /// View or change settings
    Config {
        /// Setting to change (e.g. gender)
        key: Option<String>,
        /// New value (e.g. female, male, trans-female, trans-male, any)
        value: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    let db_path = get_db_path()?;
    let db = Database::new(&db_path)?;

    match cli.command {
        Commands::Add { names } => {
            add_performers(&db, names).await?;
        }
        Commands::List => {
            list_performers(&db)?;
        }
        Commands::Search {
            body_type,
            age_min,
            age_max,
            ethnicity,
            hair_color,
            show_images,
            limit,
        } => {
            search_performers(
                &db, body_type, age_min, age_max,
                ethnicity, hair_color, show_images, limit,
            )
            .await?;
        }
        Commands::View { name, gallery } => {
            view_performer(&db, &name, gallery).await?;
        }
        Commands::Remove { name } => {
            remove_performer(&db, &name)?;
        }
        Commands::Stats => {
            show_stats(&db)?;
        }
        Commands::ClearCache => {
            clear_cache()?;
        }
        Commands::Import { file } => {
            import_from_json(&db, &file)?;
        }
        Commands::Profile => {
            show_profile(&db)?;
        }
        Commands::Recommend { limit } => {
            recommend(&db, limit).await?;
        }
        Commands::Find { looks_like, body_like, hair, eye, ethnicity, cup, hips, waist, age_min, age_max, limit } => {
            find(&db, looks_like, body_like, hair, eye, ethnicity, cup, hips, waist, age_min, age_max, limit).await?;
        }
        Commands::Similar { name, limit } => {
            similar(&db, &name, limit).await?;
        }
        Commands::Embed => {
            embed_all(&db)?;
        }
        Commands::Config { key, value } => {
            configure(key, value)?;
        }
    }

    Ok(())
}

fn get_db_path() -> anyhow::Result<String> {
    let data_dir = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
    let db_dir = data_dir.join("starfinder");
    std::fs::create_dir_all(&db_dir)?;
    Ok(db_dir.join("starfinder.db").to_string_lossy().to_string())
}

async fn add_performers(db: &Database, names: Vec<String>) -> anyhow::Result<()> {
    println!("{}", "Adding performers to your profile...".bright_cyan().bold());
    println!();

    let api_key = std::env::var("TPDB_API_KEY").ok();
    let tpdb_client = api_key.as_ref().map(|key| TpdbClient::new(key.clone()));
    let scraper = Scraper::new();

    if tpdb_client.is_some() {
        println!("{}", "Using ThePornDB API".bright_green());
    } else {
        println!("{}", "No TPDB_API_KEY found, using web scraping (may fail)".yellow());
        println!("{}", "   Set TPDB_API_KEY environment variable for better results".bright_black());
    }
    println!();

    for name in names {
        print!("{} ", format!("Fetching data for {}...", name).bright_black());
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
                        println!("\r{} {} {} {}",
                            "[OK]".green(),
                            "Added:".white(),
                            name.bright_white().bold(),
                            format!("({}, {})", performer.body_type, source).bright_black()
                        );
                        // Try to generate face embedding silently
                        let face_url = performer.face_url.as_deref()
                            .or(performer.profile_image_url.as_deref());
                        if let Some(url) = face_url {
                            match embedder::generate_embedding(url) {
                                Ok(emb) => {
                                    let _ = db.save_embedding(&performer.name, &emb);
                                    println!("     {} face embedding stored",
                                        "↳".bright_black());
                                }
                                Err(e) => {
                                    log::debug!("Embedding skipped for {}: {}", name, e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("\r{} {} {}: {}",
                            "[X]".red(), "Failed to save:".white(), name, e);
                    }
                }
            }
            None => {
                println!("\r{} {} {}",
                    "[X]".red(), "Not found:".white(), name);
            }
        }
    }

    println!();
    println!("{}", "Tip: Use 'starfinder search' to find similar performers".bright_black());
    Ok(())
}

fn list_performers(db: &Database) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;

    if performers.is_empty() {
        println!("{}", "No performers in database yet.".yellow());
        println!("{}", "Use 'starfinder add <names...>' to add some!".bright_black());
        return Ok(());
    }

    println!("{}", format!("Your Performers ({} total)", performers.len()).bright_cyan().bold());
    println!();

    for performer in performers {
        println!("{} {} {}",
            "*".bright_black(),
            performer.name.bright_white().bold(),
            format!("({})", performer.body_type).bright_black()
        );
    }

    Ok(())
}

async fn search_performers(
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
        body_type, age_min, age_max, ethnicity, hair_color,
        categories: Vec::new(),
        min_score: None,
    };

    let results = db.search(&filters)?;

    if results.is_empty() {
        println!("{}", "No matches found.".yellow());
        return Ok(());
    }

    println!("{}", format!("Search Results ({} matches)", results.len()).bright_cyan().bold());
    println!();

    for (i, performer) in results.iter().take(limit).enumerate() {
        println!("{}. {} {}",
            (i + 1).to_string().bright_black(),
            performer.name.bright_white().bold(),
            format!("(Age: {}, Body: {})",
                performer.age.map(|a| a.to_string()).unwrap_or_else(|| "?".to_string()),
                performer.body_type
            ).bright_black()
        );
    }

    Ok(())
}

async fn view_performer(db: &Database, name: &str, _gallery: bool) -> anyhow::Result<()> {
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
                println!("  {} {}", "Measurements:".bright_black(), meas);
            }
            if let Some(h) = &performer.height {
                println!("  {} {}", "Height:".bright_black(), h);
            }
            println!();
        }
        None => {
            println!("{}", format!("Performer '{}' not found in database.", name).red());
        }
    }
    Ok(())
}

fn remove_performer(db: &Database, name: &str) -> anyhow::Result<()> {
    db.remove_performer(name)?;
    println!("{} {}", "Removed:".green(), name);
    Ok(())
}

fn show_stats(db: &Database) -> anyhow::Result<()> {
    let count = db.count()?;
    let cache = image_cache::ImageCache::new()?;
    let cache_size = cache.cache_size()?;
    let cache_count = cache.cache_count()?;

    println!("{}", "Starfinder Statistics".bright_cyan().bold());
    println!();
    println!("  {} {}", "Performers:".bright_black(), count.to_string().bright_white());
    println!("  {} {}", "Cached Images:".bright_black(), cache_count.to_string().bright_white());
    println!("  {} {:.2} MB", "Cache Size:".bright_black(),
        cache_size as f64 / 1024.0 / 1024.0);
    println!();

    Ok(())
}

fn clear_cache() -> anyhow::Result<()> {
    let cache = image_cache::ImageCache::new()?;
    cache.clear()?;
    println!("{}", "Image cache cleared".green());
    Ok(())
}

fn show_profile(db: &Database) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;

    if performers.is_empty() {
        println!("{}", "No performers in database yet. Add some with 'starfinder add'.".yellow());
        return Ok(());
    }

    let total = performers.len();
    let tree = recommender::build_preference_tree(&performers);
    let path = recommender::dominant_query_path(&tree);

    println!("{}", "Your Taste Profile".bright_cyan().bold());
    println!("{}", "═".repeat(42).bright_black());
    println!("{}", format!("  Based on {} liked performers", total).bright_black());
    println!();

    recommender::print_tree(&tree, "  ", total);

    println!();
    if path.is_empty() {
        println!("{}", "  Add more performers to refine your type.".bright_black());
    } else {
        println!("{} {}",
            "  Your type:".bright_black(),
            path.join(" → ").bright_white().bold()
        );
        println!("{}", "  (The deeper the tree, the more specific your recommendations)".bright_black());
    }
    println!();

    Ok(())
}

async fn recommend(db: &Database, limit: usize) -> anyhow::Result<()> {
    let performers = db.get_all_performers()?;

    if performers.is_empty() {
        println!("{}", "No performers in database yet. Add some with 'starfinder add'.".yellow());
        return Ok(());
    }

    let api_key = std::env::var("TPDB_API_KEY")
        .context("TPDB_API_KEY not set — needed for recommendations")?;
    let cfg = config::Config::load();

    let known_names: std::collections::HashSet<String> = performers
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();

    let tree = recommender::build_preference_tree(&performers);
    let path = recommender::dominant_query_path(&tree);

    println!("{}", "Finding performers you might like...".bright_cyan().bold());
    println!("{} {}",
        "  Profile:".bright_black(),
        path.join(" → ").bright_white().bold()
    );
    println!();

    // Collect TPDB numeric IDs from liked performers
    let liked_ids: Vec<i64> = performers.iter()
        .filter_map(|p| p.tpdb_id)
        .collect();

    // Top cup size from preferences (most common cup among liked performers)
    let top_cup = performers.iter()
        .filter_map(|p| p.measurements.as_deref())
        .filter_map(|m| {
            let bust = m.split('-').next()?;
            let cup = bust.trim_start_matches(|c: char| c.is_ascii_digit());
            if cup.is_empty() { None } else { Some(cup.to_uppercase()) }
        })
        .fold(std::collections::HashMap::<String, usize>::new(), |mut acc, cup| {
            *acc.entry(cup).or_insert(0) += 1; acc
        })
        .into_iter()
        .max_by_key(|(_, v)| *v)
        .map(|(cup, _)| cup);

    let top_ethnicity = path.get(1).map(|s| s.as_str());

    let client = TpdbClient::new(api_key);
    let pool = client.get_recommendations(
        &liked_ids,
        top_ethnicity,
        top_cup.as_deref(),
        &cfg.gender_filter,
    ).await?;

    let mut scored: Vec<(f64, models::Performer)> = pool
        .into_iter()
        .filter(|p| !known_names.contains(&p.name.to_lowercase()))
        .map(|p| (recommender::score_performer(&p, &tree), p))
        .filter(|(score, _)| *score > 0.0)
        .collect();

    // Sort best-match first
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    if scored.is_empty() {
        println!("{}", "No matching recommendations found. Try adding more performers to refine your profile.".yellow());
        return Ok(());
    }

    println!("{}", format!("Top {} Recommendations for you:", scored.len()).bright_cyan().bold());
    println!();

    for (i, (score, p)) in scored.iter().enumerate() {
        let age_str = p.age
            .map(|a| format!(", {}", recommender::age_bucket(a)))
            .unwrap_or_default();
        println!("{}. {} {}  {}",
            (i + 1).to_string().bright_black(),
            p.name.bright_white().bold(),
            format!("({}, {}{}{})",
                p.body_type,
                p.ethnicity.as_deref().unwrap_or("?"),
                p.hair_color.as_ref().map(|h| format!(", {}", h)).unwrap_or_default(),
                age_str,
            ).bright_black(),
            format!("match {:.0}%", score / 10.0 * 100.0).bright_cyan()
        );
    }

    println!();
    println!("{}", "Use 'starfinder add <name>' to add any to your profile.".bright_black());

    Ok(())
}

async fn find(
    db: &Database,
    looks_like:  Option<String>,
    body_like:   Option<String>,
    hair_arg:    Option<String>,
    eye_arg:     Option<String>,
    eth_arg:     Option<String>,
    cup_arg:     Option<String>,
    hips_arg:    Option<u32>,
    waist_arg:   Option<u32>,
    age_min:     Option<u32>,
    age_max:     Option<u32>,
    limit:       usize,
) -> anyhow::Result<()> {
    let api_key = std::env::var("TPDB_API_KEY").context("TPDB_API_KEY not set")?;
    let cfg = config::Config::load();

    // ── Pull attributes from named performers ─────────────────────────────
    let mut ethnicity = eth_arg;
    let mut hair      = hair_arg;
    let mut eye       = eye_arg;
    let mut cup       = cup_arg;
    let mut hips      = hips_arg;
    let mut waist     = waist_arg;

    if let Some(ref name) = looks_like {
        let p = db.get_performer(name)?
            .ok_or_else(|| anyhow::anyhow!("'{}' not in database", name))?;
        if ethnicity.is_none() { ethnicity = p.ethnicity.clone(); }
        if hair.is_none()      { hair = p.hair_color.clone(); }
        if eye.is_none()       { eye = p.eye_color.clone(); }
    }

    if let Some(ref name) = body_like {
        let p = db.get_performer(name)?
            .ok_or_else(|| anyhow::anyhow!("'{}' not in database", name))?;

        if let Some(ref m) = p.measurements {
            let parts: Vec<&str> = m.split('-').collect();
            // Cup from bust (e.g. "34B" → "B")
            if cup.is_none() {
                let cup_letter = parts.first()
                    .map(|s| s.trim_start_matches(|c: char| c.is_ascii_digit()).to_uppercase())
                    .filter(|s| !s.is_empty());
                cup = cup_letter;
            }
            // Waist (middle measurement)
            if waist.is_none() {
                waist = parts.get(1).and_then(|s| s.trim().parse().ok());
            }
            // Hips (last measurement, strip trailing letters)
            if hips.is_none() {
                hips = parts.get(2).and_then(|s| {
                    s.trim().trim_end_matches(|c: char| !c.is_ascii_digit()).parse().ok()
                });
            }
        }
    }

    // ── Display criteria ──────────────────────────────────────────────────
    println!("{}", "Searching for performers with:".bright_cyan().bold());
    let mut criteria: Vec<String> = vec![];
    if let Some(ref n) = looks_like  { criteria.push(format!("face like {}", n.bright_white())); }
    if let Some(ref n) = body_like   { criteria.push(format!("body like {}", n.bright_white())); }
    if let Some(ref v) = ethnicity   { criteria.push(format!("ethnicity: {}", v.bright_white())); }
    if let Some(ref v) = hair        { criteria.push(format!("hair: {}", v.bright_white())); }
    if let Some(ref v) = eye         { criteria.push(format!("eyes: {}", v.bright_white())); }
    if let Some(ref v) = cup         { criteria.push(format!("cup: {}", v.bright_white())); }
    if let Some(v) = hips            { criteria.push(format!("hips: {}\" ±4", v.to_string().bright_white())); }
    if let Some(v) = waist           { criteria.push(format!("waist: {}\" ±4", v.to_string().bright_white())); }
    if let Some(v) = age_min         { criteria.push(format!("age ≥ {}", v.to_string().bright_white())); }
    if let Some(v) = age_max         { criteria.push(format!("age ≤ {}", v.to_string().bright_white())); }
    for c in &criteria { println!("  · {}", c); }
    println!();

    let known_names: std::collections::HashSet<String> = db.get_all_performers()?
        .iter().map(|p| p.name.to_lowercase()).collect();

    let client = TpdbClient::new(api_key);
    let mut results = client.search_by_attributes(
        ethnicity.as_deref(), hair.as_deref(), eye.as_deref(),
        cup.as_deref(), hips, waist, age_min, age_max, &cfg.gender_filter, limit * 5,
    ).await?;

    results.retain(|p| !known_names.contains(&p.name.to_lowercase()));

    // ── Face similarity scoring (if looks_like has a stored embedding) ────
    let ref_embedding = looks_like.as_deref()
        .and_then(|name| db.get_embedding(name).ok().flatten());

    let mut scored: Vec<(Option<f32>, models::Performer)> = results
        .into_iter()
        .map(|p| {
            let sim = ref_embedding.as_ref().and_then(|ref_emb| {
                // Use cached embedding if available, otherwise generate on the fly
                let emb = db.get_embedding(&p.name).ok().flatten()
                    .or_else(|| {
                        p.face_url.as_deref()
                            .or(p.profile_image_url.as_deref())
                            .and_then(|url| embedder::generate_embedding(url).ok()
                                .inspect(|e| { let _ = db.save_embedding(&p.name, e); }))
                    });
                emb.map(|e| embedder::cosine_similarity(ref_emb, &e))
            });
            (sim, p)
        })
        .collect();

    // Sort by face similarity if available, otherwise keep original order
    if ref_embedding.is_some() {
        scored.sort_by(|a, b| b.0.unwrap_or(-1.0)
            .partial_cmp(&a.0.unwrap_or(-1.0))
            .unwrap_or(std::cmp::Ordering::Equal));
    }

    scored.truncate(limit);

    if scored.is_empty() {
        println!("{}", "No results found. Try relaxing some filters.".yellow());
        return Ok(());
    }

    let using_face_ml = ref_embedding.is_some();
    println!("{}", format!("{} results{}:",
        scored.len(),
        if using_face_ml { " (sorted by face similarity)" } else { "" }
    ).bright_cyan().bold());
    println!();

    for (i, (sim, p)) in scored.iter().enumerate() {
        let age_str = p.age.map(|a| format!(", {}", recommender::age_bucket(a))).unwrap_or_default();
        let meas_str = p.measurements.as_deref().map(|m| {
            let parts: Vec<&str> = m.split('-').collect();
            if parts.len() >= 3 {
                format!(", {}w {}h",
                    parts[1].trim(),
                    parts[2].trim().trim_end_matches(|c: char| !c.is_ascii_digit()))
            } else { String::new() }
        }).unwrap_or_default();
        let sim_str = sim.map(|s| format!("  face {:.0}%", embedder::similarity_pct(s)))
            .unwrap_or_default();
        println!("{}. {} {}{}",
            (i + 1).to_string().bright_black(),
            p.name.bright_white().bold(),
            format!("({}, {}{}{}{}{})",
                p.body_type,
                p.ethnicity.as_deref().unwrap_or("?"),
                p.hair_color.as_ref().map(|h| format!(", {}", h)).unwrap_or_default(),
                p.eye_color.as_ref().map(|e| format!(", {} eyes", e)).unwrap_or_default(),
                meas_str,
                age_str,
            ).bright_black(),
            sim_str.bright_cyan(),
        );
    }
    println!();
    if !using_face_ml && looks_like.is_some() {
        println!("{}", "  Tip: run 'starfinder embed' to generate face embeddings for ML-powered face similarity".bright_black());
    }
    println!("{}", "Use 'starfinder add <name>' to add any to your profile.".bright_black());
    Ok(())
}

async fn similar(db: &Database, name: &str, limit: usize) -> anyhow::Result<()> {
    let performer = db.get_performer(name)?
        .ok_or_else(|| anyhow::anyhow!("'{}' not found in your database. Add them first with 'starfinder add'.", name))?;

    let tpdb_uuid = performer.source_url
        .as_deref()
        .and_then(|url| url.split('/').last())
        .ok_or_else(|| anyhow::anyhow!("No TPDB ID stored for '{}'", name))?
        .to_string();

    let api_key = std::env::var("TPDB_API_KEY")
        .context("TPDB_API_KEY not set")?;
    let cfg = config::Config::load();
    let client = TpdbClient::new(api_key);

    println!("{} {}",
        "Finding performers similar to".bright_cyan(),
        performer.name.bright_white().bold()
    );
    println!("{} {}  {}  {}",
        " ".bright_black(),
        performer.body_type.bright_black(),
        performer.ethnicity.as_deref().unwrap_or("?").bright_black(),
        performer.hair_color.as_deref().unwrap_or("?").bright_black(),
    );
    println!();

    let known_names: std::collections::HashSet<String> = db.get_all_performers()?
        .iter()
        .map(|p| p.name.to_lowercase())
        .collect();

    let mut results = client.similar_to(&tpdb_uuid, &cfg.gender_filter).await?;
    results.retain(|p| !known_names.contains(&p.name.to_lowercase()));
    results.truncate(limit);

    if results.is_empty() {
        println!("{}", "No similar performers found.".yellow());
        return Ok(());
    }

    for (i, p) in results.iter().enumerate() {
        let age_str = p.age.map(|a| format!(", {}", recommender::age_bucket(a))).unwrap_or_default();
        println!("{}. {} {}",
            (i + 1).to_string().bright_black(),
            p.name.bright_white().bold(),
            format!("({}, {}{}{})",
                p.body_type,
                p.ethnicity.as_deref().unwrap_or("?"),
                p.hair_color.as_ref().map(|h| format!(", {}", h)).unwrap_or_default(),
                age_str,
            ).bright_black()
        );
    }

    println!();
    println!("{}", "Use 'starfinder add <name>' to add any to your profile.".bright_black());
    Ok(())
}

fn embed_all(db: &Database) -> anyhow::Result<()> {
    let pending = db.get_performers_without_embedding()?;

    if pending.is_empty() {
        println!("{}", "All performers already have face embeddings.".green());
        return Ok(());
    }

    println!("{}", format!("Generating embeddings for {} performers...", pending.len())
        .bright_cyan().bold());
    println!("{}", "  Note: first run downloads the ArcFace model (~100 MB)".bright_black());
    println!();

    let mut ok = 0;
    let mut skipped = 0;

    for p in &pending {
        let face_url = p.face_url.as_deref().or(p.profile_image_url.as_deref());
        let Some(url) = face_url else {
            println!("  {} {} — no image URL", "–".bright_black(), p.name.bright_black());
            skipped += 1;
            continue;
        };

        print!("  {} {}... ", "→".bright_black(), p.name.bright_white());
        std::io::Write::flush(&mut std::io::stdout()).ok();

        match embedder::generate_embedding(url) {
            Ok(emb) => {
                db.save_embedding(&p.name, &emb)?;
                println!("{}", "done".green());
                ok += 1;
            }
            Err(e) => {
                println!("{} {}", "failed:".red(), e.to_string().bright_black());
                skipped += 1;
            }
        }
    }

    println!();
    println!("{}", format!("Done: {} embedded, {} skipped", ok, skipped).green());
    Ok(())
}

fn configure(key: Option<String>, value: Option<String>) -> anyhow::Result<()> {
    use colored::*;
    let mut cfg = config::Config::load();

    match (key.as_deref(), value.as_deref()) {
        (None, _) => {
            // Show current config
            println!("{}", "Starfinder Settings".bright_cyan().bold());
            println!("{}", "═".repeat(35).bright_black());
            println!("  {} {}",
                "gender:".bright_black(),
                cfg.gender_filter.display().bright_white()
            );
            println!();
            println!("{}", "  Valid values: female, male, trans-female, trans-male, any".bright_black());
        }
        (Some("gender"), Some(val)) => {
            match config::GenderFilter::from_str(val) {
                Some(filter) => {
                    cfg.gender_filter = filter;
                    cfg.save()?;
                    println!("{} gender = {}",
                        "Updated:".green(),
                        cfg.gender_filter.display().bright_white()
                    );
                }
                None => {
                    println!("{} Unknown value '{}'. Use: female, male, trans-female, trans-male, any",
                        "Error:".red(), val);
                }
            }
        }
        (Some(k), _) => {
            println!("{} Unknown setting '{}'. Available: gender", "Error:".red(), k);
        }
    }

    Ok(())
}

fn import_from_json(db: &Database, file_path: &str) -> anyhow::Result<()> {
    use std::fs;
    use models::Performer;

    println!("{}", format!("Importing performers from {}...", file_path).bright_cyan().bold());
    println!();

    let json_data = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read {}", file_path))?;

    let performers: Vec<Performer> = serde_json::from_str(&json_data)
        .with_context(|| format!("Failed to parse JSON from {}", file_path))?;

    let mut success_count = 0;
    let mut fail_count = 0;

    for performer in performers {
        match db.add_performer(&performer) {
            Ok(_) => {
                println!("{} {} {}",
                    "[OK]".green(), "Imported:".white(),
                    performer.name.bright_white().bold());
                success_count += 1;
            }
            Err(e) => {
                println!("{} {} {}: {}",
                    "[X]".red(), "Failed:".white(), performer.name, e);
                fail_count += 1;
            }
        }
    }

    println!();
    println!("{}", format!("Imported {} performers ({} failed)", success_count, fail_count).green());
    Ok(())
}
