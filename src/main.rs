// CLI command handlers naturally take many args (one per flag), and the
// scored-result tuples are intentionally inline rather than newtype'd. The
// Find subcommand has many optional flags, making its enum variant large.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::large_enum_variant
)]

use anyhow::Context;
use clap::{Parser, Subcommand};
use colored::*;

// Core logic lives in the `luminary` library crate (src/lib.rs).
use luminary::database::Database;
use luminary::image_cache::ImageCache;
use luminary::models::SearchFilters;
use luminary::pornpics::PornpicsClient;
use luminary::scraper::Scraper;
use luminary::stashdb::StashdbClient;
use luminary::tpdb::TpdbClient;
use luminary::{config, embedder, image_cache, models, recommender};

/// Builds the list of images to feed the centroid embedder: StashDB's multi-image
/// array first (when a key is configured), then TPDB's face/profile/gallery.
/// Capped to bound CPU cost (~5s per image on CPU).
async fn build_centroid_urls(p: &models::Performer, stash: Option<&StashdbClient>) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    // StashDB multi-image gallery (clean, person-centric) — best for faces.
    if let Some(s) = stash {
        for u in s.image_urls(&p.name, 3).await {
            if !urls.contains(&u) {
                urls.push(u);
            }
        }
    }
    // TPDB stored profile media: face crop, profile poster, and the extra
    // `posters[]` shoots (all of *this* performer — safe for the face centroid).
    // Scene stills are deliberately excluded here: they often contain a co-star,
    // and "largest face" can grab the wrong person. They return only once we add
    // identity-gating (match a scene face against the clean profile seed).
    for u in face_image_urls(p) {
        if !urls.contains(&u) {
            urls.push(u);
        }
    }
    urls.truncate(8);
    urls
}

/// Collects up to a few distinct image URLs for a performer, best face first,
/// for building a robust centroid face embedding.
fn face_image_urls(p: &models::Performer) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    let mut push = |u: &str| {
        let s = u.to_string();
        if !u.is_empty() && !urls.contains(&s) {
            urls.push(s);
        }
    };
    if let Some(u) = &p.face_url {
        push(u);
    }
    if let Some(u) = &p.profile_image_url {
        push(u);
    }
    for u in p.gallery_urls.iter().take(2) {
        push(u);
    }
    urls
}

/// Downloads (cached) and renders a small inline thumbnail for a performer.
/// Best-effort: silently does nothing if the terminal can't display images.
async fn render_thumbnail(cache: &ImageCache, url: &str) {
    if let Ok(path) = cache.get_image(url).await {
        let conf = viuer::Config {
            absolute_offset: false,
            width: Some(20),
            height: Some(10),
            ..Default::default()
        };
        let _ = viuer::print_from_file(&path, &conf);
    }
}

#[derive(Parser)]
#[command(name = "luminary")]
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
    Remove { name: String },
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
    Profile {
        /// Output the tree as a Mermaid diagram instead of ASCII
        #[arg(long, default_value_t = false)]
        mermaid: bool,
    },
    /// Get performer recommendations based on your profile
    Recommend {
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Render a thumbnail image inline for each result
        #[arg(long, default_value_t = false)]
        images: bool,
        /// Recommend separately per taste cluster (best for multi-modal taste)
        #[arg(long, default_value_t = false)]
        by_cluster: bool,
    },
    /// Detect and label your taste clusters (k-means over your library)
    Clusters {
        /// Number of clusters (default: auto from library size)
        #[arg(long)]
        k: Option<usize>,
    },
    /// Search by mixing attributes from stored performers or manual values
    Find {
        /// Find performers like this one; use --match to pick face/body/both
        #[arg(long)]
        like: Option<String>,
        /// What to match on for --like: face, body, or both (default: both)
        #[arg(long = "match", value_parser = ["face", "body", "both"])]
        match_mode: Option<String>,
        /// Match this performer's face (repeatable — blends multiple faces)
        #[arg(long)]
        looks_like: Vec<String>,
        /// Match this performer's build (repeatable — blends multiple bodies)
        #[arg(long)]
        body_like: Vec<String>,
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
        /// Target waist-to-hip ratio (searches ±0.05), e.g. 0.667 for Dee Siren's build
        #[arg(long)]
        whr: Option<f64>,
        /// Require a tattoo at this location, e.g. "lower back" (tramp stamp)
        #[arg(long)]
        tattoo: Option<String>,
        /// Only performers from a region/nationality group:
        /// slavic, nordic, latina, asian, western-european
        #[arg(long)]
        region: Option<String>,
        /// Match the face only — ignore body, boobs, and tattoo entirely
        #[arg(long, default_value_t = false)]
        face_only: bool,
        /// Render a thumbnail image inline for each result (terminal permitting)
        #[arg(long, default_value_t = false)]
        images: bool,
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
        /// Render a thumbnail image inline for each result
        #[arg(long, default_value_t = false)]
        images: bool,
    },
    /// Manage name aliases (e.g. "Goldie McHawn" → "Goldie Blair")
    Alias {
        /// The alias name to add or look up
        alias: Option<String>,
        /// The canonical name it maps to (omit to look up or list all)
        canonical: Option<String>,
        /// Remove this alias instead of adding it
        #[arg(long)]
        remove: bool,
    },
    /// Generate face embeddings for all performers missing one
    Embed {
        /// Re-embed everyone (e.g. after enabling StashDB enrichment)
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Pre-fetch and embed a pool of candidates into the local face corpus,
    /// so later face searches are instant (no API calls or re-embedding).
    Warm {
        /// How many candidates to embed into the corpus
        #[arg(long, default_value_t = 40)]
        limit: usize,
    },
    /// Build the cached body-vector index: for a roster of popular performers,
    /// gather full-body images (pornpics + TPDB scenes + StashDB), gate them, and
    /// store pose (frame) + seg (shape) centroids. Lets body-search/find rank
    /// against a rich candidate pool instantly. One-time, resumable.
    Index {
        /// Roster size (how many popular performers to index)
        #[arg(long, default_value_t = 500)]
        limit: usize,
        /// Images to gather per performer (caps cost; pornpics prioritised)
        #[arg(long, default_value_t = 18)]
        images: usize,
        /// Re-index performers already present in the index
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Find performers with a similar body frame (MediaPipe pose, via StashDB)
    BodySearch {
        /// A performer in your library to match build against
        name: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Render a thumbnail image inline for each result
        #[arg(long, default_value_t = false)]
        images: bool,
        /// Match by silhouette *volume* (waist/hip/thigh fullness) instead of
        /// skeletal frame — captures butt & thigh shape the pose vector misses.
        #[arg(long, default_value_t = false)]
        shape: bool,
    },
    /// Search for performers who look like someone (by face)
    FaceSearch {
        /// A performer in your library to match faces against
        name: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Render a thumbnail image inline for each result
        #[arg(long, default_value_t = false)]
        images: bool,
        /// Fetch a fresh candidate pool from StashDB (by the reference's
        /// attributes) and embed it before searching. Requires a StashDB key.
        #[arg(long)]
        source: Option<String>,
    },
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
                &db,
                body_type,
                age_min,
                age_max,
                ethnicity,
                hair_color,
                show_images,
                limit,
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
        Commands::Profile { mermaid } => {
            show_profile(&db, mermaid)?;
        }
        Commands::Recommend {
            limit,
            images,
            by_cluster,
        } => {
            recommend(&db, limit, images, by_cluster).await?;
        }
        Commands::Clusters { k } => {
            show_clusters(&db, k)?;
        }
        Commands::Find {
            like,
            match_mode,
            looks_like,
            body_like,
            hair,
            eye,
            ethnicity,
            cup,
            hips,
            waist,
            whr,
            tattoo,
            region,
            face_only,
            images,
            age_min,
            age_max,
            limit,
        } => {
            find(
                &db, like, match_mode, looks_like, body_like, hair, eye, ethnicity, cup, hips,
                waist, whr, tattoo, region, face_only, images, age_min, age_max, limit,
            )
            .await?;
        }
        Commands::Similar {
            name,
            limit,
            images,
        } => {
            similar(&db, &name, limit, images).await?;
        }
        Commands::Alias {
            alias,
            canonical,
            remove,
        } => {
            manage_alias(&db, alias, canonical, remove)?;
        }
        Commands::Embed { force } => {
            embed_all(&db, force).await?;
        }
        Commands::Warm { limit } => {
            warm(&db, limit).await?;
        }
        Commands::Index {
            limit,
            images,
            force,
        } => {
            build_index(&db, limit, images, force).await?;
        }
        Commands::BodySearch {
            name,
            limit,
            images,
            shape,
        } => {
            body_search(&db, &name, limit, images, shape).await?;
        }
        Commands::FaceSearch {
            name,
            limit,
            images,
            source,
        } => {
            face_search(&db, &name, limit, images, source).await?;
        }
        Commands::Config { key, value } => {
            configure(key, value)?;
        }
    }

    Ok(())
}

fn get_db_path() -> anyhow::Result<String> {
    let data_dir =
        dirs::data_local_dir().ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
    let db_dir = data_dir.join("luminary");
    std::fs::create_dir_all(&db_dir)?;
    Ok(db_dir.join("luminary.db").to_string_lossy().to_string())
}

async fn add_performers(db: &Database, names: Vec<String>) -> anyhow::Result<()> {
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
                                Some(emb) => {
                                    let _ = db.save_embedding(&performer.name, &emb);
                                    println!(
                                        "     {} face embedding stored ({} image{})",
                                        "↳".bright_black(),
                                        urls.len(),
                                        if urls.len() == 1 { "" } else { "s" }
                                    );
                                }
                                None => {
                                    log::debug!("No face detected for {}", name);
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

fn list_performers(db: &Database) -> anyhow::Result<()> {
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

fn clear_cache() -> anyhow::Result<()> {
    let cache = image_cache::ImageCache::new()?;
    cache.clear()?;
    println!("{}", "Image cache cleared".green());
    Ok(())
}

fn show_profile(db: &Database, mermaid: bool) -> anyhow::Result<()> {
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

/// Core recommendation: score a TPDB candidate pool against a set of liked
/// performers (tree + IDF), returning scored results sorted best-first.
/// Reused by global `recommend` and per-cluster recommendation.
async fn recommend_pool(
    performers: &[models::Performer],
    cfg: &config::Config,
    api_key: &str,
    known_names: &std::collections::HashSet<String>,
) -> anyhow::Result<Vec<(f64, models::Performer)>> {
    let tree = recommender::build_preference_tree(performers);
    let path = recommender::dominant_query_path(&tree);
    let idf = recommender::compute_idf_weights(performers);
    let liked_ids: Vec<i64> = performers.iter().filter_map(|p| p.tpdb_id).collect();

    let top_cup = performers
        .iter()
        .filter_map(|p| p.measurements.as_deref())
        .filter_map(|m| {
            let cup = m
                .split('-')
                .next()?
                .trim_start_matches(|c: char| c.is_ascii_digit());
            (!cup.is_empty()).then(|| cup.to_uppercase())
        })
        .fold(
            std::collections::HashMap::<String, usize>::new(),
            |mut a, c| {
                *a.entry(c).or_insert(0) += 1;
                a
            },
        )
        .into_iter()
        .max_by_key(|(_, v)| *v)
        .map(|(cup, _)| cup);
    let top_ethnicity = path.get(1).map(|s| s.as_str());

    let client = TpdbClient::new(api_key.to_string());
    let pool = client
        .get_recommendations(
            &liked_ids,
            top_ethnicity,
            top_cup.as_deref(),
            &cfg.gender_filter,
        )
        .await?;

    let mut scored: Vec<(f64, models::Performer)> = pool
        .into_iter()
        .filter(|p| !known_names.contains(&p.name.to_lowercase()))
        .map(|p| (recommender::score_performer_idf(&p, &tree, &idf), p))
        .filter(|(s, _)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}

/// Prints a scored recommendation list (with profile links + optional images).
async fn print_recs(scored: &[(f64, models::Performer)], img_cache: &Option<ImageCache>) {
    for (i, (score, p)) in scored.iter().enumerate() {
        let age_str = p
            .age
            .map(|a| format!(", {}", recommender::age_bucket(a)))
            .unwrap_or_default();
        println!(
            "{}. {} {}  {}",
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
            format!("match {:.0}%", score / 10.0 * 100.0).bright_cyan()
        );
        if let Some(url) = &p.source_url {
            println!("   {} {}", "↳".bright_black(), url.blue().underline());
        }
        if let Some(cache) = img_cache {
            if let Some(url) = p.face_url.as_deref().or(p.profile_image_url.as_deref()) {
                render_thumbnail(cache, url).await;
            }
        }
    }
}

/// A short label for a cluster from its members' dominant traits.
fn cluster_label(members: &[models::Performer]) -> String {
    let mode = |vals: Vec<Option<String>>| -> Option<String> {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for v in vals.into_iter().flatten() {
            *counts.entry(v).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k)
    };
    let bt = mode(members.iter().map(|p| Some(p.body_type.clone())).collect());
    let eth = mode(members.iter().map(|p| p.ethnicity.clone()).collect());
    let hair = mode(members.iter().map(|p| p.hair_color.clone()).collect());
    [bt, eth, hair]
        .into_iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" · ")
}

async fn recommend(
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

/// Auto-pick a cluster count from library size.
fn auto_k(n: usize) -> usize {
    (n / 8).clamp(2, 5)
}

/// Detect and print the taste clusters in the library.
fn show_clusters(db: &Database, k: Option<usize>) -> anyhow::Result<()> {
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

async fn find(
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

async fn similar(db: &Database, name: &str, limit: usize, images: bool) -> anyhow::Result<()> {
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

fn manage_alias(
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

async fn embed_all(db: &Database, force: bool) -> anyhow::Result<()> {
    let pending = if force {
        db.get_all_performers()?
    } else {
        db.get_performers_without_embedding()?
    };

    if pending.is_empty() {
        println!(
            "{}",
            "All performers already have face embeddings. Use --force to re-embed.".green()
        );
        return Ok(());
    }

    let cfg = config::Config::load();
    let stash = cfg
        .stashdb_key
        .as_ref()
        .filter(|k| !k.is_empty())
        .map(|k| StashdbClient::new(k.clone()));

    println!(
        "{}",
        format!("Generating embeddings for {} performers...", pending.len())
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "  Note: first run downloads the ArcFace model (~100 MB)".bright_black()
    );
    println!();

    let mut ok = 0;
    let mut skipped = 0;

    for p in &pending {
        let urls = build_centroid_urls(p, stash.as_ref()).await;
        if urls.is_empty() {
            println!(
                "  {} {} — no image URL",
                "–".bright_black(),
                p.name.bright_black()
            );
            skipped += 1;
            continue;
        }

        print!("  {} {}... ", "→".bright_black(), p.name.bright_white());
        std::io::Write::flush(&mut std::io::stdout()).ok();

        match embedder::generate_centroid_embedding(&urls) {
            Some(emb) => {
                db.save_embedding(&p.name, &emb)?;
                println!("{} ({} img)", "done".green(), urls.len());
                ok += 1;
            }
            None => {
                println!("{}", "no face detected".red());
                skipped += 1;
            }
        }
    }

    println!();
    println!(
        "{}",
        format!("Done: {} embedded, {} skipped", ok, skipped).green()
    );
    Ok(())
}

/// Pre-fetch a candidate pool and embed faces into the local corpus.
async fn warm(db: &Database, limit: usize) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let api_key = cfg.resolve_api_key()?;
    let performers = db.get_all_performers()?;
    if performers.is_empty() {
        println!(
            "{}",
            "Add some performers first with 'luminary add'.".yellow()
        );
        return Ok(());
    }

    let tree = recommender::build_preference_tree(&performers);
    let path = recommender::dominant_query_path(&tree);
    let liked_ids: Vec<i64> = performers.iter().filter_map(|p| p.tpdb_id).collect();
    let top_ethnicity = path.get(1).map(|s| s.as_str());
    let known: std::collections::HashSet<String> =
        performers.iter().map(|p| p.name.to_lowercase()).collect();

    println!(
        "{}",
        format!("Warming face corpus (up to {} candidates)...", limit)
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "  This embeds faces now so searches are instant later.".bright_black()
    );
    println!();

    let client = TpdbClient::new(api_key);
    let pool = client
        .get_recommendations(&liked_ids, top_ethnicity, None, &cfg.gender_filter)
        .await?;

    // Collect new candidates + their best image URL, then embed them all in ONE
    // batched sidecar call (model loads once instead of per-candidate).
    let targets: Vec<(models::Performer, String)> = pool
        .into_iter()
        .filter(|p| !known.contains(&p.name.to_lowercase()))
        .filter_map(|p| {
            p.face_url
                .clone()
                .or_else(|| p.profile_image_url.clone())
                .map(|u| (p, u))
        })
        .take(limit)
        .collect();

    let urls: Vec<String> = targets.iter().map(|(_, u)| u.clone()).collect();
    let embeddings = embedder::generate_embeddings(&urls).unwrap_or_default();

    let mut done = 0usize;
    for ((p, _), emb) in targets.iter().zip(embeddings) {
        if let Some(e) = emb {
            db.save_candidate(p, &e)?;
            done += 1;
        }
    }

    println!();
    println!(
        "{}",
        format!(
            "Embedded {} new faces. Corpus now holds {} total.",
            done,
            db.candidate_count()?
        )
        .green()
    );
    println!(
        "{}",
        "  Run 'luminary face-search <name>' to search it.".bright_black()
    );
    Ok(())
}

/// Build (or extend) the cached body-vector index. For a roster of popular
/// performers it gathers full-body images from every source, runs ONE dual
/// embedding pass per chunk (pose + seg together), and stores gated centroids.
/// Resumable: already-indexed performers are skipped unless `force`.
async fn build_index(
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

        let all = embedder::generate_dual_embeddings(&flat).unwrap_or_default();

        for (p, (s, e)) in group.iter().zip(ranges) {
            let slice = all.get(s..e).unwrap_or(&[]);
            let pose_vecs: Vec<Vec<f32>> =
                slice.iter().filter_map(|(pose, _)| pose.clone()).collect();
            let seg_vecs: Vec<Vec<f32>> = slice.iter().filter_map(|(_, seg)| seg.clone()).collect();
            let n = pose_vecs.len().max(seg_vecs.len());
            let pose_c = embedder::body_centroid(&pose_vecs);
            let seg_c = embedder::body_centroid(&seg_vecs);
            db.save_body_index(p, pose_c.as_deref(), seg_c.as_deref(), n)?;
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

/// Find performers with a similar build from full-body images, over a combined
/// (pornpics + TPDB scenes + StashDB) pool, gated to clean standing frames.
///
/// Two modes: the default *frame* uses the skeletal pose vector (shoulder/hip/
/// leg proportions); `shape` uses the silhouette *volume* vector (waist/hip/
/// thigh fullness from the body outline) — for butt & thigh shape the skeleton
/// can't see.
async fn body_search(
    db: &Database,
    name: &str,
    limit: usize,
    images: bool,
    shape: bool,
) -> anyhow::Result<()> {
    let cfg = config::Config::load();
    let reference = db
        .get_performer(name)?
        .ok_or_else(|| anyhow::anyhow!("'{}' not found in your library. Add them first.", name))?;
    // Noun used throughout the output for whichever signal we're matching on.
    let kind = if shape { "shape" } else { "frame" };
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
        if shape {
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
                let cv = if shape { e.seg } else { e.pose }?;
                let pct = if shape {
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
                let pct = if shape {
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

async fn face_search(
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

fn configure(key: Option<String>, value: Option<String>) -> anyhow::Result<()> {
    use colored::*;
    let mut cfg = config::Config::load();

    match (key.as_deref(), value.as_deref()) {
        (None, _) => {
            // Show current config
            println!("{}", "Luminary Settings".bright_cyan().bold());
            println!("{}", "═".repeat(35).bright_black());
            println!(
                "  {} {}",
                "gender: ".bright_black(),
                cfg.gender_filter.display().bright_white()
            );
            let key_status = if std::env::var("TPDB_API_KEY").is_ok() {
                "set (via TPDB_API_KEY env var)".to_string()
            } else if cfg.api_key.as_deref().is_some_and(|k| !k.is_empty()) {
                "set (stored in config)".to_string()
            } else {
                "not set".to_string()
            };
            println!(
                "  {} {}",
                "api-key:".bright_black(),
                key_status.bright_white()
            );
            let stash_status = if cfg.stashdb_key.as_deref().is_some_and(|k| !k.is_empty()) {
                "set (image enrichment on)"
            } else {
                "not set"
            };
            println!(
                "  {} {}",
                "stashdb-key:".bright_black(),
                stash_status.bright_white()
            );
            println!();
            println!(
                "{}",
                "  gender: female, male, trans-female, trans-male, any".bright_black()
            );
            println!(
                "{}",
                "  api-key <key>: store your ThePornDB API key".bright_black()
            );
            println!(
                "{}",
                "  stashdb-key <key>: store a StashDB key for extra face images".bright_black()
            );
        }
        (Some("api-key"), Some(val)) => {
            cfg.api_key = Some(val.to_string());
            cfg.save()?;
            println!("{} api-key stored", "Updated:".green());
        }
        (Some("stashdb-key"), Some(val)) => {
            cfg.stashdb_key = Some(val.to_string());
            cfg.save()?;
            println!("{} stashdb-key stored", "Updated:".green());
        }
        (Some("gender"), Some(val)) => match config::GenderFilter::from_str(val) {
            Some(filter) => {
                cfg.gender_filter = filter;
                cfg.save()?;
                println!(
                    "{} gender = {}",
                    "Updated:".green(),
                    cfg.gender_filter.display().bright_white()
                );
            }
            None => {
                println!(
                    "{} Unknown value '{}'. Use: female, male, trans-female, trans-male, any",
                    "Error:".red(),
                    val
                );
            }
        },
        (Some(k), _) => {
            println!(
                "{} Unknown setting '{}'. Available: gender, api-key, stashdb-key",
                "Error:".red(),
                k
            );
        }
    }

    Ok(())
}

fn import_from_json(db: &Database, file_path: &str) -> anyhow::Result<()> {
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
