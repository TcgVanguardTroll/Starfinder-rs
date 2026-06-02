// CLI command handlers naturally take many args (one per flag), and the
// scored-result tuples are intentionally inline rather than newtype'd. The
// Find subcommand has many optional flags, making its enum variant large.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::large_enum_variant
)]

use clap::{Parser, Subcommand};
use colored::*;

// Core logic lives in the `luminary` library crate (src/lib.rs).
use luminary::database::Database;
use luminary::image_cache::ImageCache;
use luminary::stashdb::StashdbClient;
use luminary::tpdb::TpdbClient;
use luminary::{config, models, recommender};

mod cli;
use cli::library::*;
use cli::mlindex::*;
use cli::profile::*;
use cli::search::*;
use cli::settings::*;

/// Builds the list of images to feed the centroid embedder: StashDB's multi-image
/// array first (when a key is configured), then TPDB's face/profile/gallery.
/// Capped to bound CPU cost (~5s per image on CPU).
pub(crate) async fn build_centroid_urls(
    p: &models::Performer,
    stash: Option<&StashdbClient>,
) -> Vec<String> {
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
pub(crate) fn face_image_urls(p: &models::Performer) -> Vec<String> {
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
pub(crate) async fn render_thumbnail(cache: &ImageCache, url: &str) {
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
    /// Find performers with a similar body, ranked against the cached index
    BodySearch {
        /// A performer in your library to match against
        name: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Render a thumbnail image inline for each result
        #[arg(long, default_value_t = false)]
        images: bool,
        /// Which lens to match by:
        ///   build        = skeletal proportions (shoulder/hip/leg), from pose
        ///   volume       = silhouette fullness (butt/thigh), from segmentation
        ///   measurements = recorded WHR/hips/cup (no images; works for niche refs)
        #[arg(long = "by", default_value = "build", value_parser = ["build", "volume", "measurements"])]
        by: String,
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
            by,
        } => {
            body_search(&db, &name, limit, images, &by).await?;
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

/// Core recommendation: score a TPDB candidate pool against a set of liked
/// performers (tree + IDF), returning scored results sorted best-first.
/// Reused by global `recommend` and per-cluster recommendation.
pub(crate) async fn recommend_pool(
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
pub(crate) async fn print_recs(
    scored: &[(f64, models::Performer)],
    img_cache: &Option<ImageCache>,
) {
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
pub(crate) fn cluster_label(members: &[models::Performer]) -> String {
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

/// Auto-pick a cluster count from library size.
pub(crate) fn auto_k(n: usize) -> usize {
    (n / 8).clamp(2, 5)
}
