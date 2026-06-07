//! CLI argument definitions (clap): the `Cli` parser and the `Commands` enum.
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "luminary")]
#[command(about = "Find your stars - A Rust-powered recommendation engine", long_about = None)]
#[command(version)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
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
    /// Build the per-image corpus for specific performers: gather multi-angle
    /// images from every source, identity-verify each face against a seed,
    /// classify its view (front/rear/side), quality-score it, and store the
    /// pose/seg/face vectors. The foundation the view-aware body index is built
    /// from. Incremental and resumable.
    Ingest {
        /// Performer(s) to ingest images for (omit when using --roster)
        names: Vec<String>,
        /// Max images to gather per source
        #[arg(long, default_value_t = 24)]
        images: usize,
        /// A hand-picked image URL to include (repeatable) — the `manual` source
        #[arg(long = "url")]
        urls: Vec<String>,
        /// Face-match cosine threshold; below it a detected face is treated as a
        /// different performer and the image is dropped
        #[arg(long, default_value_t = 0.3)]
        id_threshold: f32,
        /// Re-embed images already in the corpus
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Ingest every performer in the cached body index (the full roster)
        #[arg(long, default_value_t = false)]
        roster: bool,
        /// Ingest without a seed face to verify identity (unsafe at scale —
        /// bypasses the bogus-name guard)
        #[arg(long, default_value_t = false)]
        allow_unverified: bool,
    },
    /// Rebuild the cached body index from the per-image corpus that `ingest`
    /// fills — quality-weighted and view-aware (only frontal frames feed the
    /// pose/seg centroids). Cheap and pure; re-run to re-tune without
    /// re-embedding. With no names, rebuilds everyone who has ingested images.
    Aggregate {
        /// Performer(s) to rebuild (default: everyone with ingested images)
        names: Vec<String>,
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
        ///   overall   = multi-modal fusion of face + frame + curves + projection + stats (default)
        ///   lookalike = same fusion but FACE-LED — closest-looking actress (looks like + built like)
        ///   body      = same fusion but face EXCLUDED — pure body-type match
        ///   frame     = skeletal proportions (shoulder/hip/leg), from pose
        ///   curves    = silhouette fullness (butt/thigh), from segmentation
        ///   stats     = recorded WHR/hips/cup (no images; works for niche refs)
        #[arg(long = "by", default_value = "overall", value_parser = ["overall", "lookalike", "body", "frame", "curves", "stats"])]
        by: String,
        /// Only match performers within ± this many cm of the reference's height,
        /// so results share the same *stature*, not just proportions. Without it,
        /// a tall performer with the same proportions ranks as high as a short one
        /// (the body vectors are scale-free). E.g. `--height-tol 8`.
        #[arg(long = "height-tol")]
        height_tol: Option<f64>,
        /// Only match performers whose recorded hair colour contains this word
        /// (case-insensitive), so results share the reference's hair. Matches as a
        /// substring, so `--hair blond` also catches "Dark Blonde". E.g. `--hair blond`.
        #[arg(long)]
        hair: Option<String>,
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
    /// Plain-English search — translates a sentence into `find`'s filters, e.g.
    ///   luminary query blue-eyed blondes that look like Naughty Alysha
    ///   luminary query blondes with a butt like Dee Siren
    Query {
        /// The query text (no quotes needed)
        #[arg(required = true)]
        text: Vec<String>,
        /// Render a thumbnail image inline for each result
        #[arg(long, default_value_t = false)]
        images: bool,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Evaluate ranking quality of the `overall` blend against your liked set
    /// (leave-one-out): mean precision@k / recall@k / MAP / nDCG@k. Local, no
    /// network — a number to tune the blend/proj/bust against.
    Eval,
    /// View or change settings
    Config {
        /// Setting to change (e.g. gender)
        key: Option<String>,
        /// New value (e.g. female, male, trans-female, trans-male, any)
        value: Option<String>,
    },
}
