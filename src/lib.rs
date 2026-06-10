//! Luminary — a privacy-first, local-first recommendation engine for
//! discovering adult performers, powered by ThePornDB and ArcFace.
//!
//! The crate is split into a library (these modules) plus a thin binary
//! (`main.rs`) that wires up the CLI. Keeping the logic in the library makes
//! it unit- and integration-testable and reusable by other front-ends.

// GenderFilter::from_str predates (and differs from) std::str::FromStr — it
// returns Option, not Result — and is intentionally an inherent method.
// search_by_attributes takes one arg per filter; that's clearer than a struct.
#![allow(clippy::should_implement_trait, clippy::too_many_arguments)]

pub mod blend;
pub mod config;
pub mod database;
pub mod embedder;
pub mod eval;
pub mod http;
pub mod image_cache;
pub mod models;
pub mod pichunter;
pub mod pornpics;
pub mod query;
pub mod recommender;
pub mod region;
pub mod scraper;
pub mod source;
pub mod stashdb;
pub mod tpdb;
