//! CLI command handlers, split out of `main.rs` by command group.
//!
//! `main.rs` keeps the clap definition + dispatch and the small shared rendering
//! helpers; each submodule here holds the handlers for one group and pulls this
//! shared prelude via `use super::*`. The `allow(unused_imports)` covers the
//! re-exports that only some submodules use.
#![allow(unused_imports)]

// Shared rendering/scoring helpers that still live in main.rs.
pub(crate) use crate::{
    auto_k, build_centroid_urls, cluster_label, face_image_urls, print_recs, recommend_pool,
    render_thumbnail,
};

// Library crate surface used across the handlers.
pub(crate) use anyhow::Context;
pub(crate) use colored::*;
pub(crate) use luminary::blend;
pub(crate) use luminary::config;
pub(crate) use luminary::database::Database;
pub(crate) use luminary::embedder;
pub(crate) use luminary::image_cache::{self, ImageCache};
pub(crate) use luminary::models::{self, SearchFilters};
pub(crate) use luminary::pichunter::PichunterClient;
pub(crate) use luminary::pornpics::PornpicsClient;
pub(crate) use luminary::recommender;
pub(crate) use luminary::scraper::Scraper;
pub(crate) use luminary::stashdb::StashdbClient;
pub(crate) use luminary::tpdb::TpdbClient;

pub mod library;
pub mod mlindex;
pub mod profile;
pub mod search;
pub mod settings;
