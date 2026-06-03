//! Corpus-building handlers, split by corpus: face (`embed`, `warm`) and body
//! (`index`, `ingest`, `aggregate`).
mod body;
mod face;

pub(crate) use body::{aggregate, build_index, ingest};
pub(crate) use face::{embed_all, warm};
