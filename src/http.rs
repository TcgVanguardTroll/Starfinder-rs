//! Shared reqwest client construction. Every source (TPDB, StashDB, pornpics,
//! pichunter, FreeOnes) uses the same 30s-timeout client and differs only in
//! User-Agent, so the builder lives in one place.

/// API sources identify as the app itself.
pub const APP_UA: &str = "Luminary/0.1.0";
/// Scraped sites serve an empty page to non-browser agents.
pub const BROWSER_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";

/// Standard client for all HTTP sources. Building can only fail if the TLS
/// backend can't initialise — unrecoverable at startup — so panic with a clear
/// message rather than threading a Result through every `Client::new()`.
pub fn client(user_agent: &str) -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(user_agent)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("failed to initialise HTTP client (TLS backend unavailable)")
}
