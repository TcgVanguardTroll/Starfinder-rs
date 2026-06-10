//! Minimal StashDB (stash-box) GraphQL client used to enrich a performer with
//! additional face images. StashDB exposes an `images` array per performer,
//! which gives the centroid face embedding several real photos to average —
//! noticeably more robust than a single TPDB face crop.
//!
//! Used purely for image enrichment; TPDB remains the primary metadata source.

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::models::Performer;

const STASHDB_GRAPHQL: &str = "https://stashdb.org/graphql";

/// "CAUCASIAN" -> "Caucasian", "TRANSGENDER_FEMALE" -> "Transgender Female"
fn titlecase_enum(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// "Caucasian" / "caucasian" -> "CAUCASIAN" (StashDB enum form)
fn to_enum(s: &str) -> String {
    s.to_uppercase().replace([' ', '-'], "_")
}

pub struct StashdbClient {
    client: reqwest::Client,
    api_key: String,
}

#[derive(Deserialize)]
struct GqlResponse {
    data: Option<SearchData>,
}

#[derive(Deserialize)]
struct SearchData {
    #[serde(rename = "searchPerformers")]
    search_performers: SearchResult,
}

#[derive(Deserialize)]
struct SearchResult {
    performers: Vec<StashPerformer>,
}

#[derive(Deserialize)]
struct StashPerformer {
    #[serde(default)]
    images: Vec<StashImage>,
}

#[derive(Deserialize)]
struct StashImage {
    url: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
}

impl StashdbClient {
    pub fn new(api_key: String) -> Self {
        let client = crate::http::client(crate::http::APP_UA);
        StashdbClient { client, api_key }
    }

    /// Returns up to `max` image URLs for the best name match, largest first.
    /// Best-effort: returns an empty vec on any error so callers can fall back.
    pub async fn image_urls(&self, name: &str, max: usize) -> Vec<String> {
        match self.try_image_urls(name, max).await {
            Ok(urls) => urls,
            Err(e) => {
                log::warn!("StashDB lookup failed for {}: {}", name, e);
                Vec::new()
            }
        }
    }

    async fn try_image_urls(&self, name: &str, max: usize) -> Result<Vec<String>> {
        let query = r#"
            query($term: String!) {
                searchPerformers(term: $term, limit: 1) {
                    performers { images { url width height } }
                }
            }
        "#;
        let body = serde_json::json!({
            "query": query,
            "variables": { "term": name },
        });

        let resp = self
            .client
            .post(STASHDB_GRAPHQL)
            .header("ApiKey", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("StashDB request failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("StashDB returned {}", resp.status());
        }

        let parsed: GqlResponse = resp.json().await.context("parse StashDB response")?;
        let performers = parsed
            .data
            .map(|d| d.search_performers.performers)
            .unwrap_or_default();

        let Some(p) = performers.into_iter().next() else {
            return Ok(Vec::new());
        };

        // Largest images first — better for face detection / embedding.
        let mut imgs = p.images;
        imgs.sort_by_key(|i| std::cmp::Reverse(i.width as u64 * i.height as u64));
        Ok(imgs.into_iter().take(max).map(|i| i.url).collect())
    }

    /// Queries StashDB for a candidate pool matching the given attributes,
    /// mapped to our Performer model (with their images attached for embedding).
    /// `gender`, `ethnicity`, `hair` are human-readable (e.g. "Female", "Caucasian").
    pub async fn query_similar(
        &self,
        gender: Option<&str>,
        ethnicity: Option<&str>,
        hair: Option<&str>,
        per_page: usize,
    ) -> Result<Vec<Performer>> {
        // Build the input inline (enums must be unquoted in GraphQL).
        // Sort by popularity so we surface real, well-documented performers
        // (good names + multiple photos) rather than sparse placeholder entries.
        let mut fields = vec![
            format!("per_page: {}", per_page),
            "sort: POPULARITY".to_string(),
            "direction: DESC".to_string(),
        ];
        if let Some(g) = gender {
            fields.push(format!("gender: {}", to_enum(g)));
        }
        if let Some(e) = ethnicity {
            fields.push(format!("ethnicity: {}", to_enum(e)));
        }
        if let Some(h) = hair {
            fields.push(format!(
                "hair_color: {{ value: {}, modifier: EQUALS }}",
                to_enum(h)
            ));
        }

        let query = format!(
            "query {{ queryPerformers(input: {{ {} }}) {{ performers {{ \
                id name gender ethnicity hair_color eye_color \
                cup_size band_size waist_size hip_size \
                images {{ url width height }} }} }} }}",
            fields.join(", ")
        );

        self.run_performer_query(query).await
    }

    /// Fetches one page of the most-popular performers (for building the body
    /// index roster). `page` is 1-based. Filtered to real, named performers.
    pub async fn query_popular(
        &self,
        gender: Option<&str>,
        page: usize,
        per_page: usize,
    ) -> Result<Vec<Performer>> {
        let mut fields = vec![
            format!("per_page: {}", per_page),
            format!("page: {}", page.max(1)),
            "sort: POPULARITY".to_string(),
            "direction: DESC".to_string(),
        ];
        if let Some(g) = gender {
            fields.push(format!("gender: {}", to_enum(g)));
        }
        let query = format!(
            "query {{ queryPerformers(input: {{ {} }}) {{ performers {{ \
                id name gender ethnicity hair_color eye_color \
                cup_size band_size waist_size hip_size \
                images {{ url width height }} }} }} }}",
            fields.join(", ")
        );
        self.run_performer_query(query).await
    }

    /// Shared POST + parse for queryPerformers, dropping placeholder entries
    /// (empty or purely-numeric names — scene-only stubs with no real identity).
    async fn run_performer_query(&self, query: String) -> Result<Vec<Performer>> {
        let resp = self
            .client
            .post(STASHDB_GRAPHQL)
            .header("ApiKey", &self.api_key)
            .json(&serde_json::json!({ "query": query }))
            .send()
            .await
            .context("StashDB query failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("StashDB returned {}", resp.status());
        }

        let parsed: QueryResponse = resp.json().await.context("parse StashDB query")?;
        let performers = parsed
            .data
            .map(|d| d.query_performers.performers)
            .unwrap_or_default();

        Ok(performers
            .into_iter()
            .map(|p| p.into_performer())
            .filter(|p| !p.name.trim().is_empty() && !p.name.chars().all(|c| c.is_ascii_digit()))
            .collect())
    }
}

#[derive(Deserialize)]
struct QueryResponse {
    data: Option<QueryData>,
}

#[derive(Deserialize)]
struct QueryData {
    #[serde(rename = "queryPerformers")]
    query_performers: QueryResult,
}

#[derive(Deserialize)]
struct QueryResult {
    performers: Vec<FullPerformer>,
}

#[derive(Deserialize)]
struct FullPerformer {
    id: String,
    name: String,
    gender: Option<String>,
    ethnicity: Option<String>,
    hair_color: Option<String>,
    eye_color: Option<String>,
    cup_size: Option<String>,
    band_size: Option<u32>,
    waist_size: Option<u32>,
    hip_size: Option<u32>,
    #[serde(default)]
    images: Vec<StashImage>,
}

impl FullPerformer {
    fn into_performer(self) -> Performer {
        let mut p = Performer::new(self.name);
        p.gender = self.gender.map(|g| titlecase_enum(&g));
        p.ethnicity = self.ethnicity.map(|e| titlecase_enum(&e));
        p.hair_color = self.hair_color.map(|h| titlecase_enum(&h));
        p.eye_color = self.eye_color.map(|e| titlecase_enum(&e));

        // Reconstruct a "34DD-24-36"-style measurements string when possible.
        if let (Some(band), Some(cup)) = (self.band_size, &self.cup_size) {
            let waist = self.waist_size.map(|w| w.to_string()).unwrap_or_default();
            let hip = self.hip_size.map(|h| h.to_string()).unwrap_or_default();
            p.measurements = Some(format!("{}{}-{}-{}", band, cup, waist, hip));
        }

        // Largest image first → profile; all → gallery (InsightFace finds the face).
        let mut imgs = self.images;
        imgs.sort_by_key(|i| std::cmp::Reverse(i.width as u64 * i.height as u64));
        let urls: Vec<String> = imgs.into_iter().map(|i| i.url).collect();
        p.profile_image_url = urls.first().cloned();
        p.gallery_urls = urls;

        p.source = Some("StashDB".to_string());
        p.source_url = Some(format!("https://stashdb.org/performers/{}", self.id));
        p
    }
}
