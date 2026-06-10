use crate::config::GenderFilter;
use crate::models::Performer;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const TPDB_API_BASE: &str = "https://api.theporndb.net";

/// Converts "CAUCASIAN", "caucasian", or "Caucasian" → "Caucasian"
fn to_title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
    }
}

/// ThePornDB API client
pub struct TpdbClient {
    client: reqwest::Client,
    api_key: String,
}

#[derive(Debug, Deserialize)]
struct TpdbSearchResponse {
    data: Vec<TpdbPerformer>,
}

#[derive(Debug, Deserialize)]
struct TpdbPerformerResponse {
    data: TpdbPerformer,
}

#[derive(Debug, Deserialize, Serialize)]
struct TpdbPerformer {
    #[serde(rename = "_id", default)]
    numeric_id: i64,
    id: String,
    name: String,
    #[serde(default)]
    gender: Option<String>,
    #[serde(default)]
    extras: TpdbExtras,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    face: Option<String>,
    #[serde(default)]
    images: Vec<String>,
    #[serde(default)]
    posters: Vec<TpdbPoster>,
}

/// One entry of a performer's `posters[]` array. TPDB returns several profile
/// posters per performer (different shoots/sites), not just the single `image`.
#[derive(Debug, Deserialize, Serialize, Default)]
struct TpdbPoster {
    #[serde(default)]
    url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TpdbSceneResponse {
    data: Vec<TpdbScene>,
}

#[derive(Debug, Deserialize)]
struct TpdbScene {
    /// `background` is usually `{ full, large, ... }`, but some scenes return
    /// `[]` or null — keep it as a Value and extract `full` defensively.
    #[serde(default)]
    background: serde_json::Value,
    #[serde(default)]
    performers: Vec<TpdbScenePerformer>,
}

#[derive(Debug, Deserialize)]
struct TpdbScenePerformer {
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct TpdbExtras {
    #[serde(default)]
    birthday: Option<String>,
    #[serde(default)]
    age: Option<u32>,
    #[serde(default)]
    ethnicity: Option<String>,
    #[serde(default)]
    nationality: Option<String>,
    #[serde(default)]
    birthplace_code: Option<String>,
    #[serde(default, alias = "hair_colour")]
    hair_color: Option<String>,
    #[serde(default, alias = "eye_colour")]
    eye_color: Option<String>,
    #[serde(default)]
    height: Option<String>,
    #[serde(default)]
    weight: Option<String>,
    #[serde(default)]
    measurements: Option<String>,
    #[serde(default)]
    cupsize: Option<String>,
    #[serde(default)]
    waist: Option<String>,
    #[serde(default)]
    hips: Option<String>,
    #[serde(default)]
    gender: Option<String>,
    #[serde(default)]
    tattoos: Option<String>,
    #[serde(default)]
    piercings: Option<String>,
    #[serde(default, deserialize_with = "flexible_bool")]
    fake_boobs: Option<bool>,
}

/// TPDB returns fake_boobs as a JSON bool, a string ("True"/"1"), or null.
/// Accept all of them gracefully.
fn flexible_bool<'de, D>(deserializer: D) -> std::result::Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let v = serde_json::Value::deserialize(deserializer)?;
    Ok(match v {
        serde_json::Value::Bool(b) => Some(b),
        serde_json::Value::String(s) => {
            let s = s.to_lowercase();
            Some(s == "true" || s == "1" || s == "yes")
        }
        serde_json::Value::Number(n) => Some(n.as_i64().unwrap_or(0) != 0),
        _ => None,
    })
}

impl TpdbClient {
    pub fn new(api_key: String) -> Self {
        let client = crate::http::client(crate::http::APP_UA);
        TpdbClient { client, api_key }
    }

    /// Search for a performer by name
    pub async fn search_performer(&self, name: &str) -> Result<Option<Performer>> {
        let url = format!(
            "{}/performers?q={}",
            TPDB_API_BASE,
            urlencoding::encode(name)
        );

        log::info!("Searching ThePornDB: {}", name);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to search ThePornDB")?;

        if !response.status().is_success() {
            anyhow::bail!("ThePornDB returned status: {}", response.status());
        }

        let search_result: TpdbSearchResponse = response
            .json()
            .await
            .context("Failed to parse ThePornDB search response")?;

        if let Some(tpdb_performer) = search_result.data.first() {
            self.get_performer(&tpdb_performer.id).await
        } else {
            Ok(None)
        }
    }

    /// Collects clean full-body scene background images for a performer.
    ///
    /// Scene backgrounds are action stills (whole body in frame) hosted on the
    /// TPDB CDN — far better for pose/body matching than the cropped profile
    /// poster, and there are many per performer. This rescues niche performers
    /// who have only one or two profile photos. Best-effort: returns an empty
    /// vec on any error so callers can fall back to other sources.
    pub async fn scene_image_urls(&self, name: &str, max: usize) -> Vec<String> {
        match self.try_scene_image_urls(name, max).await {
            Ok(urls) => urls,
            Err(e) => {
                log::warn!("TPDB scene lookup failed for {}: {}", name, e);
                Vec::new()
            }
        }
    }

    async fn try_scene_image_urls(&self, name: &str, max: usize) -> Result<Vec<String>> {
        // `q` is a fuzzy text search, so it returns scenes that merely mention
        // the name. We keep only scenes that actually list this performer.
        let url = format!(
            "{}/scenes?q={}&per_page=30",
            TPDB_API_BASE,
            urlencoding::encode(name)
        );
        log::info!("TPDB scenes: {}", url);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to query TPDB scenes")?;

        if !resp.status().is_success() {
            anyhow::bail!("TPDB scenes returned {}", resp.status());
        }

        let parsed: TpdbSceneResponse = resp.json().await.context("parse TPDB scenes")?;
        let target = name.trim().to_lowercase();

        let mut urls: Vec<String> = Vec::new();
        for scene in parsed.data {
            let features = scene.performers.iter().any(|p| {
                p.name
                    .as_deref()
                    .is_some_and(|n| n.trim().to_lowercase() == target)
            });
            if !features {
                continue;
            }
            // Prefer the clean CDN background; skip watermarked posters and
            // external studio-hosted images. `.get` tolerates [] / null shapes.
            if let Some(full) = scene.background.get("full").and_then(|v| v.as_str()) {
                let full = full.to_string();
                if !urls.contains(&full) {
                    urls.push(full);
                    if urls.len() >= max {
                        break;
                    }
                }
            }
        }
        Ok(urls)
    }

    /// Get detailed performer information by ID
    pub async fn get_performer(&self, id: &str) -> Result<Option<Performer>> {
        let url = format!("{}/performers/{}", TPDB_API_BASE, id);

        log::info!("Fetching performer details: {}", id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to fetch performer from ThePornDB")?;

        if !response.status().is_success() {
            anyhow::bail!("ThePornDB returned status: {}", response.status());
        }

        let performer_response: TpdbPerformerResponse = response
            .json()
            .await
            .context("Failed to parse ThePornDB performer response")?;

        Ok(Some(self.convert_to_performer(performer_response.data)))
    }

    /// Convert ThePornDB performer to our Performer model
    fn convert_to_performer(&self, tpdb: TpdbPerformer) -> Performer {
        let mut performer = Performer::new(tpdb.name.clone());

        performer.body_type = self.infer_body_type(&tpdb);

        performer.age = tpdb.extras.age.or_else(|| {
            tpdb.extras.birthday.as_deref().and_then(|bd| {
                let dob = chrono::NaiveDate::parse_from_str(bd, "%Y-%m-%d").ok()?;
                let today = chrono::Utc::now().date_naive();
                let years = today.years_since(dob)?;
                Some(years)
            })
        });
        performer.ethnicity = tpdb.extras.ethnicity;
        performer.nationality = tpdb.extras.nationality;
        performer.birthplace_code = tpdb.extras.birthplace_code;
        performer.hair_color = tpdb.extras.hair_color;
        performer.eye_color = tpdb.extras.eye_color;
        performer.height = tpdb.extras.height;
        performer.weight = tpdb.extras.weight;
        performer.measurements = tpdb.extras.measurements;
        performer.birthdate = tpdb.extras.birthday;

        performer.face_url = tpdb.face;
        // Merge the multi-image `posters[]` array into the gallery — TPDB returns
        // several profile posters, but we previously kept only the single `image`.
        let mut gallery = tpdb.images;
        for poster in tpdb.posters {
            if let Some(u) = poster.url {
                if !gallery.contains(&u) {
                    gallery.push(u);
                }
            }
        }
        performer.profile_image_url = tpdb.image.or_else(|| gallery.first().cloned());
        performer.gallery_urls = gallery;

        performer.gender = tpdb.gender.or(tpdb.extras.gender);
        performer.tpdb_id = Some(tpdb.numeric_id);
        performer.tattoos = tpdb.extras.tattoos;
        performer.piercings = tpdb.extras.piercings;
        performer.fake_boobs = tpdb.extras.fake_boobs;
        performer.source = Some("ThePornDB".to_string());
        performer.source_url = Some(format!("https://theporndb.net/performers/{}", tpdb.id));
        performer.last_updated = Some(chrono::Utc::now().to_rfc3339());

        performer
    }

    /// Attribute-based search — all params optional, combined server-side where supported.
    /// Hip min is applied server-side; waist range and hip max are filtered client-side.
    pub async fn search_by_attributes(
        &self,
        ethnicity: Option<&str>,
        hair_colour: Option<&str>,
        eye_colour: Option<&str>,
        cup: Option<&str>,
        hips_target: Option<u32>,
        waist_target: Option<u32>,
        whr_target: Option<f64>,
        age_min: Option<u32>,
        age_max: Option<u32>,
        gender_filter: &GenderFilter,
        fetch: usize,
    ) -> Result<Vec<Performer>> {
        let mut url = format!("{}/performers?per_page={}", TPDB_API_BASE, fetch);

        if let Some(g) = gender_filter.tpdb_value() {
            url.push_str(&format!("&gender={}", g));
        }
        if let Some(e) = ethnicity {
            url.push_str(&format!(
                "&ethnicity={}",
                urlencoding::encode(&to_title_case(e))
            ));
        }
        if let Some(h) = hair_colour {
            url.push_str(&format!(
                "&hair_colour={}",
                urlencoding::encode(&to_title_case(h))
            ));
        }
        if let Some(e) = eye_colour {
            url.push_str(&format!(
                "&eye_colour={}",
                urlencoding::encode(&to_title_case(e))
            ));
        }
        if let Some(c) = cup {
            let cup_letter = c
                .trim_start_matches(|ch: char| ch.is_ascii_digit())
                .to_uppercase();
            url.push_str(&format!("&cup={}", urlencoding::encode(&cup_letter)));
        }
        // Hip: use server-side minimum (target - 4 inches)
        if let Some(h) = hips_target {
            let hip_min = h.saturating_sub(4);
            url.push_str(&format!("&hip={}&hip_operation=%3E%3D", hip_min));
        }
        if let Some(a) = age_min {
            url.push_str(&format!("&age={}&age_operation=%3E%3D", a));
        }

        log::info!("Attribute search: {}", url);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to query TPDB")?;

        if !resp.status().is_success() {
            anyhow::bail!("TPDB returned {}", resp.status());
        }

        let result: TpdbSearchResponse =
            resp.json().await.context("Failed to parse TPDB response")?;

        let mut performers: Vec<Performer> = result
            .data
            .into_iter()
            .filter(|p| gender_filter.matches(p.gender.as_deref().or(p.extras.gender.as_deref())))
            .map(|p| self.convert_to_performer(p))
            .collect();

        // Client-side: WHR range, hip upper bound, waist range, age max
        if let Some(whr) = whr_target {
            let tolerance = 0.05;
            performers.retain(|p| {
                crate::recommender::performer_whr(p).is_some_and(|r| (r - whr).abs() <= tolerance)
            });
        }
        if let Some(h) = hips_target {
            let hip_max = h + 4;
            performers.retain(|p| {
                // Keep if no hip data (can't verify) or within range
                p.measurements
                    .as_deref()
                    .and_then(|m| m.split('-').nth(2))
                    .and_then(|s| {
                        s.trim_end_matches(|c: char| !c.is_ascii_digit())
                            .parse::<u32>()
                            .ok()
                    })
                    .is_some_and(|hip| hip >= h.saturating_sub(4) && hip <= hip_max)
            });
        }
        if let Some(w) = waist_target {
            performers.retain(|p| {
                p.measurements
                    .as_deref()
                    .and_then(|m| m.split('-').nth(1))
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .is_some_and(|waist| waist >= w.saturating_sub(4) && waist <= w + 4)
            });
        }
        if let Some(max) = age_max {
            performers.retain(|p| p.age.is_none_or(|a| a <= max));
        }

        Ok(performers)
    }

    /// Find performers similar to a single performer by their TPDB UUID.
    pub async fn similar_to(
        &self,
        tpdb_uuid: &str,
        gender_filter: &GenderFilter,
    ) -> Result<Vec<Performer>> {
        let url = format!("{}/performers/{}/similar", TPDB_API_BASE, tpdb_uuid);
        log::info!("Similar to {}: {}", tpdb_uuid, url);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to query TPDB similar")?;

        if !resp.status().is_success() {
            anyhow::bail!("TPDB returned {}", resp.status());
        }

        let result: TpdbSearchResponse = resp
            .json()
            .await
            .context("Failed to parse similar response")?;

        Ok(result
            .data
            .into_iter()
            .filter(|p| gender_filter.matches(p.gender.as_deref().or(p.extras.gender.as_deref())))
            .map(|p| self.convert_to_performer(p))
            .collect())
    }

    /// Primary: use /performers/similar?ids=... with liked performer IDs.
    /// Fallback: filtered search by gender + ethnicity + cup.
    pub async fn get_recommendations(
        &self,
        liked_ids: &[i64],
        ethnicity: Option<&str>,
        top_cup: Option<&str>,
        gender_filter: &GenderFilter,
    ) -> Result<Vec<Performer>> {
        // ── Primary: similar performers ──────────────────────────────────────
        if !liked_ids.is_empty() {
            let ids_str = liked_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let url = format!("{}/performers/similar?ids={}", TPDB_API_BASE, ids_str);
            log::info!("Similar performers: {}", url);

            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await;

            if let Ok(r) = resp {
                if r.status().is_success() {
                    if let Ok(result) = r.json::<TpdbSearchResponse>().await {
                        let performers: Vec<Performer> = result
                            .data
                            .into_iter()
                            .filter(|p| {
                                gender_filter
                                    .matches(p.gender.as_deref().or(p.extras.gender.as_deref()))
                            })
                            .map(|p| self.convert_to_performer(p))
                            .collect();
                        if !performers.is_empty() {
                            return Ok(performers);
                        }
                    }
                }
            }
        }

        // ── Fallback: filtered search ─────────────────────────────────────────
        let mut url = format!("{}/performers?per_page=100", TPDB_API_BASE);
        if let Some(g) = gender_filter.tpdb_value() {
            url.push_str(&format!("&gender={}", g));
        }
        if let Some(eth) = ethnicity {
            url.push_str(&format!(
                "&ethnicity={}",
                urlencoding::encode(&to_title_case(eth))
            ));
        }
        if let Some(cup) = top_cup {
            let cup_letter = cup
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .to_uppercase();
            url.push_str(&format!("&cup={}", urlencoding::encode(&cup_letter)));
        }

        log::info!("Fallback search: {}", url);

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to query ThePornDB")?;

        if !resp.status().is_success() {
            return Ok(vec![]);
        }

        let result: TpdbSearchResponse =
            resp.json().await.context("Failed to parse TPDB response")?;

        Ok(result
            .data
            .into_iter()
            .filter(|p| gender_filter.matches(p.gender.as_deref().or(p.extras.gender.as_deref())))
            .map(|p| self.convert_to_performer(p))
            .collect())
    }

    /// Infer body type from cup size, waist-to-hip ratio, and weight.
    /// Prefers direct cupsize/waist/hips fields (available in search results)
    /// over parsing the measurements string.
    fn infer_body_type(&self, tpdb: &TpdbPerformer) -> String {
        // Cup score: prefer direct cupsize field, fall back to parsing measurements
        let cup = tpdb
            .extras
            .cupsize
            .as_deref()
            .map(Self::cup_score)
            .or_else(|| tpdb.extras.measurements.as_deref().map(Self::cup_score))
            .unwrap_or(2);

        // WHR: prefer direct waist/hips fields, fall back to parsing measurements
        let whr = match (tpdb.extras.waist.as_deref(), tpdb.extras.hips.as_deref()) {
            (Some(w), Some(h)) => {
                let waist: f64 = w.parse().unwrap_or(0.0);
                let hips: f64 = h.parse().unwrap_or(0.0);
                if hips > 0.0 {
                    Some(waist / hips)
                } else {
                    None
                }
            }
            _ => tpdb
                .extras
                .measurements
                .as_deref()
                .and_then(Self::waist_hip_ratio),
        };

        let weight = tpdb
            .extras
            .weight
            .as_deref()
            .and_then(Self::parse_weight_lbs);

        if weight.is_some_and(|w| w > 180.0) {
            return "BBW".to_string();
        }
        if cup >= 5 || whr.is_some_and(|r| r < 0.70) {
            return "Curvy".to_string();
        }
        if cup >= 4 || whr.is_some_and(|r| r < 0.75) {
            return "Full-Figured".to_string();
        }
        if cup <= 1 && weight.is_some_and(|w| w < 110.0) {
            return "Petite".to_string();
        }
        if cup <= 2 && whr.is_some_and(|r| r > 0.78) {
            return "Slim".to_string();
        }
        "Average".to_string()
    }

    /// Converts cup letter(s) to a numeric score
    fn cup_score(measurements: &str) -> u8 {
        let bust = measurements.split('-').next().unwrap_or("");
        let cup = bust.trim_start_matches(|c: char| c.is_ascii_digit());
        match cup.to_uppercase().as_str() {
            "AA" | "AAA" => 0,
            "A" => 1,
            "B" => 2,
            "C" => 3,
            "D" => 4,
            "DD" | "E" => 5,
            "DDD" | "F" => 6,
            "G" | "H" | "I" | "J" | "K" => 7,
            _ => 2,
        }
    }

    /// Parses waist-to-hip ratio from "34B-24-36" style strings
    fn waist_hip_ratio(measurements: &str) -> Option<f64> {
        let parts: Vec<&str> = measurements.split('-').collect();
        if parts.len() < 3 {
            return None;
        }
        let waist: f64 = parts[1].trim().parse().ok()?;
        let hips: f64 = parts[2]
            .trim()
            .trim_end_matches(|c: char| !c.is_ascii_digit())
            .parse()
            .ok()?;
        if hips == 0.0 {
            return None;
        }
        Some(waist / hips)
    }

    /// Parses weight string ("130 lbs", "59 kg") to pounds
    fn parse_weight_lbs(weight: &str) -> Option<f64> {
        let num: f64 = weight.split_whitespace().next()?.parse().ok()?;
        if weight.to_lowercase().contains("kg") {
            Some(num * 2.205)
        } else {
            Some(num)
        }
    }
}
