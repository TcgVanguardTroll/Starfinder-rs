use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use crate::config::GenderFilter;
use crate::models::Performer;

const TPDB_API_BASE: &str = "https://api.theporndb.net";

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
    images: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct TpdbExtras {
    #[serde(default)]
    birthday: Option<String>,
    #[serde(default)]
    age: Option<u32>,
    #[serde(default)]
    ethnicity: Option<String>,
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
}

impl TpdbClient {
    pub fn new(api_key: String) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Starfinder/0.1.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        TpdbClient { client, api_key }
    }

    /// Search for a performer by name
    pub async fn search_performer(&self, name: &str) -> Result<Option<Performer>> {
        let url = format!("{}/performers?q={}", TPDB_API_BASE, urlencoding::encode(name));

        log::info!("Searching ThePornDB: {}", name);

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to search ThePornDB")?;

        if !response.status().is_success() {
            anyhow::bail!("ThePornDB returned status: {}", response.status());
        }

        let search_result: TpdbSearchResponse = response.json().await
            .context("Failed to parse ThePornDB search response")?;

        if let Some(tpdb_performer) = search_result.data.first() {
            self.get_performer(&tpdb_performer.id).await
        } else {
            Ok(None)
        }
    }

    /// Get detailed performer information by ID
    pub async fn get_performer(&self, id: &str) -> Result<Option<Performer>> {
        let url = format!("{}/performers/{}", TPDB_API_BASE, id);

        log::info!("Fetching performer details: {}", id);

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .context("Failed to fetch performer from ThePornDB")?;

        if !response.status().is_success() {
            anyhow::bail!("ThePornDB returned status: {}", response.status());
        }

        let performer_response: TpdbPerformerResponse = response.json().await
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
        performer.hair_color = tpdb.extras.hair_color;
        performer.eye_color = tpdb.extras.eye_color;
        performer.height = tpdb.extras.height;
        performer.weight = tpdb.extras.weight;
        performer.measurements = tpdb.extras.measurements;
        performer.birthdate = tpdb.extras.birthday;

        performer.profile_image_url = tpdb.image.or_else(|| tpdb.images.first().cloned());
        performer.gallery_urls = tpdb.images;

        performer.gender = tpdb.gender.or(tpdb.extras.gender);
        performer.tpdb_id = Some(tpdb.numeric_id);
        performer.source = Some("ThePornDB".to_string());
        performer.source_url = Some(format!("https://theporndb.net/performers/{}", tpdb.id));
        performer.last_updated = Some(chrono::Utc::now().to_rfc3339());

        performer
    }

    /// Find performers similar to a single performer by their TPDB UUID.
    pub async fn similar_to(&self, tpdb_uuid: &str, gender_filter: &GenderFilter) -> Result<Vec<Performer>> {
        let url = format!("{}/performers/{}/similar", TPDB_API_BASE, tpdb_uuid);
        log::info!("Similar to {}: {}", tpdb_uuid, url);

        let resp = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send().await
            .context("Failed to query TPDB similar")?;

        if !resp.status().is_success() {
            anyhow::bail!("TPDB returned {}", resp.status());
        }

        let result: TpdbSearchResponse = resp.json().await
            .context("Failed to parse similar response")?;

        Ok(result.data.into_iter()
            .filter(|p| gender_filter.matches(
                p.gender.as_deref().or(p.extras.gender.as_deref())
            ))
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
            let ids_str = liked_ids.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let url = format!("{}/performers/similar?ids={}", TPDB_API_BASE, ids_str);
            log::info!("Similar performers: {}", url);

            let resp = self.client.get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send().await;

            if let Ok(r) = resp {
                if r.status().is_success() {
                    if let Ok(result) = r.json::<TpdbSearchResponse>().await {
                        let performers: Vec<Performer> = result.data.into_iter()
                            .filter(|p| gender_filter.matches(
                                p.gender.as_deref().or(p.extras.gender.as_deref())
                            ))
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
            url.push_str(&format!("&ethnicity={}", urlencoding::encode(&eth.to_uppercase())));
        }
        if let Some(cup) = top_cup {
            url.push_str(&format!("&cup={}", urlencoding::encode(cup)));
        }

        log::info!("Fallback search: {}", url);

        let resp = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send().await
            .context("Failed to query ThePornDB")?;

        if !resp.status().is_success() {
            return Ok(vec![]);
        }

        let result: TpdbSearchResponse = resp.json().await
            .context("Failed to parse TPDB response")?;

        Ok(result.data.into_iter()
            .filter(|p| gender_filter.matches(
                p.gender.as_deref().or(p.extras.gender.as_deref())
            ))
            .map(|p| self.convert_to_performer(p))
            .collect())
    }

    /// Infer body type from cup size, waist-to-hip ratio, and weight.
    /// Prefers direct cupsize/waist/hips fields (available in search results)
    /// over parsing the measurements string.
    fn infer_body_type(&self, tpdb: &TpdbPerformer) -> String {
        // Cup score: prefer direct cupsize field, fall back to parsing measurements
        let cup = tpdb.extras.cupsize.as_deref()
            .map(Self::cup_score)
            .or_else(|| tpdb.extras.measurements.as_deref().map(Self::cup_score))
            .unwrap_or(2);

        // WHR: prefer direct waist/hips fields, fall back to parsing measurements
        let whr = match (tpdb.extras.waist.as_deref(), tpdb.extras.hips.as_deref()) {
            (Some(w), Some(h)) => {
                let waist: f64 = w.parse().unwrap_or(0.0);
                let hips: f64 = h.parse().unwrap_or(0.0);
                if hips > 0.0 { Some(waist / hips) } else { None }
            }
            _ => tpdb.extras.measurements.as_deref().and_then(Self::waist_hip_ratio),
        };

        let weight = tpdb.extras.weight.as_deref().and_then(Self::parse_weight_lbs);

        if weight.map_or(false, |w| w > 180.0) {
            return "BBW".to_string();
        }
        if cup >= 5 || whr.map_or(false, |r| r < 0.70) {
            return "Curvy".to_string();
        }
        if cup >= 4 || whr.map_or(false, |r| r < 0.75) {
            return "Full-Figured".to_string();
        }
        if cup <= 1 && weight.map_or(false, |w| w < 110.0) {
            return "Petite".to_string();
        }
        if cup <= 2 && whr.map_or(false, |r| r > 0.78) {
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
            "A"          => 1,
            "B"          => 2,
            "C"          => 3,
            "D"          => 4,
            "DD" | "E"   => 5,
            "DDD" | "F"  => 6,
            "G" | "H" | "I" | "J" | "K" => 7,
            _            => 2,
        }
    }

    /// Parses waist-to-hip ratio from "34B-24-36" style strings
    fn waist_hip_ratio(measurements: &str) -> Option<f64> {
        let parts: Vec<&str> = measurements.split('-').collect();
        if parts.len() < 3 { return None; }
        let waist: f64 = parts[1].trim().parse().ok()?;
        let hips: f64 = parts[2].trim()
            .trim_end_matches(|c: char| !c.is_ascii_digit())
            .parse().ok()?;
        if hips == 0.0 { return None; }
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
