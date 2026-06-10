use crate::models::Performer;
use anyhow::{Context, Result};
use scraper::{Html, Selector};

/// Scrapes performer data from various sources
pub struct Scraper {
    client: reqwest::Client,
}

impl Scraper {
    pub fn new() -> Self {
        let client = crate::http::client(crate::http::BROWSER_UA);
        Scraper { client }
    }

    /// Attempts to scrape performer data from multiple sources
    pub async fn scrape_performer(&self, name: &str) -> Result<Performer> {
        match self.scrape_freeones(name).await {
            Ok(performer) => return Ok(performer),
            Err(e) => {
                log::warn!("FreeOnes scrape failed for {}: {}", name, e);
            }
        }

        Ok(Performer::new(name.to_string()))
    }

    /// Scrapes data from FreeOnes
    async fn scrape_freeones(&self, name: &str) -> Result<Performer> {
        let url_name = name.to_lowercase().replace(" ", "-").replace("'", "");

        let url = format!("https://www.freeones.com/{}", url_name);

        log::info!("Fetching: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch FreeOnes page")?;

        if !response.status().is_success() {
            anyhow::bail!("FreeOnes returned status: {}", response.status());
        }

        let html = response.text().await?;
        let document = Html::parse_document(&html);

        let mut performer = Performer::new(name.to_string());
        performer.source = Some("FreeOnes".to_string());
        performer.source_url = Some(url);

        if let Some(age_elem) = document.select(&Selector::parse(".age").unwrap()).next() {
            let age_text = age_elem.text().collect::<String>();
            if let Ok(age) = age_text.trim().parse::<u32>() {
                performer.age = Some(age);
            }
        }

        if let Some(eth_elem) = document
            .select(&Selector::parse(".ethnicity").unwrap())
            .next()
        {
            performer.ethnicity = Some(eth_elem.text().collect::<String>().trim().to_string());
        }

        if let Some(hair_elem) = document
            .select(&Selector::parse(".hair-color").unwrap())
            .next()
        {
            performer.hair_color = Some(hair_elem.text().collect::<String>().trim().to_string());
        }

        performer.body_type = "Average".to_string();

        if let Some(img_elem) = document
            .select(&Selector::parse("img.profile-pic").unwrap())
            .next()
        {
            if let Some(src) = img_elem.value().attr("src") {
                performer.profile_image_url = Some(src.to_string());
            }
        }

        performer.last_updated = Some(chrono::Utc::now().to_rfc3339());

        Ok(performer)
    }

    /// Scrapes data from IAFD (Internet Adult Film Database)
    #[allow(dead_code)]
    async fn scrape_iafd(&self, name: &str) -> Result<Performer> {
        let url_name = name.to_lowercase().replace(" ", "").replace("'", "");

        let url = format!("https://www.iafd.com/person.rme/perfid={}/", url_name);

        log::info!("Fetching: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch IAFD page")?;

        if !response.status().is_success() {
            anyhow::bail!("IAFD returned status: {}", response.status());
        }

        let html = response.text().await?;
        let _document = Html::parse_document(&html);

        let mut performer = Performer::new(name.to_string());
        performer.source = Some("IAFD".to_string());
        performer.source_url = Some(url);

        performer.last_updated = Some(chrono::Utc::now().to_rfc3339());

        Ok(performer)
    }
}

impl Default for Scraper {
    fn default() -> Self {
        Self::new()
    }
}
