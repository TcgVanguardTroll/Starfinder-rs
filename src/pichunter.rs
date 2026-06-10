//! pichunter.com image source.
//!
//! Its robots.txt is fully open (`User-agent: * / Disallow:` — no AI/crawler
//! restriction at all). Per-performer pages live at
//! `/models/<TitleCase_Underscore>` and link full `/gallery/<id>/` photoshoots;
//! we pull the `post_single_big` (full-size) images for multi-angle coverage.
//! Coverage often differs from pornpics, so it's genuinely additive. Used for
//! personal, local model-building only — vectors + metadata, never redistributed.

use anyhow::{Context, Result};

const BASE: &str = "https://www.pichunter.com";

pub struct PichunterClient {
    client: reqwest::Client,
}

impl Default for PichunterClient {
    fn default() -> Self {
        Self::new()
    }
}

impl PichunterClient {
    pub fn new() -> Self {
        let client = crate::http::client(crate::http::BROWSER_UA);
        PichunterClient { client }
    }

    /// Multi-angle full-size images from a performer's pichunter galleries.
    /// Visits up to `max_galleries` of their `/gallery/<id>/` shoots and takes up
    /// to `per_gallery` full-size (`post_single_big`) images from each.
    /// Best-effort: empty vec on any error.
    pub async fn gallery_image_urls(
        &self,
        name: &str,
        max_galleries: usize,
        per_gallery: usize,
    ) -> Vec<String> {
        match self
            .try_gallery_image_urls(name, max_galleries, per_gallery)
            .await
        {
            Ok(urls) => urls,
            Err(e) => {
                log::warn!("pichunter lookup failed for {}: {}", name, e);
                Vec::new()
            }
        }
    }

    async fn try_gallery_image_urls(
        &self,
        name: &str,
        max_galleries: usize,
        per_gallery: usize,
    ) -> Result<Vec<String>> {
        let url = format!("{}/models/{}", BASE, slugify(name));
        log::info!("pichunter: {}", url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("pichunter request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("pichunter returned {}", resp.status());
        }
        let body = resp.text().await.context("read pichunter body")?;

        let mut out: Vec<String> = Vec::new();
        for link in extract_gallery_links(&body).into_iter().take(max_galleries) {
            let g = match self.client.get(&link).send().await {
                Ok(r) if r.status().is_success() => r.text().await.unwrap_or_default(),
                _ => continue,
            };
            for u in extract_big_images(&g, per_gallery) {
                if !out.contains(&u) {
                    out.push(u);
                }
            }
        }
        Ok(out)
    }
}

/// "christina sapphire" -> "Christina_Sapphire" (TitleCase words joined by `_`,
/// pichunter's `/models/` slug form). First letter of each word upper-cased,
/// the rest left as-is.
fn slugify(name: &str) -> String {
    name.split_whitespace()
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join("_")
}

/// `/gallery/<id>/<slug>` page URLs linked from a performer page.
fn extract_gallery_links(html: &str) -> Vec<String> {
    const MARKER: &str = "https://www.pichunter.com/gallery/";
    let mut links: Vec<String> = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find(MARKER) {
        let tail = &rest[pos..];
        let end = tail
            .find(|c: char| c == '"' || c == '\'' || c == '\\' || c.is_whitespace())
            .unwrap_or(tail.len());
        let link = tail[..end].to_string();
        if !links.contains(&link) {
            links.push(link);
        }
        rest = &tail[end..];
    }
    links
}

/// Full-size (`post_single_big`) pichunter CDN image URLs in the HTML.
fn extract_big_images(html: &str, max: usize) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find("https://cdn") {
        let tail = &rest[pos..];
        let end = tail
            .find(|c: char| c == '"' || c == '\'' || c == '\\' || c == ')' || c.is_whitespace())
            .unwrap_or(tail.len());
        let raw = &tail[..end];
        if raw.contains(".pichunter.com/") && raw.ends_with("post_single_big.jpg") {
            let s = raw.to_string();
            if !urls.contains(&s) {
                urls.push(s);
                if urls.len() >= max {
                    break;
                }
            }
        }
        rest = &tail[end..];
    }
    urls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_titlecase_underscore() {
        assert_eq!(slugify("christina sapphire"), "Christina_Sapphire");
        assert_eq!(slugify("Lisa Ann"), "Lisa_Ann");
        assert_eq!(slugify("  merce   palau "), "Merce_Palau");
    }

    #[test]
    fn gallery_links_extracted() {
        let html = r#"
            <a href="https://www.pichunter.com/gallery/151591/All-Over-30-Christina">a</a>
            <a href="https://www.pichunter.com/gallery/152502/Other">b</a>
            <a href="https://www.pichunter.com/models/Christina_Sapphire">not-gallery</a>
        "#;
        let links = extract_gallery_links(html);
        assert_eq!(links.len(), 2);
        assert!(links[0].contains("/gallery/151591/"));
    }

    #[test]
    fn big_images_only_full_size() {
        let html = "x https://cdn1337.pichunter.com/media/posts/12--1/conversions/1-a-post_archive_thumb.jpg \
                    y https://cdn1337.pichunter.com/media/posts/12--1/conversions/1-a-post_single_big.jpg \
                    z https://cdn1337.pichunter.com/media/posts/12--2/conversions/2-b-post_single_big.jpg";
        let urls = extract_big_images(html, 10);
        assert_eq!(urls.len(), 2); // only the two post_single_big, not the thumb
        assert!(urls.iter().all(|u| u.ends_with("post_single_big.jpg")));
    }
}
