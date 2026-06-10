//! Minimal pornpics.com image source.
//!
//! Used purely as an extra pool of *full-body* images for building body-frame
//! centroids — niche performers often have only one or two profile photos
//! elsewhere, and a pornstar gallery page yields ~20 distinct full-body shoots.
//!
//! Scope/ethics: pornpics.com's robots.txt permits `/pornstars/` and sets no
//! AI/crawler restriction (unlike IAFD, which we deliberately do not touch).
//! We fetch one public gallery-index page and read the image URLs already in it
//! — for personal, local model-building only, never redistributed.

use anyhow::{Context, Result};

const BASE: &str = "https://www.pornpics.com";
/// Gallery thumbnails are served at this width; we swap it for a larger one so
/// pose estimation has enough pixels to work with.
const THUMB_SEG: &str = "/460/";
const FULL_SEG: &str = "/1280/";

pub struct PornpicsClient {
    client: reqwest::Client,
}

impl Default for PornpicsClient {
    fn default() -> Self {
        Self::new()
    }
}

impl PornpicsClient {
    pub fn new() -> Self {
        // Browser UA — the site serves an empty page to non-browser agents.
        let client = crate::http::client(crate::http::BROWSER_UA);
        PornpicsClient { client }
    }

    /// Returns up to `max` full-body image URLs for a performer from their
    /// pornpics.com pornstar page. Best-effort: returns an empty vec on any
    /// error (no such page, network failure) so callers can fall back.
    pub async fn image_urls(&self, name: &str, max: usize) -> Vec<String> {
        match self.try_image_urls(name, max).await {
            Ok(urls) => urls,
            Err(e) => {
                log::warn!("pornpics lookup failed for {}: {}", name, e);
                Vec::new()
            }
        }
    }

    async fn try_image_urls(&self, name: &str, max: usize) -> Result<Vec<String>> {
        let url = format!("{}/pornstars/{}/", BASE, slugify(name));
        log::info!("pornpics: {}", url);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("pornpics request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("pornpics returned {}", resp.status());
        }
        let body = resp.text().await.context("read pornpics body")?;
        Ok(extract_image_urls(&body, max))
    }

    /// Returns multi-angle image URLs from a performer's full galleries — the
    /// photo *sets* (front/rear/side within a shoot), not just one cover each.
    /// Visits up to `max_galleries` of the performer's galleries and takes up to
    /// `per_gallery` images from each, upscaled. Far richer angle coverage than
    /// `image_urls` (which returns one cover per gallery). Best-effort.
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
                log::warn!("pornpics galleries failed for {}: {}", name, e);
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
        let page_url = format!("{}/pornstars/{}/", BASE, slugify(name));
        let resp = self
            .client
            .get(&page_url)
            .send()
            .await
            .context("pornpics request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("pornpics returned {}", resp.status());
        }
        let body = resp.text().await.context("read pornpics body")?;

        let mut out: Vec<String> = Vec::new();
        for link in extract_gallery_links(&body).into_iter().take(max_galleries) {
            // Each gallery page is one photoshoot; pull its images (best-effort).
            let g = match self.client.get(&link).send().await {
                Ok(r) if r.status().is_success() => r.text().await.unwrap_or_default(),
                _ => continue,
            };
            for u in extract_cdni_urls(&g, per_gallery) {
                if !out.contains(&u) {
                    out.push(u);
                }
            }
        }
        Ok(out)
    }
}

/// "Christina Sapphire" -> "christina-sapphire" (lowercase, runs of non-alnum
/// collapsed to a single hyphen, trimmed).
fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_dash = false;
    for ch in name.trim().to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Scans raw HTML for `cdni.pornpics.com<THUMB_SEG>...jpg` gallery thumbnails
/// and upscales each to a pose-usable width by swapping the size segment.
/// Deduplicated, preserving page order, capped at `max`.
fn extract_image_urls(html: &str, max: usize) -> Vec<String> {
    let marker = format!("https://cdni.pornpics.com{}", THUMB_SEG);
    let mut urls: Vec<String> = Vec::new();
    let mut rest = html;

    while let Some(pos) = rest.find(&marker) {
        let tail = &rest[pos..];
        // The URL ends at the first delimiter (quote/space/escape/paren).
        let end = tail
            .find(|c: char| c == '"' || c == '\'' || c == '\\' || c == ')' || c.is_whitespace())
            .unwrap_or(tail.len());
        let raw = &tail[..end];

        if raw.ends_with(".jpg") || raw.ends_with(".jpeg") || raw.ends_with(".webp") {
            let big = raw.replacen(THUMB_SEG, FULL_SEG, 1);
            if !urls.contains(&big) {
                urls.push(big);
                if urls.len() >= max {
                    break;
                }
            }
        }
        rest = &tail[end..];
    }
    urls
}

/// Gallery *page* URLs (`/galleries/<slug>/`) linked from a performer page.
fn extract_gallery_links(html: &str) -> Vec<String> {
    const MARKER: &str = "https://www.pornpics.com/galleries/";
    let mut links: Vec<String> = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find(MARKER) {
        let tail = &rest[pos..];
        let end = tail
            .find(|c: char| c == '"' || c == '\'' || c == '\\' || c.is_whitespace())
            .unwrap_or(tail.len());
        let link = tail[..end].to_string();
        if link.ends_with('/') && !links.contains(&link) {
            links.push(link);
        }
        rest = &tail[end..];
    }
    links
}

/// Scans raw HTML for any `cdni.pornpics.com/<width>/...` image (gallery interior
/// images use varying widths, not just the cover thumbnail size) and upscales
/// each to 1280. Deduplicated, page order, capped at `max`.
fn extract_cdni_urls(html: &str, max: usize) -> Vec<String> {
    const HOST: &str = "https://cdni.pornpics.com/";
    let mut urls: Vec<String> = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find(HOST) {
        let tail = &rest[pos..];
        let end = tail
            .find(|c: char| c == '"' || c == '\'' || c == '\\' || c == ')' || c.is_whitespace())
            .unwrap_or(tail.len());
        let raw = &tail[..end];
        if raw.ends_with(".jpg") || raw.ends_with(".jpeg") || raw.ends_with(".webp") {
            if let Some(big) = upscale_cdni(raw) {
                if !urls.contains(&big) {
                    urls.push(big);
                    if urls.len() >= max {
                        break;
                    }
                }
            }
        }
        rest = &tail[end..];
    }
    urls
}

/// Rewrites a cdni URL's leading size segment to 1280:
/// `…/460/a/b.jpg` or `…/925/a/b.jpg` → `…/1280/a/b.jpg`. None if not size-prefixed.
fn upscale_cdni(url: &str) -> Option<String> {
    const HOST: &str = "https://cdni.pornpics.com/";
    let after = url.strip_prefix(HOST)?;
    let slash = after.find('/')?;
    let width = &after[..slash];
    if width.is_empty() || !width.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(format!("{}1280/{}", HOST, &after[slash + 1..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_handles_spaces_and_punctuation() {
        assert_eq!(slugify("Christina Sapphire"), "christina-sapphire");
        assert_eq!(slugify("  Anna  Claire   Clouds "), "anna-claire-clouds");
        assert_eq!(slugify("Vicky O'Hara"), "vicky-o-hara");
    }

    #[test]
    fn extract_upscales_and_dedupes_in_order() {
        let html = r#"
            <img src="https://cdni.pornpics.com/460/7/445/111/111_001_a.jpg">
            <img data-src="https://cdni.pornpics.com/460/7/445/222/222_002_b.jpg"/>
            <img src="https://cdni.pornpics.com/460/7/445/111/111_001_a.jpg">
        "#;
        let urls = extract_image_urls(html, 10);
        assert_eq!(
            urls,
            vec![
                "https://cdni.pornpics.com/1280/7/445/111/111_001_a.jpg".to_string(),
                "https://cdni.pornpics.com/1280/7/445/222/222_002_b.jpg".to_string(),
            ]
        );
    }

    #[test]
    fn extract_respects_max() {
        let html = "a https://cdni.pornpics.com/460/1/1/1/1_1_a.jpg \
                    b https://cdni.pornpics.com/460/1/1/2/2_2_b.jpg \
                    c https://cdni.pornpics.com/460/1/1/3/3_3_c.jpg";
        assert_eq!(extract_image_urls(html, 2).len(), 2);
    }

    #[test]
    fn extract_ignores_non_images() {
        let html = "https://cdni.pornpics.com/460/1/1/1/page.html and text";
        assert!(extract_image_urls(html, 10).is_empty());
    }

    #[test]
    fn gallery_links_extracted_and_deduped() {
        let html = r#"
            <a href="https://www.pornpics.com/galleries/some-shoot-12345/">x</a>
            <a href="https://www.pornpics.com/galleries/other-67890/">y</a>
            <a href="https://www.pornpics.com/galleries/some-shoot-12345/">dup</a>
            <a href="https://www.pornpics.com/pornstars/jane/">not-a-gallery</a>
        "#;
        let links = extract_gallery_links(html);
        assert_eq!(links.len(), 2);
        assert!(links[0].ends_with("some-shoot-12345/"));
    }

    #[test]
    fn cdni_upscales_any_width_to_1280() {
        assert_eq!(
            upscale_cdni("https://cdni.pornpics.com/460/7/445/1/1_2_a.jpg").as_deref(),
            Some("https://cdni.pornpics.com/1280/7/445/1/1_2_a.jpg")
        );
        assert_eq!(
            upscale_cdni("https://cdni.pornpics.com/925/1/2/3/x.jpg").as_deref(),
            Some("https://cdni.pornpics.com/1280/1/2/3/x.jpg")
        );
        // not a size-prefixed URL
        assert_eq!(upscale_cdni("https://cdni.pornpics.com/media/x.jpg"), None);
    }

    #[test]
    fn extract_cdni_pulls_varied_widths() {
        let html = "a https://cdni.pornpics.com/460/1/1/1/a.jpg \
                    b https://cdni.pornpics.com/1280/1/1/2/b.jpg \
                    c https://cdni.pornpics.com/930/1/1/3/c.webp";
        let urls = extract_cdni_urls(html, 10);
        assert_eq!(urls.len(), 3);
        assert!(urls.iter().all(|u| u.contains("/1280/")));
    }
}
