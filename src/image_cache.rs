use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Manages local caching of performer images
pub struct ImageCache {
    cache_dir: PathBuf,
}

impl ImageCache {
    /// Creates a new ImageCache instance
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .context("Could not find cache directory")?
            .join("luminary")
            .join("images");

        std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;

        Ok(ImageCache { cache_dir })
    }

    /// Gets an image from cache or downloads it
    pub async fn get_image(&self, url: &str) -> Result<PathBuf> {
        let cache_path = self.get_cache_path(url);

        if cache_path.exists() {
            return Ok(cache_path);
        }

        self.download_image(url, &cache_path).await?;

        Ok(cache_path)
    }

    /// Downloads an image from URL
    async fn download_image(&self, url: &str, dest: &PathBuf) -> Result<()> {
        let response = reqwest::get(url)
            .await
            .context("Failed to download image")?;

        let bytes = response
            .bytes()
            .await
            .context("Failed to read image bytes")?;

        std::fs::write(dest, bytes).context("Failed to write image to cache")?;

        Ok(())
    }

    /// Generates cache path for a URL
    fn get_cache_path(&self, url: &str) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(url.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        let ext = url
            .split('.')
            .next_back()
            .and_then(|s| s.split('?').next())
            .filter(|s| ["jpg", "jpeg", "png", "gif", "webp"].contains(s))
            .unwrap_or("jpg");

        self.cache_dir.join(format!("{}.{}", hash, ext))
    }

    /// Clears all cached images
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir).context("Failed to clear cache")?;
            std::fs::create_dir_all(&self.cache_dir)
                .context("Failed to recreate cache directory")?;
        }
        Ok(())
    }

    /// Gets the size of the cache in bytes
    pub fn cache_size(&self) -> Result<u64> {
        let mut total_size = 0u64;

        if !self.cache_dir.exists() {
            return Ok(0);
        }

        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().is_file() {
                total_size += entry.metadata()?.len();
            }
        }

        Ok(total_size)
    }

    /// Gets the number of cached images
    pub fn cache_count(&self) -> Result<usize> {
        if !self.cache_dir.exists() {
            return Ok(0);
        }

        let count = std::fs::read_dir(&self.cache_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .count();

        Ok(count)
    }
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new().expect("Failed to create image cache")
    }
}
