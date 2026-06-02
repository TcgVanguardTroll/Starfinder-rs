use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenderFilter {
    Female,
    Male,
    TransFemale,
    TransMale,
    Any,
}

impl GenderFilter {
    /// The value sent to TPDB's gender filter param
    pub fn tpdb_value(&self) -> Option<&'static str> {
        match self {
            GenderFilter::Female      => Some("FEMALE"),
            GenderFilter::Male        => Some("MALE"),
            GenderFilter::TransFemale => Some("TRANSGENDER_FEMALE"),
            GenderFilter::TransMale   => Some("TRANSGENDER_MALE"),
            GenderFilter::Any         => None,
        }
    }

    /// Whether a TPDB gender string matches this filter
    pub fn matches(&self, gender: Option<&str>) -> bool {
        match self {
            GenderFilter::Any => true,
            _ => {
                let Some(g) = gender else { return false };
                let g = g.to_uppercase();
                match self {
                    GenderFilter::Female      => g == "FEMALE",
                    GenderFilter::Male        => g == "MALE",
                    GenderFilter::TransFemale => g == "TRANSGENDER_FEMALE",
                    GenderFilter::TransMale   => g == "TRANSGENDER_MALE",
                    GenderFilter::Any         => true,
                }
            }
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "female" | "f"           => Some(GenderFilter::Female),
            "male"   | "m"           => Some(GenderFilter::Male),
            "trans-female" | "tf"    => Some(GenderFilter::TransFemale),
            "trans-male"   | "tm"    => Some(GenderFilter::TransMale),
            "any"    | "all"         => Some(GenderFilter::Any),
            _                        => None,
        }
    }

    pub fn display(&self) -> &'static str {
        match self {
            GenderFilter::Female      => "Female (biological)",
            GenderFilter::Male        => "Male",
            GenderFilter::TransFemale => "Transgender Female",
            GenderFilter::TransMale   => "Transgender Male",
            GenderFilter::Any         => "Any",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub gender_filter: GenderFilter,
}

impl Default for Config {
    fn default() -> Self {
        Config { gender_filter: GenderFilter::Female }
    }
}

impl Config {
    pub fn load() -> Self {
        match Self::path().and_then(|p| {
            let data = std::fs::read_to_string(&p)
                .context("read config")?;
            serde_json::from_str(&data).context("parse config")
        }) {
            Ok(cfg) => cfg,
            Err(_)  => Self::default(),
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path()?;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).context("Failed to write config")?;
        Ok(())
    }

    fn path() -> Result<PathBuf> {
        let dir = dirs::data_local_dir()
            .context("Could not find data directory")?
            .join("starfinder");
        std::fs::create_dir_all(&dir)?;
        Ok(dir.join("config.json"))
    }
}
