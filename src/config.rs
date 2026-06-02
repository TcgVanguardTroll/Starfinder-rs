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
            GenderFilter::Female => Some("Female"),
            GenderFilter::Male => Some("Male"),
            GenderFilter::TransFemale => Some("Transgender Female"),
            GenderFilter::TransMale => Some("Transgender Male"),
            GenderFilter::Any => None,
        }
    }

    /// Whether a TPDB gender string matches this filter
    pub fn matches(&self, gender: Option<&str>) -> bool {
        match self {
            GenderFilter::Any => true,
            _ => {
                let Some(g) = gender else { return false };
                // TPDB returns gender as "Female", "Transgender Female", etc.
                // Normalise separators so "Transgender_Female" and
                // "Transgender Female" compare equal.
                let g = g.to_uppercase().replace('_', " ");
                match self {
                    GenderFilter::Female => g == "FEMALE",
                    GenderFilter::Male => g == "MALE",
                    GenderFilter::TransFemale => g == "TRANSGENDER FEMALE",
                    GenderFilter::TransMale => g == "TRANSGENDER MALE",
                    GenderFilter::Any => true,
                }
            }
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "female" | "f" => Some(GenderFilter::Female),
            "male" | "m" => Some(GenderFilter::Male),
            "trans-female" | "tf" => Some(GenderFilter::TransFemale),
            "trans-male" | "tm" => Some(GenderFilter::TransMale),
            "any" | "all" => Some(GenderFilter::Any),
            _ => None,
        }
    }

    pub fn display(&self) -> &'static str {
        match self {
            GenderFilter::Female => "Female (biological)",
            GenderFilter::Male => "Male",
            GenderFilter::TransFemale => "Transgender Female",
            GenderFilter::TransMale => "Transgender Male",
            GenderFilter::Any => "Any",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub gender_filter: GenderFilter,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            gender_filter: GenderFilter::Female,
        }
    }
}

impl Config {
    pub fn load() -> Self {
        Self::path()
            .and_then(|p| {
                let data = std::fs::read_to_string(&p).context("read config")?;
                serde_json::from_str(&data).context("parse config")
            })
            .unwrap_or_default()
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
            .join("luminary");
        std::fs::create_dir_all(&dir)?;
        Ok(dir.join("config.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_biological_female() {
        assert_eq!(Config::default().gender_filter, GenderFilter::Female);
    }

    #[test]
    fn female_excludes_transgender_female() {
        let f = GenderFilter::Female;
        assert!(f.matches(Some("Female")));
        assert!(f.matches(Some("female"))); // case-insensitive
        assert!(!f.matches(Some("Transgender Female")));
        assert!(!f.matches(Some("Male")));
        assert!(!f.matches(None)); // unknown gender excluded
    }

    #[test]
    fn trans_female_does_not_match_biological_female() {
        let tf = GenderFilter::TransFemale;
        assert!(tf.matches(Some("Transgender Female")));
        assert!(!tf.matches(Some("Female")));
    }

    #[test]
    fn any_matches_everything() {
        let any = GenderFilter::Any;
        assert!(any.matches(Some("Female")));
        assert!(any.matches(Some("Transgender Female")));
        assert!(any.matches(Some("Male")));
        assert!(any.matches(None));
    }

    #[test]
    fn tpdb_value_uses_titlecase_enum() {
        assert_eq!(GenderFilter::Female.tpdb_value(), Some("Female"));
        assert_eq!(
            GenderFilter::TransFemale.tpdb_value(),
            Some("Transgender Female")
        );
        assert_eq!(GenderFilter::Any.tpdb_value(), None);
    }

    #[test]
    fn from_str_parses_aliases() {
        assert_eq!(GenderFilter::from_str("female"), Some(GenderFilter::Female));
        assert_eq!(GenderFilter::from_str("F"), Some(GenderFilter::Female));
        assert_eq!(
            GenderFilter::from_str("trans-female"),
            Some(GenderFilter::TransFemale)
        );
        assert_eq!(GenderFilter::from_str("any"), Some(GenderFilter::Any));
        assert_eq!(GenderFilter::from_str("nonsense"), None);
    }
}
