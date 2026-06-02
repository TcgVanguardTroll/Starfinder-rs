use serde::{Deserialize, Serialize};

/// Represents a performer/content item in the recommendation system
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Performer {
    pub id: Option<i64>,
    pub name: String,
    pub body_type: String,
    pub measurements: Option<String>,
    pub height: Option<String>,
    pub weight: Option<String>,
    pub ethnicity: Option<String>,
    pub nationality: Option<String>,
    pub birthplace_code: Option<String>,
    pub hair_color: Option<String>,
    pub eye_color: Option<String>,
    pub age: Option<u32>,
    pub birthdate: Option<String>,
    pub categories: Vec<String>,
    pub active_years: Option<String>,
    // Images
    pub profile_image_url: Option<String>,
    pub face_url: Option<String>,
    pub gallery_urls: Vec<String>,
    pub gender: Option<String>,
    pub tpdb_id: Option<i64>,
    pub tattoos: Option<String>,
    pub piercings: Option<String>,
    pub fake_boobs: Option<bool>,
    // Metadata
    pub source: Option<String>,
    pub source_url: Option<String>,
    pub last_updated: Option<String>,
    // Recommendation score (calculated, not stored)
    #[serde(skip)]
    pub match_score: f64,
}

impl Performer {
    pub fn new(name: String) -> Self {
        Performer {
            id: None,
            name,
            body_type: String::new(),
            measurements: None,
            height: None,
            weight: None,
            ethnicity: None,
            nationality: None,
            birthplace_code: None,
            hair_color: None,
            eye_color: None,
            age: None,
            birthdate: None,
            categories: Vec::new(),
            active_years: None,
            profile_image_url: None,
            face_url: None,
            gallery_urls: Vec::new(),
            gender: None,
            tpdb_id: None,
            tattoos: None,
            piercings: None,
            fake_boobs: None,
            source: None,
            source_url: None,
            last_updated: None,
            match_score: 0.0,
        }
    }
}

/// User's preference profile
#[derive(Debug, Serialize, Deserialize)]
pub struct UserProfile {
    pub liked_performers: Vec<String>,
    pub preferences: PreferenceAnalysis,
}

/// Analysis of user preferences
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreferenceAnalysis {
    pub common_body_types: Vec<(String, usize)>,
    pub common_ethnicities: Vec<(String, usize)>,
    pub common_hair_colors: Vec<(String, usize)>,
    pub common_categories: Vec<(String, usize)>,
    pub age_range: (u32, u32),
    pub average_age: f64,
}

impl Default for PreferenceAnalysis {
    fn default() -> Self {
        PreferenceAnalysis {
            common_body_types: Vec::new(),
            common_ethnicities: Vec::new(),
            common_hair_colors: Vec::new(),
            common_categories: Vec::new(),
            age_range: (0, 0),
            average_age: 0.0,
        }
    }
}

/// Search filters
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub body_type: Option<String>,
    pub age_min: Option<u32>,
    pub age_max: Option<u32>,
    pub ethnicity: Option<String>,
    pub hair_color: Option<String>,
    pub categories: Vec<String>,
    pub min_score: Option<f64>,
}
