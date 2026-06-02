//! Region / nationality grouping — lets searches match a *group* of
//! nationalities indiscriminately (e.g. "Slavic" = Russian, Polish, Ukrainian…).
//!
//! Matching is exact on the nationality adjective (TPDB stores a clean single
//! word like "Russian") and on the ISO-ish birthplace country code, so there
//! are no substring false positives (e.g. "Peruvian" never matches "ru").

/// (nationality adjectives, 2-letter country codes) for a named region.
fn region_data(region: &str) -> Option<(&'static [&'static str], &'static [&'static str])> {
    let r = region.to_lowercase();
    let data: (&[&str], &[&str]) = match r.as_str() {
        "slavic" => (
            &[
                "russian",
                "polish",
                "belarusian",
                "ukrainian",
                "czech",
                "slovak",
                "slovakian",
                "serbian",
                "croatian",
                "bulgarian",
                "slovenian",
                "slovene",
                "bosnian",
                "macedonian",
                "montenegrin",
                "slovakian",
            ],
            &[
                "ru", "pl", "by", "ua", "cz", "sk", "rs", "hr", "bg", "si", "ba", "mk", "me",
            ],
        ),
        "nordic" | "scandinavian" => (
            &["swedish", "norwegian", "danish", "finnish", "icelandic"],
            &["se", "no", "dk", "fi", "is"],
        ),
        "latina" | "latin-american" | "latin_american" => (
            &[
                "brazilian",
                "colombian",
                "mexican",
                "venezuelan",
                "argentine",
                "argentinian",
                "cuban",
                "peruvian",
                "chilean",
                "dominican",
                "puerto rican",
                "ecuadorian",
                "bolivian",
            ],
            &[
                "br", "co", "mx", "ve", "ar", "cu", "pe", "cl", "do", "ec", "bo",
            ],
        ),
        "asian" | "east-asian" => (
            &[
                "japanese",
                "chinese",
                "korean",
                "thai",
                "filipino",
                "filipina",
                "vietnamese",
                "taiwanese",
                "indonesian",
                "malaysian",
                "singaporean",
            ],
            &["jp", "cn", "kr", "th", "ph", "vn", "tw", "id", "my", "sg"],
        ),
        "western-european" | "western_european" => (
            &[
                "italian",
                "spanish",
                "french",
                "portuguese",
                "german",
                "british",
                "english",
                "irish",
                "dutch",
                "belgian",
                "austrian",
                "swiss",
            ],
            &[
                "it", "es", "fr", "pt", "de", "gb", "uk", "ie", "nl", "be", "at", "ch",
            ],
        ),
        _ => return None,
    };
    Some(data)
}

/// The set of region names callers can use.
pub fn known_regions() -> &'static [&'static str] {
    &["slavic", "nordic", "latina", "asian", "western-european"]
}

/// True if a performer's nationality or birthplace code falls in the region.
pub fn in_region(nationality: Option<&str>, birthplace_code: Option<&str>, region: &str) -> bool {
    let Some((nats, codes)) = region_data(region) else {
        return false;
    };
    if let Some(n) = nationality {
        let n = n.trim().to_lowercase();
        if nats.contains(&n.as_str()) {
            return true;
        }
    }
    if let Some(c) = birthplace_code {
        let c = c.trim().to_lowercase();
        if codes.contains(&c.as_str()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slavic_matches_russian_and_polish() {
        assert!(in_region(Some("Russian"), Some("RU"), "slavic"));
        assert!(in_region(Some("Polish"), None, "slavic"));
        assert!(in_region(Some("Ukrainian"), None, "slavic"));
        assert!(in_region(None, Some("BY"), "slavic")); // Belarus by code only
    }

    #[test]
    fn slavic_excludes_others_and_avoids_substring_traps() {
        assert!(!in_region(Some("Peruvian"), Some("PE"), "slavic")); // not "ru"
        assert!(!in_region(Some("American"), Some("US"), "slavic"));
        assert!(!in_region(Some("French"), Some("FR"), "slavic"));
    }

    #[test]
    fn unknown_region_matches_nothing() {
        assert!(!in_region(Some("Russian"), Some("RU"), "klingon"));
    }
}
