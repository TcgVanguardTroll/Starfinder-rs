//! Settings handler: `config`.
use super::*;
pub(crate) fn configure(key: Option<String>, value: Option<String>) -> anyhow::Result<()> {
    use colored::*;
    let mut cfg = config::Config::load();

    match (key.as_deref(), value.as_deref()) {
        (None, _) => {
            // Show current config
            println!("{}", "Luminary Settings".bright_cyan().bold());
            println!("{}", "═".repeat(35).bright_black());
            println!(
                "  {} {}",
                "gender: ".bright_black(),
                cfg.gender_filter.display().bright_white()
            );
            let key_status = if std::env::var("TPDB_API_KEY").is_ok() {
                "set (via TPDB_API_KEY env var)".to_string()
            } else if cfg.api_key.as_deref().is_some_and(|k| !k.is_empty()) {
                "set (stored in config)".to_string()
            } else {
                "not set".to_string()
            };
            println!(
                "  {} {}",
                "api-key:".bright_black(),
                key_status.bright_white()
            );
            let stash_status = if cfg.stashdb_key.as_deref().is_some_and(|k| !k.is_empty()) {
                "set (image enrichment on)"
            } else {
                "not set"
            };
            println!(
                "  {} {}",
                "stashdb-key:".bright_black(),
                stash_status.bright_white()
            );
            println!();
            println!(
                "{}",
                "  gender: female, male, trans-female, trans-male, any".bright_black()
            );
            println!(
                "{}",
                "  api-key <key>: store your ThePornDB API key".bright_black()
            );
            println!(
                "{}",
                "  stashdb-key <key>: store a StashDB key for extra face images".bright_black()
            );
        }
        (Some("api-key"), Some(val)) => {
            cfg.api_key = Some(val.to_string());
            cfg.save()?;
            println!("{} api-key stored", "Updated:".green());
        }
        (Some("stashdb-key"), Some(val)) => {
            cfg.stashdb_key = Some(val.to_string());
            cfg.save()?;
            println!("{} stashdb-key stored", "Updated:".green());
        }
        (Some("gender"), Some(val)) => match config::GenderFilter::from_str(val) {
            Some(filter) => {
                cfg.gender_filter = filter;
                cfg.save()?;
                println!(
                    "{} gender = {}",
                    "Updated:".green(),
                    cfg.gender_filter.display().bright_white()
                );
            }
            None => {
                println!(
                    "{} Unknown value '{}'. Use: female, male, trans-female, trans-male, any",
                    "Error:".red(),
                    val
                );
            }
        },
        (Some(k), _) => {
            println!(
                "{} Unknown setting '{}'. Available: gender, api-key, stashdb-key",
                "Error:".red(),
                k
            );
        }
    }

    Ok(())
}
