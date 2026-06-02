use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use crate::models::{Performer, SearchFilters};

/// Database manager for storing performers
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Creates a new database connection
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)
            .context("Failed to open database")?;
        let db = Database { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Initializes the database schema
    fn init_schema(&self) -> Result<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS performers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                body_type TEXT,
                measurements TEXT,
                height TEXT,
                weight TEXT,
                ethnicity TEXT,
                hair_color TEXT,
                eye_color TEXT,
                age INTEGER,
                birthdate TEXT,
                categories TEXT,
                active_years TEXT,
                profile_image_url TEXT,
                gallery_urls TEXT,
                source TEXT,
                source_url TEXT,
                last_updated TEXT,
                data TEXT
            )",
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_body_type ON performers(body_type)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_age ON performers(age)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ethnicity ON performers(ethnicity)",
            [],
        )?;

        Ok(())
    }

    /// Adds a performer to the database
    pub fn add_performer(&self, performer: &Performer) -> Result<i64> {
        let categories_json = serde_json::to_string(&performer.categories)?;
        let gallery_urls_json = serde_json::to_string(&performer.gallery_urls)?;
        let data_json = serde_json::to_string(performer)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO performers (
                name, body_type, measurements, height, weight,
                ethnicity, hair_color, eye_color, age, birthdate,
                categories, active_years, profile_image_url, gallery_urls,
                source, source_url, last_updated, data
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                performer.name,
                performer.body_type,
                performer.measurements,
                performer.height,
                performer.weight,
                performer.ethnicity,
                performer.hair_color,
                performer.eye_color,
                performer.age,
                performer.birthdate,
                categories_json,
                performer.active_years,
                performer.profile_image_url,
                gallery_urls_json,
                performer.source,
                performer.source_url,
                performer.last_updated,
                data_json,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Gets a performer by name, falling back to word-order-insensitive match
    pub fn get_performer(&self, name: &str) -> Result<Option<Performer>> {
        // Exact match first
        let mut stmt = self.conn.prepare(
            "SELECT data FROM performers WHERE name = ?1"
        )?;
        let mut rows = stmt.query(params![name])?;
        if let Some(row) = rows.next()? {
            let data: String = row.get(0)?;
            return Ok(Some(serde_json::from_str(&data)?));
        }

        // Fallback: all words present in name (any order, case-insensitive)
        let words: Vec<String> = name.split_whitespace()
            .map(|w| format!("%{}%", w.to_lowercase()))
            .collect();

        if words.is_empty() {
            return Ok(None);
        }

        let conditions = words.iter()
            .enumerate()
            .map(|(i, _)| format!("LOWER(name) LIKE ?{}", i + 1))
            .collect::<Vec<_>>()
            .join(" AND ");

        let query = format!("SELECT data FROM performers WHERE {}", conditions);
        let mut stmt = self.conn.prepare(&query)?;

        let param_refs: Vec<&dyn rusqlite::ToSql> = words.iter()
            .map(|w| w as &dyn rusqlite::ToSql)
            .collect();

        let mut rows = stmt.query(param_refs.as_slice())?;
        if let Some(row) = rows.next()? {
            let data: String = row.get(0)?;
            return Ok(Some(serde_json::from_str(&data)?));
        }

        Ok(None)
    }

    /// Gets all performers
    pub fn get_all_performers(&self) -> Result<Vec<Performer>> {
        let mut stmt = self.conn.prepare("SELECT data FROM performers")?;
        let performers = stmt
            .query_map([], |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str(&data).unwrap())
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(performers)
    }

    /// Searches for performers matching filters
    pub fn search(&self, filters: &SearchFilters) -> Result<Vec<Performer>> {
        let mut query = String::from("SELECT data FROM performers WHERE 1=1");
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(body_type) = &filters.body_type {
            query.push_str(" AND body_type = ?");
            params.push(Box::new(body_type.clone()));
        }
        if let Some(ethnicity) = &filters.ethnicity {
            query.push_str(" AND ethnicity = ?");
            params.push(Box::new(ethnicity.clone()));
        }
        if let Some(age_min) = filters.age_min {
            query.push_str(" AND age >= ?");
            params.push(Box::new(age_min));
        }
        if let Some(age_max) = filters.age_max {
            query.push_str(" AND age <= ?");
            params.push(Box::new(age_max));
        }
        if let Some(hair_color) = &filters.hair_color {
            query.push_str(" AND hair_color = ?");
            params.push(Box::new(hair_color.clone()));
        }

        let mut stmt = self.conn.prepare(&query)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = params
            .iter()
            .map(|p| p.as_ref() as &dyn rusqlite::ToSql)
            .collect();

        let performers = stmt
            .query_map(param_refs.as_slice(), |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str(&data).unwrap())
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(performers)
    }

    /// Removes a performer by name
    pub fn remove_performer(&self, name: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM performers WHERE name = ?1",
            params![name],
        )?;
        Ok(())
    }

    /// Gets the count of performers
    pub fn count(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM performers",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }
}
