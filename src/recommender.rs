use std::collections::HashMap;
use crate::models::{Performer, PreferenceAnalysis};

// ── Tree ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PreferenceNode {
    pub label: String,
    pub attribute: String,
    pub count: usize,
    pub parent_count: usize,
    pub children: Vec<PreferenceNode>,
}

impl PreferenceNode {
    pub fn pct(&self) -> f64 {
        if self.parent_count == 0 { 0.0 }
        else { self.count as f64 / self.parent_count as f64 * 100.0 }
    }
}

pub fn age_bucket(age: u32) -> &'static str {
    match age {
        0..=25  => "18-25",
        26..=35 => "26-35",
        36..=45 => "36-45",
        _       => "46+",
    }
}

fn attribute_label(p: &Performer, depth: usize) -> String {
    match depth {
        0 => p.body_type.clone(),
        1 => p.ethnicity.as_deref().unwrap_or("Unknown").to_string(),
        2 => p.hair_color.as_deref().unwrap_or("Unknown").to_string(),
        3 => p.age.map(age_bucket).unwrap_or("Unknown").to_string(),
        4 => p.eye_color.as_deref().unwrap_or("Unknown").to_string(),
        _ => unreachable!(),
    }
}

pub fn build_preference_tree(performers: &[Performer]) -> Vec<PreferenceNode> {
    build_level(performers, performers.len(), 0)
}

fn build_level(performers: &[Performer], parent_count: usize, depth: usize) -> Vec<PreferenceNode> {
    if depth >= 5 || performers.is_empty() { return vec![]; }
    let mut groups = group_by_attribute(performers, depth);
    let mut nodes: Vec<PreferenceNode> = groups
        .drain()
        .map(|(label, group)| {
            let count = group.len();
            PreferenceNode {
                label: label.clone(),
                attribute: attribute_name(depth).to_string(),
                count,
                parent_count,
                children: build_level(&group, count, depth + 1),
            }
        })
        .collect();
    nodes.sort_by(|a, b| b.count.cmp(&a.count));
    nodes
}

fn group_by_attribute(performers: &[Performer], depth: usize) -> HashMap<String, Vec<Performer>> {
    let mut map: HashMap<String, Vec<Performer>> = HashMap::new();
    for p in performers {
        map.entry(attribute_label(p, depth)).or_default().push(p.clone());
    }
    map
}

fn attribute_name(depth: usize) -> &'static str {
    match depth {
        0 => "body_type", 1 => "ethnicity", 2 => "hair_color",
        3 => "age_range",  4 => "eye_color",  _ => "unknown",
    }
}

pub fn print_tree(nodes: &[PreferenceNode], prefix: &str, total: usize) {
    for (i, node) in nodes.iter().enumerate() {
        let is_last      = i == nodes.len() - 1;
        let connector    = if is_last { "└──" } else { "├──" };
        let child_prefix = if is_last { "    " } else { "│   " };
        println!("{}{} {} {}/{}  {:.0}%",
            prefix, connector, node.label, node.count, total, node.pct());
        if !node.children.is_empty() {
            print_tree(&node.children, &format!("{}{}", prefix, child_prefix), node.count);
        }
    }
}

pub fn dominant_query_path(nodes: &[PreferenceNode]) -> Vec<String> {
    let mut path = vec![];
    let mut current = nodes;
    loop {
        let Some(best) = current.first() else { break };
        if best.pct() < 50.0 && !path.is_empty() { break; }
        path.push(best.label.clone());
        current = &best.children;
        if current.is_empty() { break; }
    }
    path
}

// ── Algorithm 1: WHR extraction ───────────────────────────────────────────────

/// Extracts waist-to-hip ratio from measurements string (e.g. "34B-24-36" → 0.667)
pub fn performer_whr(p: &Performer) -> Option<f64> {
    let m = p.measurements.as_deref()?;
    let parts: Vec<&str> = m.split('-').collect();
    if parts.len() < 3 { return None; }
    let waist: f64 = parts[1].trim().parse().ok()?;
    let hips: f64 = parts[2].trim()
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse().ok()?;
    if hips == 0.0 { return None; }
    Some((waist / hips * 1000.0).round() / 1000.0)
}

// ── Algorithm 2: IDF-weighted scoring ─────────────────────────────────────────
//
// Attributes that ALL your liked performers share carry less information
// (age 46+ at 100% doesn't help distinguish good recs from bad ones).
// Rare attributes are more diagnostic — weight them higher.
//
// Formula: idf(v) = ln(N / df(v)) + 1
//   where N = total liked performers, df(v) = performers with this value

#[derive(Debug, Clone)]
pub struct IdfWeights {
    pub body_type:  HashMap<String, f64>,
    pub ethnicity:  HashMap<String, f64>,
    pub hair_color: HashMap<String, f64>,
    pub eye_color:  HashMap<String, f64>,
    pub age_bucket: HashMap<String, f64>,
}

pub fn compute_idf_weights(performers: &[Performer]) -> IdfWeights {
    let n = performers.len() as f64;

    let idf_map = |vals: Vec<Option<String>>| -> HashMap<String, f64> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for v in vals.into_iter().flatten() {
            *counts.entry(v).or_insert(0) += 1;
        }
        counts.into_iter()
            .map(|(k, df)| (k, (n / df as f64).ln() + 1.0))
            .collect()
    };

    IdfWeights {
        body_type:  idf_map(performers.iter().map(|p| Some(p.body_type.clone())).collect()),
        ethnicity:  idf_map(performers.iter().map(|p| p.ethnicity.clone()).collect()),
        hair_color: idf_map(performers.iter().map(|p| p.hair_color.clone()).collect()),
        eye_color:  idf_map(performers.iter().map(|p| p.eye_color.clone()).collect()),
        age_bucket: idf_map(performers.iter().map(|p| {
            p.age.map(|a| age_bucket(a).to_string())
        }).collect()),
    }
}

/// Score using IDF weights — rare matching attributes score higher than common ones.
/// Body type remains a hard gate.
pub fn score_performer_idf(
    performer: &Performer,
    tree: &[PreferenceNode],
    idf: &IdfWeights,
) -> f64 {
    // Hard gate: body type must match
    let bt_node = tree.iter().find(|n| n.label == performer.body_type);
    let Some(bt_node) = bt_node else { return 0.0; };

    let bt_idf = idf.body_type.get(&performer.body_type).copied().unwrap_or(1.0);
    let mut score = bt_node.pct() / 100.0 * bt_idf * 3.0;

    let eth = performer.ethnicity.as_deref().unwrap_or("Unknown");
    if let Some(eth_node) = bt_node.children.iter().find(|n| n.label == eth) {
        let eth_idf = idf.ethnicity.get(eth).copied().unwrap_or(1.0);
        score += eth_node.pct() / 100.0 * eth_idf * 2.0;

        if let Some(age) = performer.age {
            let bucket = age_bucket(age);
            let age_idf = idf.age_bucket.get(bucket).copied().unwrap_or(1.0);
            let age_node = eth_node.children.iter()
                .flat_map(|h| h.children.iter())
                .find(|n| n.label == bucket);
            if let Some(age_node) = age_node {
                score += age_node.pct() / 100.0 * age_idf * 1.5;
            }
        }

        let hair = performer.hair_color.as_deref().unwrap_or("Unknown");
        if let Some(hair_node) = eth_node.children.iter().find(|n| n.label == hair) {
            let hair_idf = idf.hair_color.get(hair).copied().unwrap_or(1.0);
            score += hair_node.pct() / 100.0 * hair_idf * 0.5;

            let eye = performer.eye_color.as_deref().unwrap_or("Unknown");
            let eye_idf = idf.eye_color.get(eye).copied().unwrap_or(1.0);
            let eye_node = hair_node.children.iter()
                .flat_map(|n| n.children.iter())
                .find(|n| n.label == eye);
            if let Some(eye_node) = eye_node {
                score += eye_node.pct() / 100.0 * eye_idf * 0.3;
            }
        }
    }

    score
}

// ── Algorithm 3: k-NN feature vectors ─────────────────────────────────────────
//
// Encode each performer as a normalised numeric vector:
//   [cup_score, inverse_whr, hip_size, age, ethnicity_id, hair_id, eye_id]
//
// WHR is inverted so lower WHR (more pronounced hips) = higher value.
// Euclidean distance in this space = physical similarity.

fn cup_score_f64(p: &Performer) -> f64 {
    let s = p.measurements.as_deref()
        .map(|m| {
            let bust = m.split('-').next().unwrap_or("");
            let cup = bust.trim_start_matches(|c: char| c.is_ascii_digit());
            match cup.to_uppercase().as_str() {
                "AA" | "AAA" => 0, "A" => 1, "B" => 2, "C" => 3,
                "D" => 4, "DD" | "E" => 5, "DDD" | "F" => 6, _ => 2,
            }
        })
        .unwrap_or(2) as f64;
    s / 6.0  // normalise 0–1
}

fn hip_f64(p: &Performer) -> Option<f64> {
    let m = p.measurements.as_deref()?;
    let parts: Vec<&str> = m.split('-').collect();
    let hips: f64 = parts.get(2)?
        .trim().trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse().ok()?;
    // normalise: typical range 28–48 inches
    Some(((hips - 28.0) / 20.0).clamp(0.0, 1.0))
}

fn age_f64(p: &Performer) -> f64 {
    p.age.map(|a| ((a as f64 - 18.0) / 52.0).clamp(0.0, 1.0)).unwrap_or(0.5)
}

fn str_to_id(val: Option<&str>, options: &[&str]) -> f64 {
    let v = val.unwrap_or("Unknown");
    options.iter().position(|&o| o == v)
        .map(|i| i as f64 / (options.len() - 1).max(1) as f64)
        .unwrap_or(0.5)
}

/// Encodes a performer as a normalised feature vector for k-NN.
/// Returns None if insufficient measurement data.
pub fn feature_vector(p: &Performer) -> Option<FeatureVec> {
    let whr = performer_whr(p)?;
    let hip = hip_f64(p)?;
    let cup = cup_score_f64(p);
    let age = age_f64(p);
    let inv_whr = (1.0 - whr.clamp(0.5, 1.0) / 0.5).clamp(0.0, 1.0);

    let eth = str_to_id(p.ethnicity.as_deref(),
        &["Asian","Black","Caucasian","Indian","Latin","Middle Eastern","Mixed"]);
    let hair = str_to_id(p.hair_color.as_deref(),
        &["Auburn","Bald","Black","Blonde","Brunette","Grey","Red","White"]);
    let eye = str_to_id(p.eye_color.as_deref(),
        &["Blue","Brown","Green","Grey","Hazel","Red"]);

    // Weights: WHR and hips most important for "similar butt/build"
    Some(FeatureVec {
        name: p.name.clone(),
        // physique dims (higher weight via repetition)
        values: vec![
            inv_whr * 3.0,  // WHR × 3 — most important for butt/lower body shape
            hip    * 2.0,   // hip size × 2
            cup    * 1.5,   // cup × 1.5
            age    * 1.0,   // age
            // appearance dims (lower weight)
            eth    * 0.5,
            hair   * 0.3,
            eye    * 0.2,
        ],
    })
}

#[derive(Debug, Clone)]
pub struct FeatureVec {
    pub name: String,
    pub values: Vec<f64>,
}

impl FeatureVec {
    pub fn distance(&self, other: &FeatureVec) -> f64 {
        self.values.iter().zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Max possible distance given the weight vector
    fn max_distance() -> f64 {
        // sqrt(sum of squared max differences per dimension)
        let max_vals = [3.0_f64, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2];
        max_vals.iter().map(|v| v.powi(2)).sum::<f64>().sqrt()
    }

    /// Convert distance to a 0–100% similarity
    pub fn similarity_pct(&self, other: &FeatureVec) -> f64 {
        let d = self.distance(other);
        let sim = 1.0 - (d / Self::max_distance()).clamp(0.0, 1.0);
        (sim * 100.0).round()
    }
}

// ── Original fixed-weight scorer (kept for backward compat) ───────────────────

pub fn score_performer(performer: &Performer, tree: &[PreferenceNode]) -> f64 {
    let bt_node = tree.iter().find(|n| n.label == performer.body_type);
    let Some(bt_node) = bt_node else { return 0.0; };
    let mut score = bt_node.pct() / 100.0 * 5.0;
    let eth = performer.ethnicity.as_deref().unwrap_or("Unknown");
    let eth_node = bt_node.children.iter().find(|n| n.label == eth);
    if let Some(eth_node) = eth_node {
        score += eth_node.pct() / 100.0 * 3.0;
        if let Some(age) = performer.age {
            let bucket = age_bucket(age);
            let age_node = eth_node.children.iter()
                .flat_map(|h| h.children.iter())
                .find(|n| n.label == bucket);
            if let Some(age_node) = age_node { score += age_node.pct() / 100.0 * 2.0; }
        }
        let hair = performer.hair_color.as_deref().unwrap_or("Unknown");
        if let Some(hair_node) = eth_node.children.iter().find(|n| n.label == hair) {
            score += hair_node.pct() / 100.0 * 0.5;
            let eye = performer.eye_color.as_deref().unwrap_or("Unknown");
            let eye_node = hair_node.children.iter()
                .flat_map(|n| n.children.iter())
                .find(|n| n.label == eye);
            if let Some(eye_node) = eye_node { score += eye_node.pct() / 100.0 * 0.3; }
        }
    }
    score
}

pub fn score_against(candidate: &Performer, reference: &Performer) -> f64 {
    let mut score = 0.0;
    let mut max   = 0.0;
    max += 5.0; if candidate.body_type == reference.body_type { score += 5.0; }
    max += 3.0; if candidate.ethnicity == reference.ethnicity  { score += 3.0; }
    max += 2.0;
    if let (Some(ca), Some(ra)) = (candidate.age, reference.age) {
        if age_bucket(ca) == age_bucket(ra) { score += 2.0; }
    }
    max += 1.0; if candidate.hair_color == reference.hair_color { score += 1.0; }
    max += 0.5; if candidate.eye_color  == reference.eye_color  { score += 0.5; }
    if max == 0.0 { return 0.0; }
    (score / max * 100.0_f64).round()
}

// ── Flat analysis ─────────────────────────────────────────────────────────────

pub fn analyze_preferences(performers: &[Performer]) -> PreferenceAnalysis {
    let mut body_types: HashMap<String, usize> = HashMap::new();
    let mut ethnicities: HashMap<String, usize> = HashMap::new();
    let mut hair_colors: HashMap<String, usize> = HashMap::new();
    let mut categories: HashMap<String, usize> = HashMap::new();
    let mut ages: Vec<u32> = Vec::new();
    for p in performers {
        if !p.body_type.is_empty() { *body_types.entry(p.body_type.clone()).or_insert(0) += 1; }
        if let Some(ref e) = p.ethnicity  { *ethnicities.entry(e.clone()).or_insert(0) += 1; }
        if let Some(ref h) = p.hair_color { *hair_colors.entry(h.clone()).or_insert(0) += 1; }
        for cat in &p.categories { *categories.entry(cat.clone()).or_insert(0) += 1; }
        if let Some(a) = p.age { ages.push(a); }
    }
    let sort_map = |map: HashMap<String, usize>| -> Vec<(String, usize)> {
        let mut v: Vec<_> = map.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    };
    let age_range    = if ages.is_empty() { (0, 0) } else { (*ages.iter().min().unwrap(), *ages.iter().max().unwrap()) };
    let average_age  = if ages.is_empty() { 0.0 } else { ages.iter().sum::<u32>() as f64 / ages.len() as f64 };
    PreferenceAnalysis {
        common_body_types: sort_map(body_types),
        common_ethnicities: sort_map(ethnicities),
        common_hair_colors: sort_map(hair_colors),
        common_categories: sort_map(categories),
        age_range,
        average_age,
    }
}
