use crate::models::{Performer, PreferenceAnalysis};
use std::collections::HashMap;

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
        if self.parent_count == 0 {
            0.0
        } else {
            self.count as f64 / self.parent_count as f64 * 100.0
        }
    }
}

pub fn age_bucket(age: u32) -> &'static str {
    match age {
        0..=25 => "18-25",
        26..=35 => "26-35",
        36..=45 => "36-45",
        _ => "46+",
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
    if depth >= 5 || performers.is_empty() {
        return vec![];
    }
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
    nodes.sort_by_key(|n| std::cmp::Reverse(n.count));
    nodes
}

fn group_by_attribute(performers: &[Performer], depth: usize) -> HashMap<String, Vec<Performer>> {
    let mut map: HashMap<String, Vec<Performer>> = HashMap::new();
    for p in performers {
        map.entry(attribute_label(p, depth))
            .or_default()
            .push(p.clone());
    }
    map
}

fn attribute_name(depth: usize) -> &'static str {
    match depth {
        0 => "body_type",
        1 => "ethnicity",
        2 => "hair_color",
        3 => "age_range",
        4 => "eye_color",
        _ => "unknown",
    }
}

pub fn print_tree(nodes: &[PreferenceNode], prefix: &str, total: usize) {
    for (i, node) in nodes.iter().enumerate() {
        let is_last = i == nodes.len() - 1;
        let connector = if is_last { "└──" } else { "├──" };
        let child_prefix = if is_last { "    " } else { "│   " };
        println!(
            "{}{} {} {}/{}  {:.0}%",
            prefix,
            connector,
            node.label,
            node.count,
            total,
            node.pct()
        );
        if !node.children.is_empty() {
            print_tree(
                &node.children,
                &format!("{}{}", prefix, child_prefix),
                node.count,
            );
        }
    }
}

/// Renders the preference tree as a Mermaid flowchart (paste into any Mermaid
/// renderer / Markdown that supports it for a visual diagram).
pub fn to_mermaid(nodes: &[PreferenceNode], total: usize) -> String {
    let mut out = String::from("```mermaid\nflowchart TD\n");
    out.push_str(&format!("  root[\"You · {} liked\"]\n", total));
    let mut counter = 0usize;
    for n in nodes {
        emit_mermaid(n, "root", &mut counter, &mut out);
    }
    out.push_str("```\n");
    out
}

fn emit_mermaid(node: &PreferenceNode, parent: &str, counter: &mut usize, out: &mut String) {
    let id = format!("n{}", *counter);
    *counter += 1;
    // Labels are quoted, so spaces / '?' are fine; strip any stray quotes.
    let label = format!(
        "{} · {}/{} · {:.0}%",
        node.label.replace('"', ""),
        node.count,
        node.parent_count,
        node.pct()
    );
    out.push_str(&format!("  {}[\"{}\"]\n", id, label));
    out.push_str(&format!("  {} --> {}\n", parent, id));
    for c in &node.children {
        emit_mermaid(c, &id, counter, out);
    }
}

pub fn dominant_query_path(nodes: &[PreferenceNode]) -> Vec<String> {
    let mut path = vec![];
    let mut current = nodes;
    // Stops naturally when a node has no children (first() returns None).
    while let Some(best) = current.first() {
        if best.pct() < 50.0 && !path.is_empty() {
            break;
        }
        path.push(best.label.clone());
        current = &best.children;
    }
    path
}

// ── Algorithm 1: WHR extraction ───────────────────────────────────────────────

/// Extracts waist-to-hip ratio from measurements string (e.g. "34B-24-36" → 0.667)
pub fn performer_whr(p: &Performer) -> Option<f64> {
    let m = p.measurements.as_deref()?;
    let parts: Vec<&str> = m.split('-').collect();
    if parts.len() < 3 {
        return None;
    }
    let waist: f64 = parts[1].trim().parse().ok()?;
    let hips: f64 = parts[2]
        .trim()
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .ok()?;
    if hips == 0.0 {
        return None;
    }
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
    pub body_type: HashMap<String, f64>,
    pub ethnicity: HashMap<String, f64>,
    pub hair_color: HashMap<String, f64>,
    pub eye_color: HashMap<String, f64>,
    pub age_bucket: HashMap<String, f64>,
}

pub fn compute_idf_weights(performers: &[Performer]) -> IdfWeights {
    let n = performers.len() as f64;

    let idf_map = |vals: Vec<Option<String>>| -> HashMap<String, f64> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for v in vals.into_iter().flatten() {
            *counts.entry(v).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .map(|(k, df)| (k, (n / df as f64).ln() + 1.0))
            .collect()
    };

    IdfWeights {
        body_type: idf_map(
            performers
                .iter()
                .map(|p| Some(p.body_type.clone()))
                .collect(),
        ),
        ethnicity: idf_map(performers.iter().map(|p| p.ethnicity.clone()).collect()),
        hair_color: idf_map(performers.iter().map(|p| p.hair_color.clone()).collect()),
        eye_color: idf_map(performers.iter().map(|p| p.eye_color.clone()).collect()),
        age_bucket: idf_map(
            performers
                .iter()
                .map(|p| p.age.map(|a| age_bucket(a).to_string()))
                .collect(),
        ),
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
    let Some(bt_node) = bt_node else {
        return 0.0;
    };

    let bt_idf = idf
        .body_type
        .get(&performer.body_type)
        .copied()
        .unwrap_or(1.0);
    let mut score = bt_node.pct() / 100.0 * bt_idf * 3.0;

    let eth = performer.ethnicity.as_deref().unwrap_or("Unknown");
    if let Some(eth_node) = bt_node.children.iter().find(|n| n.label == eth) {
        let eth_idf = idf.ethnicity.get(eth).copied().unwrap_or(1.0);
        score += eth_node.pct() / 100.0 * eth_idf * 2.0;

        if let Some(age) = performer.age {
            let bucket = age_bucket(age);
            let age_idf = idf.age_bucket.get(bucket).copied().unwrap_or(1.0);
            let age_node = eth_node
                .children
                .iter()
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
            let eye_node = hair_node
                .children
                .iter()
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
    let s = p
        .measurements
        .as_deref()
        .map(|m| {
            let bust = m.split('-').next().unwrap_or("");
            let cup = bust.trim_start_matches(|c: char| c.is_ascii_digit());
            match cup.to_uppercase().as_str() {
                "AA" | "AAA" => 0,
                "A" => 1,
                "B" => 2,
                "C" => 3,
                "D" => 4,
                "DD" | "E" => 5,
                "DDD" | "F" => 6,
                _ => 2,
            }
        })
        .unwrap_or(2) as f64;
    s / 6.0 // normalise 0–1
}

fn hip_f64(p: &Performer) -> Option<f64> {
    let m = p.measurements.as_deref()?;
    let parts: Vec<&str> = m.split('-').collect();
    let hips: f64 = parts
        .get(2)?
        .trim()
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .ok()?;
    // normalise: typical range 28–48 inches
    Some(((hips - 28.0) / 20.0).clamp(0.0, 1.0))
}

fn age_f64(p: &Performer) -> f64 {
    p.age
        .map(|a| ((a as f64 - 18.0) / 52.0).clamp(0.0, 1.0))
        .unwrap_or(0.5)
}

/// Parses height to centimetres, e.g. "154cm" -> 154.0.
pub fn performer_height_cm(p: &Performer) -> Option<f64> {
    let s = p.height.as_deref()?;
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n: f64 = digits.parse().ok()?;
    if n > 90.0 && n < 230.0 {
        Some(n)
    } else {
        None
    }
}

/// Parses weight to kilograms, e.g. "59kg" -> 59.0, "130 lbs" -> 59.0.
pub fn performer_weight_kg(p: &Performer) -> Option<f64> {
    let s = p.weight.as_deref()?;
    let num: f64 = s
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .parse()
        .ok()?;
    if num <= 0.0 {
        return None;
    }
    if s.to_lowercase().contains("lb") {
        Some(num / 2.205)
    } else {
        Some(num)
    }
}

/// Raw BMI (weight ÷ height²) — the absolute "build size" / fuller-vs-slimmer
/// signal the scale-free body vectors miss (a 36-band fuller frame vs a 32-band
/// slimmer one at the same proportions). None when height or weight is unknown.
pub fn performer_bmi(p: &Performer) -> Option<f64> {
    match (performer_height_cm(p), performer_weight_kg(p)) {
        (Some(h), Some(w)) if h > 0.0 => Some(w / (h / 100.0).powi(2)),
        _ => None,
    }
}

/// Normalised height (0–1 over ~145–185cm); 0.5 (neutral) when unknown.
fn height_f64(p: &Performer) -> f64 {
    performer_height_cm(p)
        .map(|h| ((h - 145.0) / 40.0).clamp(0.0, 1.0))
        .unwrap_or(0.5)
}

/// Normalised weight (0–1 over ~45–95kg); 0.5 (neutral) when unknown.
fn weight_f64(p: &Performer) -> f64 {
    performer_weight_kg(p)
        .map(|w| ((w - 45.0) / 50.0).clamp(0.0, 1.0))
        .unwrap_or(0.5)
}

/// Normalised BMI (0–1 over ~16–32) = weight ÷ height² — the "physics" of build
/// that neither height nor weight captures alone (55kg reads slim at 175cm but
/// curvy at 160cm). 0.5 (neutral) when either height or weight is unknown.
fn bmi_f64(p: &Performer) -> f64 {
    match (performer_height_cm(p), performer_weight_kg(p)) {
        (Some(h), Some(w)) if h > 0.0 => {
            let m = h / 100.0;
            ((w / (m * m) - 16.0) / 16.0).clamp(0.0, 1.0)
        }
        _ => 0.5,
    }
}

fn str_to_id(val: Option<&str>, options: &[&str]) -> f64 {
    let v = val.unwrap_or("Unknown");
    options
        .iter()
        .position(|&o| o == v)
        .map(|i| i as f64 / (options.len() - 1).max(1) as f64)
        .unwrap_or(0.5)
}

fn body_type_id(p: &Performer) -> f64 {
    let order = ["Petite", "Slim", "Average", "Curvy", "Full-Figured", "BBW"];
    order
        .iter()
        .position(|&o| o == p.body_type)
        .map(|i| i as f64 / (order.len() - 1) as f64)
        .unwrap_or(0.5)
}

// ── Clustering: detect taste sub-types ────────────────────────────────────────
//
// A weighted feature vector per performer for k-means. Body type / ethnicity /
// hair dominate (that's how the tree splits), with build + age as finer signal.
// Always returns a vector (neutral defaults for missing data) so every
// performer can be clustered.

const ETHNICITIES: &[&str] = &[
    "Asian",
    "Black",
    "Caucasian",
    "Indian",
    "Latin",
    "Middle Eastern",
    "Mixed",
];
const HAIRS: &[&str] = &[
    "Auburn", "Bald", "Black", "Blonde", "Brunette", "Grey", "Red", "Various", "White",
];
const EYES: &[&str] = &["Blue", "Brown", "Green", "Grey", "Hazel", "Red"];

pub fn cluster_vector(p: &Performer) -> Vec<f32> {
    let whr = performer_whr(p).unwrap_or(0.72);
    let inv_whr = (1.0 - whr.clamp(0.5, 1.0) / 0.5).clamp(0.0, 1.0);
    vec![
        (body_type_id(p) * 3.0) as f32, // primary split
        (str_to_id(p.ethnicity.as_deref(), ETHNICITIES) * 2.0) as f32,
        (str_to_id(p.hair_color.as_deref(), HAIRS) * 1.5) as f32,
        (age_f64(p) * 1.0) as f32,
        (inv_whr * 1.5) as f32,
        (hip_f64(p).unwrap_or(0.4) * 1.0) as f32,
        (cup_score_f64(p) * 1.0) as f32,
        (height_f64(p) * 1.0) as f32,
    ]
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// k-means clustering. Deterministic farthest-point init (reproducible, no RNG).
/// Returns a cluster id per input point.
pub fn kmeans(points: &[Vec<f32>], k: usize) -> Vec<usize> {
    let n = points.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let k = k.min(n);
    let dims = points[0].len();

    // Init: point 0, then repeatedly the point farthest from all chosen centroids.
    let mut centroids: Vec<Vec<f32>> = vec![points[0].clone()];
    while centroids.len() < k {
        let mut best = 0usize;
        let mut best_d = -1.0_f32;
        for (i, p) in points.iter().enumerate() {
            let d = centroids
                .iter()
                .map(|c| dist2(p, c))
                .fold(f32::MAX, f32::min);
            if d > best_d {
                best_d = d;
                best = i;
            }
        }
        centroids.push(points[best].clone());
    }

    let mut assign = vec![0usize; n];
    for _ in 0..50 {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let mut bj = 0usize;
            let mut bd = f32::MAX;
            for (j, c) in centroids.iter().enumerate() {
                let d = dist2(p, c);
                if d < bd {
                    bd = d;
                    bj = j;
                }
            }
            if assign[i] != bj {
                assign[i] = bj;
                changed = true;
            }
        }
        let mut sums = vec![vec![0.0_f32; dims]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            counts[assign[i]] += 1;
            for d in 0..dims {
                sums[assign[i]][d] += p[d];
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dims {
                    centroids[j][d] = sums[j][d] / counts[j] as f32;
                }
            }
        }
        if !changed {
            break;
        }
    }
    assign
}

/// Encodes a performer as a normalised feature vector for k-NN.
/// Returns None if insufficient measurement data.
pub fn feature_vector(p: &Performer) -> Option<FeatureVec> {
    let whr = performer_whr(p)?;
    let hip = hip_f64(p)?;
    let cup = cup_score_f64(p);
    let age = age_f64(p);
    let height = height_f64(p);
    let weight = weight_f64(p);
    let bmi = bmi_f64(p);
    let inv_whr = (1.0 - whr.clamp(0.5, 1.0) / 0.5).clamp(0.0, 1.0);

    // Shared taxonomies — these previously re-declared the lists inline and
    // had drifted from the cluster_vector ones (missing "Various" hair), so a
    // performer's hair id differed between clustering and k-NN.
    let eth = str_to_id(p.ethnicity.as_deref(), ETHNICITIES);
    let hair = str_to_id(p.hair_color.as_deref(), HAIRS);
    let eye = str_to_id(p.eye_color.as_deref(), EYES);

    // Weights: WHR and hips dominate; height/weight capture overall stature
    // (e.g. a "shortstack" — short + curvy — vs the same measurements on a
    // tall frame), so they carry real weight too.
    Some(FeatureVec {
        name: p.name.clone(),
        values: vec![
            inv_whr * 3.0, // WHR × 3 — butt / lower-body shape
            hip * 2.0,     // hip size × 2
            height * 2.0,  // stature × 2 — short vs tall changes the whole build
            cup * 1.5,     // cup × 1.5
            weight * 1.5,  // weight × 1.5 — slight vs heavy frame
            bmi * 2.0,     // BMI × 2 — build (slim↔heavy) from height+weight together
            age * 1.0,     // age
            // appearance dims (lower weight)
            eth * 0.5,
            hair * 0.3,
            eye * 0.2,
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
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Max possible distance given the weight vector
    fn max_distance() -> f64 {
        // sqrt(sum of squared max differences per dimension); order matches feature_vector
        let max_vals = [3.0_f64, 2.0, 2.0, 1.5, 1.5, 2.0, 1.0, 0.5, 0.3, 0.2];
        max_vals.iter().map(|v| v.powi(2)).sum::<f64>().sqrt()
    }

    /// Averages several feature vectors into one (for multi-reference body
    /// matching — "build like the blend of these performers").
    pub fn average(vecs: &[FeatureVec]) -> Option<FeatureVec> {
        let first = vecs.first()?;
        let dims = first.values.len();
        let mut sum = vec![0.0_f64; dims];
        let mut count = 0usize;
        for v in vecs {
            if v.values.len() != dims {
                continue;
            }
            for (s, x) in sum.iter_mut().zip(v.values.iter()) {
                *s += *x;
            }
            count += 1;
        }
        if count == 0 {
            return None;
        }
        for s in sum.iter_mut() {
            *s /= count as f64;
        }
        Some(FeatureVec {
            name: "<blend>".to_string(),
            values: sum,
        })
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
    let Some(bt_node) = bt_node else {
        return 0.0;
    };
    let mut score = bt_node.pct() / 100.0 * 5.0;
    let eth = performer.ethnicity.as_deref().unwrap_or("Unknown");
    let eth_node = bt_node.children.iter().find(|n| n.label == eth);
    if let Some(eth_node) = eth_node {
        score += eth_node.pct() / 100.0 * 3.0;
        if let Some(age) = performer.age {
            let bucket = age_bucket(age);
            let age_node = eth_node
                .children
                .iter()
                .flat_map(|h| h.children.iter())
                .find(|n| n.label == bucket);
            if let Some(age_node) = age_node {
                score += age_node.pct() / 100.0 * 2.0;
            }
        }
        let hair = performer.hair_color.as_deref().unwrap_or("Unknown");
        if let Some(hair_node) = eth_node.children.iter().find(|n| n.label == hair) {
            score += hair_node.pct() / 100.0 * 0.5;
            let eye = performer.eye_color.as_deref().unwrap_or("Unknown");
            let eye_node = hair_node
                .children
                .iter()
                .flat_map(|n| n.children.iter())
                .find(|n| n.label == eye);
            if let Some(eye_node) = eye_node {
                score += eye_node.pct() / 100.0 * 0.3;
            }
        }
    }
    score
}

pub fn score_against(candidate: &Performer, reference: &Performer) -> f64 {
    let mut score = 0.0;
    let mut max = 0.0;

    // A dimension only counts toward the max when it's actually applicable
    // (the reference has the data), so missing data never penalises a match
    // and an identical performer scores a clean 100%.

    // Build / physique — always present
    max += 5.0;
    if candidate.body_type == reference.body_type {
        score += 5.0;
    }

    // Bust: cup size match (only when reference has a cup)
    if let Some(rc) = cup_letter(reference) {
        max += 2.0;
        if let Some(cc) = cup_letter(candidate) {
            if rc == cc {
                score += 2.0;
            } else if cup_rank(&rc).abs_diff(cup_rank(&cc)) == 1 {
                score += 1.0;
            }
        }
    }

    // Natural vs enhanced (only when reference knows)
    if reference.fake_boobs.is_some() {
        max += 1.0;
        if candidate.fake_boobs == reference.fake_boobs {
            score += 1.0;
        }
    }

    // WHR / butt shape (only when reference has measurements)
    if let Some(rw) = performer_whr(reference) {
        max += 2.0;
        if let Some(cw) = performer_whr(candidate) {
            let diff = (rw - cw).abs();
            if diff <= 0.03 {
                score += 2.0;
            } else if diff <= 0.06 {
                score += 1.0;
            }
        }
    }

    // Height / stature (only when reference has it) — within ~5cm = full credit
    if let Some(rh) = performer_height_cm(reference) {
        max += 2.0;
        if let Some(ch) = performer_height_cm(candidate) {
            let diff = (rh - ch).abs();
            if diff <= 3.0 {
                score += 2.0;
            } else if diff <= 7.0 {
                score += 1.0;
            }
        }
    }

    // Weight / frame (only when reference has it) — within ~5kg = full credit
    if let Some(rw) = performer_weight_kg(reference) {
        max += 1.5;
        if let Some(cw) = performer_weight_kg(candidate) {
            let diff = (rw - cw).abs();
            if diff <= 4.0 {
                score += 1.5;
            } else if diff <= 9.0 {
                score += 0.75;
            }
        }
    }

    // Tattoos (only when reference has any)
    if reference.tattoos.is_some() {
        max += 2.0;
        score += tattoo_overlap(reference.tattoos.as_deref(), candidate.tattoos.as_deref()) * 2.0;
    }

    // Demographics
    if reference.ethnicity.is_some() {
        max += 3.0;
        if candidate.ethnicity == reference.ethnicity {
            score += 3.0;
        }
    }
    if reference.age.is_some() {
        max += 2.0;
        if let (Some(ca), Some(ra)) = (candidate.age, reference.age) {
            if age_bucket(ca) == age_bucket(ra) {
                score += 2.0;
            }
        }
    }
    if reference.hair_color.is_some() {
        max += 1.0;
        if candidate.hair_color == reference.hair_color {
            score += 1.0;
        }
    }
    if reference.eye_color.is_some() {
        max += 0.5;
        if candidate.eye_color == reference.eye_color {
            score += 0.5;
        }
    }

    if max == 0.0 {
        return 0.0;
    }
    (score / max * 100.0_f64).round()
}

/// Extracts the cup letter(s) from measurements, e.g. "36DD-27-38" → "DD"
pub fn cup_letter(p: &Performer) -> Option<String> {
    let bust = p.measurements.as_deref()?.split('-').next()?;
    let cup = bust
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .to_uppercase();
    if cup.is_empty() {
        None
    } else {
        Some(cup)
    }
}

fn cup_rank(cup: &str) -> u32 {
    match cup {
        "AA" | "AAA" => 0,
        "A" => 1,
        "B" => 2,
        "C" => 3,
        "D" => 4,
        "DD" | "E" => 5,
        "DDD" | "F" => 6,
        "G" | "H" | "I" | "J" | "K" => 7,
        _ => 2,
    }
}

/// Parses a semicolon-delimited tattoo string into normalised location tokens.
pub fn parse_tattoos(s: Option<&str>) -> Vec<String> {
    s.map(|t| {
        t.split(';')
            .map(|x| x.trim().to_lowercase())
            .filter(|x| !x.is_empty())
            .collect()
    })
    .unwrap_or_default()
}

/// Jaccard similarity (0–1) between two performers' tattoo location sets.
pub fn tattoo_overlap(a: Option<&str>, b: Option<&str>) -> f64 {
    let sa = parse_tattoos(a);
    let sb = parse_tattoos(b);
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let intersection = sa.iter().filter(|x| sb.contains(x)).count();
    let union = sa.len() + sb.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// True if a performer has a tattoo whose location contains the keyword.
pub fn has_tattoo(p: &Performer, keyword: &str) -> bool {
    let kw = keyword.to_lowercase();
    parse_tattoos(p.tattoos.as_deref())
        .iter()
        .any(|t| t.contains(&kw))
}

// ── Flat analysis ─────────────────────────────────────────────────────────────

pub fn analyze_preferences(performers: &[Performer]) -> PreferenceAnalysis {
    let mut body_types: HashMap<String, usize> = HashMap::new();
    let mut ethnicities: HashMap<String, usize> = HashMap::new();
    let mut hair_colors: HashMap<String, usize> = HashMap::new();
    let mut categories: HashMap<String, usize> = HashMap::new();
    let mut ages: Vec<u32> = Vec::new();
    for p in performers {
        if !p.body_type.is_empty() {
            *body_types.entry(p.body_type.clone()).or_insert(0) += 1;
        }
        if let Some(ref e) = p.ethnicity {
            *ethnicities.entry(e.clone()).or_insert(0) += 1;
        }
        if let Some(ref h) = p.hair_color {
            *hair_colors.entry(h.clone()).or_insert(0) += 1;
        }
        for cat in &p.categories {
            *categories.entry(cat.clone()).or_insert(0) += 1;
        }
        if let Some(a) = p.age {
            ages.push(a);
        }
    }
    let sort_map = |map: HashMap<String, usize>| -> Vec<(String, usize)> {
        let mut v: Vec<_> = map.into_iter().collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        v
    };
    let age_range = if ages.is_empty() {
        (0, 0)
    } else {
        (*ages.iter().min().unwrap(), *ages.iter().max().unwrap())
    };
    let average_age = if ages.is_empty() {
        0.0
    } else {
        ages.iter().sum::<u32>() as f64 / ages.len() as f64
    };
    PreferenceAnalysis {
        common_body_types: sort_map(body_types),
        common_ethnicities: sort_map(ethnicities),
        common_hair_colors: sort_map(hair_colors),
        common_categories: sort_map(categories),
        age_range,
        average_age,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a performer with the fields the algorithms care about.
    fn perf(
        name: &str,
        body: &str,
        eth: &str,
        hair: &str,
        eye: &str,
        age: u32,
        measurements: &str,
        tattoos: Option<&str>,
    ) -> Performer {
        let mut p = Performer::new(name.to_string());
        p.body_type = body.to_string();
        p.ethnicity = Some(eth.to_string());
        p.hair_color = Some(hair.to_string());
        p.eye_color = Some(eye.to_string());
        p.age = Some(age);
        p.measurements = Some(measurements.to_string());
        p.tattoos = tattoos.map(|s| s.to_string());
        p
    }

    #[test]
    fn age_buckets() {
        assert_eq!(age_bucket(22), "18-25");
        assert_eq!(age_bucket(30), "26-35");
        assert_eq!(age_bucket(40), "36-45");
        assert_eq!(age_bucket(55), "46+");
    }

    #[test]
    fn kmeans_separates_two_groups() {
        // Two well-separated clusters of body-type vectors.
        let people = [
            perf(
                "a",
                "Curvy",
                "Caucasian",
                "Blonde",
                "Green",
                50,
                "34DD-26-36",
                None,
            ),
            perf(
                "b",
                "Curvy",
                "Caucasian",
                "Blonde",
                "Blue",
                48,
                "34DD-25-36",
                None,
            ),
            perf(
                "c",
                "Full-Figured",
                "Asian",
                "Black",
                "Brown",
                35,
                "38G-30-44",
                None,
            ),
            perf(
                "d",
                "Full-Figured",
                "Asian",
                "Black",
                "Brown",
                36,
                "38G-31-45",
                None,
            ),
        ];
        let vecs: Vec<Vec<f32>> = people.iter().map(cluster_vector).collect();
        let assign = kmeans(&vecs, 2);
        // a,b land together; c,d land together; the two pairs differ.
        assert_eq!(assign[0], assign[1]);
        assert_eq!(assign[2], assign[3]);
        assert_ne!(assign[0], assign[2]);
    }

    #[test]
    fn whr_from_measurements() {
        let dee = perf(
            "Dee",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Brown",
            40,
            "34B-24-36",
            None,
        );
        // 24 / 36 = 0.667
        assert_eq!(performer_whr(&dee), Some(0.667));

        let mut no_meas = Performer::new("x".into());
        no_meas.measurements = None;
        assert_eq!(performer_whr(&no_meas), None);
    }

    #[test]
    fn cup_extraction() {
        let p = perf(
            "a",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Green",
            50,
            "36DD-27-38",
            None,
        );
        assert_eq!(cup_letter(&p), Some("DD".to_string()));
        let b = perf(
            "b",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Green",
            50,
            "34B-24-36",
            None,
        );
        assert_eq!(cup_letter(&b), Some("B".to_string()));
    }

    #[test]
    fn tattoo_parsing_and_overlap() {
        let a = Some("back of neck; Lower back");
        let b = Some("lower back; ankle");
        // shared: "lower back" → 1 of 3 union = 0.333
        let ov = tattoo_overlap(a, b);
        assert!((ov - 0.3333).abs() < 0.01, "overlap was {}", ov);

        // no tattoos → 0
        assert_eq!(tattoo_overlap(None, b), 0.0);
    }

    #[test]
    fn has_tattoo_keyword() {
        let p = perf(
            "a",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Green",
            50,
            "36DD-27-38",
            Some("back of neck; Lower back"),
        );
        assert!(has_tattoo(&p, "lower back"));
        assert!(has_tattoo(&p, "neck"));
        assert!(!has_tattoo(&p, "ankle"));
    }

    #[test]
    fn similar_butt_scores_high() {
        // Two performers with near-identical WHR + cup should score highly.
        let dee = perf(
            "Dee",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Brown",
            40,
            "34B-24-36",
            None,
        );
        let twin = perf(
            "Twin",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Brown",
            41,
            "34B-24-36",
            None,
        );
        let other = perf(
            "Diff",
            "Slim",
            "Asian",
            "Black",
            "Brown",
            22,
            "32A-26-34",
            None,
        );
        assert!(score_against(&twin, &dee) > score_against(&other, &dee));
        assert_eq!(score_against(&dee, &dee), 100.0); // identical = perfect
    }

    #[test]
    fn idf_downweights_universal_attributes() {
        // All curvy → body_type IDF should be the minimum (ln(1)+1 = 1.0)
        let people = vec![
            perf(
                "a",
                "Curvy",
                "Caucasian",
                "Blonde",
                "Green",
                50,
                "36DD-27-38",
                None,
            ),
            perf(
                "b",
                "Curvy",
                "Caucasian",
                "Brunette",
                "Blue",
                50,
                "34C-26-36",
                None,
            ),
            perf(
                "c",
                "Curvy",
                "Latin",
                "Blonde",
                "Brown",
                30,
                "34D-25-36",
                None,
            ),
        ];
        let idf = compute_idf_weights(&people);
        // Curvy appears in all 3 → idf = ln(3/3)+1 = 1.0
        assert!((idf.body_type["Curvy"] - 1.0).abs() < 1e-9);
        // Latin appears in 1/3 → idf = ln(3/1)+1 ≈ 2.0986, higher than Caucasian (2/3)
        assert!(idf.ethnicity["Latin"] > idf.ethnicity["Caucasian"]);
    }

    #[test]
    fn feature_vector_distance_orders_by_build() {
        let dee = perf(
            "Dee",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Brown",
            40,
            "34B-24-36",
            None,
        );
        let near = perf(
            "Near",
            "Curvy",
            "Caucasian",
            "Blonde",
            "Brown",
            40,
            "34B-25-36",
            None,
        );
        let far = perf(
            "Far",
            "Slim",
            "Asian",
            "Black",
            "Brown",
            22,
            "32A-28-32",
            None,
        );
        let (vd, vn, vf) = (
            feature_vector(&dee).unwrap(),
            feature_vector(&near).unwrap(),
            feature_vector(&far).unwrap(),
        );
        assert!(vd.distance(&vn) < vd.distance(&vf));
        assert!(vd.similarity_pct(&vn) > vd.similarity_pct(&vf));
    }

    #[test]
    fn preference_tree_dominant_path() {
        let people = vec![
            perf(
                "a",
                "Curvy",
                "Caucasian",
                "Blonde",
                "Green",
                50,
                "36DD-27-38",
                None,
            ),
            perf(
                "b",
                "Curvy",
                "Caucasian",
                "Blonde",
                "Blue",
                50,
                "34DD-26-36",
                None,
            ),
            perf(
                "c",
                "Curvy",
                "Caucasian",
                "Blonde",
                "Green",
                50,
                "34D-25-36",
                None,
            ),
        ];
        let tree = build_preference_tree(&people);
        let path = dominant_query_path(&tree);
        assert_eq!(path[0], "Curvy");
        assert_eq!(path[1], "Caucasian");
        assert_eq!(path[2], "Blonde");
    }
}
