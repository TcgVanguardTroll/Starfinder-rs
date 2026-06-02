use std::collections::HashMap;
use crate::models::{Performer, PreferenceAnalysis};

// ── Tree ─────────────────────────────────────────────────────────────────────

/// A node in the preference tree.
/// Levels: body_type → ethnicity → hair_color → age_bucket
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

pub fn build_preference_tree(performers: &[Performer]) -> Vec<PreferenceNode> {
    build_level(performers, performers.len(), 0)
}

fn build_level(performers: &[Performer], parent_count: usize, depth: usize) -> Vec<PreferenceNode> {
    if depth >= 4 || performers.is_empty() { return vec![]; }

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
        let key = match depth {
            0 => p.body_type.clone(),
            1 => p.ethnicity.as_deref().unwrap_or("Unknown").to_string(),
            2 => p.hair_color.as_deref().unwrap_or("Unknown").to_string(),
            3 => p.age.map(age_bucket).unwrap_or("Unknown").to_string(),
            _ => unreachable!(),
        };
        map.entry(key).or_default().push(p.clone());
    }
    map
}

fn attribute_name(depth: usize) -> &'static str {
    match depth {
        0 => "body_type",
        1 => "ethnicity",
        2 => "hair_color",
        3 => "age_range",
        _ => "unknown",
    }
}

/// Prints the tree with ASCII connectors and percentages.
pub fn print_tree(nodes: &[PreferenceNode], prefix: &str, total: usize) {
    for (i, node) in nodes.iter().enumerate() {
        let is_last = i == nodes.len() - 1;
        let connector    = if is_last { "└──" } else { "├──" };
        let child_prefix = if is_last { "    " } else { "│   " };

        println!("{}{} {} {}/{}  {:.0}%",
            prefix, connector, node.label, node.count, total, node.pct());

        if !node.children.is_empty() {
            print_tree(&node.children, &format!("{}{}", prefix, child_prefix), node.count);
        }
    }
}

/// Follows the highest-count child at each level; stops when confidence < 50%.
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

// ── Scoring ───────────────────────────────────────────────────────────────────

/// Score a candidate performer against the preference tree.
/// Physique (body type) is the primary signal.
/// Appearance (hair, eye colour) is a small bonus — already captured by the tree.
pub fn score_performer(performer: &Performer, tree: &[PreferenceNode]) -> f64 {
    let mut score = 0.0;

    // Body type — PRIMARY (weight 5). No match = still eligible but low score.
    if let Some(bt) = tree.iter().find(|n| n.label == performer.body_type) {
        score += bt.pct() / 100.0 * 5.0;
    }

    // Ethnicity (weight 2)
    let eth = performer.ethnicity.as_deref().unwrap_or("Unknown");
    let eth_node = tree.iter()
        .flat_map(|bt| bt.children.iter())
        .find(|n| n.label == eth);

    let Some(eth_node) = eth_node else {
        return score;
    };
    score += eth_node.pct() / 100.0 * 2.0;

    // Age bucket (weight 2) — physique changes with age
    if let Some(age) = performer.age {
        let bucket = age_bucket(age);
        let age_node = eth_node.children.iter()
            .flat_map(|hair| hair.children.iter())
            .find(|n| n.label == bucket);
        if let Some(age_node) = age_node {
            score += age_node.pct() / 100.0 * 2.0;
        }
    }

    // Hair colour — low bonus (weight 1)
    let hair = performer.hair_color.as_deref().unwrap_or("Unknown");
    if let Some(hair_node) = eth_node.children.iter().find(|n| n.label == hair) {
        score += hair_node.pct() / 100.0 * 1.0;
    }

    score
}

// ── Flat analysis (for stats) ─────────────────────────────────────────────────

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
        if let Some(ref eth) = p.ethnicity {
            *ethnicities.entry(eth.clone()).or_insert(0) += 1;
        }
        if let Some(ref hair) = p.hair_color {
            *hair_colors.entry(hair.clone()).or_insert(0) += 1;
        }
        for cat in &p.categories {
            *categories.entry(cat.clone()).or_insert(0) += 1;
        }
        if let Some(age) = p.age { ages.push(age); }
    }

    let sort_map = |map: HashMap<String, usize>| -> Vec<(String, usize)> {
        let mut vec: Vec<_> = map.into_iter().collect();
        vec.sort_by(|a, b| b.1.cmp(&a.1));
        vec
    };

    let age_range = if ages.is_empty() { (0, 0) } else {
        (*ages.iter().min().unwrap(), *ages.iter().max().unwrap())
    };
    let average_age = if ages.is_empty() { 0.0 } else {
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
