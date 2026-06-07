//! Multi-modal fusion for the `body-search` `overall` lens (the default).
//!
//! Each search modality (face, build/pose, volume/seg, posterior projection,
//! recorded measurements) scores on its own scale — a face cosine reads ~55–75%
//! for a genuine match while a body-proportion similarity reads ~85–98%, so a
//! naive weighted average of the raw percentages lets the body scale drown the
//! face signal. [`blend_scores`] fixes this by rank-normalising each modality to
//! a percentile within the candidate pool *before* blending, so a modality's
//! influence is set by its weight, not by where its numbers happen to sit.

/// One candidate's raw per-modality similarity scores (each a 0–100% value, or
/// None when that modality is unavailable for the candidate or the reference).
#[derive(Default, Clone)]
pub struct ModalityScores {
    pub face: Option<f64>,
    pub build: Option<f64>,
    pub volume: Option<f64>,
    pub proj: Option<f64>,
    /// Bust shape/projection (chest analog of `proj`). None until a performer is
    /// re-embedded with the bust CV (sidecar work is #16).
    pub bust: Option<f64>,
    pub meas: Option<f64>,
    /// Stature proximity (how close the recorded heights are). A *soft* term, so
    /// a same-proportion-but-taller performer ranks lower rather than being
    /// dropped — the body vectors are scale-free, so without this "build like X"
    /// returns bodies that wouldn't *look* like X. None when either height is
    /// unknown. (The hard `--height-tol` band is a separate, opt-in filter.)
    pub height: Option<f64>,
    /// Build-size proximity (closeness of absolute BMI). The body vectors are
    /// scale-free *ratios*, so they match proportions+cup but miss whether the
    /// frame is fuller or slimmer (a 36-band vs a 32-band at the same hourglass).
    /// This captures the absolute build humans see. None when BMI is unknown.
    pub size: Option<f64>,
}

/// Relative importance of each modality in the blend.
pub struct Weights {
    pub face: f64,
    pub build: f64,
    pub volume: f64,
    pub proj: f64,
    pub bust: f64,
    pub meas: f64,
    pub height: f64,
    pub size: f64,
}

impl Weights {
    /// Body-type-only weights: face excluded (0.0) so ranking is driven purely by
    /// how similar the *body* is — skeletal frame, silhouette fullness, butt
    /// projection, bust, and recorded measurements. For "find an extremely
    /// similar body type" rather than "looks like + built like" (the default).
    pub fn body_only() -> Self {
        Weights {
            face: 0.0,
            build: 0.22,
            volume: 0.22,
            proj: 0.18,
            bust: 0.13,
            meas: 0.18,
            height: 0.12,
            size: 0.13,
        }
    }

    /// "Closest-looking actress" weights: face dominates so the ranking is led by
    /// the *look* (identity/facial geometry), but every body modality stays in —
    /// so it's "looks like her AND built like her", not pure face-search. Stature
    /// is weighted a touch higher too, since two people of very different heights
    /// rarely read as look-alikes. Pair with `--hair`/`--height-tol` to lock the
    /// obvious shared traits. Face ≈ 35% of total weight here vs ≈ 26% in default.
    pub fn lookalike() -> Self {
        Weights {
            face: 0.50,
            build: 0.15,
            volume: 0.15,
            proj: 0.12,
            bust: 0.10,
            meas: 0.10,
            height: 0.16,
            size: 0.12,
        }
    }
}

impl Default for Weights {
    fn default() -> Self {
        // Face (identity) leads; build/volume/projection describe the body;
        // recorded measurements are a weaker, noisier signal. Tunable once a
        // corpus carrying every modality (esp. side projection) actually exists.
        Weights {
            face: 0.35,
            build: 0.20,
            volume: 0.20,
            proj: 0.15,
            // Added on top (weights need not sum to 1): bust is structurally
            // absent until #16, so it changes no current ranking, while reserving
            // a sensible influence (~bust/total) once a bust corpus exists.
            bust: 0.12,
            meas: 0.10,
            // Soft stature term (~10% of the blend): same-proportion-but-taller
            // candidates rank lower instead of vanishing. Hard band = --height-tol.
            height: 0.12,
            // Build-size term: fuller-vs-slimmer frame (absolute BMI) that the
            // scale-free body vectors miss — closes the gap with human perception.
            size: 0.13,
        }
    }
}

/// Blends each candidate's per-modality scores into one 0–100 score that is
/// robust to the modalities living on different scales. Each modality is
/// rank-normalised to a 0–1 percentile within the candidates that have it, then
/// combined as a weighted percentile sum.
///
/// Coverage is rewarded: rather than renormalising by each candidate's own
/// present weight (which gives a lone modality full credit and lets a
/// measurements-only guess outrank a build+volume+measurements match), the
/// weighted sum is divided by the BEST present-weight in the pool. So a
/// candidate missing modalities is scored down relative to the best-covered one,
/// while structurally-absent modalities — ones *no* candidate has, e.g.
/// projection before any side shots are ingested — penalise nobody, since the
/// pool maximum never includes them. A perfectly-matched, best-covered candidate
/// still reaches ~100. Returns one score per input candidate, in input order.
pub fn blend_scores(cands: &[ModalityScores], w: &Weights) -> Vec<f64> {
    let face = percentiles(&cands.iter().map(|c| c.face).collect::<Vec<_>>());
    let build = percentiles(&cands.iter().map(|c| c.build).collect::<Vec<_>>());
    let volume = percentiles(&cands.iter().map(|c| c.volume).collect::<Vec<_>>());
    let proj = percentiles(&cands.iter().map(|c| c.proj).collect::<Vec<_>>());
    let bust = percentiles(&cands.iter().map(|c| c.bust).collect::<Vec<_>>());
    let meas = percentiles(&cands.iter().map(|c| c.meas).collect::<Vec<_>>());
    let height = percentiles(&cands.iter().map(|c| c.height).collect::<Vec<_>>());
    let size = percentiles(&cands.iter().map(|c| c.size).collect::<Vec<_>>());

    // Per candidate: the weighted percentile sum, and the total weight it covers.
    let scored: Vec<(f64, f64)> = (0..cands.len())
        .map(|i| {
            let parts = [
                (face[i], w.face),
                (build[i], w.build),
                (volume[i], w.volume),
                (proj[i], w.proj),
                (bust[i], w.bust),
                (meas[i], w.meas),
                (height[i], w.height),
                (size[i], w.size),
            ];
            let mut num = 0.0;
            let mut covered = 0.0;
            for (p, weight) in parts {
                if let Some(p) = p {
                    num += weight * p;
                    covered += weight;
                }
            }
            (num, covered)
        })
        .collect();

    // Divide by the best coverage in the pool, not each candidate's own — so
    // broader matches outrank single-modality ones.
    let max_covered = scored.iter().map(|(_, c)| *c).fold(0.0_f64, f64::max);
    if max_covered <= 0.0 {
        return vec![0.0; cands.len()];
    }
    scored
        .iter()
        .map(|(num, _)| 100.0 * num / max_covered)
        .collect()
}

/// Maps each present score to its percentile (0–1) among the present values,
/// leaving None as None. Ties share their averaged rank. With fewer than two
/// present values the percentile is degenerate, so a lone present value maps to
/// 1.0 (it is the best — and only — of its modality).
fn percentiles(scores: &[Option<f64>]) -> Vec<Option<f64>> {
    let present: Vec<f64> = scores.iter().filter_map(|s| *s).collect();
    if present.len() < 2 {
        return scores.iter().map(|s| s.map(|_| 1.0)).collect();
    }
    let n = present.len() as f64;
    scores
        .iter()
        .map(|s| {
            s.map(|v| {
                let less = present.iter().filter(|&&x| x < v).count() as f64;
                let equal = present.iter().filter(|&&x| (x - v).abs() < 1e-9).count() as f64;
                // Midpoint of the tied block — a smooth, symmetric percentile.
                (less + 0.5 * equal) / n
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(face: Option<f64>, build: Option<f64>, meas: Option<f64>) -> ModalityScores {
        ModalityScores {
            face,
            build,
            volume: None,
            proj: None,
            bust: None,
            meas,
            height: None,
            size: None,
        }
    }

    #[test]
    fn strong_everywhere_beats_weak_everywhere() {
        let cands = vec![
            ms(Some(70.0), Some(95.0), Some(90.0)),
            ms(Some(55.0), Some(82.0), Some(70.0)),
        ];
        let s = blend_scores(&cands, &Weights::default());
        assert!(s[0] > s[1]);
    }

    #[test]
    fn face_is_not_drowned_by_the_body_scale() {
        // cand0 wins face (the higher-weighted modality) but loses build; cand1 is
        // the reverse. Rank-normalisation + the face weight should tilt to cand0,
        // which a raw average (where build's ~90s dwarf face's ~60s) would not.
        let cands = vec![
            ms(Some(75.0), Some(85.0), None), // best face, weaker build
            ms(Some(55.0), Some(98.0), None), // weak face, best build
        ];
        let s = blend_scores(&cands, &Weights::default());
        assert!(s[0] > s[1]);
    }

    #[test]
    fn coverage_is_rewarded() {
        // A broader match outranks a single-modality one even when the latter
        // tops its modality: build+meas beats a higher meas-only candidate.
        let cands = vec![
            ms(None, Some(80.0), Some(80.0)), // build + measurements
            ms(None, None, Some(99.0)),       // measurements only, highest meas
        ];
        let s = blend_scores(&cands, &Weights::default());
        assert!(s[0] > s[1]);

        // Between two single-modality candidates, the heavier-weighted modality
        // (build 0.20 > meas 0.10) wins — neither gets full credit for a lone hit.
        let solo = vec![ms(None, Some(95.0), None), ms(None, None, Some(95.0))];
        let s2 = blend_scores(&solo, &Weights::default());
        assert!(s2[0] > s2[1]);
        // The best-covered candidate in a pool still reaches ~100.
        assert!((s2[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn body_only_ignores_face() {
        // cand0 has the best face but weaker build; cand1 the reverse. Under
        // body-only weights (face = 0), the better BUILD wins — the opposite of
        // the default blend, where the leading face weight would tilt to cand0.
        let cands = vec![
            ms(Some(95.0), Some(70.0), None), // best face, weaker build
            ms(Some(50.0), Some(98.0), None), // weak face, best build
        ];
        let s = blend_scores(&cands, &Weights::body_only());
        assert!(s[1] > s[0]); // build decides; face ignored
                              // Sanity: the default blend (face leads) tilts the other way here.
        let d = blend_scores(&cands, &Weights::default());
        assert!(d[0] > d[1]);
    }

    #[test]
    fn empty_pool_is_empty() {
        assert!(blend_scores(&[], &Weights::default()).is_empty());
    }
}
