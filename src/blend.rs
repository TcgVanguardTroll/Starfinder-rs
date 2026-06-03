//! Multi-modal fusion for the `body-search --by blend` lens.
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
    pub meas: Option<f64>,
}

/// Relative importance of each modality in the blend.
pub struct Weights {
    pub face: f64,
    pub build: f64,
    pub volume: f64,
    pub proj: f64,
    pub meas: f64,
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
            meas: 0.10,
        }
    }
}

/// Blends each candidate's per-modality scores into one 0–100 score that is
/// robust to the modalities living on different scales. Each modality is
/// rank-normalised to a 0–1 percentile within the candidates that have it, then
/// combined as a weighted average over only the modalities a candidate actually
/// carries (so a missing modality neither helps nor hurts beyond dropping its
/// weight). Returns one score per input candidate, in input order.
pub fn blend_scores(cands: &[ModalityScores], w: &Weights) -> Vec<f64> {
    let face = percentiles(&cands.iter().map(|c| c.face).collect::<Vec<_>>());
    let build = percentiles(&cands.iter().map(|c| c.build).collect::<Vec<_>>());
    let volume = percentiles(&cands.iter().map(|c| c.volume).collect::<Vec<_>>());
    let proj = percentiles(&cands.iter().map(|c| c.proj).collect::<Vec<_>>());
    let meas = percentiles(&cands.iter().map(|c| c.meas).collect::<Vec<_>>());

    (0..cands.len())
        .map(|i| {
            let parts = [
                (face[i], w.face),
                (build[i], w.build),
                (volume[i], w.volume),
                (proj[i], w.proj),
                (meas[i], w.meas),
            ];
            let mut num = 0.0;
            let mut den = 0.0;
            for (p, weight) in parts {
                if let Some(p) = p {
                    num += weight * p;
                    den += weight;
                }
            }
            if den > 0.0 {
                100.0 * num / den
            } else {
                0.0
            }
        })
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
            meas,
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
    fn missing_modalities_drop_their_weight_without_penalty() {
        // One candidate has only build, the other only measurements; each is the
        // sole holder of its modality (percentile 1.0), so both blend to the same
        // positive score rather than collapsing to zero from absent weights.
        let cands = vec![ms(None, Some(95.0), None), ms(None, None, Some(60.0))];
        let s = blend_scores(&cands, &Weights::default());
        assert!((s[0] - s[1]).abs() < 1e-9);
        assert!(s[0] > 0.0);
    }

    #[test]
    fn empty_pool_is_empty() {
        assert!(blend_scores(&[], &Weights::default()).is_empty());
    }
}
