# Luminary

> Privacy-first CLI recommendation engine for discovering adult performers you'll enjoy — built in Rust, powered by [ThePornDB](https://theporndb.net) and ArcFace face recognition.

![Rust](https://img.shields.io/badge/Rust-stable-orange?logo=rust)
![License](https://img.shields.io/badge/license-MIT-blue)
![Local First](https://img.shields.io/badge/data-local--first-green)
![ML](https://img.shields.io/badge/face%20ML-ArcFace%20%2B%20InsightFace-purple)

All data — performer profiles, face embeddings, preferences — stays on your machine. No accounts, no telemetry, no cloud.

---

## Features

- **Preference tree** — builds a `body_type → ethnicity → hair → age → eye colour` tree from performers you like, showing percentages at every branch
- **Smart recommendations** — IDF-weighted scoring that emphasises what's *distinctive* about your taste, not just what's common; body type is a hard gate
- **Face similarity** — ArcFace embeddings via InsightFace + ONNX Runtime; `find --looks-like` sorts by actual facial geometry
- **Build similarity** — waist-to-hip ratio (WHR) + k-NN feature vectors find performers with a genuinely similar physique, not just the same cup size
- **Mix-and-match search** — `find --looks-like "A" --body-like "B"` combines one performer's face with another's build
- **Configurable gender filter** — defaults to biological female; supports trans, male, any
- **Fully offline after first fetch** — all data cached in SQLite locally

---

## Requirements

| Dependency | Purpose | Install |
|---|---|---|
| **Rust** (stable) | Build the binary | [rustup.rs](https://rustup.rs) |
| **ThePornDB API key** | Performer data | [theporndb.net](https://theporndb.net) — free |
| **Python 3.9+** | Face embeddings (optional) | [python.org](https://python.org) |
| **InsightFace + ONNX** | ArcFace model (optional) | `pip install insightface onnxruntime` |

Face similarity is optional — all other commands work without Python.

---

## Installation

```powershell
git clone https://github.com/TcgVanguardTroll/Luminary-rs.git
cd Luminary-rs
cargo build --release
```

Binary: `target/release/luminary.exe`

Set your API key (add to your profile to persist):

```powershell
$env:TPDB_API_KEY = "your-key-here"
```

---

## Quick Start

```powershell
# Add performers you like
luminary add "Naughty Alysha" "Seka Black" "Dee Siren" "Lisa Ann"

# See your taste profile
luminary profile

# Get recommendations
luminary recommend

# Find performers with Naughty Alysha's face and Lisa Ann's body
luminary find --looks-like "Naughty Alysha" --body-like "Lisa Ann"
```

---

## Commands

### Managing your library

```powershell
luminary add "Name" ["Name2" ...]   # fetch from ThePornDB + auto-embed if Python available
luminary view "Name"                # show stored profile
luminary list                       # list all performers
luminary remove "Name"              # remove a performer
luminary stats                      # DB size, image cache size
luminary clear-cache                # clear downloaded images
```

### Preference tree

```powershell
luminary profile
```

```
Your Taste Profile
══════════════════════════════════════════
  Based on 8 liked performers

  ├── Curvy 7/8  88%
  │   ├── Caucasian 6/7  86%
  │   │   ├── Blonde 3/6  50%
  │   │   │   └── 46+ 3/3  100%
  │   │   │       ├── Green 1/3  33%
  │   │   │       └── Blue  1/3  33%
  │   │   └── Brunette 2/6  33%
  │   │       └── 46+ 2/2  100%
  ...

  Your type: Curvy → Caucasian → Blonde → 46+
```

The tree drills through **body type → ethnicity → hair → age range → eye colour**. Each level shows counts and percentages. The more performers you add, the more specific it becomes.

### Recommendations

```powershell
# Based on your full preference tree
luminary recommend [--limit 10]

# Performers similar to one specific person (uses ThePornDB /similar API)
luminary similar "Seka Black"
```

`recommend` scores every candidate against your tree. Body type is a **hard exclusion gate** — wrong physique means excluded entirely. Hair and eye colour are small bonuses.

### Advanced search — `find`

Mix attributes from stored performers or set them manually:

```powershell
# Face attributes from one, body/build from another
luminary find --looks-like "Naughty Alysha" --body-like "Dee Siren"
luminary find --looks-like "Naughty Alysha" --body-like "Lisa Ann"

# Find by butt/build shape — waist-to-hip ratio
luminary find --body-like "Dee Siren"      # auto-derives WHR 0.667
luminary find --whr 0.667                   # set the ratio directly

# Manual filters
luminary find --ethnicity Caucasian --hair Blonde --cup DD --age-min 40

# Combine
luminary find --looks-like "Naughty Alysha" --cup DD --age-min 46 --age-max 60
```

**`--looks-like`** copies ethnicity, hair colour, and eye colour.  
**`--body-like`** copies cup size, hips, and waist-to-hip ratio (WHR).

| Flag | Values | Notes |
|------|--------|-------|
| `--ethnicity` | `Caucasian`, `Latin`, `Black`, `Asian`, `Indian` | Title case |
| `--hair` | `Blonde`, `Brunette`, `Black`, `Red`, `Auburn` | Title case |
| `--eye` | `Blue`, `Green`, `Brown`, `Hazel`, `Grey` | Title case |
| `--cup` | `A` `B` `C` `D` `DD` `DDD` | Letter only |
| `--hips` | `36` | Inches, ±4 tolerance |
| `--waist` | `24` | Inches, ±4 tolerance |
| `--whr` | `0.667` | Waist-to-hip ratio, ±0.05 — captures butt/build shape |
| `--age-min` | `40` | |
| `--age-max` | `55` | |
| `--limit` | `10` | Number of results |

**Ranking:** results are sorted by face similarity when `--looks-like` is set (and embeddings exist), otherwise by **body/build similarity** (k-NN over a weighted feature vector) when `--body-like` is set. Each result shows a `face %` and/or `body %` along with its WHR.

```
1. Carina (Curvy, Caucasian, Blonde, 24w 36h whr 0.67)  body 99%
2. Erin   (Curvy, Caucasian, Blonde, 25w 36h whr 0.69)  body 98%
```

### Face similarity (ML)

```powershell
# Install once
pip install insightface onnxruntime

# Generate ArcFace embeddings for all performers in your DB
# Downloads buffalo_l model on first run (~300 MB, cached forever after)
luminary embed
```

Once embeddings exist, `find --looks-like` automatically re-ranks results by **cosine similarity of 512-dim ArcFace vectors** — actual facial geometry, not just hair/ethnicity attributes. New performers added via `luminary add` are auto-embedded.

### Settings

```powershell
luminary config                        # show current settings
luminary config gender female          # biological female (default)
luminary config gender trans-female
luminary config gender male
luminary config gender any
```

---

## How it works

### Preference tree

Every performer you add becomes a data point. The tree aggregates them level by level:

```
body_type → ethnicity → hair_color → age_bucket → eye_color
```

The **dominant path** (highest-count child at each level, confidence ≥ 50%) becomes your "type" and drives recommendation queries.

### Recommendation algorithms

Luminary uses several complementary algorithms depending on the command:

**1. IDF-weighted scoring (`recommend`)**

Rather than fixed attribute weights, `recommend` weights each attribute by how *rare* it is among your liked performers — borrowed from TF-IDF in search engines:

```
idf(value) = ln(total_liked / liked_with_this_value) + 1
```

If every performer you like is 46+, that attribute is uninformative and gets down-weighted. If only a couple have Green eyes, that's a strong distinguishing signal and gets up-weighted. Body type remains a **hard exclusion gate** — wrong physique is removed entirely.

**2. Waist-to-hip ratio, WHR (`find --whr`, `--body-like`)**

Body shape is captured by the ratio `waist ÷ hips`, not just absolute size. A low WHR (~0.67) is the signature of a pronounced hourglass / bubble-butt build. Searching by WHR finds that shape regardless of overall frame size or cup size.

**3. k-NN feature vectors (`find --body-like`)**

Each performer is encoded as a normalised, weighted feature vector:

```
[ inv_whr ×3, hips ×2, cup ×1.5, age ×1, ethnicity ×0.5, hair ×0.3, eye ×0.2 ]
```

WHR and hips carry the most weight, so nearest-neighbour search by Euclidean distance surfaces performers with a genuinely similar build. The result `body %` is derived from this distance.

**4. Face similarity (`find --looks-like`, `similar`)**

512-dim ArcFace embeddings compared by cosine similarity (see below). Re-ranks results on top of the attribute scores when available.

### Face similarity

Uses **InsightFace buffalo_l** (ArcFace R50 backbone) via ONNX Runtime — no TensorFlow, no GPU required, works on Python 3.14+.

```
add performer
  → download face image from ThePornDB
  → InsightFace: detect → align → ArcFace embed → 512-dim vector
  → store in SQLite

find --looks-like "X"
  → load X's 512-vector
  → for each candidate: generate/load their vector
  → cosine similarity → sort → top-k
```

Embeddings are generated once and cached — subsequent searches are instant.

---

## Data & privacy

| What | Where |
|------|-------|
| Performer database | `%LOCALAPPDATA%\luminary\luminary.db` |
| Face embeddings | Stored inside the same SQLite DB |
| Image cache | `%LOCALAPPDATA%\luminary\images\` |
| Settings | `%LOCALAPPDATA%\luminary\config.json` |
| InsightFace model | `~\.insightface\models\buffalo_l\` |

Nothing leaves your machine except outbound API calls to ThePornDB when you explicitly run a command. Face embeddings are biometric data — keeping them local is intentional.

---

## Architecture

Luminary is a **local-first, single-node** application:

- **SQLite** — embedded, zero-infrastructure database
- **ThePornDB REST API** — external data source (performer profiles, similar-performer queries)
- **InsightFace + ONNX Runtime** — in-process face embedding via Python subprocess
- **No server, no sync, no accounts**

The only distributed systems concern is cache staleness — your local performer snapshots drift from ThePornDB over time. Re-adding a performer refreshes their data.

---

## License

MIT
