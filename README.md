# Starfinder

> Privacy-first CLI recommendation engine for discovering adult performers you'll enjoy — built in Rust, powered by [ThePornDB](https://theporndb.net) and ArcFace face recognition.

![Rust](https://img.shields.io/badge/Rust-stable-orange?logo=rust)
![License](https://img.shields.io/badge/license-MIT-blue)
![Local First](https://img.shields.io/badge/data-local--first-green)
![ML](https://img.shields.io/badge/face%20ML-ArcFace%20%2B%20InsightFace-purple)

All data — performer profiles, face embeddings, preferences — stays on your machine. No accounts, no telemetry, no cloud.

---

## Features

- **Preference tree** — builds a `body_type → ethnicity → hair → age → eye colour` tree from performers you like, showing percentages at every branch
- **Smart recommendations** — uses ThePornDB's similarity engine seeded with your liked performers, then scores results against your tree (body type is a hard gate)
- **Face similarity** — ArcFace embeddings via InsightFace + ONNX Runtime; `find --looks-like` sorts by actual facial geometry
- **Mix-and-match search** — `find --looks-like "A" --body-like "B"` combines face attributes from one performer with body measurements from another
- **Body-shape search** — waist and hip measurements queried server-side; tolerance filtering client-side
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
git clone https://github.com/TcgVanguardTroll/Starfinder-rs.git
cd Starfinder-rs
cargo build --release
```

Binary: `target/release/starfinder.exe`

Set your API key (add to your profile to persist):

```powershell
$env:TPDB_API_KEY = "your-key-here"
```

---

## Quick Start

```powershell
# Add performers you like
starfinder add "Naughty Alysha" "Seka Black" "Dee Siren" "Lisa Ann"

# See your taste profile
starfinder profile

# Get recommendations
starfinder recommend

# Find performers with Naughty Alysha's face and Lisa Ann's body
starfinder find --looks-like "Naughty Alysha" --body-like "Lisa Ann"
```

---

## Commands

### Managing your library

```powershell
starfinder add "Name" ["Name2" ...]   # fetch from ThePornDB + auto-embed if Python available
starfinder view "Name"                # show stored profile
starfinder list                       # list all performers
starfinder remove "Name"              # remove a performer
starfinder stats                      # DB size, image cache size
starfinder clear-cache                # clear downloaded images
```

### Preference tree

```powershell
starfinder profile
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
starfinder recommend [--limit 10]

# Performers similar to one specific person (uses ThePornDB /similar API)
starfinder similar "Seka Black"
```

`recommend` scores every candidate against your tree. Body type is a **hard exclusion gate** — wrong physique means excluded entirely. Hair and eye colour are small bonuses.

### Advanced search — `find`

Mix attributes from stored performers or set them manually:

```powershell
# Face attributes from one, body measurements from another
starfinder find --looks-like "Naughty Alysha" --body-like "Dee Siren"
starfinder find --looks-like "Naughty Alysha" --body-like "Lisa Ann"

# Manual filters
starfinder find --ethnicity Caucasian --hair Blonde --cup DD --age-min 40

# Combine
starfinder find --looks-like "Naughty Alysha" --cup DD --age-min 46 --age-max 60
```

**`--looks-like`** copies ethnicity, hair colour, and eye colour.  
**`--body-like`** copies cup size, waist (±4"), and hip measurements (±4").

| Flag | Values | Notes |
|------|--------|-------|
| `--ethnicity` | `Caucasian`, `Latin`, `Black`, `Asian`, `Indian` | Title case |
| `--hair` | `Blonde`, `Brunette`, `Black`, `Red`, `Auburn` | Title case |
| `--eye` | `Blue`, `Green`, `Brown`, `Hazel`, `Grey` | Title case |
| `--cup` | `A` `B` `C` `D` `DD` `DDD` | Letter only |
| `--hips` | `36` | Inches, ±4 tolerance |
| `--waist` | `24` | Inches, ±4 tolerance |
| `--age-min` | `40` | |
| `--age-max` | `55` | |
| `--limit` | `10` | Number of results |

### Face similarity (ML)

```powershell
# Install once
pip install insightface onnxruntime

# Generate ArcFace embeddings for all performers in your DB
# Downloads buffalo_l model on first run (~300 MB, cached forever after)
starfinder embed
```

Once embeddings exist, `find --looks-like` automatically re-ranks results by **cosine similarity of 512-dim ArcFace vectors** — actual facial geometry, not just hair/ethnicity attributes. New performers added via `starfinder add` are auto-embedded.

### Settings

```powershell
starfinder config                        # show current settings
starfinder config gender female          # biological female (default)
starfinder config gender trans-female
starfinder config gender male
starfinder config gender any
```

---

## How it works

### Preference tree

Every performer you add becomes a data point. The tree aggregates them level by level:

```
body_type → ethnicity → hair_color → age_bucket → eye_color
```

The **dominant path** (highest-count child at each level, confidence ≥ 50%) becomes your "type" and drives recommendation queries.

### Recommendation scoring

| Attribute | Weight | Hard gate? |
|-----------|--------|:---:|
| Body type | 5 | ✓ |
| Ethnicity | 3 | — |
| Age range | 2 | — |
| Hair colour | 0.5 | — |
| Eye colour | 0.3 | — |

When face embeddings are available, cosine similarity re-ranks the results on top of this score.

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
| Performer database | `%LOCALAPPDATA%\starfinder\starfinder.db` |
| Face embeddings | Stored inside the same SQLite DB |
| Image cache | `%LOCALAPPDATA%\starfinder\images\` |
| Settings | `%LOCALAPPDATA%\starfinder\config.json` |
| InsightFace model | `~\.insightface\models\buffalo_l\` |

Nothing leaves your machine except outbound API calls to ThePornDB when you explicitly run a command. Face embeddings are biometric data — keeping them local is intentional.

---

## Architecture

Starfinder is a **local-first, single-node** application:

- **SQLite** — embedded, zero-infrastructure database
- **ThePornDB REST API** — external data source (performer profiles, similar-performer queries)
- **InsightFace + ONNX Runtime** — in-process face embedding via Python subprocess
- **No server, no sync, no accounts**

The only distributed systems concern is cache staleness — your local performer snapshots drift from ThePornDB over time. Re-adding a performer refreshes their data.

---

## License

MIT
