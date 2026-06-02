# Starfinder

A privacy-focused, local-first recommendation engine for discovering adult performers you'll enjoy — powered by [ThePornDB](https://theporndb.net) and (optionally) ArcFace face recognition.

All data stays on your machine. No accounts, no tracking.

---

## Requirements

- **Rust** (stable) — [rustup.rs](https://rustup.rs)
- **A ThePornDB API key** — free at [theporndb.net](https://theporndb.net)
- **Python 3 + deepface** *(optional, for face similarity)* — `pip install deepface tf-keras`

---

## Build

```powershell
git clone https://github.com/TcgVanguardTroll/Starfinder-rs.git
cd Starfinder-rs
cargo build --release
```

Binary ends up at `target/release/starfinder.exe`.

Set your API key once:

```powershell
$env:TPDB_API_KEY = "your-key-here"
```

---

## Commands

### Building your profile

```powershell
# Add performers you like — fetches full profile from ThePornDB
starfinder add "Naughty Alysha" "Seka Black" "Dee Siren"

# View a performer's stored profile
starfinder view "Naughty Alysha"

# List everyone in your database
starfinder list

# Remove a performer
starfinder remove "Naughty Alysha"
```

### Your taste profile

```powershell
starfinder profile
```

Displays a preference tree built from everyone you've added:

```
Your Taste Profile
══════════════════════════════════════════
  Based on 7 liked performers

  ├── Curvy 6/7  86%
  │   ├── Caucasian 5/6  83%
  │   │   ├── Blonde 3/5  60%
  │   │   │   └── 46+ 3/3  100%
  │   │   │       ├── Green 1/3  33%
  │   │   │       └── Blue  1/3  33%
  ...

  Your type: Curvy → Caucasian → Blonde → 46+
```

The tree drills down through **body type → ethnicity → hair color → age range → eye color**. The more performers you add, the more specific and accurate it becomes.

### Recommendations

```powershell
# Recommendations based on your full taste profile
starfinder recommend

# Performers similar to one specific person
starfinder similar "Seka Black"
```

`recommend` uses ThePornDB's similarity engine seeded with your liked performers, then scores every result against your preference tree. Body type is a hard requirement — wrong physique is excluded entirely.

### Advanced search

Mix and match attributes from different performers or set them manually:

```powershell
# Face of Naughty Alysha, body of Dee Siren (waist/hips)
starfinder find --looks-like "Naughty Alysha" --body-like "Dee Siren"

# Face of Naughty Alysha, body of Lisa Ann (DD cup, tight waist)
starfinder find --looks-like "Naughty Alysha" --body-like "Lisa Ann"

# Manual attribute filters
starfinder find --ethnicity Caucasian --hair Blonde --cup DD --age-min 40

# Combine both
starfinder find --looks-like "Naughty Alysha" --cup DD --age-min 40
```

**`--looks-like`** copies ethnicity, hair color, and eye color from a stored performer.  
**`--body-like`** copies cup size, waist, and hip measurements (±4 inch tolerance).

Available manual flags:

| Flag | Example | Notes |
|------|---------|-------|
| `--ethnicity` | `Caucasian`, `Latin`, `Black`, `Asian` | Title case |
| `--hair` | `Blonde`, `Brunette`, `Black`, `Red` | Title case |
| `--eye` | `Blue`, `Green`, `Brown`, `Hazel` | Title case |
| `--cup` | `B`, `D`, `DD`, `DDD` | Letter only |
| `--hips` | `36` | Target in inches, ±4 tolerance |
| `--waist` | `24` | Target in inches, ±4 tolerance |
| `--age-min` | `40` | |
| `--age-max` | `55` | |

### Face similarity (ML-powered)

Requires Python + deepface installed.

```powershell
# Generate ArcFace embeddings for all performers in your database
# First run downloads the ArcFace model (~100 MB, cached after that)
starfinder embed
```

Once embeddings exist, `find --looks-like` automatically sorts results by **actual facial geometry similarity** (cosine similarity of 512-dim ArcFace vectors) rather than just matching ethnicity/hair/eye attributes.

New performers added via `starfinder add` are automatically embedded on the way in.

### Settings

```powershell
# Show current settings
starfinder config

# Set gender filter (default: Female)
starfinder config gender female
starfinder config gender trans-female
starfinder config gender male
starfinder config gender any
```

### Stats

```powershell
starfinder stats        # performer count + image cache size
starfinder clear-cache  # clear downloaded images
```

---

## How the preference tree works

Every time you run `starfinder profile`, the app builds a tree from your liked performers:

```
body_type → ethnicity → hair_color → age_range → eye_color
```

Each node shows how many of your liked performers fall into that branch and what percentage they represent. The **dominant path** (followed by picking the highest-count child at each level) becomes your "type" and drives recommendations.

As you add more performers the tree forks and becomes more specific. For example, adding a mix of Blonde and Brunette performers splits the hair node and shows you which you prefer more.

---

## How recommendations are scored

Each candidate performer from ThePornDB is scored against your preference tree:

| Attribute | Weight | Hard gate? |
|-----------|--------|------------|
| Body type | 5 | ✓ Wrong body type = excluded |
| Ethnicity | 3 | No |
| Age range | 2 | No |
| Hair color | 0.5 | No (bonus) |
| Eye color | 0.3 | No (bonus) |

Face similarity (when embeddings are generated) re-ranks results by ArcFace cosine similarity, overriding the attribute score.

---

## Data & privacy

- All performer data is stored locally in SQLite at `%LOCALAPPDATA%\starfinder\starfinder.db`
- Face embeddings (512 floats per performer) are stored in the same database
- Downloaded images are cached at `%LOCALAPPDATA%\starfinder\images\`
- Settings are stored at `%LOCALAPPDATA%\starfinder\config.json`
- Nothing is sent anywhere except to ThePornDB's API when you explicitly run a command

---

## License

MIT
