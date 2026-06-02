# Luminary — System Design

A privacy-first, local-first recommendation engine for discovering adult
performers, powered by ThePornDB metadata and ArcFace face recognition.
Everything runs on one machine; nothing leaves it except explicit API calls.

---

## 1. High-level architecture

```mermaid
graph TB
    subgraph User["User"]
        CLI["luminary CLI<br/>(clap commands)"]
    end

    subgraph Binary["luminary binary (src/main.rs)"]
        CMD["Command handlers<br/>add · find · recommend · similar<br/>face-search · warm · embed · config"]
    end

    subgraph Lib["luminary library crate (src/lib.rs)"]
        MODELS["models<br/>Performer, SearchFilters"]
        DB["database<br/>SQLite access"]
        TPDB["tpdb<br/>ThePornDB REST client"]
        STASH["stashdb<br/>StashDB GraphQL client"]
        REC["recommender<br/>tree · IDF · WHR · k-NN"]
        EMB["embedder<br/>ArcFace via sidecar"]
        CFG["config<br/>gender filter · API keys"]
        IMG["image_cache"]
        SCR["scraper<br/>(fallback)"]
    end

    subgraph Local["Local storage (%LOCALAPPDATA%/luminary)"]
        SQLITE[("SQLite<br/>performers · candidates<br/>aliases · embeddings")]
        IMGCACHE[("image cache")]
        CONFIG[("config.json")]
    end

    subgraph External["External services"]
        TPDBAPI["api.theporndb.net<br/>(REST, authorized key)"]
        STASHAPI["stashdb.org/graphql<br/>(optional, ApiKey)"]
        PY["face_embed.py<br/>InsightFace + ONNX<br/>(local subprocess)"]
    end

    CLI --> CMD
    CMD --> MODELS & DB & TPDB & STASH & REC & EMB & CFG & IMG & SCR
    DB --> SQLITE
    IMG --> IMGCACHE
    CFG --> CONFIG
    TPDB --> TPDBAPI
    STASH --> STASHAPI
    EMB --> PY
    SCR -.fallback.-> TPDBAPI
```

---

## 2. Data model

```mermaid
erDiagram
    PERFORMERS {
        int id PK
        string name UK
        string body_type
        string measurements
        string ethnicity
        string hair_color
        string eye_color
        int age
        string tattoos
        string piercings
        bool fake_boobs
        int tpdb_id
        string face_url
        string embedding "JSON 512-float ArcFace vector"
        string data "full JSON snapshot"
    }
    CANDIDATES {
        string name PK
        string data "full JSON snapshot"
        string embedding "JSON 512-float vector"
    }
    ALIASES {
        string alias PK
        string canonical FK
    }

    PERFORMERS ||--o{ ALIASES : "resolves to"
    PERFORMERS }o--o{ CANDIDATES : "discovered from"
```

- **performers** — your liked library; each row carries a cached centroid embedding.
- **candidates** — the local face corpus: every candidate ever embedded during
  `find` / `warm`, reused for instant `face-search`.
- **aliases** — maps a name you type ("Goldie McHawn") to the canonical TPDB name
  ("Goldie Blair").

---

## 3. Workflow: `add` (build your library)

```mermaid
sequenceDiagram
    actor U as User
    participant CLI as luminary add
    participant TPDB as ThePornDB
    participant STASH as StashDB (opt)
    participant PY as face_embed.py
    participant DB as SQLite

    U->>CLI: luminary add "Naughty Alysha"
    CLI->>TPDB: search + fetch performer
    TPDB-->>CLI: metadata (body, measurements, tattoos, face_url)
    alt typed name ≠ canonical name
        CLI->>DB: save alias
    end
    CLI->>DB: insert performer
    opt StashDB key configured
        CLI->>STASH: searchPerformers(term)
        STASH-->>CLI: images[] (multiple photos)
    end
    CLI->>PY: centroid embed (StashDB imgs + TPDB face/gallery)
    PY-->>CLI: 512-float vector
    CLI->>DB: save embedding
    CLI-->>U: "Added (N images)"
```

---

## 4. Workflow: `find` (mix-and-match discovery)

The flagship command — combine one performer's face with another's build,
filter on attributes, rank by facial similarity.

```mermaid
flowchart TD
    START([luminary find --looks-like A --body-like B ...]) --> RESOLVE

    RESOLVE["Resolve references<br/>face embedding from A<br/>cup/WHR/hips from B"] --> FILTERS

    FILTERS["Build HARD filters<br/>gender · ethnicity · hair · cup · WHR · hips · age"] --> QUERY

    QUERY["TPDB search_by_attributes<br/>(server-side + client-side filters)"] --> POOL

    POOL["Candidate pool"] --> PRERANK

    PRERANK["Pre-rank by body k-NN distance<br/>(cheap — picks who to embed)"] --> LOOP

    LOOP{"For top candidates<br/>(cap 16 on-the-fly embeds)"} --> CACHED

    CACHED{"Embedding<br/>cached?"} -->|yes| SCORE
    CACHED -->|no| GEN["face_embed.py → vector<br/>save to candidates corpus"]
    GEN --> SCORE

    SCORE["Score:<br/>face cosine (primary band)<br/>+ tattoo bonus"] --> RANK

    RANK["Sort: facial matches first,<br/>body-only below"] --> OUT([Ranked results + profile links])
```

**Key rule:** face-bearing candidates are lifted into a higher score band than
body-only ones, so genuine facial similarity always wins — the body filters
constrain *who* is eligible, the face decides the *order*.

---

## 5. Workflow: `recommend` (profile-driven)

```mermaid
flowchart LR
    A([luminary recommend]) --> B["Load liked performers"]
    B --> C["Build preference tree<br/>body→ethnicity→hair→age→eye"]
    B --> D["Compute IDF weights<br/>(rare attrs weigh more)"]
    B --> E["Collect TPDB ids"]
    E --> F["TPDB /performers/similar<br/>seeded by your library"]
    F --> G["Score each vs tree × IDF<br/>(body type = hard gate)"]
    C --> G
    D --> G
    G --> H([Top matches, sorted])
```

---

## 6. Workflow: `warm` + `face-search` (instant face lookup)

```mermaid
sequenceDiagram
    actor U as User
    participant W as luminary warm
    participant TPDB as ThePornDB
    participant PY as face_embed.py
    participant DB as candidates corpus

    Note over U,DB: One-time / periodic priming
    U->>W: luminary warm --limit 40
    W->>TPDB: similar-to-your-library pool
    TPDB-->>W: candidates
    loop each candidate
        W->>PY: embed face
        PY-->>W: vector
        W->>DB: save_candidate(perf, vector)
    end

    Note over U,DB: Later — instant, no network
    U->>DB: luminary face-search "Naughty Alysha"
    DB-->>U: cosine rank over corpus (~0.15s)
```

---

## 7. Recommendation algorithms

```mermaid
mindmap
  root((Matching))
    Face
      ArcFace 512-dim embedding
      Centroid over multiple images
      Cosine similarity
    Build
      Waist-to-hip ratio (WHR)
      k-NN feature vector
        inv_whr x3
        hips x2
        cup x1.5
        age x1
    Attributes
      Preference tree
      IDF weighting
      Body type = hard gate
    Bonuses
      Tattoo overlap (Jaccard)
      Natural vs enhanced
```

| Signal | Where used | Weighting |
|--------|------------|-----------|
| Body type | recommend (hard gate), find (filter) | excludes if wrong |
| WHR / butt shape | find filter + k-NN | ×3 in feature vector |
| Cup / bust | find filter, similar | ×1.5 |
| Face (ArcFace) | find --looks-like, face-search | primary ranker |
| Ethnicity / age | recommend, find | IDF-weighted |
| Hair / eye | bonuses | small |
| Tattoo (tramp stamp) | find bonus | +5, never required |

---

## 8. Privacy & trust boundaries

```mermaid
flowchart TB
    subgraph trusted["Your machine (trusted)"]
        direction TB
        app["luminary"]
        store[("SQLite + embeddings<br/>+ image cache + config")]
        py["InsightFace (local ONNX)"]
        app --- store
        app --- py
    end

    subgraph net["Network (explicit calls only)"]
        tpdb["ThePornDB API"]
        stash["StashDB API"]
    end

    app -->|"only on add/find/recommend"| tpdb
    app -->|"only if key set"| stash

    classDef trust fill:#1b3a1b,stroke:#4caf50,color:#fff;
    classDef untrust fill:#3a1b1b,stroke:#f44336,color:#fff;
    class app,store,py trust;
    class tpdb,stash untrust;
```

- Face embeddings (biometric data) **never leave the machine**.
- Gender filter defaults to biological female and is enforced server- and
  client-side.
- IAFD was rejected as a source: its robots.txt sets `ai-train=no`.

---

## 9. Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `main.rs` | CLI parsing + command handlers (thin) |
| `models` | `Performer`, `SearchFilters`, preference types |
| `database` | SQLite: performers, candidates, aliases, embeddings |
| `tpdb` | ThePornDB REST client + body-type inference |
| `stashdb` | StashDB GraphQL client (image enrichment) |
| `recommender` | preference tree, IDF, WHR, k-NN, scoring |
| `embedder` | ArcFace via `face_embed.py`, cosine math, centroids |
| `config` | gender filter, API keys, key resolution |
| `image_cache` | local image download cache |
| `scraper` | FreeOnes fallback when no API key |
