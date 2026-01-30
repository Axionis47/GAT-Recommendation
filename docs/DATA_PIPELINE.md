# Data Pipeline

This document explains how raw e-commerce events become training data for the recommendation models.

## Dataset: RetailRocket

The RetailRocket dataset contains user behavior from a real e-commerce website over 4.5 months.

**Raw data:**
- 2.76 million events
- 1.4 million unique visitors
- 235,000 unique items

**Event distribution:**
- Views: 92% (browsing behavior)
- Add to cart: 6% (purchase intent)
- Transactions: 2% (actual purchases)

**The challenge:** No persistent user IDs. Each visitor is anonymous. Traditional collaborative filtering cannot work here because there is no user history to leverage.

## Pipeline Overview

```
Raw Events (2.76M)
       |
       v
[1. Sessionization] -----> 172,066 sessions
       |
       v
[2. Temporal Split] -----> Train/Val/Test with blackout periods
       |
       v
[3. Graph Construction] -> 737,716 co-occurrence edges
```

## Stage 1: Sessionization

**Script:** `scripts/data/02_sessionize.py`

**What it does:**
Groups raw events into browsing sessions based on user activity patterns.

**Key parameters:**
- Session gap: 30 minutes of inactivity starts a new session
- Minimum length: 3 events (shorter sessions are dropped)

**Why 30 minutes?**
This is a standard threshold in e-commerce analytics. If someone browses, leaves for coffee, and comes back after 45 minutes, that should be two separate shopping intents.

**Why minimum 3 events?**
Sessions with 1-2 events provide almost no signal. A user who viewed one item and left tells us nothing about item relationships.

**Results from actual run:**

| Metric | Value |
|--------|-------|
| Sessions created | 172,066 |
| Events retained | 955,778 |
| Unique visitors | 140,569 |
| Unique items | 99,439 |
| Avg session length | 5.6 events |
| Median session length | 4 events |
| Avg session duration | 13 minutes |

**Code reference:**
```python
# From 02_sessionize.py
SESSION_GAP_MINUTES = 30
MIN_SESSION_LENGTH = 3

# Time gaps within each visitor
df["time_gap"] = df.groupby("visitorid")["timestamp"].diff()

# New session if gap > 30 min or new visitor
df["new_session"] = (df["time_gap"].isna()) | (df["time_gap"] > gap_ms)
```

## Stage 2: Temporal Split

**Script:** `scripts/data/03_temporal_split.py`

**What it does:**
Splits sessions into train/validation/test sets based on time, not randomly.

**Split ratios:**
- Train: 70%
- Validation: 15%
- Test: 15%

**Critical feature: Blackout periods**

Between each split, I remove 2 days of data. This prevents temporal leakage.

```
|------- Train (70%) -------|-- Blackout --|--- Val (15%) ---|-- Blackout --|--- Test (15%) ---|
                            |   2 days     |                 |   2 days     |
```

**Why blackout periods matter:**

Without blackouts, an item popular on day 30 (end of train) would still be popular on day 31 (start of validation). The model could "cheat" by memorizing recent popularity rather than learning real item relationships.

The 2-day gap forces the model to generalize beyond short-term trends.

**Results from actual run:**

| Split | Sessions | Events | % of Total |
|-------|----------|--------|------------|
| Train | 120,436 | 679,365 | 70% |
| Validation | 23,861 | 127,461 | 15% |
| Test | 23,408 | 125,363 | 15% |
| Blackout 1 | 1,984 | - | removed |
| Blackout 2 | 2,377 | - | removed |

**Leakage verification:**
The script checks that max(train timestamp) < min(validation timestamp) and logs warnings if there is any overlap.

## Stage 3: Graph Construction

**Script:** `scripts/data/04_build_graph.py`

**What it does:**
Builds a co-occurrence graph from training sessions.

**Graph structure:**
- Nodes: Items (products)
- Edges: Items that appear together in sessions
- Edge weight: Number of co-occurrences

**Co-occurrence window: 5 items**

If a session contains items [A, B, C, D, E, F], I create edges between items within 5 steps of each other:
- A connects to B, C, D, E, F
- B connects to C, D, E, F, (and A from above)
- etc.

**Why 5 items?**

Empirically, items more than 5 clicks apart in a session often have weak relationships. The user may have changed their shopping intent midway.

**Results from actual run:**

| Metric | Value |
|--------|-------|
| Nodes (items) | 82,173 |
| Edges | 737,716 |
| Average degree | 18 |
| Max edge weight | 1,424 |
| Median edge weight | 1 |

The max edge weight of 1,424 means some item pairs appeared together in 1,424 different sessions. These are likely popular items that everyone browses (homepage items, sale items).

**Code reference:**
```python
# From 04_build_graph.py
CO_EVENT_WINDOW = 5

for i in range(len(items)):
    for j in range(i + 1, min(i + window + 1, len(items))):
        # Create edge between items[i] and items[j]
        edge_key = (item_i, item_j)
        edges[edge_key]["count"] += 1
```

## Running the Pipeline

```bash
# Download data (requires Kaggle API key)
python scripts/data/01_download_retailrocket.py

# Sessionize
python scripts/data/02_sessionize.py

# Split
python scripts/data/03_temporal_split.py

# Build graph
python scripts/data/04_build_graph.py
```

Or use the Makefile:
```bash
make data
```

## Output Files

```
data/
├── raw/
│   └── events.csv              # Original Kaggle download
├── interim/
│   ├── sessions.csv            # Sessionized events
│   └── session_stats.json      # Session statistics
└── processed/
    ├── train.csv               # Training sessions
    ├── val.csv                 # Validation sessions
    ├── test.csv                # Test sessions
    ├── split_info.json         # Split statistics
    ├── graph_edges.csv         # Co-occurrence edges
    └── graph_stats.json        # Graph statistics
```

## Why This Pipeline Matters

Many recommendation tutorials use random train/test splits. This is wrong for time-series data.

If you randomly split sessions, your training set contains future data. The model learns patterns like "item X became popular in September" and then gets tested on August data where X was not yet popular. This inflates metrics artificially.

Temporal splits with blackout periods simulate the real deployment scenario: you train on historical data and predict future behavior.
