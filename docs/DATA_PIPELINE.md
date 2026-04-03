# Data Pipeline: From Raw Events to Graph

This document explains every step of the data pipeline. Each stage includes actual data samples, statistics, parameter justifications, and design decisions.

## Pipeline Overview

```
Raw Events (2.76M rows)
       |
       v
[1. Sessionize]  -- 30-min gap, min 3 events
       |
       v
Sessions (172,066 sessions, 955K events)
       |
       v
[2. Temporal Split]  -- 70/15/15, 2-day blackout
       |
       v
Train (120K sessions) / Val (24K) / Test (23K)
       |
       v
[3. Build Graph]  -- co-occurrence window of 5
       |
       v
Graph (82,173 nodes, 737,716 edges)
```

---

## Stage 0: Raw Data

**Source:** RetailRocket e-commerce dataset from Kaggle
**File:** `data/raw/events.csv`
**Script:** `scripts/data/01_download_retailrocket.py`

### What the data looks like

```csv
timestamp,visitorid,event,itemid,transactionid
1433221332117,257597,view,355908,
1433224214164,992329,view,248676,
1433224603498,992329,view,248676,
1433226571398,992329,view,248676,
1433227005006,992329,addtocart,248676,
1433227005007,992329,transaction,248676,12345
```

Each row is one event: a visitor viewed, added to cart, or purchased an item.

### Statistics

| Metric | Value |
|--------|-------|
| Total events | 2,756,102 |
| Unique visitors | ~1.4M |
| Unique items | ~235K |
| Date range | ~4.5 months |
| Views | 94.4% |
| Add to cart | 5.8% |
| Transactions | 2.1% |

### Key observation

The data is heavily skewed toward views. Most visitors browse and leave without buying. This is typical for e-commerce and is exactly the cold-start problem we are solving.

---

## Stage 1: Sessionization

**Script:** `scripts/data/02_sessionize.py`
**Input:** `data/raw/events.csv`
**Output:** `data/interim/sessions.csv`
**Metrics:** `data/interim/session_stats.json`

### What it does

Groups raw events into browsing sessions based on two rules:
1. **30-minute gap:** If a visitor is inactive for more than 30 minutes, the next event starts a new session.
2. **Minimum 3 events:** Sessions with fewer than 3 events are dropped.

### Before and after

**Before (raw events):**
```csv
timestamp,visitorid,event,itemid,transactionid
1442004589439,0,view,285930,
1442004759591,0,view,357564,
1442004917175,0,view,67045,
1442089498498,0,view,310498,         <-- 23 hours later, new session
1442089652498,0,view,241014,
```

**After (sessionized):**
```csv
timestamp,visitorid,event,itemid,transactionid,session_id
1442004589439,0,view,285930,,sess_1
1442004759591,0,view,357564,,sess_1
1442004917175,0,view,67045,,sess_1
1442089498498,0,view,310498,,sess_2     <-- different session
1442089652498,0,view,241014,,sess_2
```

Wait, that second session only has 2 events. It would be dropped by the minimum length filter. Only sessions with 3+ events survive.

### Why 30 minutes?

This is the industry-standard threshold for e-commerce session boundaries. The reasoning:

- Under 5 minutes: Someone switching between tabs. Same intent.
- 5-30 minutes: Could be reading reviews, comparing prices. Still the same shopping intent.
- Over 30 minutes: Likely left the site. If they come back, it is probably a different shopping intent.

Google Analytics uses 30 minutes. Most e-commerce studies use 30 minutes. We use 30 minutes.

### Why minimum 3 events?

A session with 1-2 events is not useful for training:
- We need at least 1 item as context and 1 item as the prediction target.
- With only 2 items, there is no graph structure (just one edge). The GNN has nothing to work with.
- 3 events gives us 2 context items and 1 target, plus at least one edge in the subgraph.

### Statistics after sessionization

| Metric | Value |
|--------|-------|
| Sessions | 172,066 |
| Events | 955,778 |
| Unique visitors | 140,569 |
| Unique items | 99,439 |
| Avg session length | 5.55 events |
| Median session length | 4.0 events |
| Min session length | 3 |
| Max session length | 417 |
| Views per session | 880,098 (92.1%) |
| Add-to-cart per session | 55,245 (5.8%) |
| Transactions per session | 20,435 (2.1%) |

---

## Stage 2: Temporal Split

**Script:** `scripts/data/03_temporal_split.py`
**Input:** `data/interim/sessions.csv`
**Output:** `data/processed/train.csv`, `val.csv`, `test.csv`
**Metrics:** `data/processed/split_info.json`

### What it does

Splits sessions into train/val/test by time, not randomly. Inserts blackout periods between splits.

### Why temporal splits instead of random splits?

Random splits cause **data leakage**. Here is the problem:

```
Random split (BAD):
  Train: [Session from Aug 15, Session from Sep 1, Session from Jul 20, ...]
  Test:  [Session from Aug 10, Session from Aug 25, ...]

  The model trains on future data and is tested on past data.
  Metrics look great. Deployment performance is terrible.
```

```
Temporal split (CORRECT):
  Train: [All sessions from Jul 1 - Aug 20]
  Val:   [All sessions from Aug 23 - Sep 5]
  Test:  [All sessions from Sep 8 - Sep 20]

  The model only sees past data during training.
  Metrics reflect real deployment performance.
```

### Why blackout periods?

Even with temporal splits, sessions near the boundary can leak information. A session that starts at the end of training and overlaps with the validation period would contaminate the split.

```
Timeline:
  |---- Train ----|##|---- Val ----|##|---- Test ----|
                  2-day            2-day
                blackout         blackout

Sessions in blackout periods are dropped entirely.
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Train ratio | 70% |
| Validation ratio | 15% |
| Test ratio | 15% |
| Blackout days (min) | 1 |
| Blackout days (max) | 3 |

### Split results

| Split | Sessions | Events |
|-------|----------|--------|
| Train | 120,436 | 679,365 |
| Validation | 23,861 | 127,461 |
| Test | 23,408 | 125,363 |
| Dropped (blackout 1) | 1,984 | - |
| Dropped (blackout 2) | 2,377 | - |

---

## Stage 3: Graph Construction

**Script:** `scripts/data/04_build_graph.py`
**Input:** `data/processed/train.csv` (training sessions only)
**Output:** `data/processed/graph_edges.csv`
**Metrics:** `data/processed/graph_stats.json`

### What it does

Builds an undirected, weighted item co-occurrence graph from training sessions. Two items get an edge if they appear within 5 steps of each other in any session.

### Step-by-step example

Take this session:

```
Session sess_42:
  Step 0: view item_100
  Step 1: view item_200
  Step 2: addtocart item_300
  Step 3: view item_400
  Step 4: view item_500
  Step 5: view item_600
  Step 6: transaction item_300
```

With a window of 5, for each item we look at the next 5 items:

```
From item_100 (step 0):
  -> item_200 (step 1): edge(100, 200) view_view
  -> item_300 (step 2): edge(100, 300) view_addtocart
  -> item_400 (step 3): edge(100, 400) view_view
  -> item_500 (step 4): edge(100, 500) view_view
  -> item_600 (step 5): edge(100, 600) view_view

From item_200 (step 1):
  -> item_300 (step 2): edge(200, 300) view_addtocart
  -> item_400 (step 3): edge(200, 400) view_view
  -> item_500 (step 4): edge(200, 500) view_view
  -> item_600 (step 5): edge(200, 600) view_view
  -> item_300 (step 6): edge(200, 300) view_transaction  [count increments!]

From item_300 (step 2):
  -> item_400 (step 3): edge(300, 400) addtocart_view
  -> item_500 (step 4): edge(300, 500) addtocart_view
  -> item_600 (step 5): edge(300, 600) addtocart_view
  -> item_300 (step 6): edge(300, 300) addtocart_transaction  [self-loop!]

... and so on for steps 3-5.
```

**Key details:**
- Edges are undirected: `edge(A, B)` and `edge(B, A)` are the same edge. Canonical ordering: smaller ID first.
- Counts accumulate: if items A and B appear together in 50 different sessions, the edge count is 50+.
- Self-loops are preserved: item_300 appears at step 2 and step 6 in the same session.
- Event pairs are tracked: `view_addtocart` counts separately from `view_view`.

### Why a co-occurrence window of 5?

- Sessions are short: median length is 4, mean is 5.55.
- A window of 5 captures nearly all items in most sessions.
- Larger windows would connect unrelated items in long sessions.
- Smaller windows would miss meaningful relationships.

### What the output looks like

```csv
item_i,item_j,count,last_ts,event_pair_hist
40870,40870,1424,1436309982317,"{'view_view': 1334, 'view_addtocart': 23, ...}"
461686,461686,617,1438294597997,"{'view_addtocart': 167, 'view_transaction': 45, ...}"
396064,396064,580,1437207411308,"{'view_view': 580}"
100,200,3,1435000000000,"{'view_view': 3}"
```

### Graph statistics

| Property | Value |
|----------|-------|
| Nodes (unique items) | 82,173 |
| Edges | 737,716 |
| Average degree | 17.96 |
| Edge count mean | 2.40 |
| Edge count median | 1.0 |
| Edge count min | 1 |
| Edge count max | 1,424 |

### Graph properties to note

1. **Sparse graph.** Density is about 0.0002. Most items are not connected to most other items.
2. **Power-law degree distribution.** A few popular items have hundreds of connections. Most items have fewer than 20.
3. **Self-loops exist.** Items that appear multiple times in the same session get self-loops. This is intentional: it signals repeated interest.
4. **Only training data.** The graph is built from training sessions only. Validation and test sessions are never used for graph construction. This prevents data leakage.

---

## Why This Pipeline Design?

### No random splitting anywhere

Every split is temporal. The graph is built from training data only. The model never sees future information during training. This is how the system would actually work in production: train on history, predict the future.

### Blackout periods prevent subtle leakage

Without blackout periods, the last training session and the first validation session could be from the same visitor, minutes apart. The blackout period creates a clean separation.

### Graph from training data only

If we built the graph from all data, the graph would contain co-occurrence patterns from the validation and test periods. The model would implicitly learn about the future. Building the graph from training data only ensures the evaluation is honest.

### DVC tracks everything

Every stage is defined in `dvc.yaml`. Change a parameter, run `dvc repro`, and only the affected stages re-execute. Data versions are pushed to GCS. Any experiment can be reproduced.

```bash
# Reproduce the entire pipeline
dvc repro

# See the pipeline DAG
dvc dag

# Check metrics
dvc metrics show
```
