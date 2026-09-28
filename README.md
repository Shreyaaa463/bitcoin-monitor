# TRACE-X — Offline Bitcoin Transaction Forensic Intelligence Platform

An offline, Linux-based forensic intelligence system built for the problem statement
**"AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic"** (NTRO,
Blockchain & Cybersecurity track).

It ingests Bitcoin network + blockchain metadata (the TRACE-X dataset), correlates
IPs / transactions / wallets / ASNs / countries, detects anomalies with an
unsupervised ML model, clusters similar behavior, builds an explainable investigative
priority score, groups related alerts into investigation cases, fingerprints each
case's evidence with a Merkle tree anchored in a **local/private append-only
tamper-evident ledger**, and presents everything in an interactive Streamlit
dashboard with an explorable IP → Transaction → Wallet entity graph.

> **Note: the dataset is synthetic.** No output of this system should ever be
> interpreted as identifying a real person, real wallet, or real criminal activity.
> Every score is a **model-generated investigative signal**, not proof of anything.

---

## 1. Quick start (one command, fully offline)

```bash
./run.sh
# -> validates the environment, fingerprints the dataset (SHA-256),
#    reuses the identical completed run if it exists -- otherwise runs the pipeline once --
#    then starts the dashboard at  http://localhost:8501
```

| Command | What it does |
|---|---|
| `./run.sh` | validate → reuse-or-run pipeline → dashboard |
| `./run.sh --input data/raw/other.json` | same, for another dataset (file **or directory**) |
| `./run.sh --dashboard-only` | open the latest completed run; never touches the pipeline |
| `./run.sh --run <RUN_ID>` | open a specific completed run |
| `./run.sh --pipeline-only` | run/reuse the pipeline, no dashboard |
| `./run.sh --force` | produce a **new** run even if an identical completed one exists (never overwrites it) |
| `./run.sh --check` | environment validation only |
| `python run.py [--input …] [--force]` | pipeline only (same as before) |

If `run.sh` is not executable after copying (e.g. from a Windows machine), run
`chmod +x run.sh scripts/*.sh` once, or call it as `bash run.sh`.

Default dataset: `data/raw/trace_x_bitcoin_12000.json` (`paths.raw_data` in
`config.yaml`; override with `--input` or `BTC_MONITOR_DATASET`). Supported: `.json`
(array), `.jsonl`/`.ndjson`, `.csv` (list fields `|`-delimited), `.xml`, or a directory
containing any mix of these.

---

## 2. Architecture

```
                 ./run.sh  ──►  src/launcher.py
                                  │ 1. validate env (python, packages, dataset, disk, port)
                                  │ 2. dataset SHA-256 + PIPELINE_VERSION + analytical-config hash = run_key
                                  │ 3. completed run with that run_key?  yes → reuse (no recompute)
                                  ▼                                       no  → run pipeline once
                       src/pipeline.py:run_pipeline()   (under an OS file lock per run_key)
                                  │   writes ONLY into runs/<RUN_ID>/, tracked by manifest.json
 dataset (file or dir)            ▼
   │   [1] ingest   stream in chunks, per-source-file cache  (src/ingestion/loader.py, staging.py)
   │   [2] clean    row rules per chunk + global txid dedup   (src/preprocessing/clean.py)
   │                + offline GeoIP/ASN                        (src/geoip/lookup.py)
   │                → runs/<RUN_ID>/data/cleaned/date=YYYY-MM-DD/*.parquet
   │   [3] correlate IP/TX/Wallet/ASN/Country edges            (src/correlation/correlate.py)
   │                → edges/ (Parquet) + graph_index.sqlite    (src/graph/store.py)
   │   [4] features  behavioral + graph-topology               (src/features/engineer.py)
   │   [5] Isolation Forest: train (optionally sampled), then chunked inference (src/models/anomaly.py)
   │   [6] risk score (transparent weighted 0-100)             (src/scoring/risk_score.py)
   │   [7] candidate funnel → DBSCAN                           (src/models/clustering.py)
   │   [8] explanations (feature-grounded reasons)             (src/explainability/explain.py)
   │   [9] evaluation vs synthetic_label (evaluation only)     (src/models/evaluate.py)
   │  [10] alerts   → alerts/date=…/*.parquet (every transaction, all scores + reasons)
   │  [11] investigation cases (shared wallet / IP / cluster+time) (src/forensics/cases.py)
   │  [12] evidence → SHA-256 leaves → Merkle root → local ledger bound to RUN_ID
   │                                                           (src/forensics/*)
   │  [13] finalize: checksum every artifact → status = completed → runs/latest.json
   ▼
 Streamlit dashboard (dashboard/app.py): read-only view of ONE completed run, chosen in
 the sidebar; queries artifacts (Parquet filter pushdown, SQLite graph index) instead of
 loading them; never runs the pipeline.
```

**Design principles (unchanged from the prototype):**
- `synthetic_label` is used **only** in evaluation. The Isolation Forest is trained
  purely unsupervised; it never sees the label.
- The forensic layer (cases → evidence → Merkle → ledger) **reuses** the alerts table;
  no ML score, cluster label or graph metric is recomputed there.

**Equivalence with the prototype.** On a 12,000-record synthetic TRACE-X-format
dataset, a run of this version was compared artifact-by-artifact against the original
code: all 54 alert columns, `data_quality.json`, `graph_summary.json`,
`evaluation_metrics.json`, `cluster_summary.json`, `cases.json`,
`evidence_store.json`, and every case's Merkle root and leaf hashes are **identical**
(the ledger additionally binds blocks to the run ID). Wall time on the same machine:
161 s → 24 s.

---

## 3. Runs, reproducibility and safety

Every pipeline execution gets its own directory; nothing overwrites a previous run.

```
runs/
  <RUN_ID>/                      e.g. 20260927T125638Z-3e879ae9c473
    manifest.json                status, identity, provenance, per-stage status, artifact checksums
    pipeline.log  (error.log)    run log (traceback if it failed)
    data/cleaned/date=…/         cleaned, validated transactions (partitioned Parquet)
    edges/date=…/                typed correlation edges (Parquet)
    graph_index.sqlite           on-disk entity-graph index (dashboard graph + lookups)
    alerts/date=…/               every transaction + features + scores + priority + reasons
    models/isolation_forest.joblib
    data_quality.json  graph_summary.json  cluster_summary.json
    evaluation_metrics.json  run_metadata.json  cases.json
    entity_graph.pickle          (small runs only; legacy NetworkX artifact)
    forensics/evidence_store.json  forensics/blockchain_ledger.json
  latest.json                    pointer to the most recently COMPLETED run
  .locks/                        OS file locks (one per run_key being produced)
data/processed/ingest_cache/     per-source-file staging cache shared by runs (safe to delete)
```

**Run identity.** `run_key = sha256(dataset SHA-256 | PIPELINE_VERSION | config_hash)`.
- *dataset SHA-256*: the file's SHA-256 (same as `sha256sum`), or for a directory the
  SHA-256 over every file's relative path + SHA-256.
- *PIPELINE_VERSION*: `src/version.py` (bump it when a code change can alter results).
- *config_hash*: SHA-256 of the **analytical** config sections (`ingestion`,
  `features`, `model`, `clustering`, `scoring`, `explainability`, `cases`, `graph`,
  `geoip`, `scale`) plus the content hash of any offline GeoIP database. Paths,
  dashboard, `runtime` (chunk sizes), `runs` and `launcher` settings are
  result-neutral and excluded — tests verify that chunk size does not change results.

**manifest.json** records: `run_id`, `run_key`, `status` (`running` / `completed` /
`failed`), start/end timestamps, dataset SHA-256 + per-file hashes, pipeline version,
code fingerprint, config hash + analytical config snapshot, GeoIP DB fingerprint,
Python/package versions, record counts, resolved scale settings, candidate-funnel
counts, **per-stage status and timing**, forensic ledger head/genesis hashes, and the
size + SHA-256 of every artifact.

**Completion and failure safety.**
- A run is marked `completed` only after every stage succeeded and every artifact
  was checksummed. The existence of `alerts/` (or any file) is never used as a
  completion signal.
- Any exception (or Ctrl+C) marks the run `failed`, writes `error.log`, and exits
  non-zero. A killed process leaves the run `running`; the next launch detects that no
  process holds its lock and marks it `failed` ("interrupted").
- The dashboard, `tamper_demo.py` and `reset_case_evidence.py` only ever open
  **completed** runs.
- Two launchers started for the same dataset/config: one produces the run, the other
  waits on the lock and then reuses the finished run (verified).
- Before reusing a completed run, its artifacts are checked (existence + size;
  `--verify-deep` adds SHA-256). If they were modified, the old run is left untouched
  and a new run is produced.
- JSON artifacts, the evidence store and the ledger are written atomically
  (temp file + fsync + rename).

**Legacy `outputs/`.** Outputs produced by the earlier single-directory prototype are
still viewable: the dashboard lists `outputs/` as "legacy-outputs (unversioned)" if it
contains `alerts.parquet`. New runs never write there.

---

## 4. Repository tree

```
bitcoin-monitor/
├── run.sh                        # ONE-command launcher (env check → reuse/run → dashboard)
├── run.py                        # pipeline-only CLI (python run.py)
├── config.yaml
├── requirements.txt              # dependency ranges
├── requirements.lock.txt         # exact versions verified together (offline wheelhouse)
├── Dockerfile  .dockerignore     # optional container (no Internet at runtime)
├── .streamlit/config.toml        # offline Streamlit defaults (no telemetry, localhost)
├── data/raw/                     # your datasets (never modified)
├── data/geoip/                   # optional GeoLite2 / DB-IP .mmdb files
├── data/processed/ingest_cache/  # shared per-source-file staging cache
├── runs/                         # one directory per pipeline run (see §3)
├── src/
│   ├── version.py                # PIPELINE_VERSION + code fingerprint
│   ├── config.py                 # config + path/env resolution
│   ├── pipeline.py               # staged, manifest-tracked orchestration (run_pipeline)
│   ├── launcher.py               # environment validation + run selection + Streamlit start
│   ├── runs/                     # identity.py, manifest.py, registry.py, locks.py, fsutil.py
│   ├── storage/parquet_store.py  # partitioned Parquet schema + read/write helpers
│   ├── ingestion/loader.py       # JSON/JSONL/CSV/XML → normalized records (whole-file + streaming)
│   ├── ingestion/staging.py      # chunked, cached ingestion + global dedup
│   ├── preprocessing/clean.py    # validation rules (chunk-level + dedup)
│   ├── geoip/lookup.py           # offline-only GeoIP/ASN resolver
│   ├── correlation/correlate.py  # typed relationship edges
│   ├── graph/build_graph.py      # NetworkX graph + offline PyVis export
│   ├── graph/store.py            # SQLite graph index: neighborhoods, lookups, edge arrays
│   ├── features/engineer.py      # behavioral + graph features (networkx / edge_table backends)
│   ├── models/                   # anomaly.py (Isolation Forest), clustering.py (DBSCAN + funnel), evaluate.py
│   ├── scoring/risk_score.py
│   ├── explainability/explain.py
│   ├── serving/run_store.py      # read-only, query-based access to a completed run
│   └── forensics/                # canonical, cases, evidence, merkle, blockchain, anchoring,
│                                 # verification, restore
├── dashboard/app.py              # Streamlit UI (8 tabs, unchanged charts)
├── scripts/
│   ├── generate_synthetic_dataset.py   # --format json|jsonl
│   ├── tamper_demo.py / reset_case_evidence.py   # forensic demo (--run RUN_ID)
│   ├── benchmark_scalability.py        # 10K → 100K → 1M benchmark
│   ├── build_offline_bundle.sh         # (online machine) wheelhouse + tarball
│   └── install_offline.sh              # (offline machine) .venv from wheelhouse
└── tests/                        # 89 tests
```

---

## 5. Dependencies

`requirements.txt` (ranges) / `requirements.lock.txt` (exact, verified): pandas,
numpy, scikit-learn, scipy, joblib, networkx, pyvis, streamlit, plotly, pyyaml,
pyarrow, pytest.

**No database server, no Docker, no Spark/Kafka/Neo4j, no cloud service is required.**
The scalable graph index is a SQLite file created with Python's built-in `sqlite3`
module (nothing to install or run). Optional: `geoip2` (only if you drop `.mmdb`
files into `data/geoip/`).

---

## 6. Offline deployment

The application never needs network access at runtime: no public APIs, no LLM APIs,
no CDN (PyVis' Bootstrap CDN tags are stripped from the generated graph HTML;
Streamlit telemetry is off; the server binds to `127.0.0.1`), no runtime package
downloads.

### 6.1 Package the project (on a machine WITH Internet, Linux or WSL)

```bash
# wheels for the TARGET machine's Python (3.11-3.13 for the lock file) + a tarball
PY_VERSION=3.12 scripts/build_offline_bundle.sh           # add INCLUDE_DATA=1 to ship data/raw/
# -> dist/bitcoin-monitor-offline-2.0.0-py3.12.tar.gz  (prints its SHA-256)
```

### 6.2 Install and start on the offline Linux machine

```bash
tar xzf bitcoin-monitor-offline-2.0.0-py3.12.tar.gz
cd bitcoin-monitor
scripts/install_offline.sh                   # creates .venv from wheelhouse/ (pip --no-index)
cp /media/usb/trace_x_bitcoin_12000.json data/raw/   # your dataset
./run.sh                                     # → http://localhost:8501
```

Requirements on the target: Linux x86_64, Python 3.11–3.13 with `venv`, ~1 GB free
disk for the 12K demo (see §8 for larger sizes). For Python 3.10 build the wheelhouse
from `requirements.txt` instead of the lock file.

### 6.3 Optional: Docker (still offline at runtime)

```bash
docker build -t trace-x:2.0.0 .                               # online build host
docker save trace-x:2.0.0 | gzip > trace-x-2.0.0.tar.gz        # copy this file
docker load < trace-x-2.0.0.tar.gz                             # offline host
docker run --rm -p 127.0.0.1:8501:8501 \
    -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" trace-x:2.0.0
```

Docker is optional; the image was not built as part of this change's verification
(no Docker on the development machine).

---

## 7. Forensic demo (tamper evidence)

```bash
python scripts/tamper_demo.py --list                              # cases of the latest completed run
python scripts/tamper_demo.py --case CASE-XXXXXXXXXXXX --field risk_score --value 0
#   Dashboard → Forensic Blockchain → Verify Evidence  ⇒  FAILED (names the txid)
python run.py            # does NOT repair it: the run is reused, evidence stays FAILED
python scripts/reset_case_evidence.py --case CASE-XXXXXXXXXXXX    # explicit, deliberate
python run.py            # restores that case's evidence from the run's own alerts table
#   Verify Evidence  ⇒  VERIFIED   (ledger unchanged: nothing is re-anchored)
```

All scripts take `--run <RUN_ID>` (default: latest completed run). Restoration only
happens if the re-derived evidence hashes to exactly the Merkle root already anchored
for that case; otherwise it is refused and reported.

---

## 8. Scalability — what is implemented, measured, and not

**Implemented (single machine):**
- *Streaming ingestion*: JSON arrays are decoded incrementally, JSONL/CSV/XML are read
  in chunks; the raw file is never held in memory.
- *Partitioned storage*: cleaned data, edges and alerts are date-partitioned Parquet
  with an enforced schema; downstream stages read column projections per partition.
- *Incremental ingestion*: each source file is staged once per content hash; a
  directory dataset that gains a new file only parses the new file.
- *Run reuse*: an identical (dataset, version, config) run is never recomputed.
- *Training/inference split*: Isolation Forest training can use a deterministic sample
  (`model.isolation_forest.max_train_rows`); inference is chunked over all rows.
- *Candidate funnel*: above `scale.exact_mode_max_rows` (20,000 valid records):
  every transaction is scored (features → Isolation Forest → risk); only candidates
  (priority in `cases.seed_priorities` or ML-flagged, capped by `clustering.max_rows`,
  highest risk first) go to DBSCAN; non-candidates get cluster label `-2`
  ("not clustered", distinct from DBSCAN noise `-1`); cases are built only from
  candidates, and an unlinked candidate becomes its own case only if HIGH priority
  (`cases.singleton_min_priority`). All alerts stay in the alert queue.
- *Graph*: the full relationship set is stored in `graph_index.sqlite` (indexed by
  source and destination node). The dashboard retrieves only the 1–3-hop
  neighborhood of the selected entity, with a per-hub-node neighbor cap and a total
  node cap. Graph features use the `edge_table` backend: sparse-matrix degree and
  PageRank over the index (same algorithm and tolerance as NetworkX), and IP
  betweenness = 0 exactly because no edge points into an IP node in this graph
  schema (a node without incoming edges cannot lie on a shortest path; the backend
  checks this at runtime and refuses otherwise). This replaced a NetworkX
  betweenness computation that took ~70–80 s at 12K records and produced the same
  all-zero feature.
- *Forensic ledger*: blocks are appended in one atomic write per run (the prototype
  rewrote the whole ledger file after every block — O(n²) I/O); case-block lookup is
  indexed; pairwise case links are chained for very large groups (identical cases,
  bounded memory).

**Measured** (`scripts/benchmark_scalability.py`, synthetic data, Windows 11 laptop,
6 CPUs, 7.4 GB RAM, Python 3.13):

| records | wall time | peak RSS | run dir on disk | ingest | clean | correlate | features | IF train+score | DBSCAN | explain | alerts | cases | forensics |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 16 s | 410 MB | 65 MB | cached | 0.8 s | 2.2 s | 2.5 s | 0.9 s | 0.2 s (all rows) | 0.1 s | 1.0 s | 7.4 s | 0.6 s |
| 100,000 | 51 s | 675 MB | 350 MB | cached | 4.7 s | 18 s | 9.6 s | 3.8 s | 0.5 s (20K candidates) | 0.5 s | 2.7 s | 6.8 s | 3.4 s |
| 1,000,000 | 817 s | 2.1 GB | 3.6 GB | 65 s | 45 s | 368 s | 151 s | 31 s | 29 s (200K candidates) | 5.8 s | 28 s | 41 s | 44 s |

("cached": those two datasets had already been staged by an earlier benchmark run; the
1M row is a cold run.) For reference, the original prototype on the same machine took
161 s for 12,000 records (~70–80 s of it in the constant-zero betweenness feature, much
of the rest in per-block ledger rewrites) and **did not finish 100,000 records within
20 minutes** (it was still building/anchoring cases when stopped).

Observations: correlation (building the SQLite index for ~7.8M edges) dominates at 1M
and is the next thing to optimize; disk use is ~5× the raw JSONL size (cleaned data +
edges + alerts + graph index are all kept per run); at 1M rows the shared-IP /
cluster+time linkage rules chain ~198K seed transactions into 94 very large cases —
the linkage criteria, not the implementation, need tightening for large datasets.

**Not implemented — production-scale work, not claimed:**
- The feature stage holds per-transaction scalar columns and the distinct
  (IP, wallet) pairs of the whole dataset in memory, and scoring/explanations use
  in-memory numpy arrays: memory grows linearly with the number of records. Nothing
  here has been shown to handle billions (let alone trillions) of records.
- Analytical stages after cleaning are recomputed for every new dataset hash: the
  per-IP aggregates, graph metrics, Isolation Forest fit, z-scores and min-max
  normalizations are global statistics, so appending data changes them for all rows.
  Truly incremental analytics would need a frozen-model scoring mode and streaming
  aggregate maintenance.
- Single machine, single process per stage. Distributed execution (e.g. Spark/Dask
  over the same Parquet layout) and a server-grade graph store are the natural next
  steps; the Parquet partitions and the graph index schema were chosen so they can be
  swapped in without changing the analytical logic.
- The evidence store is one JSON file per run and the Forensic tab re-reads it on every
  render (by design, to catch tampering immediately); very large case counts would need
  a per-case store.

---

## 9. ML method

- **Anomaly detection:** `sklearn.ensemble.IsolationForest`, trained on a
  **standardized** (`StandardScaler`) feature matrix. Fully unsupervised —
  `synthetic_label` is never passed to `.fit()`. Output is converted from sklearn's
  raw `decision_function` (higher = more normal) into an inverted, min-max-scaled
  **0–100 anomaly score** (higher = more anomalous). This is explicitly **not** a
  calibrated probability.
- **Clustering:** `sklearn.cluster.DBSCAN` over the same standardized feature space
  (all rows on small datasets; the candidate funnel on large ones, §8). `-1` labels
  (DBSCAN "noise") are preserved and shown as-is.
- Configurable in `config.yaml` under `model.isolation_forest`, `model.dbscan` and
  `clustering` without touching code.

## 10. Graph method

- Typed nodes (`IP`, `Wallet`, `Transaction`, `ASN`, `Country`) and typed edges
  (`OBSERVED`, `INPUT`, `OUTPUT`, `BELONGS_TO`, `LOCATED_IN`), stored per run in
  `graph_index.sqlite` (and, for small runs, also as the prototype's NetworkX pickle).
- The **Graph Explorer** never renders the full graph: it computes the bounded k-hop
  neighborhood of the selected entity from the index (identical node/edge sets to the
  prototype's `get_neighborhood()` when no cap is hit — tested) and renders it with
  PyVis, so paths like `IP → TX → Wallet → TX → Wallet` stay explorable.
- Graph-topology features (degree, in/out-degree, PageRank, betweenness) are computed
  once per run and attached to every transaction via its source IP.

## 11. Scoring

`src/scoring/risk_score.py` combines four **min-max normalized (0–100)** components
into one weighted, transparent score:

| Component | What it captures |
|---|---|
| `ml_anomaly` (default weight 0.40) | Isolation Forest anomaly score |
| `behavioral` (0.25) | fan-out, fan-in, transaction burst count, amount variance |
| `graph` (0.20) | node degree, PageRank, betweenness centrality |
| `network_geo` (0.15) | countries touched, wallet/transaction ratio |

Weights are configurable and re-normalized if they don't sum to 1. Thresholds
(`high: 70`, `medium: 40`) map the score to `HIGH` / `MEDIUM` / `LOW`. **This is an
investigative priority signal, not proof of criminal activity.**

## 12. Explainability

Every alert's `reasons` field is built from **real numbers present in that row**:
1. Whether Isolation Forest flagged it, and its exact anomaly score.
2. Its DBSCAN cluster membership ("noise point", or "not clustered" outside the funnel).
3. Up to `top_n_reasons` (default 5) feature-level reasons, chosen by ranking each
   feature's z-score against the dataset mean/std and citing only those beyond
   `z_score_reason_threshold` (default 1.5 std devs).

`try_shap_global_importance` exists as an optional, exception-guarded helper; the
pipeline does not currently call it.

## 13. GeoIP/ASN (optional, offline)

By default the dataset's own `geo_country` / `asn` fields are used. For independent
offline enrichment, download `GeoLite2-Country.mmdb` / `GeoLite2-ASN.mmdb` (MaxMind,
free account) or the DB-IP Lite equivalents once, place them in `data/geoip/`, and
`pip install geoip2` (add its wheel to the wheelhouse for offline machines). Their
content hash becomes part of the run identity, so adding/updating a database produces
a new run. **No online IP lookup API is ever called.**

## 14. Evaluation

Precision, Recall, F1, ROC-AUC, PR-AUC and the confusion matrix (ML/Evaluation tab)
are computed **per run** by comparing `is_anomaly_prediction` / `anomaly_score_0_100`
against `synthetic_label` (`src/models/evaluate.py`). Nothing is hard-coded.

## 15. Investigation cases

`src/forensics/cases.py` groups related suspicious transactions into **investigation
cases**, reusing the alerts table. A case is deliberately **not** a DBSCAN cluster.
Seed transactions (priority HIGH/MEDIUM, or ML-flagged) are linked when:

| Link type | Signal |
|---|---|
| `shared_wallet` | two transactions touch the same input/output wallet |
| `shared_ip` | two transactions share a source IP |
| `cluster_temporal` | same non-noise DBSCAN cluster **and** within `temporal_window_hours` |

Cluster membership alone never creates a link (unit-tested). Connected components
become cases; case IDs are a SHA-256 digest of the sorted linked txids, so unchanged
data reproduces the same case ID and evidence. Large-scale controls:
`cases.singleton_min_priority` and `cases.max_link_pairs_per_group` (§8).

## 16. Merkle-tree evidence

For every case, `src/forensics/evidence.py` builds one canonical evidence record per
linked transaction (txid, IPs, wallets, timestamp, cluster, scores, priority, reasons
— values already produced upstream). `src/forensics/canonical.py` serializes each
record deterministically (sorted keys, fixed separators, floats rounded to 6 decimals)
before SHA-256 hashing. `src/forensics/merkle.py` builds a binary Merkle tree over the
case's leaves and provides `generate_merkle_proof` / `verify_merkle_proof`.

## 17. Local/private append-only tamper-evident ledger

`src/forensics/blockchain.py` is a **local, private, append-only, hash-linked JSON
ledger** — not a public blockchain (no Bitcoin/Ethereum, no network, no mining, no
consensus). One ledger per run (`runs/<RUN_ID>/forensics/blockchain_ledger.json`);
each block anchors one case:

```
{ index, timestamp, case_id, merkle_root, leaf_hashes, evidence_count,
  previous_hash, run_id, hash }
```

`run_id` is inside the hashed header of every block (genesis included), binding each
Merkle root to its run and case. Blocks store only hashes and small metadata — never
raw transaction data, wallet addresses or IPs. `Blockchain.validate_chain()` recomputes
every block hash, every `previous_hash` link, every block's Merkle root from its
leaves, and the run binding. The run manifest records the ledger's genesis and head
hashes at completion; the dashboard reports whether the ledger still matches them.

**What it guarantees:** modification of anchored evidence (or of the ledger file) is
*detected*. **What it does not:** it cannot prove the evidence is true, and it cannot
prevent a user with write access to the whole filesystem from rebuilding a consistent
ledger — protect `runs/` at the OS level (permissions, backups, write-once media).

## 18. Verification

`src/forensics/verification.py` is **read-only**. `verify_case()` re-hashes the
evidence currently on disk, compares it to the hashes frozen in the case's latest
block, recomputes the Merkle root, re-validates the ledger, and checks the block is
bound to the run being viewed. Any single-field change is reported as `FAILED`,
naming the transaction. It is never silently repaired (§7).

## 19. Limitations

- **Synthetic data only.** No real-world Bitcoin attribution is claimed or possible.
- Isolation Forest `contamination` is a prior belief (0.2), not learned.
- DBSCAN `eps`/`min_samples` are scale-dependent; on the funnel they apply to the
  candidate set only.
- Under the current graph schema `ip_in_degree` and `ip_betweenness` are always 0 and
  `ip_pagerank` is identical for every IP, so they carry no information (unchanged
  from the prototype; kept so detection logic is not altered).
- On the synthetic dataset no transaction reaches the HIGH threshold (70); cases come
  from MEDIUM and ML-flagged seeds.
- Timestamp format is inferred from the first timestamp of the dataset and applied to
  all rows (as pandas does for a whole column); mixed-format datasets should be
  normalized first.
- File locks require a local filesystem (not NFS without lock support).
- The ingest cache (`data/processed/ingest_cache/`) grows with every distinct source
  file; it is safe to delete at any time (it only speeds up ingestion).
- See §8 for what scales and what remains production work.

## 20. Demo flow (~7 minutes)

1. **Terminal:** `./run.sh` — narrate: dataset SHA-256 → run key → stages → COMPLETED
   → dashboard URL. Run `./run.sh` again: "Reusing completed run … (no recomputation)".
2. **Sidebar:** run ID, dataset hash, pipeline version, config hash, artifact check.
3. **Overview / Charts:** headline metrics, priority split, anomaly types, risk histogram.
4. **Alerts:** filter to MEDIUM, sort by risk score, read the `reasons`.
5. **Investigation:** look up a txid, source IP or wallet (prefix search).
6. **Graph Explorer:** render the 2-hop neighborhood of that IP; trace
   `IP → TX → Wallet → TX → Wallet`.
7. **ML/Evaluation** and **Data Quality** tabs.
8. **Forensic Blockchain:** pick a multi-transaction case → Verify (VERIFIED) → run the
   tamper script (§7) → Verify (FAILED, names the txid) → Merkle proof no longer
   reconstructs the root → reset + `python run.py` → VERIFIED.

## 21. Requirement → implementation mapping

| Requirement | File(s) |
|---|---|
| Ingestion (JSON/JSONL/CSV/XML, streaming) | `src/ingestion/loader.py`, `src/ingestion/staging.py` |
| Correlation | `src/correlation/correlate.py` |
| Graph (index + NetworkX/PyVis view) | `src/graph/store.py`, `src/graph/build_graph.py` |
| Features | `src/features/engineer.py` |
| ML (Isolation Forest) | `src/models/anomaly.py` |
| Clustering (DBSCAN + candidate funnel) | `src/models/clustering.py` |
| Risk scoring | `src/scoring/risk_score.py` |
| Explainability | `src/explainability/explain.py` |
| GeoIP/ASN (offline) | `src/geoip/lookup.py` |
| Dashboard | `dashboard/app.py`, `src/serving/run_store.py` |
| One-command launch | `run.sh`, `src/launcher.py` |
| Versioned, reproducible runs | `src/runs/`, `src/version.py`, `src/pipeline.py` |
| Investigation cases | `src/forensics/cases.py` |
| Merkle evidence + proofs | `src/forensics/canonical.py`, `src/forensics/merkle.py`, `src/forensics/evidence.py` |
| Local/private ledger + validation | `src/forensics/blockchain.py`, `src/forensics/anchoring.py` |
| Tamper detection, never silently fixed | `src/forensics/verification.py`, `src/forensics/restore.py`, `scripts/` |
| Offline deployment | `scripts/build_offline_bundle.sh`, `scripts/install_offline.sh`, `Dockerfile` |
| Scalability benchmark | `scripts/benchmark_scalability.py` |
| Tests | `tests/` (`pytest -q`) |
