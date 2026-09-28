"""
End-to-end pipeline (same analytical workflow as the prototype):

  ingest (streaming, per-file cached) -> clean (global dedup) + offline GeoIP
  -> correlate (IP/TX/Wallet/ASN/Country edges + on-disk graph index)
  -> features -> Isolation Forest (train, then chunked inference) -> risk scoring
  -> candidate funnel -> DBSCAN -> explanations -> evaluation -> alerts
  -> investigation cases -> evidence -> Merkle roots -> local/private ledger.

Every invocation is identified by run_key = sha256(dataset SHA-256 | PIPELINE_VERSION |
analytical-config hash). If a COMPLETED run with that key exists it is reused, not
recomputed. Otherwise a brand-new runs/<RUN_ID>/ directory is created under an
inter-process lock, every stage is recorded in runs/<RUN_ID>/manifest.json, and the
run is marked "completed" only after all stages succeeded and all artifacts were
checksummed. Failed/interrupted runs are marked "failed" and never shown by the
dashboard. Completed runs are never overwritten.

Run with:  python run.py            (or ./run.sh, which also starts the dashboard)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from src.config import get_dataset_path, get_paths, get_runs_root, load_config
from src.correlation.correlate import build_edges_frame
from src.explainability.explain import generate_all_reasons
from src.features.engineer import (_ip_behavioral_features, _ip_graph_features,
                                   _ip_graph_features_from_store, _txn_level_features,
                                   assemble_feature_matrix, ip_wallet_pairs)
from src.forensics.anchoring import anchor_cases
from src.forensics.blockchain import Blockchain
from src.forensics.cases import build_cases
from src.forensics.evidence import EvidenceStore
from src.forensics.restore import restore_missing_evidence
from src.geoip.lookup import OfflineGeoIPResolver, enrich_geo_asn
from src.graph.build_graph import build_entity_graph_from_frame
from src.graph.store import GraphStore
from src.ingestion.staging import (build_cleaned_dataset, cleanup_stale_temp_dirs,
                                   detect_dataset_timestamp_format, stage_source_file)
from src.models.anomaly import save_model, score_isolation_forest, train_isolation_forest
from src.models.clustering import (cluster_summary, run_dbscan, run_dbscan_funnel,
                                   select_cluster_candidates)
from src.models.evaluate import evaluate_against_synthetic_label
from src.runs.fsutil import atomic_write_json
from src.runs.identity import (analytical_config, compute_run_key, config_hash, dataset_fingerprint,
                               geoip_fingerprint)
from src.runs.manifest import RunManifest, verify_artifacts
from src.runs.registry import RunRegistry
from src.scoring.risk_score import compute_risk_scores
from src.storage.parquet_store import ROW_NO, TX_FIELDS, iter_fragments, lists_to_python, read_frame, write_table
from src.version import PIPELINE_VERSION, code_fingerprint

logger = logging.getLogger("pipeline")

STAGES = [
    "ingest", "clean", "correlate", "features", "anomaly_model", "anomaly_scoring",
    "risk_scoring", "clustering", "explanations", "evaluation", "alerts", "cases",
    "forensics", "finalize",
]

# Relative artifact layout inside runs/<RUN_ID>/ (mirrors the old outputs/ names).
CLEANED_DIR = "data/cleaned"
EDGES_DIR = "edges"
ALERTS_DIR = "alerts"
GRAPH_INDEX = "graph_index.sqlite"
GRAPH_PICKLE = "entity_graph.pickle"
MODEL_FILE = "models/isolation_forest.joblib"

CASE_COLUMNS = ["txid", "src_ip", "dst_ip", "input_addresses", "output_addresses", "geo_country",
                "asn", "cluster_label", "anomaly_score_0_100", "risk_score", "priority", "anomaly_type",
                "reasons", "timestamp_parsed", "timestamp_str", "is_anomaly_prediction", ROW_NO]

_TX_TYPES = dict(TX_FIELDS)
_PACKAGES = ["pandas", "numpy", "scikit-learn", "scipy", "networkx", "pyarrow", "pyvis",
             "streamlit", "plotly", "PyYAML"]


class _RunLog:
    """Mirror pipeline log lines to stdout (as before) and to runs/<RUN_ID>/pipeline.log."""

    def __init__(self):
        self.fh = None

    def open(self, path: Path):
        self.fh = open(path, "a", encoding="utf-8")

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None

    def __call__(self, msg: str):
        print(f"[pipeline] {msg}", flush=True)
        if self.fh:
            self.fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
            self.fh.flush()


log = _RunLog()


# =========================================================================== #
# Settings resolution ("auto" = exact prototype behavior on small datasets)
# =========================================================================== #
def resolve_settings(cfg: Dict[str, Any], n_rows: int) -> Dict[str, Any]:
    exact_max = int((cfg.get("scale") or {}).get("exact_mode_max_rows", 20000))
    large = n_rows > exact_max
    graph_cfg = cfg.get("graph") or {}
    clus_cfg = cfg.get("clustering") or {}
    cases_cfg = cfg.get("cases") or {}

    backend = graph_cfg.get("feature_backend", "auto")
    write_pickle = graph_cfg.get("write_networkx_pickle", "auto")
    scope = clus_cfg.get("scope", "auto")
    singleton = cases_cfg.get("singleton_min_priority", "auto")
    if singleton == "auto":
        singleton = "HIGH" if large else None
    elif singleton in (None, "none", "None", ""):
        singleton = None
    return {
        "n_rows": n_rows,
        "exact_mode_max_rows": exact_max,
        "large_dataset_mode": large,
        "graph_feature_backend": backend,          # auto resolved at feature time (needs the graph index)
        "write_networkx_pickle": (not large) if write_pickle == "auto" else bool(write_pickle),
        "clustering_scope": ("candidates" if large else "all") if scope == "auto" else scope,
        "clustering_max_rows": clus_cfg.get("max_rows"),
        "singleton_min_priority": singleton,
        "max_link_pairs_per_group": cases_cfg.get("max_link_pairs_per_group"),
        "chunk_rows": int((cfg.get("runtime") or {}).get("chunk_rows", 100_000)),
        "sqlite_cache_mb": int((cfg.get("runtime") or {}).get("sqlite_cache_mb", 256)),
    }


def _environment() -> Dict[str, Any]:
    from importlib import metadata
    pkgs = {}
    for p in _PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    return {"python": platform.python_version(), "platform": platform.platform(terse=True), "packages": pkgs}


def _dataset_files(dataset_path: Path, fp: Dict[str, Any]) -> List[Path]:
    if fp["kind"] == "file":
        return [dataset_path]
    return [dataset_path / e["name"] for e in fp["files"]]


def _table_with_schema(df: pd.DataFrame, schema: Optional[pa.Schema]) -> tuple[pa.Table, pa.Schema]:
    t = pa.Table.from_pandas(df, preserve_index=False)
    if schema is None:
        fields = []
        for f in t.schema:
            if f.name in _TX_TYPES:
                fields.append(pa.field(f.name, _TX_TYPES[f.name]))
            elif pa.types.is_large_string(f.type) or pa.types.is_string(f.type) or pa.types.is_null(f.type):
                fields.append(pa.field(f.name, pa.string()))
            else:
                fields.append(f)
        schema = pa.schema(fields)
    return t.cast(schema), schema


# =========================================================================== #
# Stage implementations
# =========================================================================== #
def _stage_ingest(cfg, settings, dataset_path, fp, cache_root) -> List[Dict[str, Any]]:
    ing = cfg["ingestion"]
    files = _dataset_files(dataset_path, fp)
    cleanup_stale_temp_dirs(cache_root)
    ts_format = detect_dataset_timestamp_format(files[0])
    entries = []
    for path, meta in zip(files, fp["files"]):
        e = stage_source_file(path, meta["sha256"], cache_root, ing["required_fields"],
                              ing.get("duplicate_subset") or ["txid"], ts_format, settings["chunk_rows"])
        log(f"  {'reused cached staging' if e['cache_hit'] else 'staged'} {meta['name']}: {e['n_rows']} records")
        entries.append(e)
    return entries


def _stage_clean(cfg, run_dir, entries):
    ing = cfg["ingestion"]
    paths = get_paths(cfg)
    resolver = OfflineGeoIPResolver(geoip_dir=paths["geoip_dir"],
                                    country_db_filename=cfg["geoip"]["country_db_filename"],
                                    asn_db_filename=cfg["geoip"]["asn_db_filename"])
    try:
        report, info = build_cleaned_dataset(
            entries, run_dir / CLEANED_DIR, ing.get("duplicate_subset") or ["txid"],
            ing.get("drop_exact_duplicates", True),
            enrich_fn=(lambda df: enrich_geo_asn(df, resolver)) if resolver.available else None)
    finally:
        resolver.close()
    atomic_write_json(run_dir / "data_quality.json", report.to_dict())
    return report, info, resolver.available


def _iter_edge_frames(run_dir: Path):
    cleaned = run_dir / CLEANED_DIR
    cols = ["txid", "src_ip", "asn", "geo_country", "timestamp_parsed", "src_port", "dst_port",
            "input_addresses", "output_addresses", "input_amounts", "output_amounts", ROW_NO]
    schema = None
    for frag in iter_fragments(cleaned):
        df = lists_to_python(read_frame(frag, columns=cols, sort_by_row=False))
        ef = build_edges_frame(df)
        table, schema = _table_with_schema(ef, schema)
        write_table(table, run_dir / EDGES_DIR / frag.relative_to(cleaned))
        yield ef


def _stage_correlate(run_dir, settings):
    GraphStore.build(run_dir / GRAPH_INDEX, _iter_edge_frames(run_dir), cache_mb=settings["sqlite_cache_mb"])
    store = GraphStore(run_dir / GRAPH_INDEX)
    try:
        summary = store.summary()
    finally:
        store.close()
    atomic_write_json(run_dir / "graph_summary.json", summary)
    return summary


def _stage_features(cfg, settings, run_dir):
    fcfg = cfg["features"]
    cols = ["txid", "src_ip", "geo_country", "timestamp_parsed", "fee", "synthetic_label",
            "input_addresses", "output_addresses", "input_amounts", "output_amounts", ROW_NO]
    txn_parts, scalar_parts, pair_parts = [], [], []
    for frag in iter_fragments(run_dir / CLEANED_DIR):
        df = lists_to_python(read_frame(frag, columns=cols, sort_by_row=False))
        t = _txn_level_features(df)
        t[ROW_NO] = df[ROW_NO].to_numpy()
        txn_parts.append(t)
        pair_parts.append(ip_wallet_pairs(df))
        scalar_parts.append(df[["txid", "src_ip", "geo_country", "timestamp_parsed", "synthetic_label", ROW_NO]])

    # Original input order, independent of partitioning.
    scalar = pd.concat(scalar_parts, ignore_index=True).sort_values(ROW_NO, kind="stable").reset_index(drop=True)
    txn = (pd.concat(txn_parts, ignore_index=True).sort_values(ROW_NO, kind="stable")
           .reset_index(drop=True).drop(columns=[ROW_NO]))
    pairs = pd.concat(pair_parts, ignore_index=True).drop_duplicates(ignore_index=True)
    del txn_parts, scalar_parts, pair_parts

    ip_feats = _ip_behavioral_features(scalar, burst_window_hours=fcfg["burst_window_hours"], wallet_pairs=pairs)
    del pairs

    backend = settings["graph_feature_backend"]
    used = backend
    graph_feats = None
    if backend in ("auto", "edge_table"):
        store = GraphStore(run_dir / GRAPH_INDEX)
        try:
            graph_feats = _ip_graph_features_from_store(scalar, store, fcfg["compute_betweenness"])
            used = "edge_table"
        except NotImplementedError as e:
            if backend == "edge_table" or settings["large_dataset_mode"]:
                raise
            log(f"  edge_table graph backend not applicable ({e}); falling back to networkx")
        finally:
            store.close()
    g = None
    if graph_feats is None or settings["write_networkx_pickle"]:
        edges = read_frame(run_dir / EDGES_DIR, sort_by_row=False).sort_values([ROW_NO, "_seq"], kind="stable")
        g = build_entity_graph_from_frame(edges)
        del edges
        if graph_feats is None:
            graph_feats = _ip_graph_features(scalar, g, compute_betweenness=fcfg["compute_betweenness"],
                                             betweenness_k_sample=fcfg["betweenness_k_sample"])
            used = "networkx"
        if settings["write_networkx_pickle"]:
            with open(run_dir / GRAPH_PICKLE, "wb") as f:
                pickle.dump(g, f)
    features = assemble_feature_matrix(txn, ip_feats, graph_feats)
    return scalar, features, used


def _stage_clustering(cfg, settings, features, risk_df, ml_scores):
    db_cfg = cfg["model"]["dbscan"]
    cases_cfg = cfg.get("cases") or {}
    candidates = risk_df["priority"].isin(cases_cfg.get("seed_priorities", ["HIGH", "MEDIUM"]))
    if cases_cfg.get("include_ml_flagged", True):
        candidates = candidates | ml_scores["is_anomaly_prediction"].astype(bool)
    funnel = {"n_scored": int(len(features)), "n_candidates": int(candidates.sum())}

    if settings["clustering_scope"] == "all":
        labels = run_dbscan(features, eps=db_cfg["eps"], min_samples=db_cfg["min_samples"])
        funnel["n_clustered"] = int(len(features))
    else:
        mask = select_cluster_candidates(candidates, risk_df["risk_score"], settings["clustering_max_rows"])
        labels = run_dbscan_funnel(features, mask, eps=db_cfg["eps"], min_samples=db_cfg["min_samples"])
        funnel["n_clustered"] = int(mask.sum())
    summary = cluster_summary(labels)
    summary["scope"] = settings["clustering_scope"]
    return labels, summary, funnel


def _stage_alerts(run_dir, results: pd.DataFrame):
    cleaned = run_dir / CLEANED_DIR
    schema = None
    n = 0
    for frag in iter_fragments(cleaned):
        base = read_frame(frag, sort_by_row=False)
        base = base[[c for c in base.columns if c != "date"]]
        row_no = base.pop(ROW_NO)
        merged = pd.concat([base.reset_index(drop=True),
                            results.loc[row_no.to_numpy()].reset_index(drop=True)], axis=1)
        merged["entity"] = merged["src_ip"]
        merged["timestamp_str"] = merged["timestamp_parsed"].astype(str)
        merged[ROW_NO] = row_no.to_numpy()
        table, schema = _table_with_schema(merged, schema)
        write_table(table, run_dir / ALERTS_DIR / frag.relative_to(cleaned))
        n += len(merged)
    return n


def _seed_filter(cases_cfg):
    expr = pads.field("priority").isin(cases_cfg.get("seed_priorities", ["HIGH", "MEDIUM"]))
    if cases_cfg.get("include_ml_flagged", True):
        expr = expr | (pads.field("is_anomaly_prediction") == True)  # noqa: E712
    return expr


# =========================================================================== #
# Orchestration
# =========================================================================== #
def run_pipeline(input_dataset: str | os.PathLike | None = None, config_path: str | None = None,
                 runs_root: str | os.PathLike | None = None, force: bool = False,
                 verify_deep: bool = False) -> Dict[str, Any]:
    """Ensure a completed run exists for (dataset, PIPELINE_VERSION, analytical config).

    Returns {"run_id", "run_dir", "reused", "manifest"}. Raises on failure (the run is
    then recorded as failed and is never used by the dashboard)."""
    cfg = load_config(config_path)
    paths = get_paths(cfg)
    dataset_path = get_dataset_path(cfg, input_dataset)
    registry = RunRegistry(get_runs_root(cfg, runs_root))

    log(f"Fingerprinting dataset {dataset_path.name} ...")
    fp = dataset_fingerprint(dataset_path)
    geo_fp = geoip_fingerprint(paths["geoip_dir"], cfg["geoip"]["country_db_filename"],
                               cfg["geoip"]["asn_db_filename"])
    cfg_hash = config_hash(cfg, geo_fp)
    run_key = compute_run_key(fp["sha256"], PIPELINE_VERSION, cfg_hash)
    log(f"dataset sha256={fp['sha256'][:16]}...  pipeline={PIPELINE_VERSION}  config={cfg_hash[:12]}  "
        f"run_key={run_key}")

    for rid in registry.reap_stale_runs():
        log(f"Marked interrupted run {rid} as failed.")

    lock = registry.lock_for(run_key)
    wait = float((cfg.get("runs") or {}).get("lock_wait_seconds", 3600))

    def _reuse(m: RunManifest) -> Optional[Dict[str, Any]]:
        check = verify_artifacts(m.run_dir, m, deep=verify_deep)
        if not check["ok"]:
            log(f"Completed run {m.run_id} failed artifact verification ({check['problems'][:3]}); "
                f"it is left untouched and a NEW run will be produced.")
            return None
        if m.data.get("code_fingerprint") != code_fingerprint():
            log(f"NOTE: run {m.run_id} was produced by different pipeline code under the same "
                f"PIPELINE_VERSION {PIPELINE_VERSION}; reusing it. Use --force to recompute.")
        run_lock = registry.lock_for(f"run-{m.run_id}")
        if run_lock.acquire(wait_seconds=wait):
            try:
                res = _restore(m)
            finally:
                run_lock.release()
            if res["restored"]:
                log(f"Restored explicitly-reset evidence for {len(res['restored'])} case(s): {res['restored']}")
            if res["refused"]:
                log(f"WARNING: refused to restore {len(res['refused'])} case(s) whose re-derived evidence "
                    f"no longer matches the anchored Merkle root: {res['refused']}")
        registry.set_latest(m.run_id)
        log(f"Reusing completed run {m.run_id} (no recomputation).")
        return {"run_id": m.run_id, "run_dir": str(m.run_dir), "reused": True, "manifest": m.data}

    if not force:
        existing = registry.find_completed(run_key)
        if existing is not None:
            reused = _reuse(existing)
            if reused:
                return reused

    if not lock.acquire(wait_seconds=wait,
                        on_wait=lambda o: log(f"Another pipeline (pid {o.get('pid')}) is producing this "
                                              f"run; waiting up to {int(wait)}s ...")):
        raise RuntimeError(f"Timed out waiting for the run lock for run_key {run_key}")
    try:
        if not force:  # another process may have completed it while we waited
            existing = registry.find_completed(run_key)
            if existing is not None:
                reused = _reuse(existing)
                if reused:
                    return reused
        return _execute(cfg, paths, dataset_path, fp, geo_fp, cfg_hash, run_key, registry)
    finally:
        lock.release()


def _restore(m: RunManifest) -> Dict[str, Any]:
    forensics = m.data.get("forensics") or {}
    return restore_missing_evidence(
        m.run_dir,
        m.run_dir / forensics.get("evidence_store", "forensics/evidence_store.json"),
        m.run_dir / forensics.get("ledger", "forensics/blockchain_ledger.json"),
        m.run_dir / ALERTS_DIR,
        m.run_id,
    )


def _execute(cfg, paths, dataset_path, fp, geo_fp, cfg_hash, run_key, registry) -> Dict[str, Any]:
    t0 = time.time()
    run_id, run_dir = registry.new_run_dir(run_key)
    manifest = RunManifest.create(
        run_dir, run_id=run_id, run_key=run_key, pipeline_version=PIPELINE_VERSION,
        code_fingerprint=code_fingerprint(), config_hash=cfg_hash,
        dataset={k: fp[k] for k in ("sha256", "display_name", "kind", "files", "total_bytes")},
        config=analytical_config(cfg), geoip_db=geo_fp, environment=_environment(),
    )
    log.open(run_dir / "pipeline.log")
    log(f"Started run {run_id}")
    forensics_cfg = cfg.get("forensics") or {}
    ev_rel = forensics_cfg.get("evidence_store_filename", "forensics/evidence_store.json")
    ledger_rel = forensics_cfg.get("blockchain_filename", "forensics/blockchain_ledger.json")
    try:
        # ------------------------------------------------------------ 1. INGEST
        with manifest.stage("ingest") as st:
            settings = resolve_settings(cfg, 0)
            cache_root = paths["processed_dir"] / "ingest_cache"
            entries = _stage_ingest(cfg, settings, dataset_path, fp, cache_root)
            st.update(n_files=len(entries), cache_hits=sum(e["cache_hit"] for e in entries),
                      n_raw_records=sum(e["n_rows"] for e in entries))
            log(f"Loaded {st['n_raw_records']} raw records from {len(entries)} file(s) "
                f"({st['cache_hits']} cached).")

        # ------------------------------------------------------------- 2. CLEAN
        with manifest.stage("clean") as st:
            dq_report, clean_info, geo_used = _stage_clean(cfg, run_dir, entries)
            st.update(**clean_info, geoip_offline_db_used=bool(geo_used))
            log(f"Valid: {dq_report.valid_records} / {dq_report.total_records} "
                f"(invalid: {dq_report.invalid_records}, duplicates: {dq_report.duplicate_records}); "
                f"GeoIP offline DB in use: {geo_used}")
            if dq_report.valid_records == 0:
                raise RuntimeError("No valid records survived cleaning — check the input dataset / "
                                   "required_fields config.")
        settings = resolve_settings(cfg, dq_report.valid_records)
        manifest.data["resolved_settings"] = settings
        manifest.data["record_counts"] = {
            "raw": dq_report.total_records, "valid": dq_report.valid_records,
            "invalid": dq_report.invalid_records, "duplicates": dq_report.duplicate_records}
        manifest.save()

        # --------------------------------------------------------- 3. CORRELATE
        with manifest.stage("correlate") as st:
            g_summary = _stage_correlate(run_dir, settings)
            st.update(num_nodes=g_summary["num_nodes"], num_edges=g_summary["num_edges"])
            log(f"Graph index: {g_summary['num_nodes']} nodes, {g_summary['num_edges']} edges. "
                f"Types: {g_summary['node_type_counts']}")

        # ---------------------------------------------------------- 4. FEATURES
        with manifest.stage("features") as st:
            scalar, features, backend_used = _stage_features(cfg, settings, run_dir)
            st.update(shape=list(features.shape), graph_feature_backend=backend_used)
            settings["graph_feature_backend"] = backend_used
            log(f"Feature matrix shape: {features.shape} (graph features: {backend_used} backend)")

        # --------------------------------------------------------------- 5. ML
        if_cfg = cfg["model"]["isolation_forest"]
        with manifest.stage("anomaly_model") as st:
            log("Training Isolation Forest (unsupervised; synthetic_label NOT used) ...")
            bundle = train_isolation_forest(
                features, n_estimators=if_cfg["n_estimators"], contamination=if_cfg["contamination"],
                random_state=if_cfg["random_state"], max_samples=if_cfg["max_samples"],
                max_train_rows=if_cfg.get("max_train_rows"))
            (run_dir / MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
            save_model(bundle, str(run_dir / MODEL_FILE))
            n_train = min(len(features), if_cfg.get("max_train_rows") or len(features))
            st.update(n_train_rows=int(n_train))

        with manifest.stage("anomaly_scoring") as st:
            ml_scores = score_isolation_forest(bundle, features, chunk_rows=settings["chunk_rows"])
            st.update(n_flagged=int(ml_scores["is_anomaly_prediction"].sum()))
            log(f"Flagged {st['n_flagged']} / {len(ml_scores)} as anomalous.")

        with manifest.stage("risk_scoring"):
            risk_df = compute_risk_scores(features, anomaly_score_0_100=ml_scores["anomaly_score_0_100"],
                                          weights=cfg["scoring"]["weights"], thresholds=cfg["scoring"]["thresholds"])

        # -------------------------------------------------------- 6. CLUSTERING
        with manifest.stage("clustering") as st:
            cluster_labels, c_summary, funnel = _stage_clustering(cfg, settings, features, risk_df, ml_scores)
            atomic_write_json(run_dir / "cluster_summary.json", c_summary)
            st.update(scope=c_summary["scope"], **funnel)
            log(f"Clustering ({c_summary['scope']}, {funnel['n_clustered']} rows): "
                f"{c_summary['n_clusters']} clusters, noise points: {c_summary['n_noise_points']}")

        # ------------------------------------------------------ 7. EXPLANATIONS
        with manifest.stage("explanations"):
            exp_cfg = cfg["explainability"]
            reasons = generate_all_reasons(
                features, is_anomaly_prediction=ml_scores["is_anomaly_prediction"],
                cluster_labels=cluster_labels, anomaly_score_0_100=ml_scores["anomaly_score_0_100"],
                top_n=exp_cfg["top_n_reasons"], z_threshold=exp_cfg["z_score_reason_threshold"])

        # -------------------------------------------------------- 8. EVALUATION
        with manifest.stage("evaluation"):
            eval_metrics = evaluate_against_synthetic_label(
                synthetic_label=scalar["synthetic_label"].fillna(0),
                is_anomaly_prediction=ml_scores["is_anomaly_prediction"],
                anomaly_score_0_100=ml_scores["anomaly_score_0_100"])
            atomic_write_json(run_dir / "evaluation_metrics.json", eval_metrics)
            log(f"Precision={eval_metrics['precision']:.3f} Recall={eval_metrics['recall']:.3f} "
                f"F1={eval_metrics['f1']:.3f} ROC-AUC={eval_metrics['roc_auc']}")

        # ------------------------------------------------------------ 9. ALERTS
        with manifest.stage("alerts") as st:
            results = pd.concat([
                features.drop(columns=["fee"]).add_prefix("feat_").reset_index(drop=True),
                ml_scores.reset_index(drop=True),
                risk_df.reset_index(drop=True),
                cluster_labels.reset_index(drop=True).rename("cluster_label"),
                reasons.reset_index(drop=True),
            ], axis=1)
            results.index = scalar[ROW_NO].to_numpy()
            n_alerts = _stage_alerts(run_dir, results)
            if n_alerts != len(features):
                raise RuntimeError(f"alerts row count mismatch ({n_alerts} != {len(features)})")
            st.update(n_alerts=n_alerts,
                      priority_counts={k: int(v) for k, v in risk_df["priority"].value_counts().items()})
            del results

        # ------------------------------------------------------------- 10. CASES
        cases_cfg = cfg.get("cases") or {}
        with manifest.stage("cases") as st:
            seed_alerts = lists_to_python(read_frame(run_dir / ALERTS_DIR, columns=CASE_COLUMNS,
                                                     filter=_seed_filter(cases_cfg)),
                                          cols=("input_addresses", "output_addresses"))
            cases, _ = build_cases(
                seed_alerts, None,
                seed_priorities=cases_cfg.get("seed_priorities", ["HIGH", "MEDIUM"]),
                include_ml_flagged=cases_cfg.get("include_ml_flagged", True),
                temporal_window_hours=cases_cfg.get("temporal_window_hours", 2),
                min_case_size=cases_cfg.get("min_case_size", 1),
                max_link_pairs_per_group=settings["max_link_pairs_per_group"],
                singleton_min_priority=settings["singleton_min_priority"])
            atomic_write_json(run_dir / "cases.json", [c.to_dict() for c in cases])
            funnel.update(n_case_seeds=int(len(seed_alerts)), n_cases=len(cases),
                          n_case_transactions=int(sum(len(c.linked_txids) for c in cases)))
            st.update(n_seeds=int(len(seed_alerts)), n_cases=len(cases))
            log(f"Built {len(cases)} investigation case(s) from {funnel['n_case_transactions']} linked "
                f"transaction(s) ({len(seed_alerts)} seed alerts).")

        # --------------------------------------------------------- 11. FORENSICS
        with manifest.stage("forensics") as st:
            # Evidence is built from the alerts table as read back from disk -- the same
            # source restore_missing_evidence() uses -- so a later restore hashes identically.
            store = EvidenceStore(run_dir / ev_rel)
            chain = Blockchain(run_dir / ledger_rel, run_id=run_id)
            anchor_summary = anchor_cases(cases, seed_alerts, store, chain)
            validation = chain.validate_chain(expected_run_id=run_id)
            if not validation["valid"]:
                raise RuntimeError(f"forensic ledger failed validation: {validation['errors'][:5]}")
            manifest.data["forensics"] = {
                "evidence_store": ev_rel, "ledger": ledger_rel, "n_cases": len(cases),
                "ledger_length": validation["length"], "ledger_tip_hash": validation["tip_hash"],
                "ledger_genesis_hash": chain.blocks[0].hash, "ledger_run_id": validation["run_id"]}
            st.update(n_new_blocks=anchor_summary["n_new_blocks"], ledger_length=validation["length"])
            log(f"Forensic anchoring: {anchor_summary['n_new_blocks']} new block(s); ledger length "
                f"{validation['length']}, valid={validation['valid']} (bound to run {run_id}).")

        # ---------------------------------------------------------- 12. FINALIZE
        with manifest.stage("finalize"):
            manifest.data["funnel"] = funnel
            meta = {
                "run_id": run_id,
                "pipeline_version": PIPELINE_VERSION,
                "dataset": fp["display_name"],
                "dataset_sha256": fp["sha256"],
                "elapsed_seconds": time.time() - t0,
                "n_raw_records": int(dq_report.total_records),
                "n_clean_records": int(dq_report.valid_records),
                "geoip_offline_db_used": bool(geo_used),
                # paths are relative to the run directory -- no absolute paths are exposed
                "alerts_path": ALERTS_DIR,
                "processed_path": CLEANED_DIR,
                "edges_path": EDGES_DIR,
                "graph_index_path": GRAPH_INDEX,
                "graph_path": GRAPH_PICKLE if settings["write_networkx_pickle"] else None,
                "feature_columns": list(features.columns),
                "n_cases": len(cases),
                "funnel": funnel,
                "resolved_settings": settings,
                "forensic_anchoring": anchor_summary,
                "forensic_chain_valid": validation["valid"],
                "forensic_chain_length": validation["length"],
                "evidence_store_path": ev_rel,
                "blockchain_path": ledger_rel,
                "cases_summary_path": "cases.json",
            }
            atomic_write_json(run_dir / "run_metadata.json", meta)
        manifest.data["elapsed_seconds"] = round(time.time() - t0, 2)
        manifest.mark_completed(STAGES)
        registry.set_latest(run_id)
        log(f"Pipeline complete in {time.time() - t0:.1f}s. Run {run_id} marked COMPLETED.")
        return {"run_id": run_id, "run_dir": str(run_dir), "reused": False, "manifest": manifest.data}
    except BaseException as e:
        manifest.mark_failed(e)
        log(f"Run {run_id} FAILED: {type(e).__name__}: {e} (details: {run_id}/error.log)")
        raise
    finally:
        log.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run (or reuse) the TRACE-X forensic pipeline.")
    ap.add_argument("--input", help="dataset file or directory (default: paths.raw_data / BTC_MONITOR_DATASET)")
    ap.add_argument("--config", help="config.yaml path (default: BTC_MONITOR_CONFIG or ./config.yaml)")
    ap.add_argument("--runs-root", help="runs directory (default: runs.root / BTC_MONITOR_RUNS_ROOT)")
    ap.add_argument("--force", action="store_true",
                    help="produce a NEW run even if an identical completed run exists (never overwrites it)")
    ap.add_argument("--verify-deep", action="store_true",
                    help="SHA-256-verify every artifact of a reused run (default: existence + size)")
    args = ap.parse_args(argv)
    result = run_pipeline(args.input, args.config, args.runs_root, args.force, args.verify_deep)
    print(json.dumps({"run_id": result["run_id"], "reused": result["reused"],
                      "status": result["manifest"]["status"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
