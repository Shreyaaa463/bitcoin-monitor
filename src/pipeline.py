"""
End-to-end pipeline:
  ingestion -> preprocessing -> geoip enrichment -> correlation -> graph ->
  features -> ML (Isolation Forest) -> clustering (DBSCAN) -> risk scoring ->
  explainability -> evaluation -> cached outputs for the dashboard.

Run with:  python run.py
All outputs are cached to disk (data/processed/, models/, outputs/) so that
`streamlit run dashboard/app.py` never needs to retrain anything — it just loads
the cached artifacts. Re-run `python run.py` to regenerate everything from scratch.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


from src.config import load_config, get_paths, ensure_dirs, PROJECT_ROOT
from src.ingestion.loader import load_raw_data
from src.preprocessing.clean import clean_dataframe
from src.geoip.lookup import OfflineGeoIPResolver, enrich_geo_asn
from src.correlation.correlate import build_edges, edges_to_dataframe
from src.graph.build_graph import build_entity_graph, graph_summary
from src.features.engineer import build_feature_matrix
from src.models.anomaly import train_isolation_forest, score_isolation_forest, save_model
from src.models.clustering import run_dbscan, cluster_summary
from src.models.evaluate import evaluate_against_synthetic_label
from src.scoring.risk_score import compute_risk_scores
from src.explainability.explain import generate_all_reasons
from src.forensics.cases import build_cases
from src.forensics.evidence import EvidenceStore
from src.forensics.blockchain import Blockchain
from src.forensics.anchoring import anchor_cases


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def run_pipeline():
    t0 = time.time()
    cfg = load_config()
    paths = get_paths()
    ensure_dirs()

    # ---------------------------------------------------------------- 1. INGEST
    log(f"Loading raw data from {paths['raw_data']} ...")
    df_raw = load_raw_data(paths["raw_data"])
    log(f"Loaded {len(df_raw)} raw records.")

    # ------------------------------------------------------------ 2. PREPROCESS
    log("Cleaning and validating records ...")
    df_clean, dq_report = clean_dataframe(
        df_raw,
        required_fields=cfg["ingestion"]["required_fields"],
        drop_exact_duplicates=cfg["ingestion"]["drop_exact_duplicates"],
        duplicate_subset=cfg["ingestion"]["duplicate_subset"],
    )
    log(f"Valid: {dq_report.valid_records} / {dq_report.total_records} "
        f"(invalid: {dq_report.invalid_records}, duplicates: {dq_report.duplicate_records})")

    if len(df_clean) == 0:
        raise RuntimeError("No valid records survived cleaning — check the input dataset / required_fields config.")

    # -------------------------------------------------------------- 3. GEOIP
    log("Attempting offline GeoIP/ASN enrichment (falls back to dataset fields if unavailable) ...")
    resolver = OfflineGeoIPResolver(
        geoip_dir=paths["geoip_dir"],
        country_db_filename=cfg["geoip"]["country_db_filename"],
        asn_db_filename=cfg["geoip"]["asn_db_filename"],
    )
    df_clean = enrich_geo_asn(df_clean, resolver)
    resolver.close()
    log(f"GeoIP offline DB in use: {resolver.available}")

    # ---------------------------------------------------------- 4. CORRELATION
    log("Building entity correlation edges ...")
    edges = build_edges(df_clean)
    log(f"Built {len(edges)} edges.")

    # ---------------------------------------------------------------- 5. GRAPH
    log("Building NetworkX entity graph ...")
    g = build_entity_graph(edges)
    g_summary = graph_summary(g)
    log(f"Graph: {g_summary['num_nodes']} nodes, {g_summary['num_edges']} edges. "
        f"Types: {g_summary['node_type_counts']}")

    # ------------------------------------------------------------- 6. FEATURES
    log("Engineering behavioral + graph features ...")
    features = build_feature_matrix(
        df_clean,
        g,
        burst_window_hours=cfg["features"]["burst_window_hours"],
        compute_betweenness=cfg["features"]["compute_betweenness"],
        betweenness_k_sample=cfg["features"]["betweenness_k_sample"],
    )
    log(f"Feature matrix shape: {features.shape}")

    # ------------------------------------------------------------------ 7. ML
    log("Training Isolation Forest (unsupervised; synthetic_label NOT used) ...")
    if_cfg = cfg["model"]["isolation_forest"]
    bundle = train_isolation_forest(
        features,
        n_estimators=if_cfg["n_estimators"],
        contamination=if_cfg["contamination"],
        random_state=if_cfg["random_state"],
        max_samples=if_cfg["max_samples"],
    )
    ml_scores = score_isolation_forest(bundle, features)
    save_model(bundle, str(paths["models_dir"] / "isolation_forest.joblib"))
    log(f"Flagged {int(ml_scores['is_anomaly_prediction'].sum())} / {len(ml_scores)} as anomalous.")

    # ------------------------------------------------------------ 8. CLUSTERING
    log("Running DBSCAN clustering ...")
    db_cfg = cfg["model"]["dbscan"]
    cluster_labels = run_dbscan(features, eps=db_cfg["eps"], min_samples=db_cfg["min_samples"])
    c_summary = cluster_summary(cluster_labels)
    log(f"Clusters: {c_summary['n_clusters']}, noise points: {c_summary['n_noise_points']}")

    # -------------------------------------------------------------- 9. SCORING
    log("Computing transparent investigative risk scores ...")
    scoring_cfg = cfg["scoring"]
    risk_df = compute_risk_scores(
        features,
        anomaly_score_0_100=ml_scores["anomaly_score_0_100"],
        weights=scoring_cfg["weights"],
        thresholds=scoring_cfg["thresholds"],
    )

    # --------------------------------------------------------- 10. EXPLANATIONS
    log("Generating explainable, feature-grounded alert reasons ...")
    exp_cfg = cfg["explainability"]
    reasons = generate_all_reasons(
        features,
        is_anomaly_prediction=ml_scores["is_anomaly_prediction"],
        cluster_labels=cluster_labels,
        anomaly_score_0_100=ml_scores["anomaly_score_0_100"],
        top_n=exp_cfg["top_n_reasons"],
        z_threshold=exp_cfg["z_score_reason_threshold"],
    )

    # ------------------------------------------------------------ 11. EVALUATION
    log("Evaluating against synthetic_label (evaluation-only; not used for training) ...")
    eval_metrics = evaluate_against_synthetic_label(
        synthetic_label=df_clean["synthetic_label"].fillna(0),
        is_anomaly_prediction=ml_scores["is_anomaly_prediction"],
        anomaly_score_0_100=ml_scores["anomaly_score_0_100"],
    )
    log(f"Precision={eval_metrics['precision']:.3f} Recall={eval_metrics['recall']:.3f} "
        f"F1={eval_metrics['f1']:.3f} ROC-AUC={eval_metrics['roc_auc']}")

    # ------------------------------------------------------------- 12. ASSEMBLE
    log("Assembling final alerts table ...")
    alerts = df_clean.copy().reset_index(drop=True)
    # 'fee' already exists on df_clean; avoid a duplicate column name when concatenating features.
    features_reset = features.drop(columns=["fee"]).reset_index(drop=True)
    features_reset = features_reset.add_prefix("feat_")
    ml_scores_reset = ml_scores.reset_index(drop=True)
    risk_df_reset = risk_df.reset_index(drop=True)
    cluster_labels_reset = cluster_labels.reset_index(drop=True)
    reasons_reset = reasons.reset_index(drop=True)

    alerts = pd.concat([
        alerts,
        features_reset,
        ml_scores_reset,
        risk_df_reset,
        cluster_labels_reset.rename("cluster_label"),
        reasons_reset,
    ], axis=1)
    alerts["entity"] = alerts["src_ip"]
    alerts["timestamp_str"] = alerts["timestamp_parsed"].astype(str)

    # ---------------------------------------------------------------- 13. SAVE
    log("Saving cached outputs for the dashboard ...")
    processed_path = paths["processed_dir"] / "cleaned_transactions.parquet"
    alerts_path = paths["outputs_dir"] / "alerts.parquet"
    edges_path = paths["outputs_dir"] / "edges.parquet"
    graph_path = paths["outputs_dir"] / "entity_graph.pickle"
    dq_path = paths["outputs_dir"] / "data_quality.json"
    graph_summary_path = paths["outputs_dir"] / "graph_summary.json"
    cluster_summary_path = paths["outputs_dir"] / "cluster_summary.json"
    eval_path = paths["outputs_dir"] / "evaluation_metrics.json"
    meta_path = paths["outputs_dir"] / "run_metadata.json"

    try:
        alerts.to_parquet(alerts_path, index=False)
        df_clean.to_parquet(processed_path, index=False)
        edges_to_dataframe(edges).to_parquet(edges_path, index=False)
    except Exception as e:
        # Parquet engine (pyarrow/fastparquet) may not be installed -> fall back to CSV/pickle.
        log(f"Parquet save failed ({e}); falling back to CSV/pickle.")
        alerts_path = alerts_path.with_suffix(".csv")
        processed_path = processed_path.with_suffix(".csv")
        edges_path = edges_path.with_suffix(".csv")
        alerts.to_csv(alerts_path, index=False)
        df_clean.to_csv(processed_path, index=False)
        edges_to_dataframe(edges).to_csv(edges_path, index=False)

    with open(graph_path, "wb") as f:
        pickle.dump(g, f)

    with open(dq_path, "w") as f:
        json.dump(dq_report.to_dict(), f, indent=2)
    with open(graph_summary_path, "w") as f:
        json.dump(g_summary, f, indent=2)
    with open(cluster_summary_path, "w") as f:
        json.dump(c_summary, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # ------------------------------------------------- 14. FORENSIC CASES + CHAIN
    # Reuses the `alerts` table and `g` graph already built above -- no ML, graph,
    # or scoring computation is repeated here.
    log("Building investigation cases from linked alerts (graph + cluster + IP/wallet/time) ...")
    cases_cfg = cfg.get("cases", {})
    cases, link_edges = build_cases(
        alerts,
        g,
        seed_priorities=cases_cfg.get("seed_priorities", ["HIGH", "MEDIUM"]),
        include_ml_flagged=cases_cfg.get("include_ml_flagged", True),
        temporal_window_hours=cases_cfg.get("temporal_window_hours", 2),
        min_case_size=cases_cfg.get("min_case_size", 1),
    )
    log(f"Built {len(cases)} investigation case(s) from "
        f"{sum(len(c.linked_txids) for c in cases)} linked transaction(s).")

    log("Anchoring case evidence into the local forensic blockchain ledger ...")
    forensics_cfg = cfg.get("forensics", {})
    evidence_store_path = paths["outputs_dir"] / forensics_cfg.get(
        "evidence_store_filename", "forensics/evidence_store.json")
    blockchain_path = paths["outputs_dir"] / forensics_cfg.get(
        "blockchain_filename", "forensics/blockchain_ledger.json")

    evidence_store = EvidenceStore(evidence_store_path)
    chain = Blockchain(blockchain_path)
    anchor_summary = anchor_cases(cases, alerts, evidence_store, chain)
    log(f"Forensic anchoring: {anchor_summary['n_new_blocks']} new block(s), "
        f"{anchor_summary['n_updated_blocks']} updated, {anchor_summary['n_unchanged']} unchanged. "
        f"Ledger length: {anchor_summary['chain_length']}.")

    chain_validation = chain.validate_chain()
    log(f"Forensic ledger validation: valid={chain_validation['valid']} "
        f"(length={chain_validation['length']}, errors={len(chain_validation['errors'])})")

    cases_summary_path = paths["outputs_dir"] / "cases.json"
    with open(cases_summary_path, "w") as f:
        json.dump([c.to_dict() for c in cases], f, indent=2)

    elapsed = time.time() - t0
    meta = {
        "elapsed_seconds": elapsed,
        "n_raw_records": int(len(df_raw)),
        "n_clean_records": int(len(df_clean)),
        "geoip_offline_db_used": bool(resolver.available),
        "alerts_path": str(alerts_path),
        "processed_path": str(processed_path),
        "edges_path": str(edges_path),
        "graph_path": str(graph_path),
        "feature_columns": list(features.columns),
        "n_cases": len(cases),
        "forensic_anchoring": anchor_summary,
        "forensic_chain_valid": chain_validation["valid"],
        "forensic_chain_length": chain_validation["length"],
        "evidence_store_path": str(evidence_store_path),
        "blockchain_path": str(blockchain_path),
        "cases_summary_path": str(cases_summary_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Pipeline complete in {elapsed:.1f}s. Outputs written to {paths['outputs_dir']}")
    return meta


if __name__ == "__main__":
    run_pipeline()
