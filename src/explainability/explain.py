"""
Explainability layer: generates real, dynamically computed reasons for each alert
based on how far the transaction's actual feature values deviate from the dataset
mean (z-score), plus direct signals from the ML model and clustering result.

No reason is ever fabricated: every string is built from an actual computed number
that is also shown to the user.

SHAP is intentionally NOT a hard dependency (optional import). If unavailable, or if
it errors on a given feature matrix, the pipeline falls back silently to the
z-score-based explanations below so the project never breaks over it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FEATURE_REASON_TEMPLATES = {
    "fan_out": "high fan-out ({value:.0f} output addresses)",
    "fan_in": "high fan-in / consolidation ({value:.0f} input addresses)",
    "ip_tx_count": "unusually high transaction count from this IP ({value:.0f} transactions)",
    "ip_burst_count": "transaction burst detected ({value:.0f} transactions in the configured time window)",
    "ip_unique_wallets": "many distinct wallets touched by this IP ({value:.0f} wallets)",
    "ip_unique_countries": "activity spans multiple countries ({value:.0f} countries)",
    "ip_degree": "high graph degree / network centrality ({value:.0f} connections)",
    "ip_betweenness": "high betweenness centrality — potential relay/bridge node ({value:.3f})",
    "ip_pagerank": "high PageRank — well-connected node in the transaction network ({value:.4f})",
    "amount_variance": "unusual amount variance across inputs/outputs ({value:.4f})",
    "avg_amount": "unusually large average transaction amount ({value:.4f} BTC)",
    "ip_avg_time_gap_seconds": "unusually short average time gap between transactions ({value:.1f}s)",
    "ip_wallet_ratio": "high wallet-per-transaction ratio, suggestive of wallet cycling ({value:.2f})",
}

# For these features, LOW values (short time gaps) are the anomalous direction, not high.
LOW_IS_ANOMALOUS = {"ip_avg_time_gap_seconds"}


def compute_zscores(features: pd.DataFrame) -> pd.DataFrame:
    mean = features.mean()
    std = features.std().replace(0, np.nan)
    z = (features - mean) / std
    return z.fillna(0.0)


def generate_reasons_for_row(
    row_features: pd.Series,
    row_z: pd.Series,
    is_anomaly_prediction: bool,
    cluster_label: int,
    anomaly_score_0_100: float,
    top_n: int = 5,
    z_threshold: float = 1.5,
) -> List[str]:
    reasons: List[str] = []

    if is_anomaly_prediction:
        reasons.append(f"flagged as anomalous by Isolation Forest (anomaly score {anomaly_score_0_100:.1f}/100)")

    if cluster_label == -1:
        reasons.append("does not belong to any dense behavioral cluster (DBSCAN noise point)")
    elif cluster_label is not None:
        reasons.append(f"grouped in behavioral cluster #{cluster_label} with similar entities")

    # Rank candidate features by |z-score| and only keep ones that clear the threshold.
    candidates = []
    for feat, template in FEATURE_REASON_TEMPLATES.items():
        if feat not in row_z.index:
            continue
        z_val = row_z[feat]
        anomalous_direction = (z_val <= -z_threshold) if feat in LOW_IS_ANOMALOUS else (z_val >= z_threshold)
        if anomalous_direction:
            candidates.append((abs(z_val), feat, template))

    candidates.sort(key=lambda t: t[0], reverse=True)
    for _, feat, template in candidates[: max(0, top_n - len(reasons))]:
        reasons.append(template.format(value=row_features[feat]))

    if not reasons:
        reasons.append("no individually extreme feature values; flagged primarily by combined model score")

    return reasons


def generate_all_reasons(
    features: pd.DataFrame,
    is_anomaly_prediction: pd.Series,
    cluster_labels: pd.Series,
    anomaly_score_0_100: pd.Series,
    top_n: int = 5,
    z_threshold: float = 1.5,
) -> pd.Series:
    z = compute_zscores(features)
    reasons = []
    for idx in features.index:
        r = generate_reasons_for_row(
            row_features=features.loc[idx],
            row_z=z.loc[idx],
            is_anomaly_prediction=bool(is_anomaly_prediction.loc[idx]),
            cluster_label=int(cluster_labels.loc[idx]) if idx in cluster_labels.index else None,
            anomaly_score_0_100=float(anomaly_score_0_100.loc[idx]),
            top_n=top_n,
            z_threshold=z_threshold,
        )
        reasons.append(" | ".join(r))
    return pd.Series(reasons, index=features.index, name="reasons")


def try_shap_global_importance(model, X: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Optional SHAP-based global feature importance. Returns None on any failure
    (missing package, incompatible model, runtime error) so it never breaks the pipeline."""
    try:
        import shap  # optional dependency

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return dict(zip(X.columns, mean_abs.tolist()))
    except Exception:
        return None
