"""
Clustering: DBSCAN over the same standardized feature space used for anomaly
detection, grouping transactions/entities with similar behavioral + network +
graph characteristics. Cluster label -1 (DBSCAN's noise label) is preserved as-is
so the dashboard can distinguish "no cluster / outlier" from a real cluster.

Candidate funnel (large datasets): DBSCAN is NOT run over every transaction.
run_dbscan_funnel() clusters only the candidate set (the same rows that can seed an
investigation case: priority in cases.seed_priorities, or ML-flagged), optionally
capped to the highest-risk max_rows. Rows outside the funnel get label -2
("not clustered"), which is distinct from DBSCAN noise (-1). On small datasets
(scope "all") this is exactly the prototype behavior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

NOT_CLUSTERED = -2


def run_dbscan(X: pd.DataFrame, eps: float = 0.8, min_samples: int = 5) -> pd.Series:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X_scaled)

    return pd.Series(labels, index=X.index, name="cluster_label")


def select_cluster_candidates(candidate_mask: pd.Series, risk_score: pd.Series,
                              max_rows: int | None) -> pd.Series:
    """Candidate rows, keeping only the top-`max_rows` by risk (ties by position) if capped."""
    mask = candidate_mask.astype(bool)
    if max_rows and int(mask.sum()) > max_rows:
        cand = risk_score[mask]
        order = np.lexsort((np.arange(len(cand)), -cand.to_numpy()))
        keep = cand.index[order[:max_rows]]
        mask = pd.Series(False, index=candidate_mask.index)
        mask.loc[keep] = True
    return mask


def run_dbscan_funnel(X: pd.DataFrame, cluster_mask: pd.Series, eps: float = 0.8,
                      min_samples: int = 5) -> pd.Series:
    labels = pd.Series(NOT_CLUSTERED, index=X.index, name="cluster_label", dtype="int64")
    if int(cluster_mask.sum()) > 0:
        labels.loc[cluster_mask] = run_dbscan(X[cluster_mask], eps=eps, min_samples=min_samples).to_numpy()
    return labels


def cluster_summary(labels: pd.Series) -> dict:
    counts = labels.value_counts().to_dict()
    n_noise = counts.get(-1, 0)
    n_clusters = len([k for k in counts.keys() if k >= 0])
    summary = {
        "n_clusters": n_clusters,
        "n_noise_points": int(n_noise),
        "cluster_sizes": {str(k): int(v) for k, v in counts.items()},
    }
    n_not = int(counts.get(NOT_CLUSTERED, 0))
    if n_not:
        summary["n_not_clustered"] = n_not
    return summary
