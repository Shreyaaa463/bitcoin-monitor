"""
Clustering: DBSCAN over the same standardized feature space used for anomaly
detection, grouping transactions/entities with similar behavioral + network +
graph characteristics. Cluster label -1 (DBSCAN's noise label) is preserved as-is
so the dashboard can distinguish "no cluster / outlier" from a real cluster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def run_dbscan(X: pd.DataFrame, eps: float = 0.8, min_samples: int = 5) -> pd.Series:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X_scaled)

    return pd.Series(labels, index=X.index, name="cluster_label")


def cluster_summary(labels: pd.Series) -> dict:
    counts = labels.value_counts().to_dict()
    n_noise = counts.get(-1, 0)
    n_clusters = len([k for k in counts.keys() if k != -1])
    return {
        "n_clusters": n_clusters,
        "n_noise_points": int(n_noise),
        "cluster_sizes": {str(k): int(v) for k, v in counts.items()},
    }
