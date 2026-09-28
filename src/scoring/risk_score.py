"""
Investigative priority scoring: combines the ML anomaly score with behavioral,
graph, and network/geographic indicators into a single transparent 0-100 score.

This score is explicitly an INVESTIGATIVE / MODEL-GENERATED PRIORITY SIGNAL,
not proof of criminal activity, and not a calibrated probability of anything.
Weights are configurable (config.yaml -> scoring.weights) and re-normalized to
sum to 1 automatically.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _minmax_0_100(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo) * 100.0


def compute_behavioral_component(features: pd.DataFrame) -> pd.Series:
    """Combine fan-out, fan-in, burst count, and amount variance into one behavioral indicator."""
    parts = [
        _minmax_0_100(features["fan_out"]),
        _minmax_0_100(features["fan_in"]),
        _minmax_0_100(features["ip_burst_count"]),
        _minmax_0_100(features["amount_variance"]),
    ]
    return sum(parts) / len(parts)


def compute_graph_component(features: pd.DataFrame) -> pd.Series:
    """Combine degree, pagerank, and betweenness into one graph-topology indicator."""
    parts = [
        _minmax_0_100(features["ip_degree"]),
        _minmax_0_100(features["ip_pagerank"]),
        _minmax_0_100(features["ip_betweenness"]),
    ]
    return sum(parts) / len(parts)


def compute_network_geo_component(features: pd.DataFrame) -> pd.Series:
    """Combine unique-country spread and IP/wallet ratio into one network/geo indicator."""
    parts = [
        _minmax_0_100(features["ip_unique_countries"]),
        _minmax_0_100(features["ip_wallet_ratio"]),
    ]
    return sum(parts) / len(parts)


def compute_risk_scores(
    features: pd.DataFrame,
    anomaly_score_0_100: pd.Series,
    weights: Dict[str, float],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    w = dict(weights)
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Scoring weights must sum to a positive number.")
    w = {k: v / total for k, v in w.items()}  # re-normalize

    behavioral = compute_behavioral_component(features)
    graph = compute_graph_component(features)
    network_geo = compute_network_geo_component(features)

    final_score = (
        w.get("ml_anomaly", 0.0) * anomaly_score_0_100
        + w.get("behavioral", 0.0) * behavioral
        + w.get("graph", 0.0) * graph
        + w.get("network_geo", 0.0) * network_geo
    )

    high_th = thresholds.get("high", 70)
    med_th = thresholds.get("medium", 40)

    def priority(score):
        if score >= high_th:
            return "HIGH"
        elif score >= med_th:
            return "MEDIUM"
        return "LOW"

    priority_labels = final_score.apply(priority)

    return pd.DataFrame({
        "component_ml_anomaly": anomaly_score_0_100,
        "component_behavioral": behavioral,
        "component_graph": graph,
        "component_network_geo": network_geo,
        "risk_score": final_score,
        "priority": priority_labels,
    }, index=features.index)
