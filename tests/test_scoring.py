import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring.risk_score import compute_risk_scores
from src.features.engineer import ALL_FEATURE_COLUMNS


def _dummy_features(n=5):
    """Build a minimal but complete feature frame with all required columns."""
    data = {col: [0.0] * n for col in ALL_FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    # give row 0 the most "suspicious" looking values across the board
    for col in ["fan_out", "fan_in", "ip_burst_count", "amount_variance",
                "ip_degree", "ip_pagerank", "ip_betweenness",
                "ip_unique_countries", "ip_wallet_ratio"]:
        df.loc[0, col] = 100.0
    return df


def test_weights_are_renormalized():
    features = _dummy_features()
    anomaly_scores = pd.Series([90.0, 10.0, 10.0, 10.0, 10.0])
    # weights sum to 2.0, not 1.0 -- should be renormalized internally without error
    weights = {"ml_anomaly": 0.8, "behavioral": 0.4, "graph": 0.4, "network_geo": 0.4}
    thresholds = {"high": 70, "medium": 40}
    result = compute_risk_scores(features, anomaly_scores, weights, thresholds)
    assert (result["risk_score"] <= 100.0).all()
    assert (result["risk_score"] >= 0.0).all()


def test_highest_component_scores_yield_highest_priority():
    features = _dummy_features()
    anomaly_scores = pd.Series([100.0, 0.0, 0.0, 0.0, 0.0])
    weights = {"ml_anomaly": 0.4, "behavioral": 0.25, "graph": 0.2, "network_geo": 0.15}
    thresholds = {"high": 70, "medium": 40}
    result = compute_risk_scores(features, anomaly_scores, weights, thresholds)
    assert result.loc[0, "priority"] == "HIGH"
    assert result.loc[0, "risk_score"] == result["risk_score"].max()


def test_priority_thresholds_respected():
    features = _dummy_features(n=3)
    anomaly_scores = pd.Series([0.0, 0.0, 0.0])
    weights = {"ml_anomaly": 1.0, "behavioral": 0.0, "graph": 0.0, "network_geo": 0.0}
    thresholds = {"high": 70, "medium": 40}
    result = compute_risk_scores(features, anomaly_scores, weights, thresholds)
    # all scores are 0 -> all LOW
    assert (result["priority"] == "LOW").all()


def test_zero_weight_sum_raises():
    features = _dummy_features(n=2)
    anomaly_scores = pd.Series([0.0, 0.0])
    weights = {"ml_anomaly": 0.0, "behavioral": 0.0, "graph": 0.0, "network_geo": 0.0}
    thresholds = {"high": 70, "medium": 40}
    with pytest.raises(ValueError):
        compute_risk_scores(features, anomaly_scores, weights, thresholds)


def test_all_components_present_in_output():
    features = _dummy_features(n=4)
    anomaly_scores = pd.Series([50.0, 20.0, 80.0, 10.0])
    weights = {"ml_anomaly": 0.4, "behavioral": 0.25, "graph": 0.2, "network_geo": 0.15}
    thresholds = {"high": 70, "medium": 40}
    result = compute_risk_scores(features, anomaly_scores, weights, thresholds)
    for col in ["component_ml_anomaly", "component_behavioral", "component_graph",
                "component_network_geo", "risk_score", "priority"]:
        assert col in result.columns
