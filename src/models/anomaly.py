"""
Anomaly detection: real (unsupervised) Isolation Forest over the engineered feature
matrix. synthetic_label is NEVER used as a training signal here — only afterwards,
in evaluation, to compute precision/recall/F1/ROC-AUC/PR-AUC.

Produces:
- raw_anomaly_score: sklearn's decision_function output (higher = more normal,
  by sklearn convention), kept for transparency/debugging.
- anomaly_score_0_100: min-max normalized, inverted so HIGHER = MORE ANOMALOUS,
  scaled to a 0-100 range. Explicitly documented as NOT a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyModelBundle:
    model: IsolationForest
    scaler: StandardScaler
    feature_columns: list


def train_isolation_forest(
    X: pd.DataFrame,
    n_estimators: int = 200,
    contamination: float = 0.2,
    random_state: int = 42,
    max_samples="auto",
) -> AnomalyModelBundle:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        max_samples=max_samples,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    return AnomalyModelBundle(model=model, scaler=scaler, feature_columns=list(X.columns))


def score_isolation_forest(bundle: AnomalyModelBundle, X: pd.DataFrame) -> pd.DataFrame:
    X_aligned = X[bundle.feature_columns]
    X_scaled = bundle.scaler.transform(X_aligned.values)

    raw_scores = bundle.model.decision_function(X_scaled)   # higher = more normal (sklearn convention)
    predictions = bundle.model.predict(X_scaled)             # -1 = anomaly, 1 = normal

    # Invert & min-max scale to 0-100 so HIGHER means MORE ANOMALOUS.
    inverted = -raw_scores
    lo, hi = float(inverted.min()), float(inverted.max())
    if hi - lo < 1e-12:
        normalized = np.zeros_like(inverted)
    else:
        normalized = (inverted - lo) / (hi - lo)
    score_0_100 = normalized * 100.0

    return pd.DataFrame({
        "raw_anomaly_score": raw_scores,
        "anomaly_score_0_100": score_0_100,
        "is_anomaly_prediction": predictions == -1,
    }, index=X.index)


def save_model(bundle: AnomalyModelBundle, path: str):
    joblib.dump(bundle, path)


def load_model(path: str) -> AnomalyModelBundle:
    return joblib.load(path)
