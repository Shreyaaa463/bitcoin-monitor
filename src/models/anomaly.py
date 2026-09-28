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


def select_training_rows(n_rows: int, max_train_rows: int | None, random_state: int = 42) -> np.ndarray | None:
    """Row positions to train on. None (= all rows, the prototype behavior) unless the
    dataset exceeds max_train_rows, in which case a deterministic uniform sample is
    used -- training and inference are separate steps, so every row is still scored."""
    if not max_train_rows or n_rows <= max_train_rows:
        return None
    rng = np.random.RandomState(random_state)
    return np.sort(rng.choice(n_rows, size=int(max_train_rows), replace=False))


def train_isolation_forest(
    X: pd.DataFrame,
    n_estimators: int = 200,
    contamination: float = 0.2,
    random_state: int = 42,
    max_samples="auto",
    max_train_rows: int | None = None,
) -> AnomalyModelBundle:
    rows = select_training_rows(len(X), max_train_rows, random_state)
    X_train = X if rows is None else X.iloc[rows]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        max_samples=max_samples,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    return AnomalyModelBundle(model=model, scaler=scaler, feature_columns=list(X.columns))


def score_isolation_forest(bundle: AnomalyModelBundle, X: pd.DataFrame,
                           chunk_rows: int | None = None) -> pd.DataFrame:
    """Inference. Scores are per-row, so scoring in chunks gives identical values while
    bounding the memory of the scaled matrix; only the final 0-100 min-max
    normalization needs the global min/max of the raw scores."""
    X_aligned = X[bundle.feature_columns]
    n = len(X_aligned)
    step = chunk_rows if chunk_rows and chunk_rows > 0 else max(n, 1)
    raw_parts, pred_parts = [], []
    for start in range(0, n, step):
        X_scaled = bundle.scaler.transform(X_aligned.values[start:start + step])
        raw_parts.append(bundle.model.decision_function(X_scaled))  # higher = more normal (sklearn convention)
        pred_parts.append(bundle.model.predict(X_scaled))           # -1 = anomaly, 1 = normal
    raw_scores = np.concatenate(raw_parts) if raw_parts else np.zeros(0)
    predictions = np.concatenate(pred_parts) if pred_parts else np.zeros(0)

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
