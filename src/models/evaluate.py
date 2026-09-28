"""
Evaluation: computes real classification metrics AT RUNTIME by comparing model
outputs (is_anomaly_prediction, anomaly_score_0_100) against synthetic_label.

synthetic_label is used ONLY here — never as a training signal for the Isolation
Forest — per the requirement that this remain a genuinely unsupervised detector.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_against_synthetic_label(
    synthetic_label: pd.Series,
    is_anomaly_prediction: pd.Series,
    anomaly_score_0_100: pd.Series,
) -> Dict:
    y_true = synthetic_label.astype(int).values
    y_pred = is_anomaly_prediction.astype(int).values
    y_score = anomaly_score_0_100.values

    result: Dict = {}
    result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC / PR-AUC require both classes present and a continuous score.
    if len(np.unique(y_true)) > 1:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            result["roc_auc"] = None
        try:
            result["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            result["pr_auc"] = None
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    result["n_total"] = int(len(y_true))
    result["n_true_anomalies"] = int(y_true.sum())
    result["n_predicted_anomalies"] = int(y_pred.sum())

    return result
