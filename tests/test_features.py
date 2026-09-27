import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import CANONICAL_FIELDS
from src.preprocessing.clean import clean_dataframe
from src.correlation.correlate import build_edges
from src.graph.build_graph import build_entity_graph
from src.features.engineer import build_feature_matrix, ALL_FEATURE_COLUMNS


def _row(txid, src_ip, ts, n_out=1, n_in=1):
    return {
        "timestamp": ts,
        "src_ip": src_ip,
        "dst_ip": "9.9.9.9",
        "src_port": 1234,
        "dst_port": 8333,
        "txid": txid,
        "input_addresses": [f"in_{txid}_{i}" for i in range(n_in)],
        "output_addresses": [f"out_{txid}_{i}" for i in range(n_out)],
        "input_amounts": [1.0] * n_in,
        "output_amounts": [0.9] * n_out,
        "fee": 0.0001,
        "script_type": "P2PKH",
        "geo_country": "US",
        "asn": "AS1",
        "transaction_status": "confirmed",
        "synthetic_label": 0,
        "anomaly_type": None,
        "anomaly_family": None,
    }


@pytest.fixture
def cleaned_df_and_graph():
    rows = [
        _row("tx1", "1.1.1.1", "2025-01-01T00:00:00Z", n_out=1),
        _row("tx2", "1.1.1.1", "2025-01-01T00:05:00Z", n_out=1),
        _row("tx3", "2.2.2.2", "2025-01-01T01:00:00Z", n_out=10),  # fan-out entity
    ]
    df = pd.DataFrame(rows, columns=CANONICAL_FIELDS)
    cleaned, _ = clean_dataframe(df, required_fields=["timestamp", "src_ip", "txid"])
    edges = build_edges(cleaned)
    g = build_entity_graph(edges)
    return cleaned, g


def test_feature_matrix_shape_and_columns(cleaned_df_and_graph):
    df, g = cleaned_df_and_graph
    features = build_feature_matrix(df, g, burst_window_hours=1.0, compute_betweenness=True)
    assert len(features) == len(df)
    assert list(features.columns) == ALL_FEATURE_COLUMNS
    assert features.isna().sum().sum() == 0


def test_fan_out_feature_reflects_output_count(cleaned_df_and_graph):
    df, g = cleaned_df_and_graph
    features = build_feature_matrix(df, g)
    tx3_idx = df[df["txid"] == "tx3"].index[0]
    assert features.loc[tx3_idx, "fan_out"] == 10


def test_ip_tx_count_aggregates_correctly(cleaned_df_and_graph):
    df, g = cleaned_df_and_graph
    features = build_feature_matrix(df, g)
    # 1.1.1.1 appears in 2 transactions, 2.2.2.2 in 1
    tx1_idx = df[df["txid"] == "tx1"].index[0]
    tx3_idx = df[df["txid"] == "tx3"].index[0]
    assert features.loc[tx1_idx, "ip_tx_count"] == 2
    assert features.loc[tx3_idx, "ip_tx_count"] == 1


def test_no_infinite_or_nan_values(cleaned_df_and_graph):
    df, g = cleaned_df_and_graph
    features = build_feature_matrix(df, g)
    import numpy as np
    assert not np.isinf(features.values).any()
    assert not np.isnan(features.values).any()
