"""
Feature engineering layer.

The primary unit of analysis is the TRANSACTION (each row of the cleaned dataset),
since that's the level at which synthetic_label/anomaly_type exist for evaluation.
Every transaction is enriched with:
  1. Direct transaction-level behavioral features (fan-in/out, amounts, timing).
  2. Aggregated behavioral features for the transaction's source IP (frequency,
     unique wallets/countries touched, average time gap, IP/wallet ratio).
  3. Graph-topology features for the source IP node (degree, in/out-degree,
     PageRank, approximate betweenness centrality).

These are combined into a single numeric feature matrix used for both the
Isolation Forest anomaly detector and DBSCAN clustering.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

BEHAVIORAL_FEATURE_COLUMNS = [
    "num_inputs", "num_outputs", "fan_in", "fan_out",
    "total_input_amount", "total_output_amount", "avg_amount", "amount_variance",
    "fee",
    "unique_wallets_in_tx",
    "hour_of_day", "day_of_week",
    "ip_tx_count", "ip_unique_wallets", "ip_unique_countries",
    "ip_avg_time_gap_seconds", "ip_wallet_ratio", "ip_burst_count",
]

GRAPH_FEATURE_COLUMNS = [
    "ip_degree", "ip_in_degree", "ip_out_degree", "ip_pagerank", "ip_betweenness",
]

ALL_FEATURE_COLUMNS = BEHAVIORAL_FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS


def _txn_level_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["num_inputs"] = df["input_addresses"].apply(len)
    out["num_outputs"] = df["output_addresses"].apply(len)
    out["fan_in"] = out["num_inputs"]
    out["fan_out"] = out["num_outputs"]

    def safe_sum(vals):
        vals = [v for v in (vals or []) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(sum(vals)) if vals else 0.0

    def safe_var(vals):
        vals = [v for v in (vals or []) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.var(vals)) if len(vals) > 1 else 0.0

    out["total_input_amount"] = df["input_amounts"].apply(safe_sum)
    out["total_output_amount"] = df["output_amounts"].apply(safe_sum)

    combined_amounts = df.apply(lambda r: (r["input_amounts"] or []) + (r["output_amounts"] or []), axis=1)
    out["avg_amount"] = combined_amounts.apply(lambda v: float(np.mean(v)) if len(v) > 0 else 0.0)
    out["amount_variance"] = combined_amounts.apply(safe_var)

    out["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)

    def unique_wallets(row):
        return len(set((row["input_addresses"] or []) + (row["output_addresses"] or [])))

    out["unique_wallets_in_tx"] = df.apply(unique_wallets, axis=1)

    ts = df["timestamp_parsed"]
    out["hour_of_day"] = ts.dt.hour.fillna(0).astype(int)
    out["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)

    return out


def _ip_behavioral_features(df: pd.DataFrame, burst_window_hours: float = 1.0) -> pd.DataFrame:
    """Aggregate per-src_ip behavioral stats, then broadcast back to each transaction row."""
    work = df[["src_ip", "timestamp_parsed", "geo_country", "input_addresses", "output_addresses"]].copy()
    work = work.reset_index().rename(columns={"index": "_orig_idx"})

    ip_groups = work.groupby("src_ip")

    ip_tx_count = ip_groups["_orig_idx"].count().rename("ip_tx_count")
    ip_unique_countries = ip_groups["geo_country"].nunique().rename("ip_unique_countries")

    def wallet_set_size(sub):
        wallets = set()
        for addrs in sub["input_addresses"]:
            wallets.update(addrs or [])
        for addrs in sub["output_addresses"]:
            wallets.update(addrs or [])
        return len(wallets)

    ip_unique_wallets = ip_groups.apply(wallet_set_size).rename("ip_unique_wallets")

    def avg_time_gap(sub):
        times = sub["timestamp_parsed"].dropna().sort_values()
        if len(times) < 2:
            return 0.0
        gaps = times.diff().dropna().dt.total_seconds()
        return float(gaps.mean()) if len(gaps) > 0 else 0.0

    ip_avg_gap = ip_groups.apply(avg_time_gap).rename("ip_avg_time_gap_seconds")

    window = pd.Timedelta(hours=burst_window_hours)

    def burst_count(sub):
        times = sub["timestamp_parsed"].dropna().sort_values().reset_index(drop=True)
        if len(times) == 0:
            return 0
        max_count = 0
        left = 0
        for right in range(len(times)):
            while times[right] - times[left] > window:
                left += 1
            max_count = max(max_count, right - left + 1)
        return int(max_count)

    ip_burst = ip_groups.apply(burst_count).rename("ip_burst_count")

    ip_stats = pd.concat([ip_tx_count, ip_unique_countries, ip_unique_wallets, ip_avg_gap, ip_burst], axis=1)
    ip_stats["ip_wallet_ratio"] = ip_stats["ip_unique_wallets"] / ip_stats["ip_tx_count"].replace(0, np.nan)
    ip_stats["ip_wallet_ratio"] = ip_stats["ip_wallet_ratio"].fillna(0.0)

    merged = df[["src_ip"]].merge(ip_stats, left_on="src_ip", right_index=True, how="left")
    merged = merged.drop(columns=["src_ip"])
    merged.index = df.index
    return merged


def _ip_graph_features(df: pd.DataFrame, g: nx.MultiDiGraph, compute_betweenness: bool = True,
                        betweenness_k_sample: int = 500) -> pd.DataFrame:
    """Pull graph-topology metrics for each transaction's src_ip node."""
    simple_g = nx.DiGraph(g)  # collapse multi-edges for centrality algorithms

    degree = dict(simple_g.degree())
    in_degree = dict(simple_g.in_degree())
    out_degree = dict(simple_g.out_degree())

    try:
        pagerank = nx.pagerank(simple_g, alpha=0.85)
    except Exception:
        pagerank = {n: 0.0 for n in simple_g.nodes()}

    betweenness: Dict[str, float] = {}
    if compute_betweenness and simple_g.number_of_nodes() > 0:
        k = min(betweenness_k_sample, simple_g.number_of_nodes())
        try:
            betweenness = nx.betweenness_centrality(simple_g, k=k, seed=42, normalized=True)
        except Exception:
            betweenness = {n: 0.0 for n in simple_g.nodes()}
    else:
        betweenness = {n: 0.0 for n in simple_g.nodes()}

    def lookup(node_id, table):
        return table.get(node_id, 0.0)

    ip_nodes = df["src_ip"].apply(lambda ip: f"ip:{ip}")

    out = pd.DataFrame(index=df.index)
    out["ip_degree"] = ip_nodes.apply(lambda n: lookup(n, degree))
    out["ip_in_degree"] = ip_nodes.apply(lambda n: lookup(n, in_degree))
    out["ip_out_degree"] = ip_nodes.apply(lambda n: lookup(n, out_degree))
    out["ip_pagerank"] = ip_nodes.apply(lambda n: lookup(n, pagerank))
    out["ip_betweenness"] = ip_nodes.apply(lambda n: lookup(n, betweenness))
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    g: nx.MultiDiGraph,
    burst_window_hours: float = 1.0,
    compute_betweenness: bool = True,
    betweenness_k_sample: int = 500,
) -> pd.DataFrame:
    """Returns a numeric feature DataFrame aligned to df's index, with columns == ALL_FEATURE_COLUMNS."""
    txn_feats = _txn_level_features(df)
    ip_feats = _ip_behavioral_features(df, burst_window_hours=burst_window_hours)
    graph_feats = _ip_graph_features(df, g, compute_betweenness=compute_betweenness,
                                      betweenness_k_sample=betweenness_k_sample)

    features = pd.concat([txn_feats, ip_feats, graph_feats], axis=1)
    features = features[ALL_FEATURE_COLUMNS].astype(float)
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features
