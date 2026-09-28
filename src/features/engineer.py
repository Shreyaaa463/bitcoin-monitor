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

Two interchangeable graph-feature backends produce the same features:
  - "networkx"   : the original prototype path over an in-memory NetworkX graph
                   (exact; used for small datasets).
  - "edge_table" : sparse-matrix computation over the on-disk graph index
                   (src/graph/store.py), no in-memory NetworkX graph. Degrees are
                   exact; PageRank uses the same power iteration as NetworkX
                   (same alpha/tol/max_iter); IP betweenness is exactly 0 whenever no
                   edge points INTO an IP node (true for the current graph schema:
                   a node without incoming edges can never lie on a shortest path),
                   which the backend verifies at runtime before relying on it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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

TXN_FEATURE_COLUMNS = BEHAVIORAL_FEATURE_COLUMNS[:12]
IP_FEATURE_COLUMNS = BEHAVIORAL_FEATURE_COLUMNS[12:]


def _clean_vals(vals):
    return [v for v in (vals or []) if v is not None and not (isinstance(v, float) and np.isnan(v))]


def _safe_sum(vals) -> float:
    vals = _clean_vals(vals)
    return float(sum(vals)) if vals else 0.0


def _safe_var(vals) -> float:
    vals = _clean_vals(vals)
    return float(np.var(vals)) if len(vals) > 1 else 0.0


def _txn_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row features. Same arithmetic as the prototype (python sum, np.mean, np.var
    on the same python lists), evaluated with list comprehensions instead of
    DataFrame.apply(axis=1) for speed."""
    out = pd.DataFrame(index=df.index)
    ins, outs = df["input_addresses"].tolist(), df["output_addresses"].tolist()
    in_amts, out_amts = df["input_amounts"].tolist(), df["output_amounts"].tolist()

    out["num_inputs"] = np.array([len(v) for v in ins], dtype=np.int64)
    out["num_outputs"] = np.array([len(v) for v in outs], dtype=np.int64)
    out["fan_in"] = out["num_inputs"]
    out["fan_out"] = out["num_outputs"]

    out["total_input_amount"] = [_safe_sum(v) for v in in_amts]
    out["total_output_amount"] = [_safe_sum(v) for v in out_amts]

    combined = [(a or []) + (b or []) for a, b in zip(in_amts, out_amts)]
    out["avg_amount"] = [float(np.mean(v)) if len(v) > 0 else 0.0 for v in combined]
    out["amount_variance"] = [_safe_var(v) for v in combined]

    out["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)

    out["unique_wallets_in_tx"] = [len(set((a or []) + (b or []))) for a, b in zip(ins, outs)]

    ts = df["timestamp_parsed"]
    out["hour_of_day"] = ts.dt.hour.fillna(0).astype(int)
    out["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)

    return out


def ip_wallet_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct (src_ip, wallet) pairs for a (chunk of the) dataset. Every element of
    the address lists counts as a wallet, exactly like the prototype's set.update()."""
    ips, wallets = [], []
    for ip, a, b in zip(df["src_ip"].tolist(), df["input_addresses"].tolist(), df["output_addresses"].tolist()):
        for w in (a or []):
            ips.append(ip); wallets.append(w)
        for w in (b or []):
            ips.append(ip); wallets.append(w)
    pairs = pd.DataFrame({"src_ip": pd.Series(ips, dtype=object), "wallet": pd.Series(wallets, dtype=object)})
    return pairs.drop_duplicates(ignore_index=True)


def _ip_behavioral_features(df: pd.DataFrame, burst_window_hours: float = 1.0,
                            wallet_pairs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Aggregate per-src_ip behavioral stats, then broadcast back to each transaction row.
    Vectorized equivalent of the prototype's per-group python loops. `wallet_pairs`
    (distinct src_ip/wallet pairs) may be supplied pre-aggregated so the address
    lists never need to be held in memory for the whole dataset."""
    work = df[["src_ip", "timestamp_parsed", "geo_country"]]
    ip_groups = work.groupby("src_ip")

    ip_tx_count = ip_groups.size().rename("ip_tx_count")
    ip_unique_countries = ip_groups["geo_country"].nunique().rename("ip_unique_countries")

    if wallet_pairs is None:
        wallet_pairs = ip_wallet_pairs(df)
    ip_unique_wallets = (wallet_pairs.drop_duplicates().groupby("src_ip").size()
                         .reindex(ip_tx_count.index, fill_value=0).rename("ip_unique_wallets"))

    # Time-ordered events per IP (NaT timestamps are ignored, as in the prototype).
    ev = work.loc[work["timestamp_parsed"].notna(), ["src_ip", "timestamp_parsed"]]
    ev = ev.sort_values(["src_ip", "timestamp_parsed"], kind="stable")
    same_ip = ev["src_ip"].eq(ev["src_ip"].shift())
    gaps = ev["timestamp_parsed"].diff().dt.total_seconds().where(same_ip)
    ip_avg_gap = (gaps.groupby(ev["src_ip"]).mean()
                  .reindex(ip_tx_count.index).fillna(0.0).rename("ip_avg_time_gap_seconds"))

    # Max number of events inside any window of length <= burst_window_hours
    # (window inclusive at both ends, like the prototype's two-pointer scan).
    window = pd.Timedelta(hours=burst_window_hours)
    group_size = ev.groupby("src_ip")["src_ip"].transform("size")
    burst = pd.Series(1, index=ip_tx_count.index[ip_tx_count.index.isin(ev["src_ip"])], dtype="int64")
    multi = ev[group_size.to_numpy() > 1]
    if len(multi):
        counts = (pd.Series(1.0, index=pd.DatetimeIndex(multi["timestamp_parsed"]))
                  .groupby(multi["src_ip"].to_numpy())
                  .rolling(window, closed="both").sum())
        max_counts = counts.groupby(level=0).max().astype("int64")
        burst.loc[max_counts.index] = max_counts.to_numpy()
    ip_burst = burst.reindex(ip_tx_count.index, fill_value=0).rename("ip_burst_count")

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


def _pagerank_sparse(src: np.ndarray, dst: np.ndarray, n: int, alpha: float = 0.85,
                     max_iter: int = 100, tol: float = 1.0e-06) -> Optional[np.ndarray]:
    """networkx.pagerank (scipy implementation, unweighted DiGraph, uniform teleport and
    dangling redistribution) on integer edge arrays. Returns None on non-convergence
    (the networkx path then falls back to zeros, and so does this one)."""
    import scipy.sparse as sp

    if n == 0:
        return np.zeros(0)
    A = sp.csr_array((np.ones(len(src)), (src, dst)), shape=(n, n))
    S = np.asarray(A.sum(axis=1)).ravel()
    nz = S != 0
    inv = np.zeros(n)
    inv[nz] = 1.0 / S[nz]
    A = sp.csr_array(sp.diags_array(inv)) @ A
    x = np.repeat(1.0 / n, n)
    p = np.repeat(1.0 / n, n)
    is_dangling = np.where(~nz)[0]
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + x[is_dangling].sum() * p) + (1 - alpha) * p
        if np.absolute(x - xlast).sum() < n * tol:
            return x
    return None


def _ip_graph_features_from_store(df: pd.DataFrame, store, compute_betweenness: bool = True) -> pd.DataFrame:
    """Edge-table backend (see module docstring). `store` is a src.graph.store.GraphStore."""
    src, dst = store.edge_arrays()
    n = store.num_nodes()
    # Collapse parallel edges, exactly like nx.DiGraph(multigraph).
    pair = np.unique(src * np.int64(n) + dst)
    usrc, udst = pair // n, pair % n
    out_deg = np.bincount(usrc, minlength=n)
    in_deg = np.bincount(udst, minlength=n)
    pr = _pagerank_sparse(usrc, udst, n, alpha=0.85)
    if pr is None:
        pr = np.zeros(n)
    del src, dst, pair

    unique_ips = pd.unique(df["src_ip"])
    ids = store.node_ids_for_keys(f"ip:{ip}" for ip in unique_ips)
    ip_id = np.array([ids.get(f"ip:{ip}", -1) for ip in unique_ips], dtype=np.int64)
    known = ip_id >= 0

    if compute_betweenness and store.count_edges_into_type("IP") > 0:
        raise NotImplementedError(
            "edge_table graph backend: IP nodes have incoming edges, so IP betweenness is "
            "not structurally zero; use graph.feature_backend: networkx for this schema.")

    def per_ip(values: np.ndarray) -> np.ndarray:
        res = np.zeros(len(unique_ips))
        res[known] = values[ip_id[known]]
        return res

    table = pd.DataFrame({
        "ip_degree": per_ip((in_deg + out_deg).astype(float)),
        "ip_in_degree": per_ip(in_deg.astype(float)),
        "ip_out_degree": per_ip(out_deg.astype(float)),
        "ip_pagerank": per_ip(pr),
        "ip_betweenness": np.zeros(len(unique_ips)),
    }, index=unique_ips)
    out = df[["src_ip"]].merge(table, left_on="src_ip", right_index=True, how="left").drop(columns=["src_ip"])
    out.index = df.index
    return out.fillna(0.0)


def assemble_feature_matrix(txn_feats: pd.DataFrame, ip_feats: pd.DataFrame,
                            graph_feats: pd.DataFrame) -> pd.DataFrame:
    features = pd.concat([txn_feats, ip_feats, graph_feats], axis=1)
    features = features[ALL_FEATURE_COLUMNS].astype(float)
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features


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
    return assemble_feature_matrix(txn_feats, ip_feats, graph_feats)
