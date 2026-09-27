"""
Investigation case builder.

A "case" groups suspicious transactions that are actually related to each
other, using several independent linkage signals that reuse the pipeline's
EXISTING outputs (alerts table + entity graph) rather than recomputing anything:

  1. shared_wallet   -- two transactions share an input or output wallet
                         (read directly off the existing entity graph's
                         Transaction -> Wallet edges).
  2. shared_ip        -- two transactions were observed from the same source IP.
  3. cluster_temporal -- two transactions fall in the SAME non-noise DBSCAN
                         cluster AND occur within a configurable time window
                         of each other.

A DBSCAN cluster on its own is deliberately NOT treated as a case: signal (3)
only fires when cluster membership is corroborated by temporal proximity, and
signals (1)/(2) don't involve clustering at all. Two transactions in the same
cluster but months apart, with no shared entity, are never merged into one case
by this logic alone.

Related transactions are connected via a lightweight linkage graph; each
connected component becomes one case. Case IDs are deterministic (derived from
the sorted set of linked txids) so re-running the pipeline against unchanged
data reproduces the exact same case_id and evidence -- which is what lets the
forensic anchoring step (src/forensics/anchoring.py) recognize "this case is
unchanged, nothing new to anchor" versus "this case's evidence changed".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from src.forensics.canonical import sha256_hex

logger = logging.getLogger(__name__)

PRIORITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass
class LinkEdge:
    txid_a: str
    txid_b: str
    link_type: str          # "shared_wallet" | "shared_ip" | "cluster_temporal"
    shared_entity: str      # the wallet address / ip / cluster id that justified the link


@dataclass
class Case:
    case_id: str
    linked_txids: List[str]
    wallets: List[str]
    source_ips: List[str]
    countries: List[str]
    asns: List[str]
    clusters_involved: List[int]
    time_start: Optional[str]
    time_end: Optional[str]
    max_risk_score: float
    avg_risk_score: float
    case_priority: str
    priority_counts: Dict[str, int]
    alert_reasons: List[str]
    link_edges: List[Dict[str, Any]]
    link_type_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "linked_txids": self.linked_txids,
            "wallets": self.wallets,
            "source_ips": self.source_ips,
            "countries": self.countries,
            "asns": self.asns,
            "clusters_involved": self.clusters_involved,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "max_risk_score": self.max_risk_score,
            "avg_risk_score": self.avg_risk_score,
            "case_priority": self.case_priority,
            "priority_counts": self.priority_counts,
            "alert_reasons": self.alert_reasons,
            "link_edges": self.link_edges,
            "link_type_counts": self.link_type_counts,
            "num_transactions": len(self.linked_txids),
        }


def _select_seed_transactions(alerts: pd.DataFrame, seed_priorities: List[str],
                               include_ml_flagged: bool) -> pd.DataFrame:
    mask = alerts["priority"].isin(seed_priorities)
    if include_ml_flagged and "is_anomaly_prediction" in alerts.columns:
        mask = mask | alerts["is_anomaly_prediction"].astype(bool)
    return alerts[mask].copy()


def _wallet_linkage_edges(seed: pd.DataFrame, g: nx.MultiDiGraph) -> List[LinkEdge]:
    """Reuse the existing entity graph's Transaction -> Wallet edges (no recomputation):
    for each seed transaction, look up its wallet neighbors directly from the graph,
    then link any two seed transactions that share a wallet."""
    wallet_to_txids: Dict[str, Set[str]] = {}
    for txid in seed["txid"]:
        tx_node = f"tx:{txid}"
        if tx_node not in g:
            continue
        for neighbor in g.successors(tx_node):
            data = g.nodes[neighbor]
            if data.get("type") == "Wallet":
                wallet_to_txids.setdefault(neighbor, set()).add(txid)

    edges: List[LinkEdge] = []
    for wallet_node, txids in wallet_to_txids.items():
        if len(txids) < 2:
            continue
        wallet_label = g.nodes[wallet_node].get("label", wallet_node)
        txid_list = sorted(txids)
        for i in range(len(txid_list)):
            for j in range(i + 1, len(txid_list)):
                edges.append(LinkEdge(txid_list[i], txid_list[j], "shared_wallet", wallet_label))
    return edges


def _ip_linkage_edges(seed: pd.DataFrame) -> List[LinkEdge]:
    """Directly reuse the existing src_ip column (already computed by ingestion) --
    no graph traversal needed for this one."""
    edges: List[LinkEdge] = []
    for ip, group in seed.groupby("src_ip"):
        txid_list = sorted(group["txid"].tolist())
        if len(txid_list) < 2:
            continue
        for i in range(len(txid_list)):
            for j in range(i + 1, len(txid_list)):
                edges.append(LinkEdge(txid_list[i], txid_list[j], "shared_ip", str(ip)))
    return edges


def _cluster_temporal_linkage_edges(seed: pd.DataFrame, temporal_window_hours: float) -> List[LinkEdge]:
    """Cluster membership ALONE never creates a link. It only contributes a link when
    corroborated by temporal proximity within `temporal_window_hours`. This is what
    keeps a case from being a simple restatement of a DBSCAN cluster."""
    edges: List[LinkEdge] = []
    window = pd.Timedelta(hours=temporal_window_hours)

    valid = seed[seed["cluster_label"] != -1]
    for cluster_label, group in valid.groupby("cluster_label"):
        g_sorted = group.sort_values("timestamp_parsed")
        rows = list(g_sorted[["txid", "timestamp_parsed"]].itertuples(index=False, name=None))
        for i in range(len(rows)):
            txid_i, ts_i = rows[i]
            for j in range(i + 1, len(rows)):
                txid_j, ts_j = rows[j]
                if pd.isna(ts_i) or pd.isna(ts_j):
                    continue
                if abs(ts_j - ts_i) <= window:
                    edges.append(LinkEdge(min(txid_i, txid_j), max(txid_i, txid_j),
                                           "cluster_temporal", f"cluster:{cluster_label}"))
                else:
                    # rows are time-sorted, so once we exceed the window every later
                    # row will too -- stop scanning forward from i.
                    break
    return edges


def _make_case_id(txids: List[str]) -> str:
    digest = sha256_hex(",".join(sorted(txids)).encode("utf-8"))
    return f"CASE-{digest[:12].upper()}"


def _to_iso(ts) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    return str(ts)


def build_cases(
    alerts: pd.DataFrame,
    g: nx.MultiDiGraph,
    seed_priorities: List[str],
    include_ml_flagged: bool,
    temporal_window_hours: float,
    min_case_size: int = 1,
) -> Tuple[List[Case], List[LinkEdge]]:
    """Build investigation cases from the EXISTING alerts table + entity graph.
    Does not recompute any ML score, cluster label, or graph metric -- it only
    reads and relates values already produced upstream in the pipeline."""
    seed = _select_seed_transactions(alerts, seed_priorities, include_ml_flagged)
    logger.info("Case builder: %d seed transactions selected out of %d total alerts.",
                len(seed), len(alerts))

    if len(seed) == 0:
        return [], []

    all_edges = (
        _wallet_linkage_edges(seed, g)
        + _ip_linkage_edges(seed)
        + _cluster_temporal_linkage_edges(seed, temporal_window_hours)
    )

    linkage_graph = nx.Graph()
    linkage_graph.add_nodes_from(seed["txid"].tolist())
    for e in all_edges:
        linkage_graph.add_edge(e.txid_a, e.txid_b)

    seed_by_txid = seed.set_index("txid")
    edges_by_pair: Dict[Tuple[str, str], List[LinkEdge]] = {}
    for e in all_edges:
        key = tuple(sorted((e.txid_a, e.txid_b)))
        edges_by_pair.setdefault(key, []).append(e)

    cases: List[Case] = []
    for component in nx.connected_components(linkage_graph):
        txids = sorted(component)
        if len(txids) < min_case_size:
            continue

        rows = seed_by_txid.loc[txids]
        wallets: Set[str] = set()
        for col in ("input_addresses", "output_addresses"):
            for addr_list in rows[col]:
                if addr_list is not None:
                    wallets.update(a for a in addr_list if a)

        source_ips = sorted(set(rows["src_ip"].dropna().astype(str)))
        countries = sorted(set(rows["geo_country"].dropna().astype(str)))
        asns = sorted(set(rows["asn"].dropna().astype(str)))
        clusters_involved = sorted(set(int(c) for c in rows["cluster_label"].dropna().unique()))

        times = rows["timestamp_parsed"].dropna().sort_values()
        time_start = _to_iso(times.iloc[0]) if len(times) else None
        time_end = _to_iso(times.iloc[-1]) if len(times) else None

        priority_counts = rows["priority"].value_counts().to_dict()
        case_priority = max(priority_counts.keys(), key=lambda p: PRIORITY_RANK.get(p, 0)) \
            if priority_counts else "LOW"

        alert_reasons = sorted(set(
            r for reasons_str in rows["reasons"].dropna().astype(str)
            for r in reasons_str.split(" | ")
        ))

        component_set = set(txids)
        relevant_link_edges: List[LinkEdge] = []
        for pair, edge_list in edges_by_pair.items():
            if pair[0] in component_set and pair[1] in component_set:
                relevant_link_edges.extend(edge_list)

        link_type_counts: Dict[str, int] = {}
        for e in relevant_link_edges:
            link_type_counts[e.link_type] = link_type_counts.get(e.link_type, 0) + 1

        case_id = _make_case_id(txids)
        case = Case(
            case_id=case_id,
            linked_txids=txids,
            wallets=sorted(wallets),
            source_ips=source_ips,
            countries=countries,
            asns=asns,
            clusters_involved=clusters_involved,
            time_start=time_start,
            time_end=time_end,
            max_risk_score=float(rows["risk_score"].max()),
            avg_risk_score=float(rows["risk_score"].mean()),
            case_priority=case_priority,
            priority_counts={k: int(v) for k, v in priority_counts.items()},
            alert_reasons=alert_reasons,
            link_edges=[
                {"txid_a": e.txid_a, "txid_b": e.txid_b, "link_type": e.link_type,
                 "shared_entity": e.shared_entity}
                for e in relevant_link_edges
            ],
            link_type_counts=link_type_counts,
        )
        cases.append(case)

    cases.sort(key=lambda c: c.max_risk_score, reverse=True)
    logger.info("Case builder: %d investigation cases built from %d seed transactions.",
                len(cases), len(seed))
    return cases, all_edges
