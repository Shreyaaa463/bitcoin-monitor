"""
Investigator-facing graph view layer.

The analytical entity graph (outputs/entity_graph.pickle) remains the comprehensive
source of truth. This module derives a *dynamic*, ranked, budgeted subgraph suitable
for investigation visualization — without mutating or replacing the analytical graph.

Pipeline:
  selected entity
    -> candidate neighborhood (via get_neighborhood)
    -> investigative relevance ranking
    -> structural preservation (selected / direct neighbors / path essentials)
    -> node/edge budgets
    -> focus-mode emphasis
    -> filtered investigator MultiDiGraph + explanations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd

from src.graph.build_graph import get_neighborhood

PRIMARY_TYPES = frozenset({"IP", "Transaction", "Wallet"})
CONTEXTUAL_TYPES = frozenset({"ASN", "Country"})

FOCUS_ALL = "All Relevant"
FOCUS_TRANSACTIONS = "Transactions"
FOCUS_WALLETS = "Wallet Connections"
FOCUS_IPS = "IP Connections"
FOCUS_CASES = "Case Connections"

FOCUS_MODES = (
    FOCUS_ALL,
    FOCUS_TRANSACTIONS,
    FOCUS_WALLETS,
    FOCUS_IPS,
    FOCUS_CASES,
)

PRIORITY_WEIGHT = {"HIGH": 30.0, "MEDIUM": 15.0, "LOW": 5.0}

DEFAULT_MAX_NODES = 75
DEFAULT_MAX_EDGES = 150
DEFAULT_MAX_TRANSACTIONS = 40
DEFAULT_HOPS = 2


@dataclass
class InvestigatorViewResult:
    """Result of build_investigator_view — a filtered copy, never a view into the original."""
    graph: nx.MultiDiGraph
    selected_node: str
    focus: str
    relevance: Dict[str, float] = field(default_factory=dict)
    explanations: Dict[str, List[str]] = field(default_factory=dict)
    preserved_reasons: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.graph.number_of_nodes() == 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _safe_str(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def _node_type(g: nx.MultiDiGraph, node_id: str) -> str:
    return g.nodes[node_id].get("type", "Unknown") if node_id in g else "Unknown"


def _undirected_neighbors(g: nx.MultiDiGraph, node_id: str) -> Set[str]:
    if node_id not in g:
        return set()
    return set(g.successors(node_id)) | set(g.predecessors(node_id))


def _tx_id_from_node(node_id: str) -> Optional[str]:
    if node_id.startswith("tx:"):
        return node_id.split(":", 1)[1]
    return None


def _label_from_node(node_id: str) -> str:
    if ":" in node_id:
        return node_id.split(":", 1)[1]
    return node_id


def _build_alert_index(alerts: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Map txid -> alert field dict. Missing columns are simply omitted."""
    if alerts is None or not isinstance(alerts, pd.DataFrame) or alerts.empty:
        return {}
    if "txid" not in alerts.columns:
        return {}

    wanted = [
        "txid", "src_ip", "dst_ip", "priority", "risk_score", "anomaly_score_0_100",
        "is_anomaly_prediction", "cluster_label", "geo_country", "asn",
        "input_addresses", "output_addresses", "reasons", "timestamp_str",
        "anomaly_type",
    ]
    cols = [c for c in wanted if c in alerts.columns]
    index: Dict[str, Dict[str, Any]] = {}
    for _, row in alerts[cols].iterrows():
        txid = _safe_str(row.get("txid"))
        if not txid:
            continue
        index[txid] = {c: row.get(c) for c in cols}
    return index


def _build_case_maps(
    cases: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, Set[str]]:
    """Map graph node id -> set of case_ids that mention that entity."""
    node_to_cases: Dict[str, Set[str]] = {}
    if not cases:
        return node_to_cases

    def _add(node_id: str, case_id: str) -> None:
        if not node_id or not case_id:
            return
        node_to_cases.setdefault(node_id, set()).add(case_id)

    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = _safe_str(case.get("case_id"))
        if not case_id:
            continue
        for txid in case.get("linked_txids") or []:
            _add(f"tx:{txid}", case_id)
        for wallet in case.get("wallets") or []:
            _add(f"wallet:{wallet}", case_id)
        for ip in case.get("source_ips") or []:
            _add(f"ip:{ip}", case_id)
        for asn in case.get("asns") or []:
            _add(f"asn:{asn}", case_id)
        for country in case.get("countries") or []:
            _add(f"country:{country}", case_id)
    return node_to_cases


def _cases_for_selected(
    selected_node: str,
    node_to_cases: Dict[str, Set[str]],
) -> Set[str]:
    return set(node_to_cases.get(selected_node, set()))


def _is_suspicious_alert(alert: Dict[str, Any]) -> bool:
    priority = _safe_str(alert.get("priority"), "LOW").upper()
    if priority in ("HIGH", "MEDIUM"):
        return True
    if _safe_bool(alert.get("is_anomaly_prediction")):
        return True
    if _safe_float(alert.get("risk_score")) >= 40.0:
        return True
    return False


def _transaction_relevance(
    alert: Optional[Dict[str, Any]],
    case_ids: Set[str],
    connected_to_selected: bool,
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0
    if alert is None:
        if connected_to_selected:
            score += 25.0
            reasons.append("Direct neighbor of selected entity")
        if case_ids:
            score += 20.0
            reasons.append(f"Appears in investigation case {sorted(case_ids)[0]}")
        return score, reasons

    risk = _safe_float(alert.get("risk_score"))
    anomaly = _safe_float(alert.get("anomaly_score_0_100"))
    priority = _safe_str(alert.get("priority"), "LOW").upper()
    is_anom = _safe_bool(alert.get("is_anomaly_prediction"))

    score += risk * 0.45
    score += anomaly * 0.25
    score += PRIORITY_WEIGHT.get(priority, 0.0)

    if is_anom:
        score += 15.0
        reasons.append("Flagged as anomaly by Isolation Forest")
    if priority == "HIGH":
        reasons.append("HIGH priority transaction")
    elif priority == "MEDIUM":
        reasons.append("MEDIUM priority transaction")
    if risk > 0:
        reasons.append(f"Transaction risk score: {risk:.1f}")
    if anomaly > 0 and "anomaly" not in " ".join(reasons).lower():
        reasons.append(f"Anomaly score: {anomaly:.1f}")
    if case_ids:
        score += 20.0
        reasons.append(f"Appears in investigation case {sorted(case_ids)[0]}")
    if connected_to_selected:
        score += 25.0
        reasons.append("Direct neighbor of selected entity")
    return score, reasons


def _aggregate_tx_stats(
    tx_nodes: Iterable[str],
    alert_index: Dict[str, Dict[str, Any]],
    node_to_cases: Dict[str, Set[str]],
) -> Dict[str, Any]:
    n_tx = 0
    n_suspicious = 0
    n_high = 0
    n_anomaly = 0
    max_risk = 0.0
    case_ids: Set[str] = set()
    for tx_node in tx_nodes:
        n_tx += 1
        txid = _tx_id_from_node(tx_node)
        alert = alert_index.get(txid or "", {})
        risk = _safe_float(alert.get("risk_score")) if alert else 0.0
        max_risk = max(max_risk, risk)
        priority = _safe_str(alert.get("priority"), "").upper() if alert else ""
        if priority == "HIGH":
            n_high += 1
        if alert and _safe_bool(alert.get("is_anomaly_prediction")):
            n_anomaly += 1
        if alert and _is_suspicious_alert(alert):
            n_suspicious += 1
        case_ids.update(node_to_cases.get(tx_node, set()))
    return {
        "n_tx": n_tx,
        "n_suspicious": n_suspicious,
        "n_high": n_high,
        "n_anomaly": n_anomaly,
        "max_risk": max_risk,
        "case_ids": case_ids,
    }


def _wallet_or_ip_relevance(
    node_id: str,
    ntype: str,
    candidate: nx.MultiDiGraph,
    alert_index: Dict[str, Dict[str, Any]],
    node_to_cases: Dict[str, Set[str]],
    selected_node: str,
    selected_cases: Set[str],
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    neighbors = _undirected_neighbors(candidate, node_id)
    tx_neighbors = [n for n in neighbors if _node_type(candidate, n) == "Transaction"]
    stats = _aggregate_tx_stats(tx_neighbors, alert_index, node_to_cases)

    score = 0.0
    score += stats["n_suspicious"] * 12.0
    score += stats["n_high"] * 10.0
    score += stats["n_anomaly"] * 8.0
    score += stats["max_risk"] * 0.35
    score += min(stats["n_tx"], 20) * 1.5

    if stats["n_high"] > 0:
        reasons.append(f"Connected to {stats['n_high']} HIGH-priority transaction(s)")
    if stats["n_suspicious"] > 0:
        reasons.append(f"Connected to {stats['n_suspicious']} suspicious transaction(s)")
    if stats["max_risk"] > 0:
        reasons.append(f"Maximum connected transaction risk: {stats['max_risk']:.1f}")
    if stats["n_anomaly"] > 0:
        reasons.append(f"Connected to {stats['n_anomaly']} anomaly-flagged transaction(s)")

    if ntype == "IP":
        wallets = {n for n in neighbors if _node_type(candidate, n) == "Wallet"}
        # Wallet diversity via transaction fan-out
        wallet_via_tx: Set[str] = set()
        for tx in tx_neighbors:
            for n in _undirected_neighbors(candidate, tx):
                if _node_type(candidate, n) == "Wallet":
                    wallet_via_tx.add(n)
        diversity = len(wallets | wallet_via_tx)
        score += min(diversity, 15) * 1.5
        if diversity > 1:
            reasons.append(f"Wallet diversity via connected transactions: {diversity}")

    own_cases = set(node_to_cases.get(node_id, set())) | stats["case_ids"]
    if own_cases:
        score += 18.0
        reasons.append(f"Appears in investigation case {sorted(own_cases)[0]}")
    if selected_cases and (own_cases & selected_cases):
        score += 12.0
        reasons.append("Shares investigation case with selected entity")

    if node_id == selected_node:
        score += 100.0
        reasons.insert(0, "Selected entity")
    elif selected_node in neighbors:
        score += 25.0
        reasons.append("Direct neighbor of selected entity")
    elif selected_node.startswith("tx:") and selected_node in neighbors:
        score += 20.0
        reasons.append("Connected to selected transaction")

    # Shared wallet with selected transaction
    if selected_node.startswith("tx:") and ntype == "Wallet":
        sel_neighbors = _undirected_neighbors(candidate, selected_node)
        if node_id in sel_neighbors:
            reasons.append("Shared wallet with selected transaction")

    return score, reasons


def _contextual_relevance(
    node_id: str,
    candidate: nx.MultiDiGraph,
    selected_node: str,
    important_ips: Set[str],
    node_to_cases: Dict[str, Set[str]],
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    neighbors = _undirected_neighbors(candidate, node_id)
    score = 5.0  # base contextual score — kept low vs primary entities

    if node_id == selected_node:
        score += 100.0
        reasons.append("Selected entity")
    if selected_node in neighbors:
        score += 20.0
        reasons.append("Direct neighbor of selected entity")

    linked_important = neighbors & important_ips
    if linked_important:
        score += 15.0 + min(len(linked_important), 5) * 2.0
        reasons.append("Contextual metadata for an important IP")

    own_cases = node_to_cases.get(node_id, set())
    if own_cases:
        score += 8.0
        reasons.append(f"Appears in investigation case {sorted(own_cases)[0]}")

    return score, reasons


def _score_candidates(
    candidate: nx.MultiDiGraph,
    selected_node: str,
    alert_index: Dict[str, Dict[str, Any]],
    node_to_cases: Dict[str, Set[str]],
    selected_cases: Set[str],
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    relevance: Dict[str, float] = {}
    explanations: Dict[str, List[str]] = {}

    # First pass: score transactions so important IPs can be identified for contextual nodes
    tx_scores: Dict[str, float] = {}
    for node_id, data in candidate.nodes(data=True):
        if data.get("type") != "Transaction":
            continue
        txid = _tx_id_from_node(node_id)
        alert = alert_index.get(txid or "") if txid else None
        connected = selected_node in _undirected_neighbors(candidate, node_id) or node_id == selected_node
        score, reasons = _transaction_relevance(
            alert, node_to_cases.get(node_id, set()), connected
        )
        if node_id == selected_node:
            score += 100.0
            reasons = ["Selected entity"] + [r for r in reasons if r != "Direct neighbor of selected entity"]
        relevance[node_id] = score
        explanations[node_id] = reasons
        tx_scores[node_id] = score

    # Important IPs: those adjacent to high-scoring transactions or the selection
    important_ips: Set[str] = set()
    ranked_txs = sorted(tx_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_tx_cutoff = ranked_txs[: max(1, len(ranked_txs) // 3)] if ranked_txs else []
    for tx_node, _ in top_tx_cutoff:
        for n in _undirected_neighbors(candidate, tx_node):
            if _node_type(candidate, n) == "IP":
                important_ips.add(n)
    if _node_type(candidate, selected_node) == "IP":
        important_ips.add(selected_node)
    for n in _undirected_neighbors(candidate, selected_node):
        if _node_type(candidate, n) == "IP":
            important_ips.add(n)

    for node_id, data in candidate.nodes(data=True):
        ntype = data.get("type", "Unknown")
        if ntype == "Transaction":
            continue
        if ntype in ("Wallet", "IP"):
            score, reasons = _wallet_or_ip_relevance(
                node_id, ntype, candidate, alert_index, node_to_cases,
                selected_node, selected_cases,
            )
        elif ntype in CONTEXTUAL_TYPES:
            score, reasons = _contextual_relevance(
                node_id, candidate, selected_node, important_ips, node_to_cases,
            )
        else:
            score, reasons = 1.0, ["Unknown node type"]
        relevance[node_id] = score
        explanations[node_id] = reasons

    return relevance, explanations


def _important_transactions(
    candidate: nx.MultiDiGraph,
    relevance: Dict[str, float],
    alert_index: Dict[str, Dict[str, Any]],
    node_to_cases: Dict[str, Set[str]],
    selected_node: str,
    max_transactions: int,
) -> Set[str]:
    """Select high-value Transaction nodes to always try to preserve."""
    txs = [n for n, d in candidate.nodes(data=True) if d.get("type") == "Transaction"]
    if not txs:
        return set()

    def _is_important(tx_node: str) -> bool:
        if tx_node == selected_node:
            return True
        if selected_node in _undirected_neighbors(candidate, tx_node):
            return True
        if node_to_cases.get(tx_node):
            return True
        txid = _tx_id_from_node(tx_node)
        alert = alert_index.get(txid or "") if txid else None
        if not alert:
            return False
        priority = _safe_str(alert.get("priority"), "").upper()
        if priority == "HIGH":
            return True
        if _safe_bool(alert.get("is_anomaly_prediction")):
            return True
        if _safe_float(alert.get("risk_score")) >= 70.0:
            return True
        return False

    important = [t for t in txs if _is_important(t)]
    important.sort(key=lambda n: relevance.get(n, 0.0), reverse=True)
    return set(important[:max_transactions])


def _path_support_nodes(
    candidate: nx.MultiDiGraph,
    keep: Set[str],
    important_txs: Set[str],
    selected_node: str,
) -> Set[str]:
    """
    Preserve nodes needed for investigative paths such as:
      IP -> Transaction -> Wallet
      Wallet -> Transaction -> Wallet
      IP -> Transaction
    """
    support: Set[str] = set()
    targets = set(important_txs)
    if _node_type(candidate, selected_node) == "Transaction":
        targets.add(selected_node)

    for tx in targets:
        for n in _undirected_neighbors(candidate, tx):
            ntype = _node_type(candidate, n)
            if ntype in ("IP", "Wallet"):
                support.add(n)
            # one hop of wallet-wallet via shared tx already covered; keep ASN/Country
            # only when attached to an IP we already keep (handled later)
    return support


def _apply_focus_boost(
    relevance: Dict[str, float],
    candidate: nx.MultiDiGraph,
    focus: str,
    selected_cases: Set[str],
    node_to_cases: Dict[str, Set[str]],
) -> Dict[str, float]:
    boosted = dict(relevance)
    if focus == FOCUS_ALL:
        return boosted

    for node_id, data in candidate.nodes(data=True):
        ntype = data.get("type", "Unknown")
        if focus == FOCUS_TRANSACTIONS:
            if ntype == "Transaction":
                boosted[node_id] = boosted.get(node_id, 0.0) * 1.5 + 20.0
            elif ntype in CONTEXTUAL_TYPES:
                boosted[node_id] = boosted.get(node_id, 0.0) * 0.4
        elif focus == FOCUS_WALLETS:
            if ntype == "Wallet":
                boosted[node_id] = boosted.get(node_id, 0.0) * 1.5 + 20.0
            elif ntype == "Transaction":
                boosted[node_id] = boosted.get(node_id, 0.0) * 1.2 + 10.0
            elif ntype == "IP":
                boosted[node_id] = boosted.get(node_id, 0.0) * 0.7
        elif focus == FOCUS_IPS:
            if ntype == "IP":
                boosted[node_id] = boosted.get(node_id, 0.0) * 1.5 + 20.0
            elif ntype == "Transaction":
                boosted[node_id] = boosted.get(node_id, 0.0) * 1.2 + 10.0
            elif ntype == "Wallet":
                boosted[node_id] = boosted.get(node_id, 0.0) * 0.7
        elif focus == FOCUS_CASES:
            own = node_to_cases.get(node_id, set())
            if selected_cases and (own & selected_cases):
                boosted[node_id] = boosted.get(node_id, 0.0) * 1.6 + 30.0
            elif own:
                boosted[node_id] = boosted.get(node_id, 0.0) * 1.2 + 10.0
            elif ntype in CONTEXTUAL_TYPES:
                boosted[node_id] = boosted.get(node_id, 0.0) * 0.5
    return boosted


def _focus_type_preference(focus: str) -> Optional[Set[str]]:
    """Optional soft preference for which types to fill remaining budget with."""
    if focus == FOCUS_TRANSACTIONS:
        return {"Transaction", "IP", "Wallet"}
    if focus == FOCUS_WALLETS:
        return {"Wallet", "Transaction"}
    if focus == FOCUS_IPS:
        return {"IP", "Transaction"}
    return None


def _select_nodes(
    candidate: nx.MultiDiGraph,
    selected_node: str,
    relevance: Dict[str, float],
    alert_index: Dict[str, Dict[str, Any]],
    node_to_cases: Dict[str, Set[str]],
    selected_cases: Set[str],
    focus: str,
    max_nodes: int,
    max_transactions: int,
) -> Tuple[Set[str], Dict[str, List[str]]]:
    preserved_reasons: Dict[str, List[str]] = {}
    keep: Set[str] = set()

    def _mark(node_id: str, reason: str) -> None:
        keep.add(node_id)
        preserved_reasons.setdefault(node_id, [])
        if reason not in preserved_reasons[node_id]:
            preserved_reasons[node_id].append(reason)

    # 1. Always preserve selected
    _mark(selected_node, "Selected entity")

    # 2. Direct neighbors (structural essentials)
    direct = _undirected_neighbors(candidate, selected_node)
    for n in direct:
        _mark(n, "Direct neighbor of selected entity")

    # If direct neighbors already exceed budget, rank them and keep the best
    # while always retaining the selected node.
    if len(keep) > max_nodes:
        ranked_direct = sorted(
            (n for n in keep if n != selected_node),
            key=lambda n: relevance.get(n, 0.0),
            reverse=True,
        )
        keep = {selected_node} | set(ranked_direct[: max(0, max_nodes - 1)])
        preserved_reasons = {
            n: preserved_reasons.get(n, ["Retained within node budget"])
            for n in keep
        }
        return keep, preserved_reasons

    # 3. Important transactions
    important_txs = _important_transactions(
        candidate, relevance, alert_index, node_to_cases, selected_node, max_transactions
    )
    for tx in important_txs:
        if len(keep) >= max_nodes:
            break
        _mark(tx, "Important transaction for investigation")

    # 4. Path support (IP/Wallet attached to important txs)
    support = _path_support_nodes(candidate, keep, important_txs, selected_node)
    for n in sorted(support, key=lambda x: relevance.get(x, 0.0), reverse=True):
        if len(keep) >= max_nodes:
            break
        if n not in keep:
            _mark(n, "Preserves investigative path to important transaction")

    # 5. Contextual ASN/Country only when attached to a kept IP
    kept_ips = {n for n in keep if _node_type(candidate, n) == "IP"}
    for ip in kept_ips:
        for n in _undirected_neighbors(candidate, ip):
            if _node_type(candidate, n) in CONTEXTUAL_TYPES and n not in keep:
                if len(keep) >= max_nodes:
                    break
                _mark(n, "Contextual metadata for a retained IP")

    # 6. Case Connections focus: prefer remaining case co-members
    if focus == FOCUS_CASES and selected_cases:
        case_nodes = [
            n for n in candidate.nodes
            if n not in keep and (node_to_cases.get(n, set()) & selected_cases)
        ]
        case_nodes.sort(key=lambda n: relevance.get(n, 0.0), reverse=True)
        for n in case_nodes:
            if len(keep) >= max_nodes:
                break
            _mark(n, "Shares investigation case with selected entity")

    # 7. Fill remaining budget by relevance (primary entities first)
    preferred_types = _focus_type_preference(focus)
    remaining = [n for n in candidate.nodes if n not in keep]
    remaining.sort(key=lambda n: relevance.get(n, 0.0), reverse=True)

    # Prefer primary entities when filling
    primary_remaining = [
        n for n in remaining
        if _node_type(candidate, n) in PRIMARY_TYPES
        and (preferred_types is None or _node_type(candidate, n) in preferred_types)
    ]
    other_remaining = [n for n in remaining if n not in primary_remaining]

    for n in primary_remaining + other_remaining:
        if len(keep) >= max_nodes:
            break
        # Cap contextual dominance: don't let ASN/Country crowd out primaries
        if _node_type(candidate, n) in CONTEXTUAL_TYPES:
            contextual_count = sum(
                1 for x in keep if _node_type(candidate, x) in CONTEXTUAL_TYPES
            )
            primary_count = sum(
                1 for x in keep if _node_type(candidate, x) in PRIMARY_TYPES
            )
            if contextual_count >= max(3, primary_count // 4):
                continue
        _mark(n, "Ranked by investigative relevance")

    return keep, preserved_reasons


def _cap_edges(
    subgraph: nx.MultiDiGraph,
    max_edges: int,
    relevance: Dict[str, float],
) -> nx.MultiDiGraph:
    """Return a copy with at most max_edges, preferring edges between high-relevance nodes."""
    if subgraph.number_of_edges() <= max_edges:
        return subgraph.copy()

    edge_list = []
    for u, v, key, data in subgraph.edges(keys=True, data=True):
        relation = data.get("relation", "")
        # Prefer primary investigative relations over metadata
        relation_bonus = 0.0
        if relation in ("OBSERVED", "INPUT", "OUTPUT"):
            relation_bonus = 50.0
        elif relation in ("BELONGS_TO", "LOCATED_IN"):
            relation_bonus = 5.0
        weight = relevance.get(u, 0.0) + relevance.get(v, 0.0) + relation_bonus
        edge_list.append((weight, u, v, key, data))

    edge_list.sort(key=lambda t: t[0], reverse=True)
    kept = edge_list[:max_edges]

    out = nx.MultiDiGraph()
    for node, data in subgraph.nodes(data=True):
        out.add_node(node, **data)
    for _, u, v, key, data in kept:
        out.add_edge(u, v, key=key, **data)
    return out


def build_investigator_view(
    g: nx.MultiDiGraph,
    alerts: Optional[pd.DataFrame],
    selected_node: str,
    hops: int = DEFAULT_HOPS,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_transactions: int = DEFAULT_MAX_TRANSACTIONS,
    focus: str = FOCUS_ALL,
    cases: Optional[Sequence[Dict[str, Any]]] = None,
) -> InvestigatorViewResult:
    """
    Build a dynamic investigator subgraph from the analytical graph.

    Does not mutate `g`. Does not write any pickle. Returns a copy/subgraph
    suitable for PyVis rendering plus relevance metadata and explanations.
    """
    empty = nx.MultiDiGraph()
    focus = focus if focus in FOCUS_MODES else FOCUS_ALL
    max_nodes = max(1, int(max_nodes))
    max_edges = max(0, int(max_edges))
    max_transactions = max(1, int(max_transactions))
    hops = max(1, int(hops))

    if selected_node is None or selected_node not in g:
        return InvestigatorViewResult(
            graph=empty,
            selected_node=selected_node or "",
            focus=focus,
            stats={
                "candidate_nodes": 0,
                "candidate_edges": 0,
                "retained_nodes": 0,
                "retained_edges": 0,
                "reason": "selected_node_not_in_graph",
            },
        )

    # Snapshot node-count of original so tests can assert non-mutation
    original_n = g.number_of_nodes()
    original_e = g.number_of_edges()

    candidate = get_neighborhood(g, selected_node, hops=hops)
    alert_index = _build_alert_index(alerts)
    node_to_cases = _build_case_maps(cases)
    selected_cases = _cases_for_selected(selected_node, node_to_cases)

    relevance, explanations = _score_candidates(
        candidate, selected_node, alert_index, node_to_cases, selected_cases
    )
    relevance = _apply_focus_boost(
        relevance, candidate, focus, selected_cases, node_to_cases
    )

    keep, preserved_reasons = _select_nodes(
        candidate,
        selected_node,
        relevance,
        alert_index,
        node_to_cases,
        selected_cases,
        focus,
        max_nodes,
        max_transactions,
    )

    # Merge preserved reasons into explanations
    for node_id, reasons in preserved_reasons.items():
        existing = explanations.setdefault(node_id, [])
        for r in reasons:
            if r not in existing:
                existing.insert(0, r)

    # Induced subgraph, then edge budget
    induced = candidate.subgraph(keep).copy()
    filtered = _cap_edges(induced, max_edges, relevance)

    # Annotate nodes with relevance for consumers (dashboard sizing)
    for node_id in list(filtered.nodes):
        filtered.nodes[node_id]["investigator_relevance"] = relevance.get(node_id, 0.0)
        filtered.nodes[node_id]["investigator_explanations"] = explanations.get(node_id, [])

    assert g.number_of_nodes() == original_n
    assert g.number_of_edges() == original_e

    return InvestigatorViewResult(
        graph=filtered,
        selected_node=selected_node,
        focus=focus,
        relevance={n: relevance.get(n, 0.0) for n in filtered.nodes},
        explanations={n: explanations.get(n, []) for n in filtered.nodes},
        preserved_reasons=preserved_reasons,
        stats={
            "candidate_nodes": candidate.number_of_nodes(),
            "candidate_edges": candidate.number_of_edges(),
            "retained_nodes": filtered.number_of_nodes(),
            "retained_edges": filtered.number_of_edges(),
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "hops": hops,
            "focus": focus,
            "selected_case_count": len(selected_cases),
        },
    )


# --------------------------------------------------------------------------- #
# Investigator PyVis export (presentation only; does not alter analytical graph)
# --------------------------------------------------------------------------- #

def export_investigator_pyvis_html(
    g: nx.MultiDiGraph,
    out_path,
    selected_node: str = "",
    relevance: Optional[Dict[str, float]] = None,
    explanations: Optional[Dict[str, List[str]]] = None,
    height: str = "640px",
) -> Any:
    """
    Render an investigator subgraph with hierarchy-aware sizing and a dark legend.
    Reuses NODE_COLORS semantics from build_graph. Edge labels are omitted to
    reduce clutter; relation names remain available via edge hover titles.
    """
    from pathlib import Path

    from pyvis.network import Network

    from src.graph.build_graph import NODE_COLORS

    relevance = relevance or {}
    explanations = explanations or {}
    out_path = Path(out_path)

    net = Network(
        height=height,
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1d23",
        font_color="#c8ccd4",
        cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=95,
        spring_strength=0.04,
        damping=0.45,
    )

    max_rel = max(relevance.values()) if relevance else 1.0
    if max_rel <= 0:
        max_rel = 1.0

    for node, data in g.nodes(data=True):
        ntype = data.get("type", "Unknown")
        rel = float(data.get("investigator_relevance", relevance.get(node, 0.0)))
        norm = rel / max_rel

        if node == selected_node:
            size = 34
            border_width = 3
            border_color = "#E8EAED"
        elif ntype == "Transaction":
            size = 12 + int(norm * 16)
            border_width = 1
            border_color = "#2a2f38"
        elif ntype in CONTEXTUAL_TYPES:
            size = 8 + int(norm * 4)
            border_width = 1
            border_color = "#2a2f38"
        else:
            size = 14 + int(norm * 10)
            border_width = 1
            border_color = "#2a2f38"

        why = explanations.get(node) or data.get("investigator_explanations") or []
        why_txt = "; ".join(why[:3]) if why else ""
        title = f"{ntype}: {node}"
        if why_txt:
            title = f"{title}\n{why_txt}"

        net.add_node(
            node,
            label=str(data.get("label", node))[:18],
            title=title,
            color={
                "background": NODE_COLORS.get(ntype, "#999999"),
                "border": border_color,
                "highlight": {
                    "background": NODE_COLORS.get(ntype, "#999999"),
                    "border": "#FFFFFF",
                },
            },
            shape="dot",
            size=size,
            borderWidth=border_width,
            font={"color": "#c8ccd4", "size": 11},
        )

    for u, v, data in g.edges(data=True):
        relation = data.get("relation", "")
        # Quiet metadata edges; primary relations slightly stronger
        if relation in ("BELONGS_TO", "LOCATED_IN"):
            width = 0.6
            color = "#3a404c"
        else:
            width = 1.2
            color = "#5a6270"
        net.add_edge(
            u,
            v,
            title=relation,
            color=color,
            width=width,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # generate_html + UTF-8 write avoids Windows cp1252 failures from pyvis write_html
    html = net.generate_html(notebook=False)
    legend = """
    <div style="
        position: fixed;
        top: 12px;
        left: 12px;
        z-index: 9999;
        background: rgba(26,29,35,0.94);
        border: 1px solid #3a404c;
        border-radius: 4px;
        padding: 10px 14px;
        font-family: 'Segoe UI', Consolas, monospace;
        font-size: 12px;
        color: #c8ccd4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    ">
        <div style="font-weight: 600; margin-bottom: 8px; color: #e8eaed; letter-spacing: 0.02em;">
            Graph Legend
        </div>
        <div style="margin: 4px 0;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#4C9AFF;margin-right:7px;"></span>IP Address</div>
        <div style="margin: 4px 0;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#F5A623;margin-right:7px;"></span>Wallet</div>
        <div style="margin: 4px 0;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#7ED321;margin-right:7px;"></span>Transaction</div>
        <div style="margin: 4px 0;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#BD10E0;margin-right:7px;"></span>ASN</div>
        <div style="margin: 4px 0;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#D0021B;margin-right:7px;"></span>Country</div>
        <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #3a404c; color: #8b919c; font-size: 11px;">
            Selected node emphasized; ASN/Country shown as context
        </div>
    </div>
    """
    html = html.replace("<body>", "<body>" + legend)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Entity detail helpers (for dashboard side panel — no fabricated fields)
# --------------------------------------------------------------------------- #

def _to_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value else []
    if hasattr(value, "__iter__"):
        return list(value)
    return []


def get_entity_details(
    g: nx.MultiDiGraph,
    node_id: str,
    alerts: Optional[pd.DataFrame] = None,
    cases: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Return a structured detail dict for the selected entity using only fields
    that exist on the graph / alerts / cases. Missing sources yield empty lists.
    """
    if node_id not in g:
        return {"node_id": node_id, "type": "Unknown", "available": False}

    ntype = _node_type(g, node_id)
    label = g.nodes[node_id].get("label", _label_from_node(node_id))
    alert_index = _build_alert_index(alerts)
    node_to_cases = _build_case_maps(cases)
    neighbors = _undirected_neighbors(g, node_id)

    details: Dict[str, Any] = {
        "node_id": node_id,
        "type": ntype,
        "label": label,
        "available": True,
        "case_ids": sorted(node_to_cases.get(node_id, set())),
    }

    if ntype == "Transaction":
        txid = _tx_id_from_node(node_id) or label
        alert = alert_index.get(txid, {})
        details.update({
            "txid": txid,
            "priority": alert.get("priority"),
            "risk_score": alert.get("risk_score"),
            "anomaly_score_0_100": alert.get("anomaly_score_0_100"),
            "cluster_label": alert.get("cluster_label"),
            "src_ip": alert.get("src_ip"),
            "dst_ip": alert.get("dst_ip"),
            "geo_country": alert.get("geo_country"),
            "asn": alert.get("asn"),
            "input_addresses": _to_list(alert.get("input_addresses")),
            "output_addresses": _to_list(alert.get("output_addresses")),
            "reasons": alert.get("reasons"),
            "timestamp_str": alert.get("timestamp_str"),
            "is_anomaly_prediction": alert.get("is_anomaly_prediction"),
        })
        # Also surface graph neighbors when alert row is missing fields
        details["connected_ips"] = sorted(
            _label_from_node(n) for n in neighbors if _node_type(g, n) == "IP"
        )
        details["connected_wallets"] = sorted(
            _label_from_node(n) for n in neighbors if _node_type(g, n) == "Wallet"
        )

    elif ntype == "IP":
        tx_neighbors = [n for n in neighbors if _node_type(g, n) == "Transaction"]
        stats = _aggregate_tx_stats(tx_neighbors, alert_index, node_to_cases)
        countries = sorted({
            _label_from_node(n) for n in neighbors if _node_type(g, n) == "Country"
        })
        asns = sorted({
            _label_from_node(n) for n in neighbors if _node_type(g, n) == "ASN"
        })
        # Countries / ASNs from alert rows of connected txs when graph edges sparse
        for tx in tx_neighbors:
            txid = _tx_id_from_node(tx)
            alert = alert_index.get(txid or "", {})
            if alert.get("geo_country"):
                countries = sorted(set(countries) | {_safe_str(alert["geo_country"])})
            if alert.get("asn"):
                asns = sorted(set(asns) | {_safe_str(alert["asn"])})
        details.update({
            "ip": label,
            "connected_transaction_count": stats["n_tx"],
            "suspicious_transaction_count": stats["n_suspicious"],
            "maximum_risk": stats["max_risk"],
            "countries": countries,
            "asns": asns,
            "case_ids": sorted(set(details["case_ids"]) | stats["case_ids"]),
        })

    elif ntype == "Wallet":
        tx_neighbors = [n for n in neighbors if _node_type(g, n) == "Transaction"]
        stats = _aggregate_tx_stats(tx_neighbors, alert_index, node_to_cases)
        related_ips: Set[str] = set()
        for tx in tx_neighbors:
            for n in _undirected_neighbors(g, tx):
                if _node_type(g, n) == "IP":
                    related_ips.add(_label_from_node(n))
        details.update({
            "wallet": label,
            "connected_transaction_count": stats["n_tx"],
            "suspicious_transaction_count": stats["n_suspicious"],
            "maximum_risk": stats["max_risk"],
            "related_ips": sorted(related_ips),
            "case_ids": sorted(set(details["case_ids"]) | stats["case_ids"]),
        })

    elif ntype == "ASN":
        ips = sorted(_label_from_node(n) for n in neighbors if _node_type(g, n) == "IP")
        countries: Set[str] = set()
        tx_labels: Set[str] = set()
        for ip_node in neighbors:
            if _node_type(g, ip_node) != "IP":
                continue
            for n in _undirected_neighbors(g, ip_node):
                if _node_type(g, n) == "Country":
                    countries.add(_label_from_node(n))
                if _node_type(g, n) == "Transaction":
                    tx_labels.add(_label_from_node(n))
        details.update({
            "asn": label,
            "connected_ips": ips,
            "countries": sorted(countries),
            "relevant_transactions": sorted(tx_labels)[:50],
        })

    elif ntype == "Country":
        ips = sorted(_label_from_node(n) for n in neighbors if _node_type(g, n) == "IP")
        tx_labels: Set[str] = set()
        for ip_node in neighbors:
            if _node_type(g, ip_node) != "IP":
                continue
            for n in _undirected_neighbors(g, ip_node):
                if _node_type(g, n) == "Transaction":
                    tx_labels.add(_label_from_node(n))
        details.update({
            "country": label,
            "connected_ips": ips,
            "relevant_transactions": sorted(tx_labels)[:50],
        })

    return details
