import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correlation.correlate import Edge
from src.graph.build_graph import build_entity_graph
from src.forensics.cases import build_cases
from src.forensics.evidence import EvidenceStore, build_case_evidence_records
from src.forensics.anchoring import anchor_cases
from src.forensics.blockchain import Blockchain
from src.forensics.verification import generate_proof_for_txid, verify_case


def _row(txid, src_ip, ts, cluster_label, priority, wallets_in, wallets_out,
         risk_score=60.0, anomaly_score=70.0, is_anomaly=True, reasons="test reason"):
    return dict(
        txid=txid, src_ip=src_ip, dst_ip="9.9.9.9",
        input_addresses=wallets_in, output_addresses=wallets_out,
        geo_country="US", asn="AS1", cluster_label=cluster_label,
        anomaly_score_0_100=anomaly_score, risk_score=risk_score, priority=priority,
        anomaly_type=None, reasons=reasons,
        timestamp_parsed=pd.Timestamp(ts), timestamp_str=ts,
        is_anomaly_prediction=is_anomaly,
    )


def _graph_from_rows(rows):
    edges = []
    for r in rows:
        edges.append(Edge(f"ip:{r['src_ip']}", f"tx:{r['txid']}", "OBSERVED", "IP", "Transaction", {}))
        for w in r["input_addresses"]:
            edges.append(Edge(f"tx:{r['txid']}", f"wallet:{w}", "INPUT", "Transaction", "Wallet", {}))
        for w in r["output_addresses"]:
            edges.append(Edge(f"tx:{r['txid']}", f"wallet:{w}", "OUTPUT", "Transaction", "Wallet", {}))
    return build_entity_graph(edges)


def test_same_cluster_far_apart_in_time_is_not_one_case():
    """Two HIGH-priority transactions in the SAME cluster, with no shared wallet/IP,
    and far outside the temporal window, must NOT be merged into one case -- proving
    a DBSCAN cluster is not simply equated with a case."""
    rows = [
        _row("tx1", "1.1.1.1", "2025-01-01T00:00:00Z", cluster_label=5, priority="HIGH",
             wallets_in=["wA"], wallets_out=["wB"]),
        _row("tx2", "2.2.2.2", "2025-06-01T00:00:00Z", cluster_label=5, priority="HIGH",
             wallets_in=["wC"], wallets_out=["wD"]),
    ]
    alerts = pd.DataFrame(rows)
    g = _graph_from_rows(rows)
    cases, _ = build_cases(alerts, g, seed_priorities=["HIGH", "MEDIUM"],
                            include_ml_flagged=True, temporal_window_hours=2, min_case_size=1)
    assert len(cases) == 2  # NOT merged just because they share a cluster
    for c in cases:
        assert len(c.linked_txids) == 1


def test_same_cluster_within_window_is_linked():
    rows = [
        _row("tx1", "1.1.1.1", "2025-01-01T00:00:00Z", cluster_label=5, priority="HIGH",
             wallets_in=["wA"], wallets_out=["wB"]),
        _row("tx2", "2.2.2.2", "2025-01-01T00:30:00Z", cluster_label=5, priority="HIGH",
             wallets_in=["wC"], wallets_out=["wD"]),
    ]
    alerts = pd.DataFrame(rows)
    g = _graph_from_rows(rows)
    cases, _ = build_cases(alerts, g, seed_priorities=["HIGH", "MEDIUM"],
                            include_ml_flagged=True, temporal_window_hours=2, min_case_size=1)
    assert len(cases) == 1
    assert set(cases[0].linked_txids) == {"tx1", "tx2"}
    assert cases[0].link_type_counts.get("cluster_temporal") == 1


def test_shared_wallet_links_transactions_across_different_ips_and_clusters():
    rows = [
        _row("tx1", "1.1.1.1", "2025-01-01T00:00:00Z", cluster_label=-1, priority="HIGH",
             wallets_in=["shared_wallet"], wallets_out=["wB"]),
        _row("tx2", "9.9.9.9", "2025-03-01T00:00:00Z", cluster_label=2, priority="MEDIUM",
             wallets_in=["wC"], wallets_out=["shared_wallet"]),
    ]
    alerts = pd.DataFrame(rows)
    g = _graph_from_rows(rows)
    cases, _ = build_cases(alerts, g, seed_priorities=["HIGH", "MEDIUM"],
                            include_ml_flagged=True, temporal_window_hours=2, min_case_size=1)
    assert len(cases) == 1
    assert cases[0].link_type_counts.get("shared_wallet") == 1


def test_case_id_is_deterministic_regardless_of_row_order():
    rows = [
        _row("tx1", "1.1.1.1", "2025-01-01T00:00:00Z", cluster_label=-1, priority="HIGH",
             wallets_in=["w1"], wallets_out=["w2"]),
        _row("tx2", "1.1.1.1", "2025-01-01T00:05:00Z", cluster_label=-1, priority="HIGH",
             wallets_in=["w3"], wallets_out=["w4"]),
    ]
    alerts_fwd = pd.DataFrame(rows)
    alerts_rev = pd.DataFrame(list(reversed(rows)))
    g = _graph_from_rows(rows)

    cases_fwd, _ = build_cases(alerts_fwd, g, ["HIGH"], True, 2, 1)
    cases_rev, _ = build_cases(alerts_rev, g, ["HIGH"], True, 2, 1)
    assert cases_fwd[0].case_id == cases_rev[0].case_id


def test_low_priority_non_flagged_transactions_are_not_seeds():
    rows = [
        _row("tx1", "1.1.1.1", "2025-01-01T00:00:00Z", cluster_label=-1, priority="LOW",
             wallets_in=["w1"], wallets_out=["w2"], is_anomaly=False),
    ]
    alerts = pd.DataFrame(rows)
    g = _graph_from_rows(rows)
    cases, _ = build_cases(alerts, g, seed_priorities=["HIGH", "MEDIUM"],
                            include_ml_flagged=True, temporal_window_hours=2, min_case_size=1)
    assert len(cases) == 0


@pytest.fixture
def anchored_case(tmp_path):
    rows = [
        _row("tx1", "1.1.1.1", "2025-01-01T00:00:00Z", cluster_label=-1, priority="HIGH",
             wallets_in=["w1"], wallets_out=["w2"]),
        _row("tx2", "1.1.1.1", "2025-01-01T00:05:00Z", cluster_label=-1, priority="MEDIUM",
             wallets_in=["w3"], wallets_out=["w4"]),
    ]
    alerts = pd.DataFrame(rows)
    g = _graph_from_rows(rows)
    cases, _ = build_cases(alerts, g, ["HIGH", "MEDIUM"], True, 2, 1)
    store = EvidenceStore(tmp_path / "evidence_store.json")
    chain = Blockchain(tmp_path / "chain.json")
    anchor_cases(cases, alerts, store, chain)
    return {"tmp_path": tmp_path, "case": cases[0], "alerts": alerts}


def test_fresh_anchor_verifies_successfully(anchored_case):
    store = EvidenceStore(anchored_case["tmp_path"] / "evidence_store.json")
    chain = Blockchain(anchored_case["tmp_path"] / "chain.json")
    result = verify_case(anchored_case["case"].case_id, store, chain)
    assert result["overall_verified"] is True
    assert result["status"] == "VERIFIED"


def test_tampering_one_field_fails_verification(anchored_case):
    path = anchored_case["tmp_path"] / "evidence_store.json"
    case_id = anchored_case["case"].case_id

    raw = json.loads(path.read_text())
    raw[case_id]["evidence_records"][0]["risk_score"] = 999.0
    path.write_text(json.dumps(raw))

    store = EvidenceStore(path)
    chain = Blockchain(anchored_case["tmp_path"] / "chain.json")
    result = verify_case(case_id, store, chain)
    assert result["overall_verified"] is False
    assert result["status"] == "FAILED"
    assert result["merkle_root_match"] is False
    assert any(not r["match"] for r in result["per_txid_results"])


def test_rerunning_pipeline_does_not_silently_fix_tampered_evidence(anchored_case):
    """Critical security property: re-anchoring the SAME unchanged source alerts
    must not overwrite evidence that was tampered directly on disk."""
    path = anchored_case["tmp_path"] / "evidence_store.json"
    case_id = anchored_case["case"].case_id

    raw = json.loads(path.read_text())
    raw[case_id]["evidence_records"][0]["risk_score"] = 999.0
    path.write_text(json.dumps(raw))

    store = EvidenceStore(path)
    chain = Blockchain(anchored_case["tmp_path"] / "chain.json")
    anchor_cases([anchored_case["case"]], anchored_case["alerts"], store, chain)

    store2 = EvidenceStore(path)
    chain2 = Blockchain(anchored_case["tmp_path"] / "chain.json")
    result = verify_case(case_id, store2, chain2)
    assert result["overall_verified"] is False


def test_merkle_proof_for_txid_fails_after_tampering(anchored_case):
    path = anchored_case["tmp_path"] / "evidence_store.json"
    case_id = anchored_case["case"].case_id
    txid = anchored_case["case"].linked_txids[0]

    store = EvidenceStore(path)
    chain = Blockchain(anchored_case["tmp_path"] / "chain.json")
    proof_before = generate_proof_for_txid(case_id, txid, store, chain)
    assert proof_before["proof_reconstructs_anchored_root"] is True

    raw = json.loads(path.read_text())
    raw[case_id]["evidence_records"][0]["priority"] = "LOW"
    path.write_text(json.dumps(raw))

    store2 = EvidenceStore(path)
    chain2 = Blockchain(anchored_case["tmp_path"] / "chain.json")
    proof_after = generate_proof_for_txid(case_id, txid, store2, chain2)
    assert proof_after["proof_reconstructs_anchored_root"] is False
    assert proof_after["hash_match"] is False


def test_verify_unanchored_case_returns_not_anchored(tmp_path):
    store = EvidenceStore(tmp_path / "evidence_store.json")
    chain = Blockchain(tmp_path / "chain.json")
    result = verify_case("CASE-DOES-NOT-EXIST", store, chain)
    assert result["status"] == "NOT_ANCHORED"
    assert result["overall_verified"] is False
