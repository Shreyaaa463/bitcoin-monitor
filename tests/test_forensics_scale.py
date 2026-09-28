"""Forensic-layer additions: backward-compatible run binding, batched atomic ledger writes,
graph-free case building, pair caps, singleton policy, and vectorized explanations."""
import json

import numpy as np
import pandas as pd
import pytest

from src.correlation.correlate import Edge
from src.explainability.explain import generate_all_reasons, generate_reasons_for_row, compute_zscores
from src.features.engineer import ALL_FEATURE_COLUMNS
from src.forensics.blockchain import Block, Blockchain
from src.forensics.canonical import canonical_json_bytes, sha256_hex
from src.forensics.cases import build_cases
from src.graph.build_graph import build_entity_graph


def test_legacy_block_hash_is_unchanged_by_run_id_support(tmp_path):
    chain = Blockchain(tmp_path / "legacy.json")
    b = chain.add_block("CASE-X", ["ab" * 32])
    legacy_header = {k: getattr(b, k) for k in ("index", "timestamp", "case_id", "merkle_root",
                                                "leaf_hashes", "evidence_count", "previous_hash")}
    assert b.hash == sha256_hex(canonical_json_bytes(legacy_header))
    assert "run_id" not in json.loads((tmp_path / "legacy.json").read_text())[1]
    assert Blockchain(tmp_path / "legacy.json").validate_chain()["valid"]


def test_run_bound_ledger_detects_run_id_edit(tmp_path):
    p = tmp_path / "bound.json"
    chain = Blockchain(p, run_id="RUN-A")
    chain.add_block("CASE-1", ["cd" * 32])
    assert Blockchain(p).validate_chain(expected_run_id="RUN-A")["valid"]
    assert not Blockchain(p).validate_chain(expected_run_id="RUN-B")["valid"]
    raw = json.loads(p.read_text())
    raw[1]["run_id"] = "RUN-B"  # move a block to another run
    p.write_text(json.dumps(raw))
    assert not Blockchain(p).validate_chain()["valid"]


def test_deferred_save_writes_once_and_rolls_back_on_error(tmp_path):
    p = tmp_path / "c.json"
    chain = Blockchain(p, run_id="R")
    with chain.deferred_save():
        for i in range(5):
            chain.add_block(f"C{i}", [f"{i:02x}" * 32])
        assert len(json.loads(p.read_text())) == 1  # nothing persisted mid-batch
    assert len(json.loads(p.read_text())) == 6
    with pytest.raises(ValueError):
        with chain.deferred_save():
            chain.add_block("C9", ["ee" * 32])
            raise ValueError("boom")
    assert len(json.loads(p.read_text())) == 6 and len(chain.blocks) == 6
    assert chain.find_latest_block_for_case("C9") is None
    assert chain.find_latest_block_for_case("C3").index == 4


def _row(txid, ip, ts, cl, prio, win, wout, flagged=True):
    return dict(txid=txid, src_ip=ip, dst_ip="9.9.9.9", input_addresses=win, output_addresses=wout,
                geo_country="US", asn="AS1", cluster_label=cl, anomaly_score_0_100=50.0, risk_score=50.0,
                priority=prio, anomaly_type=None, reasons="r", timestamp_parsed=pd.Timestamp(ts, tz="UTC"),
                timestamp_str=ts, is_anomaly_prediction=flagged)


def _graph(rows):
    edges = []
    for r in rows:
        for w in r["input_addresses"]:
            edges.append(Edge(f"tx:{r['txid']}", f"wallet:{w}", "INPUT", "Transaction", "Wallet", {}))
        for w in r["output_addresses"]:
            edges.append(Edge(f"tx:{r['txid']}", f"wallet:{w}", "OUTPUT", "Transaction", "Wallet", {}))
    return build_entity_graph(edges)


def _random_rows(n=120, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        rows.append(_row(f"tx{i:04d}", f"10.0.0.{rng.randint(0, 8)}",
                         (pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=int(rng.randint(0, 3000)))).isoformat(),
                         int(rng.randint(-1, 4)), ["HIGH", "MEDIUM", "LOW"][rng.randint(0, 3)],
                         [f"w{rng.randint(0, 60)}"], [f"w{rng.randint(0, 60)}", f"w{rng.randint(0, 60)}"],
                         flagged=bool(rng.randint(0, 2))))
    return rows


def test_graph_free_case_building_matches_graph_path():
    rows = _random_rows()
    alerts = pd.DataFrame(rows)
    a, ea = build_cases(alerts, _graph(rows), ["HIGH", "MEDIUM"], True, 2, 1)
    b, eb = build_cases(alerts, None, ["HIGH", "MEDIUM"], True, 2, 1)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_pair_cap_keeps_identical_cases():
    rows = _random_rows(n=200, seed=1)
    alerts = pd.DataFrame(rows)
    full, _ = build_cases(alerts, None, ["HIGH", "MEDIUM"], True, 2, 1)
    capped, _ = build_cases(alerts, None, ["HIGH", "MEDIUM"], True, 2, 1, max_link_pairs_per_group=3)
    assert [(c.case_id, c.linked_txids) for c in full] == [(c.case_id, c.linked_txids) for c in capped]
    assert sum(len(c.link_edges) for c in capped) < sum(len(c.link_edges) for c in full)


def test_singleton_policy_drops_only_unlinked_low_priority_cases():
    rows = [_row("a", "1.1.1.1", "2025-01-01T00:00:00", -1, "MEDIUM", ["x"], ["y"]),
            _row("b", "2.2.2.2", "2025-03-01T00:00:00", -1, "HIGH", ["p"], ["q"]),
            _row("c", "3.3.3.3", "2025-05-01T00:00:00", -1, "MEDIUM", ["m"], ["n"]),
            _row("d", "3.3.3.3", "2025-05-02T00:00:00", -1, "MEDIUM", ["k"], ["l"])]
    cases, _ = build_cases(pd.DataFrame(rows), None, ["HIGH", "MEDIUM"], True, 2, 1,
                           singleton_min_priority="HIGH")
    assert sorted(tuple(c.linked_txids) for c in cases) == [("b",), ("c", "d")]


def test_vectorized_reasons_match_per_row_reasons():
    rng = np.random.RandomState(3)
    n = 300
    feats = pd.DataFrame(rng.gamma(1.5, 2.0, size=(n, len(ALL_FEATURE_COLUMNS))), columns=ALL_FEATURE_COLUMNS)
    feats.iloc[::7, 3] = 40.0  # some extreme fan-out rows
    flagged = pd.Series(rng.rand(n) > 0.7)
    labels = pd.Series(rng.randint(-2, 5, size=n))
    scores = pd.Series(rng.rand(n) * 100)
    got = generate_all_reasons(feats, flagged, labels, scores, top_n=5, z_threshold=1.5)
    z = compute_zscores(feats)
    for i in range(n):
        exp = " | ".join(generate_reasons_for_row(feats.loc[i], z.loc[i], bool(flagged[i]), int(labels[i]),
                                                  float(scores[i]), 5, 1.5))
        assert got[i] == exp
