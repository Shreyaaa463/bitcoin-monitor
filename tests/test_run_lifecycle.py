"""Run identity, reuse, isolation, failure handling, incremental ingestion and the
forensic tamper -> reset -> restore cycle, end to end through run_pipeline()."""
import json
import shutil

import pandas as pd
import pytest

from tests.helpers import generate_dataset, write_config
from src.forensics.blockchain import Blockchain
from src.forensics.evidence import EvidenceStore
from src.forensics.verification import verify_case
from src.pipeline import STAGES, run_pipeline
from src.runs.locks import RunLock
from src.runs.manifest import RunManifest
from src.runs.registry import RunRegistry, resolve_artifact_dir
from src.storage.parquet_store import read_frame


def _run(dataset, cfg_path, **kw):
    return run_pipeline(str(dataset), str(cfg_path), **kw)


def _alerts(run_dir):
    return read_frame(run_dir / "alerts")


def test_first_run_completes_then_identical_rerun_is_reused(tmp_path, small_dataset):
    cfg = write_config(tmp_path)
    r1 = _run(small_dataset, cfg)
    assert r1["reused"] is False
    m = RunManifest.load(r1["run_dir"])
    assert m.is_completed
    assert all(m.data["stages"][s]["status"] == "completed" for s in STAGES)
    assert m.data["dataset"]["display_name"] == "small.json"
    assert m.data["artifacts"]["immutable"], "artifact checksums recorded"
    manifest_before = (tmp_path / "runs" / r1["run_id"] / "manifest.json").read_bytes()

    r2 = _run(small_dataset, cfg)
    assert r2["reused"] is True and r2["run_id"] == r1["run_id"]
    assert len(RunRegistry(tmp_path / "runs").list_manifests()) == 1
    assert (tmp_path / "runs" / r1["run_id"] / "manifest.json").read_bytes() == manifest_before


def test_config_change_creates_new_run_and_never_touches_old_one(tmp_path, small_dataset):
    r1 = _run(small_dataset, write_config(tmp_path, "a.yaml"))
    before = (tmp_path / "runs" / r1["run_id"] / "manifest.json").read_bytes()
    r2 = _run(small_dataset, write_config(tmp_path, "b.yaml", **{"model.isolation_forest.contamination": 0.1}))
    assert r2["run_id"] != r1["run_id"] and r2["reused"] is False
    assert (tmp_path / "runs" / r1["run_id"] / "manifest.json").read_bytes() == before


def test_result_neutral_settings_do_not_change_run_identity(tmp_path, small_dataset):
    r1 = _run(small_dataset, write_config(tmp_path, "a.yaml"))
    r2 = _run(small_dataset, write_config(tmp_path, "b.yaml", **{"runtime.chunk_rows": 17,
                                                                 "dashboard.alerts_page_size": 7}))
    assert r2["reused"] is True and r2["run_id"] == r1["run_id"]


def test_force_creates_a_new_run_with_identical_results(tmp_path, small_dataset):
    cfg = write_config(tmp_path, **{"runtime.chunk_rows": 53})
    r1 = _run(small_dataset, cfg)
    r2 = _run(small_dataset, cfg, force=True)
    assert r2["run_id"] != r1["run_id"]
    a, b = _alerts(tmp_path / "runs" / r1["run_id"]), _alerts(tmp_path / "runs" / r2["run_id"])
    cols = [c for c in a.columns if c != "date"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    c1 = json.loads((tmp_path / "runs" / r1["run_id"] / "cases.json").read_text())
    c2 = json.loads((tmp_path / "runs" / r2["run_id"] / "cases.json").read_text())
    assert c1 == c2


def test_chunk_size_does_not_change_results(tmp_path, small_dataset):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    r1 = _run(small_dataset, write_config(tmp_path / "x", **{"runtime.chunk_rows": 100000}))
    r2 = _run(small_dataset, write_config(tmp_path / "y", **{"runtime.chunk_rows": 29}))
    a, b = _alerts(tmp_path / "x" / "runs" / r1["run_id"]), _alerts(tmp_path / "y" / "runs" / r2["run_id"])
    cols = [c for c in a.columns if c != "date"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_failed_run_is_marked_failed_and_never_offered(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"timestamp": "2025-01-01T00:00:00Z", "src_ip": "not-an-ip", "txid": "t1",
                                "input_addresses": ["a"], "output_addresses": ["b"],
                                "input_amounts": [1], "output_amounts": [1]}]))
    cfg = write_config(tmp_path)
    with pytest.raises(RuntimeError, match="No valid records"):
        _run(bad, cfg)
    registry = RunRegistry(tmp_path / "runs")
    (m,) = registry.list_manifests()
    assert m.status == "failed" and "No valid records" in m.data["error"]
    assert m.data["stages"]["clean"]["status"] == "failed"
    assert (m.run_dir / "error.log").exists()
    assert registry.list_completed() == []
    assert resolve_artifact_dir(tmp_path, tmp_path / "runs", m.run_id, allow_legacy=False) == (None, None)


def test_interrupted_run_is_reaped_as_failed(tmp_path, small_dataset):
    registry = RunRegistry(tmp_path / "runs")
    run_id, run_dir = registry.new_run_dir("deadbeef" * 3)
    RunManifest.create(run_dir, run_id=run_id, run_key="deadbeef" * 3)  # status=running, no live owner
    assert registry.reap_stale_runs() == [run_id]
    assert RunManifest.load(run_dir).status == "failed"


def test_running_run_with_live_lock_is_not_reaped(tmp_path):
    registry = RunRegistry(tmp_path / "runs")
    key = "cafebabe" * 3
    run_id, run_dir = registry.new_run_dir(key)
    RunManifest.create(run_dir, run_id=run_id, run_key=key)
    holder = registry.lock_for(key)
    assert holder.try_acquire()
    try:
        assert registry.reap_stale_runs() == []
        assert RunManifest.load(run_dir).status == "running"
    finally:
        holder.release()


def test_run_lock_is_exclusive(tmp_path):
    a, b = RunLock(tmp_path / "k.lock"), RunLock(tmp_path / "k.lock")
    assert a.try_acquire()
    assert not b.try_acquire()
    assert b.is_locked_by_other()
    a.release()
    assert b.try_acquire()
    b.release()


def test_directory_dataset_only_stages_new_files(tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    generate_dataset(ds / "day1.json", n=300, seed=1)
    cfg = write_config(tmp_path)
    r1 = _run(ds, cfg)
    assert r1["manifest"]["stages"]["ingest"]["cache_hits"] == 0
    generate_dataset(ds / "day2.jsonl", n=200, seed=2, fmt="jsonl")
    r2 = _run(ds, cfg)
    assert r2["run_id"] != r1["run_id"]
    ingest = r2["manifest"]["stages"]["ingest"]
    assert ingest["n_files"] == 2 and ingest["cache_hits"] == 1
    assert ingest["n_raw_records"] == 500


def test_tamper_is_never_silently_fixed_but_explicit_reset_restores(tmp_path, small_dataset):
    cfg = write_config(tmp_path)
    r = _run(small_dataset, cfg)
    run_dir = tmp_path / "runs" / r["run_id"]
    store_path = run_dir / "forensics" / "evidence_store.json"
    ledger_path = run_dir / "forensics" / "blockchain_ledger.json"
    raw = json.loads(store_path.read_text())
    case_id = next(iter(raw))
    ledger_len = len(json.loads(ledger_path.read_text()))

    def verify():
        return verify_case(case_id, EvidenceStore(store_path), Blockchain(ledger_path), expected_run_id=r["run_id"])

    assert verify()["status"] == "VERIFIED"

    raw[case_id]["evidence_records"][0]["risk_score"] = 0.0
    store_path.write_text(json.dumps(raw))
    assert verify()["status"] == "FAILED"

    assert _run(small_dataset, cfg)["reused"] is True           # re-running does NOT repair it
    assert verify()["status"] == "FAILED"

    raw = json.loads(store_path.read_text())
    del raw[case_id]                                             # explicit, deliberate reset
    store_path.write_text(json.dumps(raw))
    assert verify()["status"] == "FAILED"                        # missing evidence != anchored

    assert _run(small_dataset, cfg)["reused"] is True           # restores from the run's alerts
    assert verify()["status"] == "VERIFIED"
    assert len(json.loads(ledger_path.read_text())) == ledger_len  # nothing re-anchored


def test_evidence_verification_is_bound_to_its_run(tmp_path, small_dataset):
    r = _run(small_dataset, write_config(tmp_path))
    run_dir = tmp_path / "runs" / r["run_id"]
    store = EvidenceStore(run_dir / "forensics" / "evidence_store.json")
    chain = Blockchain(run_dir / "forensics" / "blockchain_ledger.json")
    case_id = store.all_case_ids()[0]
    assert chain.run_id == r["run_id"]
    assert verify_case(case_id, store, chain, expected_run_id=r["run_id"])["status"] == "VERIFIED"
    res = verify_case(case_id, store, chain, expected_run_id="some-other-run")
    assert res["status"] == "FAILED" and not res["run_binding_ok"]


def test_modified_completed_run_is_not_reused(tmp_path, small_dataset):
    cfg = write_config(tmp_path)
    r1 = _run(small_dataset, cfg)
    (tmp_path / "runs" / r1["run_id"] / "cases.json").write_text("[]")  # artifact changed after completion
    r2 = _run(small_dataset, cfg)
    assert r2["reused"] is False and r2["run_id"] != r1["run_id"]


def test_large_dataset_mode_uses_candidate_funnel(tmp_path, small_dataset):
    cfg = write_config(tmp_path, **{"scale.exact_mode_max_rows": 100})
    r = _run(small_dataset, cfg)
    m = r["manifest"]
    assert m["resolved_settings"]["large_dataset_mode"] is True
    assert m["stages"]["clustering"]["scope"] == "candidates"
    run_dir = tmp_path / "runs" / r["run_id"]
    a = _alerts(run_dir)
    seeds = a["priority"].isin(["HIGH", "MEDIUM"]) | a["is_anomaly_prediction"]
    assert (a.loc[~seeds, "cluster_label"] == -2).all()
    assert (a.loc[seeds, "cluster_label"] >= -1).all()
    assert m["funnel"]["n_clustered"] == int(seeds.sum()) < len(a)
    cases = json.loads((run_dir / "cases.json").read_text())
    assert all(c["num_transactions"] > 1 or c["case_priority"] == "HIGH" for c in cases)
    assert not (run_dir / "entity_graph.pickle").exists()
    assert (run_dir / "graph_index.sqlite").exists()
