import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forensics.blockchain import Blockchain, GENESIS_HASH
from src.forensics.canonical import hash_record


def test_new_ledger_has_genesis_block(tmp_path):
    chain = Blockchain(tmp_path / "chain.json")
    assert len(chain.blocks) == 1
    assert chain.blocks[0].case_id == "GENESIS"
    assert chain.blocks[0].previous_hash == GENESIS_HASH
    assert (tmp_path / "chain.json").exists()


def test_add_block_links_to_previous_hash(tmp_path):
    chain = Blockchain(tmp_path / "chain.json")
    leaves = [hash_record({"txid": "a"})]
    block = chain.add_block("CASE-1", leaves)
    assert block.previous_hash == chain.blocks[0].hash
    assert block.index == 1
    assert chain.validate_chain()["valid"] is True


def test_ledger_persists_across_reload(tmp_path):
    path = tmp_path / "chain.json"
    chain = Blockchain(path)
    chain.add_block("CASE-1", [hash_record({"a": 1})])
    chain.add_block("CASE-2", [hash_record({"b": 2})])

    reloaded = Blockchain(path)
    assert len(reloaded.blocks) == 3
    assert reloaded.validate_chain()["valid"] is True


def test_find_latest_block_for_case_returns_most_recent(tmp_path):
    chain = Blockchain(tmp_path / "chain.json")
    chain.add_block("CASE-1", [hash_record({"v": 1})])
    chain.add_block("CASE-1", [hash_record({"v": 2})])  # case re-anchored with new evidence
    latest = chain.find_latest_block_for_case("CASE-1")
    assert latest.index == 2
    assert len(chain.all_blocks_for_case("CASE-1")) == 2


def test_tampering_ledger_file_directly_fails_validation(tmp_path):
    path = tmp_path / "chain.json"
    chain = Blockchain(path)
    chain.add_block("CASE-1", [hash_record({"a": 1}), hash_record({"a": 2})])

    raw = json.loads(path.read_text())
    raw[1]["leaf_hashes"][0] = "f" * 64
    path.write_text(json.dumps(raw))

    tampered_chain = Blockchain(path)
    result = tampered_chain.validate_chain()
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_broken_previous_hash_link_detected(tmp_path):
    path = tmp_path / "chain.json"
    chain = Blockchain(path)
    chain.add_block("CASE-1", [hash_record({"a": 1})])
    chain.add_block("CASE-2", [hash_record({"a": 2})])

    raw = json.loads(path.read_text())
    raw[2]["previous_hash"] = "d" * 64  # break the link to block 1
    path.write_text(json.dumps(raw))

    tampered_chain = Blockchain(path)
    result = tampered_chain.validate_chain()
    assert result["valid"] is False
    assert any("chain link broken" in e for e in result["errors"])
