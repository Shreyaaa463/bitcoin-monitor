import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_raw_data, _coerce_list_field, CANONICAL_FIELDS


SAMPLE_RECORD = {
    "timestamp": "2025-01-01T00:00:00Z",
    "src_ip": "1.2.3.4",
    "dst_ip": "5.6.7.8",
    "src_port": 1234,
    "dst_port": 8333,
    "txid": "abc123",
    "input_addresses": ["walletA"],
    "output_addresses": ["walletB", "walletC"],
    "input_amounts": [1.5],
    "output_amounts": [0.7, 0.7],
    "fee": 0.0001,
    "script_type": "P2PKH",
    "geo_country": "US",
    "asn": "AS1234",
    "transaction_status": "confirmed",
    "synthetic_label": 0,
    "anomaly_type": None,
    "anomaly_family": None,
}


def test_coerce_list_field_variants():
    assert _coerce_list_field(None) == []
    assert _coerce_list_field([]) == []
    assert _coerce_list_field(["a", "b"]) == ["a", "b"]
    assert _coerce_list_field("a|b|c") == ["a", "b", "c"]
    assert _coerce_list_field('["a", "b"]') == ["a", "b"]
    assert _coerce_list_field("") == []


def test_load_json(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text(json.dumps([SAMPLE_RECORD]))
    df = load_raw_data(p)
    assert len(df) == 1
    assert list(df.columns) == CANONICAL_FIELDS
    assert df.iloc[0]["input_addresses"] == ["walletA"]
    assert df.iloc[0]["output_addresses"] == ["walletB", "walletC"]


def test_load_json_wrapped_key(tmp_path):
    p = tmp_path / "sample_wrapped.json"
    p.write_text(json.dumps({"records": [SAMPLE_RECORD]}))
    df = load_raw_data(p)
    assert len(df) == 1


def test_load_csv_pipe_delimited(tmp_path):
    p = tmp_path / "sample.csv"
    row = dict(SAMPLE_RECORD)
    row["input_addresses"] = "walletA"
    row["output_addresses"] = "walletB|walletC"
    row["input_amounts"] = "1.5"
    row["output_amounts"] = "0.7|0.7"
    pd.DataFrame([row]).to_csv(p, index=False)
    df = load_raw_data(p)
    assert len(df) == 1
    assert df.iloc[0]["output_addresses"] == ["walletB", "walletC"]
    assert df.iloc[0]["output_amounts"] == [0.7, 0.7]


def test_load_xml(tmp_path):
    root = ET.Element("records")
    rec = ET.SubElement(root, "record")
    for field in ["timestamp", "src_ip", "dst_ip", "txid", "fee", "geo_country", "asn",
                  "transaction_status", "script_type"]:
        child = ET.SubElement(rec, field)
        child.text = str(SAMPLE_RECORD.get(field, ""))
    for addr in SAMPLE_RECORD["input_addresses"]:
        child = ET.SubElement(rec, "input_address")
        child.text = addr
    for addr in SAMPLE_RECORD["output_addresses"]:
        child = ET.SubElement(rec, "output_address")
        child.text = addr
    for amt in SAMPLE_RECORD["input_amounts"]:
        child = ET.SubElement(rec, "input_amount")
        child.text = str(amt)
    for amt in SAMPLE_RECORD["output_amounts"]:
        child = ET.SubElement(rec, "output_amount")
        child.text = str(amt)

    p = tmp_path / "sample.xml"
    tree = ET.ElementTree(root)
    tree.write(p)

    df = load_raw_data(p)
    assert len(df) == 1
    assert df.iloc[0]["src_ip"] == "1.2.3.4"
    assert df.iloc[0]["input_addresses"] == ["walletA"]
    assert df.iloc[0]["output_addresses"] == ["walletB", "walletC"]


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("nope")
    with pytest.raises(ValueError):
        load_raw_data(p)
