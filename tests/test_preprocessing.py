import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.clean import clean_dataframe
from src.ingestion.loader import CANONICAL_FIELDS


def _base_row(**overrides):
    row = {
        "timestamp": "2025-01-01T00:00:00Z",
        "src_ip": "1.2.3.4",
        "dst_ip": "5.6.7.8",
        "src_port": 1234,
        "dst_port": 8333,
        "txid": "tx1",
        "input_addresses": ["walletA"],
        "output_addresses": ["walletB"],
        "input_amounts": [1.0],
        "output_amounts": [1.0],
        "fee": 0.0001,
        "script_type": "P2PKH",
        "geo_country": "US",
        "asn": "AS1",
        "transaction_status": "confirmed",
        "synthetic_label": 0,
        "anomaly_type": None,
        "anomaly_family": None,
    }
    row.update(overrides)
    return row


def _make_df(rows):
    return pd.DataFrame(rows, columns=CANONICAL_FIELDS)


def test_clean_valid_rows_pass():
    df = _make_df([_base_row(txid="tx1"), _base_row(txid="tx2")])
    cleaned, report = clean_dataframe(df, required_fields=["timestamp", "src_ip", "txid"])
    assert report.total_records == 2
    assert report.valid_records == 2
    assert report.invalid_records == 0
    assert len(cleaned) == 2


def test_invalid_ip_is_dropped():
    df = _make_df([_base_row(txid="tx1", src_ip="999.999.999.999")])
    cleaned, report = clean_dataframe(df, required_fields=["timestamp", "src_ip", "txid"])
    assert report.invalid_ip_count == 1
    assert report.valid_records == 0
    assert len(cleaned) == 0


def test_invalid_timestamp_is_dropped():
    df = _make_df([_base_row(txid="tx1", timestamp="not-a-date")])
    cleaned, report = clean_dataframe(df, required_fields=["timestamp", "src_ip", "txid"])
    assert report.invalid_timestamp_count == 1
    assert len(cleaned) == 0


def test_duplicate_txid_is_flagged_and_dropped():
    df = _make_df([_base_row(txid="dup"), _base_row(txid="dup")])
    cleaned, report = clean_dataframe(
        df, required_fields=["timestamp", "src_ip", "txid"],
        drop_exact_duplicates=True, duplicate_subset=["txid"],
    )
    assert report.duplicate_records == 1
    assert len(cleaned) == 1


def test_empty_input_addresses_is_invalid():
    df = _make_df([_base_row(txid="tx1", input_addresses=[])])
    cleaned, report = clean_dataframe(df, required_fields=["timestamp", "src_ip", "txid"])
    assert len(cleaned) == 0
    assert "empty_input_addresses" in report.invalid_reasons


def test_missing_fee_defaults_to_zero():
    df = _make_df([_base_row(txid="tx1", fee=None)])
    cleaned, report = clean_dataframe(df, required_fields=["timestamp", "src_ip", "txid"])
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["fee"] == 0.0
