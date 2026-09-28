"""Streaming readers and chunked cleaning produce exactly what the whole-file path produces."""
import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from src.ingestion import loader
from src.ingestion.loader import CANONICAL_FIELDS, iter_record_chunks, load_raw_data
from src.ingestion.staging import build_cleaned_dataset, detect_dataset_timestamp_format, stage_source_file
from src.preprocessing.clean import clean_dataframe
from src.runs.fsutil import sha256_file
from src.storage.parquet_store import lists_to_python, read_frame

from tests.helpers import generate_dataset

REQ = ["timestamp", "src_ip", "txid"]


def _concat(chunks):
    return pd.concat(list(chunks), ignore_index=True)


def _same(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in a.columns:
        for x, y in zip(a[c].tolist(), b[c].tolist()):
            assert x == y or (pd.isna(x) and pd.isna(y)) if not isinstance(x, list) else x == y, c


@pytest.mark.parametrize("bufsize", [7, 64, 1 << 20])
def test_json_array_stream_matches_json_load(tmp_path, bufsize):
    p = generate_dataset(tmp_path / "d.json", n=150, seed=3)
    streamed = list(loader._iter_json_array_raw(p, bufsize=bufsize))
    assert streamed == json.loads(p.read_text())


def test_wrapped_json_and_empty_array(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"records": [{"txid": "a"}, {"txid": "b"}]}))
    assert [r["txid"] for r in loader._iter_json_array_raw(p)] == ["a", "b"]
    e = tmp_path / "e.json"
    e.write_text("  [ ]  ")
    assert list(loader._iter_json_array_raw(e)) == []


@pytest.mark.parametrize("chunk_rows", [1, 13, 10_000])
def test_chunks_equal_whole_file_json_and_jsonl(tmp_path, chunk_rows):
    js = generate_dataset(tmp_path / "d.json", n=120, seed=4)
    jl = tmp_path / "d.jsonl"  # the same records, one per line
    jl.write_text("".join(json.dumps(r) + "\n" for r in json.loads(js.read_text())))
    whole = load_raw_data(js)
    _same(_concat(iter_record_chunks(js, chunk_rows)), whole)
    _same(_concat(iter_record_chunks(jl, chunk_rows)), whole)


def test_xml_stream_matches_whole_file(tmp_path):
    root = ET.Element("records")
    for i in range(5):
        rec = ET.SubElement(root, "record")
        for f, v in [("timestamp", "2025-01-01T00:00:00Z"), ("src_ip", f"1.2.3.{i}"), ("txid", f"t{i}")]:
            ET.SubElement(rec, f).text = v
        ET.SubElement(rec, "input_address").text = f"w{i}"
        ET.SubElement(rec, "input_amount").text = "1.5"
    p = tmp_path / "d.xml"
    ET.ElementTree(root).write(p)
    _same(_concat(iter_record_chunks(p, 2)), load_raw_data(p))


@pytest.mark.parametrize("chunk_rows", [3, 50, 10_000])
def test_staged_cleaning_equals_single_frame_cleaning(tmp_path, chunk_rows):
    p = generate_dataset(tmp_path / "d.json", n=250, seed=5)  # includes invalid IP/timestamp/dup/empty rows
    ref_df, ref_report = clean_dataframe(load_raw_data(p), REQ, True, ["txid"])
    meta = stage_source_file(p, sha256_file(p), tmp_path / "cache", REQ, ["txid"],
                             detect_dataset_timestamp_format(p), chunk_rows)
    report, info = build_cleaned_dataset([meta], tmp_path / "cleaned", ["txid"], True)
    assert report.to_dict() == ref_report.to_dict()
    new = lists_to_python(read_frame(tmp_path / "cleaned"))
    _same(new[list(ref_df.columns)], ref_df)
    assert ref_report.invalid_records >= 4 and ref_report.duplicate_records >= 1


def test_duplicates_across_files_keep_first_file(tmp_path):
    p1 = generate_dataset(tmp_path / "a.json", n=60, seed=8)
    p2 = tmp_path / "b.json"
    p2.write_text(p1.read_text())  # every record duplicated in the second file
    fmt = detect_dataset_timestamp_format(p1)
    metas = [stage_source_file(p, sha256_file(p), tmp_path / "cache", REQ, ["txid"], fmt, 25) for p in (p1, p2)]
    assert metas[1]["cache_hit"] is True  # identical content -> staged once
    report, _ = build_cleaned_dataset(metas, tmp_path / "cleaned", ["txid"], True)
    single, single_rep = clean_dataframe(load_raw_data(p1), REQ, True, ["txid"])
    assert report.valid_records == single_rep.valid_records
    assert report.duplicate_records == single_rep.duplicate_records + 60
