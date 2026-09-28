"""
Ingestion layer: reads Bitcoin transaction traffic data from JSON, CSV, or XML and
normalizes every record into one common flat/semi-structured schema (a pandas
DataFrame with list-typed columns for input/output addresses and amounts).

Design notes:
- JSON is the "native" TRACE-X format (nested lists for addresses/amounts).
- CSV support assumes list-valued fields are encoded as '|'-delimited strings
  (a common flat-file convention), e.g. "addr1|addr2|addr3".
- XML support assumes a <records><record>...</record></records> structure where
  repeated child tags (e.g. multiple <input_address>) represent list fields.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pandas as pd

LIST_FIELDS = ["input_addresses", "output_addresses", "input_amounts", "output_amounts"]

CANONICAL_FIELDS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
    "input_addresses", "output_addresses", "input_amounts", "output_amounts",
    "fee", "script_type", "geo_country", "asn", "transaction_status",
    "synthetic_label", "anomaly_type", "anomaly_family",
]


def _coerce_list_field(value: Any) -> List:
    """Ensure list-valued fields are always actual python lists, regardless of source format."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return []
        # Try JSON-encoded list first (e.g. '["a","b"]')
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        # Fall back to pipe-delimited convention
        return [v for v in stripped.split("|") if v != ""]
    # Single scalar -> wrap in list
    return [value]


def _normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    rec = {field: raw.get(field) for field in CANONICAL_FIELDS}
    for lf in LIST_FIELDS:
        rec[lf] = _coerce_list_field(rec[lf])
    # amounts should be numeric lists
    for lf in ["input_amounts", "output_amounts"]:
        coerced = []
        for v in rec[lf]:
            try:
                coerced.append(float(v))
            except (TypeError, ValueError):
                coerced.append(None)
        rec[lf] = coerced
    return rec


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some exports wrap records under a top-level key, e.g. {"records": [...]}
        for key in ("records", "transactions", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError("Unrecognized JSON structure: expected a list of records.")
    if not isinstance(data, list):
        raise ValueError("Expected JSON top level to be a list of transaction records.")
    return [_normalize_record(r) for r in data]


def load_csv(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return [_normalize_record(r) for r in df.to_dict(orient="records")]


def _xml_child_texts(elem: ET.Element, tag_singular: str) -> List[str]:
    return [child.text for child in elem.findall(tag_singular) if child.text is not None]


def load_xml(path: Path) -> List[Dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()
    records = []
    record_elems = root.findall(".//record")
    if not record_elems:
        record_elems = list(root)  # fallback: assume root's direct children are records

    # Map plural canonical field -> singular XML child tag convention
    singular_map = {
        "input_addresses": "input_address",
        "output_addresses": "output_address",
        "input_amounts": "input_amount",
        "output_amounts": "output_amount",
    }

    for elem in record_elems:
        raw: Dict[str, Any] = {}
        for field in CANONICAL_FIELDS:
            if field in singular_map:
                raw[field] = _xml_child_texts(elem, singular_map[field])
            else:
                child = elem.find(field)
                raw[field] = child.text if child is not None else None
        records.append(_normalize_record(raw))
    return records


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [_normalize_record(r) for r in _iter_jsonl_raw(path)]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Detect format by extension and load into a normalized DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        records = load_json(path)
    elif suffix in (".jsonl", ".ndjson"):
        records = load_jsonl(path)
    elif suffix == ".csv":
        records = load_csv(path)
    elif suffix == ".xml":
        records = load_xml(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .json, .jsonl, .csv, or .xml.")

    df = pd.DataFrame(records, columns=CANONICAL_FIELDS)
    return df


# --------------------------------------------------------------------------- #
# Streaming (bounded-memory) readers. They produce exactly the same normalized
# records as the whole-file loaders above, just in chunks, so arbitrarily large
# files can be ingested without holding the raw file in memory.
# --------------------------------------------------------------------------- #
def _iter_jsonl_raw(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}: invalid JSON on line {line_no}: {e}") from e


def _iter_json_array_raw(path: Path, bufsize: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    """Incrementally decode a top-level JSON array of records with json.raw_decode.
    Wrapped exports ({"records": [...]}) fall back to the whole-file loader."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(bufsize).lstrip("﻿")
        pos = 0
        eof = len(buf) == 0

        def fill(p):
            nonlocal buf, eof
            more = f.read(bufsize)
            if not more:
                eof = True
            buf = buf[p:] + more
            return 0

        def skip_ws(p):
            while True:
                while p < len(buf) and buf[p] in " \t\r\n":
                    p += 1
                if p < len(buf) or eof:
                    return p
                p = fill(p)

        pos = skip_ws(pos)
        if pos >= len(buf):
            return
        if buf[pos] == "{":
            yield from _raw_records_from_whole_json(path)
            return
        if buf[pos] != "[":
            raise ValueError("Expected JSON top level to be a list of transaction records.")
        pos = skip_ws(pos + 1)
        if pos < len(buf) and buf[pos] == "]":
            return
        while True:
            pos = skip_ws(pos)
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                pos = fill(pos)
                continue
            if end >= len(buf) and not eof and not isinstance(obj, dict):
                # A scalar may have been cut at the buffer boundary; re-read with more data.
                pos = fill(pos)
                continue
            yield obj
            pos = skip_ws(end)
            if pos >= len(buf):
                raise ValueError("Unexpected end of JSON array.")
            if buf[pos] == ",":
                pos += 1
            elif buf[pos] == "]":
                return
            else:
                raise ValueError(f"Malformed JSON array near offset {pos}.")
            if pos > bufsize:  # drop already-consumed text so memory stays bounded
                buf = buf[pos:]
                pos = 0


def _raw_records_from_whole_json(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("records", "transactions", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError("Unrecognized JSON structure: expected a list of records.")
    if not isinstance(data, list):
        raise ValueError("Expected JSON top level to be a list of transaction records.")
    yield from data


def _iter_xml_raw(path: Path) -> Iterator[Dict[str, Any]]:
    singular_map = {
        "input_addresses": "input_address",
        "output_addresses": "output_address",
        "input_amounts": "input_amount",
        "output_amounts": "output_amount",
    }
    n = 0
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "record":
            continue
        raw: Dict[str, Any] = {}
        for field in CANONICAL_FIELDS:
            if field in singular_map:
                raw[field] = _xml_child_texts(elem, singular_map[field])
            else:
                child = elem.find(field)
                raw[field] = child.text if child is not None else None
        n += 1
        yield raw
        elem.clear()
    if n == 0:
        # No <record> tags: same fallback as load_xml (root's direct children are records).
        for rec in load_xml(path):
            yield rec


def iter_record_chunks(path: Path, chunk_rows: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield normalized DataFrames (columns == CANONICAL_FIELDS) of at most chunk_rows rows.
    CSV is read with dtype=str so type inference cannot differ between chunks; the
    staging layer coerces numeric fields explicitly."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for chunk in pd.read_csv(path, chunksize=chunk_rows, dtype=str, keep_default_na=True):
            records = [_normalize_record(r) for r in chunk.to_dict(orient="records")]
            yield pd.DataFrame(records, columns=CANONICAL_FIELDS)
        return

    if suffix == ".json":
        raw_iter = _iter_json_array_raw(path)
    elif suffix in (".jsonl", ".ndjson"):
        raw_iter = _iter_jsonl_raw(path)
    elif suffix == ".xml":
        raw_iter = _iter_xml_raw(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .json, .jsonl, .csv, or .xml.")

    batch: List[Dict[str, Any]] = []
    for raw in raw_iter:
        batch.append(_normalize_record(raw))  # idempotent for already-normalized records
        if len(batch) >= chunk_rows:
            yield pd.DataFrame(batch, columns=CANONICAL_FIELDS)
            batch = []
    if batch:
        yield pd.DataFrame(batch, columns=CANONICAL_FIELDS)
