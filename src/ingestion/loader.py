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
from typing import Any, Dict, List

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


def load_raw_data(path: Path) -> pd.DataFrame:
    """Detect format by extension and load into a normalized DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        records = load_json(path)
    elif suffix == ".csv":
        records = load_csv(path)
    elif suffix == ".xml":
        records = load_xml(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .json, .csv, or .xml.")

    df = pd.DataFrame(records, columns=CANONICAL_FIELDS)
    return df
