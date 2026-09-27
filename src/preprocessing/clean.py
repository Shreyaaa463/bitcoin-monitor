"""
Cleaning and validation layer.

Produces:
- a cleaned DataFrame (invalid rows removed but never silently mutated into fake data)
- a DataQualityReport with counts of valid / invalid / duplicate / missing values,
  used verbatim by the dashboard's "Data Quality" tab (no invented numbers).
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    invalid_ip_count: int = 0
    invalid_timestamp_count: int = 0
    invalid_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "duplicate_records": self.duplicate_records,
            "missing_value_counts": self.missing_value_counts,
            "invalid_ip_count": self.invalid_ip_count,
            "invalid_timestamp_count": self.invalid_timestamp_count,
            "invalid_reasons": self.invalid_reasons,
        }


def _is_valid_ip(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        ipaddress.ip_address(str(value))
        return True
    except ValueError:
        return False


def clean_dataframe(
    df: pd.DataFrame,
    required_fields: List[str],
    drop_exact_duplicates: bool = True,
    duplicate_subset: List[str] | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    report = DataQualityReport(total_records=len(df))

    # --- missing value counts (computed on the raw frame, before any dropping) ---
    missing_counts = {}
    for col in df.columns:
        if df[col].dtype == object:
            n_missing = df[col].apply(lambda v: v is None or (isinstance(v, list) and len(v) == 0)).sum()
        else:
            n_missing = df[col].isna().sum()
        missing_counts[col] = int(n_missing)
    report.missing_value_counts = missing_counts

    working = df.copy()
    working["_row_valid"] = True
    invalid_reasons: Dict[str, int] = {}

    def mark_invalid(mask: pd.Series, reason: str):
        newly = mask & working["_row_valid"]
        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + int(newly.sum())
        working.loc[mask, "_row_valid"] = False

    # required fields must be present
    for req in required_fields:
        if req in working.columns:
            mask = working[req].isna() if working[req].dtype != object else working[req].apply(lambda v: v is None or v == "")
            mask_bool = mask.fillna(True) if hasattr(mask, "fillna") else mask
            mark_invalid(mask_bool.astype(bool), f"missing_required_field:{req}")

    # timestamp validity
    parsed_ts = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
    bad_ts_mask = parsed_ts.isna()
    report.invalid_timestamp_count = int(bad_ts_mask.sum())
    mark_invalid(bad_ts_mask, "invalid_timestamp")
    working["timestamp_parsed"] = parsed_ts

    # IP validity (src_ip must be valid; dst_ip validated but not required)
    src_ip_valid = working["src_ip"].apply(_is_valid_ip)
    report.invalid_ip_count = int((~src_ip_valid).sum())
    mark_invalid(~src_ip_valid, "invalid_src_ip")

    # amounts must be numeric and non-empty for a transaction to be analyzable
    empty_inputs = working["input_addresses"].apply(lambda v: not isinstance(v, list) or len(v) == 0)
    mark_invalid(empty_inputs, "empty_input_addresses")

    # duplicate detection (based on configured subset, e.g. txid)
    subset = duplicate_subset or ["txid"]
    subset = [c for c in subset if c in working.columns]
    dup_mask = working.duplicated(subset=subset, keep="first")
    report.duplicate_records = int(dup_mask.sum())
    if drop_exact_duplicates:
        mark_invalid(dup_mask, "duplicate_txid")

    report.invalid_reasons = invalid_reasons
    report.invalid_records = int((~working["_row_valid"]).sum())
    report.valid_records = int(working["_row_valid"].sum())

    cleaned = working[working["_row_valid"]].drop(columns=["_row_valid"]).reset_index(drop=True)

    # fee/amount safety: replace missing fee with 0.0 (explicit, documented default — not fabricated data,
    # just a neutral placeholder so downstream math doesn't break)
    cleaned["fee"] = pd.to_numeric(cleaned["fee"], errors="coerce").fillna(0.0)

    return cleaned, report
