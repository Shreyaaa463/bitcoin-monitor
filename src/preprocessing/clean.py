"""
Cleaning and validation layer.

Produces:
- a cleaned DataFrame (invalid rows removed but never silently mutated into fake data)
- a DataQualityReport with counts of valid / invalid / duplicate / missing values,
  used verbatim by the dashboard's "Data Quality" tab (no invented numbers).

The rules are split so they can run chunk-by-chunk on arbitrarily large inputs:
  - apply_row_rules(): every rule that only looks at one row (required fields,
    timestamp validity, src_ip validity, non-empty inputs, fee default). Chunk-local.
  - duplicate detection: the only rule that needs the whole dataset; it runs over the
    (small) duplicate-key columns of ALL rows, in original input order, keep="first".
clean_dataframe() composes both for a single in-memory frame, so the chunked pipeline
and the single-frame path share exactly the same rule implementations.

Chunk-invariance notes (both are no-ops for the TRACE-X JSON format):
  - "missing" is evaluated per value (None / NaN / empty list), not per column dtype,
    because a chunk's inferred dtype can differ from the whole column's.
  - the timestamp format is inferred ONCE from the first non-null timestamp of the
    whole dataset (exactly what pandas does for a whole column) and then applied to
    every chunk, instead of being re-guessed per chunk.
"""
from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # public since pandas 2.2
    from pandas.tseries.api import guess_datetime_format
except ImportError:  # pragma: no cover - older pandas
    from pandas._libs.tslibs.parsing import guess_datetime_format

# Sentinel meaning "let pandas parse element-wise" (used when no format can be guessed).
MIXED_FORMAT = "mixed"


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


def _is_null(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    return isinstance(v, float) and math.isnan(v)


def _is_missing_value(v: Any) -> bool:
    return _is_null(v) or (isinstance(v, (list, tuple, np.ndarray)) and len(v) == 0)


def _is_blank(v: Any) -> bool:
    return _is_null(v) or (isinstance(v, str) and v == "")


# ------------------------------------------------------------------ timestamps
def infer_timestamp_format(values: Iterable[Any]) -> Optional[str]:
    """Mirror pandas' whole-column behavior: guess from the first non-null string.
    Returns None if the first non-null value is not a string (let pandas decide)."""
    for v in values:
        if _is_null(v):
            continue
        if isinstance(v, str):
            if v == "":
                continue
            return guess_datetime_format(v) or MIXED_FORMAT
        return None
    return None


def parse_timestamps(series: pd.Series, ts_format: Optional[str]) -> pd.Series:
    if ts_format is None:
        return pd.to_datetime(series, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True, format=ts_format)


# ------------------------------------------------------------------ row rules
@dataclass
class RowRuleResult:
    frame: pd.DataFrame            # input rows + timestamp_parsed + _row_valid
    total_records: int
    missing_value_counts: Dict[str, int]
    invalid_ip_count: int
    invalid_timestamp_count: int
    invalid_reasons: Dict[str, int]  # ordered exactly like the original single-pass rules


def apply_row_rules(df: pd.DataFrame, required_fields: List[str], ts_format: Optional[str]) -> RowRuleResult:
    # Non-object dtypes cannot hold lists, so for them "missing" is exactly isna().
    missing_counts = {
        col: int(sum(_is_missing_value(v) for v in df[col])) if df[col].dtype == object
        else int(df[col].isna().sum())
        for col in df.columns
    }

    working = df.copy()
    working["_row_valid"] = True
    invalid_reasons: Dict[str, int] = {}

    def mark_invalid(mask: pd.Series, reason: str):
        mask = mask.astype(bool)
        newly = mask & working["_row_valid"]
        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + int(newly.sum())
        working.loc[mask, "_row_valid"] = False

    for req in required_fields:
        if req in working.columns:
            mark_invalid(working[req].map(_is_blank), f"missing_required_field:{req}")

    parsed_ts = parse_timestamps(working["timestamp"], ts_format)
    bad_ts_mask = parsed_ts.isna()
    mark_invalid(bad_ts_mask, "invalid_timestamp")
    working["timestamp_parsed"] = parsed_ts

    src_ip_valid = working["src_ip"].apply(_is_valid_ip).astype(bool)
    mark_invalid(~src_ip_valid, "invalid_src_ip")

    empty_inputs = working["input_addresses"].apply(lambda v: not isinstance(v, list) or len(v) == 0)
    mark_invalid(empty_inputs, "empty_input_addresses")

    return RowRuleResult(
        frame=working,
        total_records=len(df),
        missing_value_counts=missing_counts,
        invalid_ip_count=int((~src_ip_valid).sum()),
        invalid_timestamp_count=int(bad_ts_mask.sum()),
        invalid_reasons=invalid_reasons,
    )


def finalize_valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep valid rows and apply the documented neutral fee default (missing -> 0.0)."""
    cleaned = frame[frame["_row_valid"]].drop(columns=["_row_valid"]).reset_index(drop=True)
    cleaned["fee"] = pd.to_numeric(cleaned["fee"], errors="coerce").fillna(0.0)
    return cleaned


def merge_row_rule_counts(parts: List[RowRuleResult | Dict[str, Any]]) -> Dict[str, Any]:
    """Sum chunk-level counts, preserving the original reason/column ordering."""
    total, ip_bad, ts_bad = 0, 0, 0
    missing: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    for p in parts:
        d = p if isinstance(p, dict) else p.__dict__
        total += d["total_records"]
        ip_bad += d["invalid_ip_count"]
        ts_bad += d["invalid_timestamp_count"]
        for k, v in d["missing_value_counts"].items():
            missing[k] = missing.get(k, 0) + int(v)
        for k, v in d["invalid_reasons"].items():
            reasons[k] = reasons.get(k, 0) + int(v)
    return {"total_records": total, "invalid_ip_count": ip_bad, "invalid_timestamp_count": ts_bad,
            "missing_value_counts": missing, "invalid_reasons": reasons}


def build_report(merged: Dict[str, Any], n_invalid_before_dedup: int, duplicate_records: int,
                 duplicates_among_valid: int, drop_exact_duplicates: bool) -> DataQualityReport:
    reasons = dict(merged["invalid_reasons"])
    invalid = n_invalid_before_dedup
    if drop_exact_duplicates:
        reasons["duplicate_txid"] = reasons.get("duplicate_txid", 0) + duplicates_among_valid
        invalid += duplicates_among_valid
    return DataQualityReport(
        total_records=merged["total_records"],
        valid_records=merged["total_records"] - invalid,
        invalid_records=invalid,
        duplicate_records=duplicate_records,
        missing_value_counts=merged["missing_value_counts"],
        invalid_ip_count=merged["invalid_ip_count"],
        invalid_timestamp_count=merged["invalid_timestamp_count"],
        invalid_reasons=reasons,
    )


def clean_dataframe(
    df: pd.DataFrame,
    required_fields: List[str],
    drop_exact_duplicates: bool = True,
    duplicate_subset: List[str] | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Single-frame cleaning (used by tests and small inputs)."""
    ts_format = infer_timestamp_format(df["timestamp"]) if "timestamp" in df.columns else None
    rr = apply_row_rules(df, required_fields, ts_format)
    working = rr.frame

    subset = [c for c in (duplicate_subset or ["txid"]) if c in working.columns]
    dup_mask = working.duplicated(subset=subset, keep="first")
    dups_among_valid = int((dup_mask & working["_row_valid"]).sum())
    n_invalid = int((~working["_row_valid"]).sum())
    if drop_exact_duplicates:
        working.loc[dup_mask, "_row_valid"] = False

    report = build_report(merge_row_rule_counts([rr]), n_invalid, int(dup_mask.sum()),
                          dups_among_valid, drop_exact_duplicates)
    return finalize_valid_rows(working), report
