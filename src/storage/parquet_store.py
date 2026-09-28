"""
Partitioned Parquet storage for large intermediate/processed datasets.

Transactions are stored as a hive-partitioned Parquet dataset partitioned by UTC
event date:

    <root>/date=YYYY-MM-DD/part-<file>-<chunk>.parquet

An explicit Arrow schema is enforced on every write, so partitions written from
different chunks/files always have identical column types (per-chunk pandas type
inference can otherwise disagree, e.g. an all-null column in one chunk).

`_row_no` is the record's global position in the original input (file order, then
row order). The analysis stages sort by it, so results do not depend on how the
data happened to be partitioned or chunked.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

ROW_NO = "_row_no"
LIST_STR_COLS = ("input_addresses", "output_addresses")
LIST_FLOAT_COLS = ("input_amounts", "output_amounts")
STR_COLS = ("timestamp", "src_ip", "dst_ip", "txid", "script_type", "geo_country", "asn",
            "transaction_status", "anomaly_type", "anomaly_family")
INT_COLS = ("src_port", "dst_port", "synthetic_label")

TX_FIELDS: List[tuple] = [
    ("timestamp", pa.string()), ("src_ip", pa.string()), ("dst_ip", pa.string()),
    ("src_port", pa.int64()), ("dst_port", pa.int64()), ("txid", pa.string()),
    ("input_addresses", pa.list_(pa.string())), ("output_addresses", pa.list_(pa.string())),
    ("input_amounts", pa.list_(pa.float64())), ("output_amounts", pa.list_(pa.float64())),
    ("fee", pa.float64()), ("script_type", pa.string()), ("geo_country", pa.string()),
    ("asn", pa.string()), ("transaction_status", pa.string()), ("synthetic_label", pa.int64()),
    ("anomaly_type", pa.string()), ("anomaly_family", pa.string()),
    ("timestamp_parsed", pa.timestamp("us", tz="UTC")),
]


def tx_schema(row_col: str = ROW_NO) -> pa.Schema:
    return pa.schema(TX_FIELDS + [(row_col, pa.int64())])


# ------------------------------------------------------------------ coercion
def _null(v: Any) -> bool:
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v))


def _to_str(v: Any) -> Optional[str]:
    return None if _null(v) else str(v)


def _to_int(v: Any) -> Optional[int]:
    if _null(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def frame_to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Coerce a transaction frame to `schema` value-by-value (identifiers -> str,
    ports/labels -> int, list elements -> str/float)."""
    arrays = []
    for f in schema:
        name = f.name
        col = df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index)
        if name in LIST_STR_COLS:
            data = [[None if _null(x) else str(x) for x in (v if v is not None else [])] for v in col]
        elif name in LIST_FLOAT_COLS:
            data = [[_to_float(x) for x in (v if v is not None else [])] for v in col]
        elif name in STR_COLS:
            data = [_to_str(v) for v in col]
        elif name in INT_COLS:
            data = [_to_int(v) for v in col]
        elif pa.types.is_timestamp(f.type):
            s = pd.to_datetime(col, utc=True)
            arrays.append(pa.array(s, type=f.type, from_pandas=True, safe=False))
            continue
        else:
            arrays.append(pa.array(col, type=f.type, from_pandas=True))
            continue
        arrays.append(pa.array(data, type=f.type, from_pandas=True))
    return pa.Table.from_arrays(arrays, schema=schema)


def lists_to_python(df: pd.DataFrame, cols: Iterable[str] = LIST_STR_COLS + LIST_FLOAT_COLS) -> pd.DataFrame:
    """Parquet round-trips list cells as numpy arrays; convert back to python lists so
    feature/case code sees exactly what the in-memory prototype saw."""
    for c in cols:
        if c in df.columns:
            df[c] = [v.tolist() if isinstance(v, np.ndarray) else (list(v) if v is not None else [])
                     for v in df[c]]
    return df


# ------------------------------------------------------------------ writing
def write_table(table: pa.Table, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def write_date_partitioned(df: pd.DataFrame, root: Path, part_name: str, schema: pa.Schema,
                           ts_col: str = "timestamp_parsed") -> List[str]:
    """Write df under root/date=YYYY-MM-DD/<part_name>.parquet (one file per date)."""
    if len(df) == 0:
        return []
    dates = pd.to_datetime(df[ts_col], utc=True).dt.strftime("%Y-%m-%d")
    written = []
    for date_value, idx in dates.groupby(dates, sort=True).groups.items():
        sub = df.loc[idx]
        rel = f"date={date_value}/{part_name}.parquet"
        write_table(frame_to_table(sub, schema), Path(root) / rel)
        written.append(rel)
    return written


# ------------------------------------------------------------------ reading
def dataset(root: Path) -> pads.Dataset:
    return pads.dataset(str(root), format="parquet", partitioning="hive")


def read_frame(root: Path, columns: Optional[List[str]] = None, filter=None,
               sort_by_row: bool = True) -> pd.DataFrame:
    """Read (a projection of) a partitioned dataset. Never reads columns not asked for."""
    root = Path(root)
    if root.is_file():  # legacy single-file artifact
        df = pd.read_parquet(root, columns=columns) if root.suffix == ".parquet" else pd.read_csv(root)
        if filter is not None:
            df = pads.dataset(pa.Table.from_pandas(df, preserve_index=False)).to_table(filter=filter).to_pandas()
        return df
    ds_ = dataset(root)
    cols = list(columns) if columns is not None else [n for n in ds_.schema.names if n != "date"]
    if sort_by_row and ROW_NO in ds_.schema.names and ROW_NO not in cols:
        cols = cols + [ROW_NO]
    table = ds_.to_table(columns=cols, filter=filter)
    df = table.to_pandas()
    if sort_by_row and ROW_NO in df.columns:
        df = df.sort_values(ROW_NO, kind="stable").reset_index(drop=True)
    return df


def iter_fragments(root: Path) -> List[Path]:
    """All parquet files of a partitioned dataset, in deterministic (path) order."""
    return sorted(Path(root).rglob("*.parquet"), key=lambda p: p.relative_to(root).as_posix())


def count_rows(root: Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    if root.is_file():
        return pq.ParquetFile(root).metadata.num_rows
    return sum(pq.ParquetFile(p).metadata.num_rows for p in iter_fragments(root))
