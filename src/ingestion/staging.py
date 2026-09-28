"""
Incremental, bounded-memory ingestion + row-level cleaning.

Stage A (per source file, cached):
    raw file --stream in chunks--> normalize --> row-level cleaning rules
    --> <cache>/<entry_key>/chunk-NNNNN.parquet   (row-valid records)
        <cache>/<entry_key>/index-NNNNN.parquet   (every record: _local_row, _row_valid, dedup keys)
        <cache>/<entry_key>/meta.json             (row counts + data-quality partial counts)

    entry_key = sha256(file SHA-256, PIPELINE_VERSION, staging schema version,
                       required_fields, duplicate_subset, timestamp format)
    A source file whose content was already staged under the same rules is never
    re-parsed: appending a new file to a directory dataset only stages the new file.

Stage B (per run, see build_cleaned_dataset):
    global duplicate detection over the dedup-key columns of ALL records, in
    original order (file order, then row order), keep="first" -- identical to the
    single-frame rule -- then the surviving rows are written, date-partitioned, into
    the run's own cleaned dataset with a global `_row_no`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.loader import iter_record_chunks
from src.preprocessing.clean import (apply_row_rules, build_report, finalize_valid_rows,
                                     infer_timestamp_format, merge_row_rule_counts, DataQualityReport)
from src.storage.parquet_store import (ROW_NO, _to_str, frame_to_table, tx_schema,
                                       write_date_partitioned, write_table)
from src.version import PIPELINE_VERSION

logger = logging.getLogger(__name__)

STAGING_SCHEMA_VERSION = 1
LOCAL_ROW = "_local_row"


def detect_dataset_timestamp_format(first_file: Path, sample_rows: int = 1000) -> Optional[str]:
    """The timestamp format pandas would infer for the WHOLE dataset column: guessed
    from the first non-null timestamp of the first file."""
    for chunk in iter_record_chunks(first_file, chunk_rows=sample_rows):
        fmt = infer_timestamp_format(chunk["timestamp"])
        if fmt is not None or chunk["timestamp"].notna().any():
            return fmt
    return None


def _entry_key(file_sha: str, required_fields: List[str], dup_subset: List[str],
               ts_format: Optional[str]) -> str:
    payload = json.dumps({
        "file_sha256": file_sha, "pipeline_version": PIPELINE_VERSION,
        "staging_schema": STAGING_SCHEMA_VERSION, "required_fields": list(required_fields),
        "duplicate_subset": list(dup_subset), "ts_format": ts_format,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _index_schema(dup_subset: List[str]) -> pa.Schema:
    return pa.schema([(LOCAL_ROW, pa.int64()), ("_row_valid", pa.bool_())]
                     + [(c, pa.string()) for c in dup_subset])


def cleanup_stale_temp_dirs(cache_root: Path, older_than_seconds: float = 6 * 3600) -> int:
    """Remove half-written staging dirs left by killed processes (never published, so
    never read). Only dirs older than the threshold are touched, so a concurrent
    in-progress staging is left alone."""
    import time
    removed = 0
    if not Path(cache_root).exists():
        return 0
    for d in Path(cache_root).glob(".tmp-*"):
        try:
            if d.is_dir() and time.time() - d.stat().st_mtime > older_than_seconds:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    return removed


def stage_source_file(path: Path, file_sha: str, cache_root: Path, required_fields: List[str],
                      dup_subset: List[str], ts_format: Optional[str], chunk_rows: int) -> Dict[str, Any]:
    """Stage one source file into the shared cache (or reuse it). Returns the cache entry meta."""
    key = _entry_key(file_sha, required_fields, dup_subset, ts_format)
    entry_dir = Path(cache_root) / key
    meta_path = entry_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["cache_hit"] = True
        meta["dir"] = str(entry_dir)
        return meta

    tmp_dir = Path(cache_root) / f".tmp-{key}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        partials = []
        chunks = []
        local_row = 0
        schema = tx_schema(LOCAL_ROW)
        idx_schema = _index_schema(dup_subset)
        for i, chunk in enumerate(iter_record_chunks(path, chunk_rows=chunk_rows)):
            rr = apply_row_rules(chunk, required_fields, ts_format)
            frame = rr.frame
            frame[LOCAL_ROW] = range(local_row, local_row + len(frame))
            local_row += len(frame)

            n = len(frame)
            idx_arrays = [pa.array(frame[LOCAL_ROW].to_numpy(dtype="int64"), pa.int64()),
                          pa.array(frame["_row_valid"].to_numpy(dtype=bool), pa.bool_())]
            for c in dup_subset:
                vals = [_to_str(v) for v in frame[c]] if c in frame.columns else [None] * n
                idx_arrays.append(pa.array(vals, pa.string()))
            write_table(pa.Table.from_arrays(idx_arrays, schema=idx_schema),
                        tmp_dir / f"index-{i:05d}.parquet")

            valid = finalize_valid_rows(frame)
            write_table(frame_to_table(valid, schema), tmp_dir / f"chunk-{i:05d}.parquet")

            partials.append({k: getattr(rr, k) for k in ("total_records", "missing_value_counts",
                                                          "invalid_ip_count", "invalid_timestamp_count",
                                                          "invalid_reasons")})
            chunks.append({"chunk": i, "rows": len(frame), "valid_rows": len(valid)})

        meta = {
            "entry_key": key, "file_sha256": file_sha, "source_name": Path(path).name,
            "n_rows": local_row, "chunks": chunks,
            "counts": merge_row_rule_counts(partials),
            "n_invalid_row_rules": sum(c["rows"] - c["valid_rows"] for c in chunks),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            os.rename(tmp_dir, entry_dir)  # atomic publish of the whole cache entry
        except OSError:
            if not meta_path.exists():
                raise
            shutil.rmtree(tmp_dir, ignore_errors=True)  # another process published it first
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    meta["cache_hit"] = False
    meta["dir"] = str(entry_dir)
    return meta


def build_cleaned_dataset(entries: List[Dict[str, Any]], out_root: Path, dup_subset: List[str],
                          drop_exact_duplicates: bool, enrich_fn=None) -> tuple[DataQualityReport, Dict[str, Any]]:
    """Stage B: global dedup across all staged files, then write the run's cleaned,
    date-partitioned dataset with a global _row_no."""
    # --- global row numbering (file order, then row order) ---
    offsets = []
    total = 0
    for e in entries:
        offsets.append(total)
        total += e["n_rows"]

    # --- global duplicate detection over the dedup-key columns only ---
    idx_parts = []
    for e, off in zip(entries, offsets):
        for c in e["chunks"]:
            t = pq.read_table(Path(e["dir"]) / f"index-{c['chunk']:05d}.parquet").to_pandas()
            t[ROW_NO] = t[LOCAL_ROW].astype("int64") + off
            idx_parts.append(t.drop(columns=[LOCAL_ROW]))
    index = pd.concat(idx_parts, ignore_index=True) if idx_parts else pd.DataFrame(
        {ROW_NO: pd.Series(dtype="int64"), "_row_valid": pd.Series(dtype=bool)})
    subset = [c for c in dup_subset if c in index.columns]
    if subset and len(index):
        dup_mask = index[subset].duplicated(keep="first")
    else:
        dup_mask = pd.Series(False, index=index.index)
    valid = index["_row_valid"].astype(bool)
    dups_among_valid_mask = dup_mask & valid
    drop_rows = set(index.loc[dups_among_valid_mask, ROW_NO].tolist()) if drop_exact_duplicates else set()

    merged = merge_row_rule_counts([e["counts"] for e in entries])
    report = build_report(merged, int((~valid).sum()), int(dup_mask.sum()),
                          int(dups_among_valid_mask.sum()), drop_exact_duplicates)
    del index

    # --- write the run's cleaned dataset ---
    schema = tx_schema(ROW_NO)
    n_written = 0
    partitions = set()
    for fi, (e, off) in enumerate(zip(entries, offsets)):
        for c in e["chunks"]:
            df = pq.read_table(Path(e["dir"]) / f"chunk-{c['chunk']:05d}.parquet").to_pandas()
            df[ROW_NO] = df[LOCAL_ROW].astype("int64") + off
            df = df.drop(columns=[LOCAL_ROW])
            if drop_rows:
                df = df[~df[ROW_NO].isin(drop_rows)]
            if len(df) == 0:
                continue
            if enrich_fn is not None:
                df = enrich_fn(df)
            for rel in write_date_partitioned(df, out_root, f"part-{fi:04d}-{c['chunk']:05d}", schema):
                partitions.add(rel.split("/")[0])
            n_written += len(df)

    if n_written != report.valid_records:
        raise RuntimeError(f"Cleaned row count mismatch: wrote {n_written}, expected {report.valid_records}")
    return report, {"n_rows_total": total, "n_rows_written": n_written, "n_partitions": len(partitions)}
