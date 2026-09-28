"""
Read-only, query-based access to ONE completed run's artifacts, for the dashboard
and the forensic scripts.

The dashboard never loads the raw dataset, the full alerts table with its list and
feature columns, or the full entity graph:
  - charts/overview read a small projection of scalar alert columns;
  - the alert queue and entity lookups push filters down into the Parquet dataset;
  - wallet lookups and graph neighborhoods go through the SQLite graph index.

Legacy single-directory outputs/ (produced by the pre-run-directory prototype) are
still readable, with the prototype's in-memory behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.dataset as pads

from src.graph.store import GraphStore
from src.runs.manifest import RunManifest
from src.runs.registry import LEGACY_RUN_ID
from src.storage.parquet_store import ROW_NO, read_frame

CHART_COLUMNS = ["priority", "risk_score", "synthetic_label", "anomaly_type", "geo_country",
                 "timestamp_parsed", "is_anomaly_prediction", "src_ip"]


class RunStore:
    def __init__(self, run_id: str, run_dir: Path, forensics_cfg: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.legacy = run_id == LEGACY_RUN_ID
        self.manifest = None if self.legacy else RunManifest.load(self.run_dir)
        forensics_cfg = forensics_cfg or {}
        mf = (self.manifest.data.get("forensics") if self.manifest else None) or {}
        self.evidence_store_path = self.run_dir / mf.get(
            "evidence_store", forensics_cfg.get("evidence_store_filename", "forensics/evidence_store.json"))
        self.ledger_path = self.run_dir / mf.get(
            "ledger", forensics_cfg.get("blockchain_filename", "forensics/blockchain_ledger.json"))

    # ------------------------------------------------------------ artifacts
    @property
    def alerts_path(self) -> Path:
        d = self.run_dir / "alerts"
        if d.is_dir():
            return d
        p = self.run_dir / "alerts.parquet"
        return p if p.exists() else p.with_suffix(".csv")

    def alerts_exist(self) -> bool:
        return self.alerts_path.exists()

    @property
    def graph_index_path(self) -> Path:
        return self.run_dir / "graph_index.sqlite"

    @property
    def graph_pickle_path(self) -> Path:
        return self.run_dir / "entity_graph.pickle"

    def load_json(self, name: str) -> Any:
        p = self.run_dir / name
        if not p.exists():
            return {} if name != "cases.json" else []
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def open_graph(self) -> Optional[GraphStore]:
        return GraphStore(self.graph_index_path) if self.graph_index_path.exists() else None

    # --------------------------------------------------------------- alerts
    def alert_columns(self) -> List[str]:
        p = self.alerts_path
        if p.is_dir():
            return [n for n in pads.dataset(str(p), format="parquet", partitioning="hive").schema.names if n != "date"]
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            return pq.ParquetFile(p).schema_arrow.names
        return list(pd.read_csv(p, nrows=0).columns)

    def _is_legacy_csv(self) -> bool:
        return self.alerts_path.suffix == ".csv"

    def read_alerts(self, columns: Optional[List[str]] = None, filter=None) -> pd.DataFrame:
        """Projected (and optionally filtered) alerts, in original row order."""
        available = set(self.alert_columns())
        cols = [c for c in dict.fromkeys(columns) if c in available] if columns is not None else None
        if self._is_legacy_csv():
            df = pd.read_csv(self.alerts_path)
            if filter is not None:
                import pyarrow as pa
                df = pads.dataset(pa.Table.from_pandas(df, preserve_index=False)).to_table(filter=filter).to_pandas()
            return df[cols] if cols is not None else df
        if cols is not None and ROW_NO in available and ROW_NO not in cols:
            cols = cols + [ROW_NO]
        return read_frame(self.alerts_path, columns=cols, filter=filter)

    def chart_frame(self) -> pd.DataFrame:
        return self.read_alerts(CHART_COLUMNS)

    def query_alerts(self, priorities: List[str], min_score: float, countries: List[str],
                     cluster: Optional[int], columns: List[str]) -> pd.DataFrame:
        expr = pads.field("risk_score") >= float(min_score)
        if priorities:
            expr = expr & pads.field("priority").isin(list(priorities))
        if countries:
            expr = expr & pads.field("geo_country").isin(list(countries))
        if cluster is not None:
            expr = expr & (pads.field("cluster_label") == int(cluster))
        return self.read_alerts(columns + ["risk_score"], filter=expr)

    def alerts_where(self, column: str, value: Any, columns: List[str]) -> pd.DataFrame:
        return self.read_alerts(columns, filter=pads.field(column) == value)

    def alerts_for_txids(self, txids: List[str], columns: Optional[List[str]] = None) -> pd.DataFrame:
        if not txids:
            return self.read_alerts(columns, filter=pads.field("txid").isin([""])).iloc[0:0]
        return self.read_alerts(columns, filter=pads.field("txid").isin(list(txids)))
