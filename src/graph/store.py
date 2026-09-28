"""
Scalable on-disk entity-graph index (SQLite, Python standard library -- no server,
no extra dependency, fully offline).

The full IP / Transaction / Wallet / ASN / Country relationship set is written once
per run to runs/<RUN_ID>/graph_index.sqlite:

    nodes(id INTEGER PRIMARY KEY, node_key TEXT UNIQUE, type TEXT, label TEXT)
        index (type, node_key)          -> entity pickers / prefix search
    edges(id INTEGER PRIMARY KEY, src_id, dst_id, relation, ts, amount, src_port, dst_port)
        index (src_id), index (dst_id)  -> k-hop neighborhood expansion in both directions

Node ids follow first-appearance order in the input (same order as the NetworkX
prototype graph); edge ids follow input row order. Parallel edges are kept, exactly
like the prototype's MultiDiGraph.

The dashboard never loads the whole graph: neighborhood() walks the indexes from the
selected entity, with a per-node neighbor cap and a total node cap so that hub nodes
(e.g. a Country connected to millions of IPs) cannot blow up a query, and returns a
small NetworkX MultiDiGraph that the existing PyVis renderer draws unchanged.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

SCHEMA_VERSION = "1"
_BATCH = 50_000


class GraphStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        uri = f"{self.path.resolve().as_uri()}?mode=ro"
        # check_same_thread=False: Streamlit may call from different script threads;
        # the connection is read-only.
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def close(self):
        self.conn.close()

    # ================================================================ build
    @staticmethod
    def build(path: Path, edge_frames: Iterable[pd.DataFrame], cache_mb: int = 256,
              timings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Build the index from an iterable of edge frames (build_edges_frame output).
        Written to a temp file and atomically renamed, so a crash never leaves a
        half-built index at `path`. cache_mb bounds SQLite's page cache (memory) during
        the build; larger is faster for big graphs."""
        import time as _time
        timings = timings if timings is not None else {}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        stage = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.stage")
        conn = sqlite3.connect(str(tmp))

        def timed(label, fn, *a):
            t0 = _time.time()
            r = fn(*a)
            timings[label] = round(timings.get(label, 0.0) + _time.time() - t0, 3)
            return r

        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute(f"PRAGMA cache_size=-{int(cache_mb) * 1024}")
            conn.execute("ATTACH DATABASE ? AS stage", (str(stage),))
            conn.execute("PRAGMA stage.journal_mode=OFF")
            conn.execute(f"PRAGMA stage.cache_size=-{int(cache_mb) * 1024}")
            conn.execute("""CREATE TABLE stage.edge_stage(
                src TEXT, dst TEXT, relation TEXT, src_type TEXT, dst_type TEXT,
                ts TEXT, amount REAL, src_port INTEGER, dst_port INTEGER, row_no INTEGER, seq INTEGER)""")
            n_stage = 0
            for frame in edge_frames:
                rows = frame[["src", "dst", "relation", "src_type", "dst_type", "timestamp",
                              "amount", "src_port", "dst_port", "_row_no", "_seq"]]
                rows = rows.astype(object).where(rows.notna(), None)
                recs = list(rows.itertuples(index=False, name=None))
                for i in range(0, len(recs), _BATCH):
                    conn.executemany("INSERT INTO stage.edge_stage VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                     recs[i:i + _BATCH])
                n_stage += len(recs)
            conn.commit()

            timed("stage_order_index", conn.execute, "CREATE INDEX stage.idx_stage_order ON edge_stage(row_no, seq)")
            conn.executescript("""
                CREATE TABLE nodes(id INTEGER PRIMARY KEY, node_key TEXT NOT NULL UNIQUE,
                                   type TEXT NOT NULL, label TEXT NOT NULL);
                CREATE TABLE edges(id INTEGER PRIMARY KEY, src_id INTEGER NOT NULL,
                                   dst_id INTEGER NOT NULL, relation TEXT NOT NULL, ts TEXT,
                                   amount REAL, src_port INTEGER, dst_port INTEGER);
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            """)
            # Nodes in first-appearance order: an edge's src is seen before its dst.
            timed("insert_nodes", conn.execute, """
                INSERT OR IGNORE INTO nodes(node_key, type, label)
                SELECT key, type, substr(key, instr(key, ':') + 1) FROM (
                    SELECT src AS key, src_type AS type, row_no, seq, 0 AS side FROM stage.edge_stage
                    UNION ALL
                    SELECT dst, dst_type, row_no, seq, 1 FROM stage.edge_stage
                ) ORDER BY row_no, seq, side""")
            timed("insert_edges", conn.execute, """
                INSERT INTO edges(src_id, dst_id, relation, ts, amount, src_port, dst_port)
                SELECT s.id, d.id, e.relation, e.ts, e.amount, e.src_port, e.dst_port
                FROM stage.edge_stage e
                JOIN nodes s ON s.node_key = e.src
                JOIN nodes d ON d.node_key = e.dst
                ORDER BY e.row_no, e.seq""")
            conn.commit()
            conn.execute("DETACH DATABASE stage")
            timed("indexes", conn.executescript, """
                CREATE INDEX idx_edges_src ON edges(src_id);
                CREATE INDEX idx_edges_dst ON edges(dst_id);
                CREATE INDEX idx_nodes_type_key ON nodes(type, node_key);
            """)
            n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            if n_edges != n_stage:
                raise RuntimeError(f"graph index edge count mismatch ({n_edges} != {n_stage})")
            conn.executemany("INSERT INTO meta VALUES (?, ?)", [
                ("schema_version", SCHEMA_VERSION), ("num_nodes", str(n_nodes)), ("num_edges", str(n_edges))])
            conn.commit()
        finally:
            conn.close()
            if stage.exists():
                os.remove(stage)
        os.replace(tmp, path)
        return {"num_nodes": n_nodes, "num_edges": n_edges}

    # ============================================================== summary
    def summary(self) -> Dict[str, Any]:
        """Same shape (and type ordering) as graph.build_graph.graph_summary()."""
        rows = self.conn.execute(
            "SELECT type, COUNT(*), MIN(id) FROM nodes GROUP BY type ORDER BY MIN(id)").fetchall()
        return {
            "num_nodes": int(self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]),
            "num_edges": int(self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
            "node_type_counts": {t: int(c) for t, c, _ in rows},
        }

    def node_types(self) -> List[str]:
        return sorted(r[0] for r in self.conn.execute("SELECT DISTINCT type FROM nodes"))

    def list_nodes(self, node_type: str, prefix: str = "", limit: int = 5000) -> List[str]:
        """Sorted node keys of one type (optionally label-prefix filtered), capped at `limit`.
        Uses the (type, node_key) index for both the ordering and the prefix range."""
        type_prefix = _TYPE_PREFIX.get(node_type)
        if prefix and type_prefix is not None:
            lo = type_prefix + prefix
            return [r[0] for r in self.conn.execute(
                "SELECT node_key FROM nodes WHERE type = ? AND node_key >= ? AND node_key < ? "
                "ORDER BY node_key LIMIT ?", (node_type, lo, lo + "\U0010ffff", limit))]
        if prefix:
            return [r[0] for r in self.conn.execute(
                "SELECT node_key FROM nodes WHERE type = ? AND instr(label, ?) = 1 "
                "ORDER BY node_key LIMIT ?", (node_type, prefix, limit))]
        return [r[0] for r in self.conn.execute(
            "SELECT node_key FROM nodes WHERE type = ? ORDER BY node_key LIMIT ?", (node_type, limit))]

    def list_labels(self, node_type: str, prefix: str = "", limit: int = 5000) -> List[str]:
        return [k.split(":", 1)[-1] for k in self.list_nodes(node_type, prefix, limit)]

    def _node_id(self, node_key: str) -> Optional[int]:
        r = self.conn.execute("SELECT id FROM nodes WHERE node_key = ?", (node_key,)).fetchone()
        return r[0] if r else None

    # ======================================================== neighborhoods
    def _neighbors(self, nid: int, cap: int) -> Tuple[List[int], bool]:
        """Distinct undirected neighbors in edge order (out-edges first, then in-edges),
        stopping after cap+1 so hub nodes are never fully scanned."""
        seen: Dict[int, None] = {}
        for sql in ("SELECT dst_id FROM edges WHERE src_id = ? ORDER BY id",
                    "SELECT src_id FROM edges WHERE dst_id = ? ORDER BY id"):
            cur = self.conn.execute(sql, (nid,))
            while True:
                batch = cur.fetchmany(1000)
                if not batch:
                    break
                for (other,) in batch:
                    if other not in seen:
                        seen[other] = None
                        if len(seen) > cap:
                            cur.close()
                            return list(seen)[:cap], True
        return list(seen), False

    def neighborhood(self, node_key: str, hops: int = 2, max_nodes: int = 5000,
                     max_neighbors_per_node: int = 2000) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
        """Induced subgraph of all nodes within `hops` (undirected) of node_key -- the same
        definition as graph.build_graph.get_neighborhood() -- bounded by the two caps."""
        info = {"truncated": False, "hub_nodes_capped": 0, "node_cap_hit": False}
        start = self._node_id(node_key)
        if start is None:
            return nx.MultiDiGraph(), info
        visited: Dict[int, None] = {start: None}
        frontier = [start]
        for _ in range(hops):
            nxt = []
            for nid in frontier:
                nbrs, capped = self._neighbors(nid, max_neighbors_per_node)
                if capped:
                    info["hub_nodes_capped"] += 1
                for o in nbrs:
                    if o in visited:
                        continue
                    if len(visited) >= max_nodes:
                        info["node_cap_hit"] = True
                        break
                    visited[o] = None
                    nxt.append(o)
                if info["node_cap_hit"]:
                    break
            frontier = nxt
            if info["node_cap_hit"] or not frontier:
                break
        info["truncated"] = info["node_cap_hit"] or info["hub_nodes_capped"] > 0
        return self._induced_subgraph(list(visited)), info

    def _induced_subgraph(self, node_ids: List[int]) -> nx.MultiDiGraph:
        cur = self.conn.cursor()
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS sel(id INTEGER PRIMARY KEY)")
        cur.execute("DELETE FROM sel")
        cur.executemany("INSERT OR IGNORE INTO sel(id) VALUES (?)", [(i,) for i in node_ids])
        g = nx.MultiDiGraph()
        for nid, key, ntype, label in cur.execute(
                "SELECT n.id, n.node_key, n.type, n.label FROM sel JOIN nodes n ON n.id = sel.id ORDER BY n.id"):
            g.add_node(key, type=ntype, label=label)
        for src, dst, rel, ts, amt, sp, dp in cur.execute("""
                SELECT s.node_key, d.node_key, e.relation, e.ts, e.amount, e.src_port, e.dst_port
                FROM sel JOIN edges e ON e.src_id = sel.id
                JOIN nodes s ON s.id = e.src_id JOIN nodes d ON d.id = e.dst_id
                WHERE e.dst_id IN (SELECT id FROM sel)
                ORDER BY e.id"""):
            attrs = {"relation": rel}
            if rel == "OBSERVED":
                attrs.update(timestamp=ts, src_port=sp, dst_port=dp)
            elif rel in ("INPUT", "OUTPUT"):
                attrs.update(amount=amt, timestamp=ts)
            g.add_edge(src, dst, **attrs)
        cur.execute("DELETE FROM sel")
        return g

    # =============================================================== lookups
    def txids_for_wallet(self, address: str, limit: int = 100_000) -> List[str]:
        rows = self.conn.execute("""
            SELECT DISTINCT s.label FROM nodes w
            JOIN edges e ON e.dst_id = w.id
            JOIN nodes s ON s.id = e.src_id
            WHERE w.node_key = ? AND e.relation IN ('INPUT', 'OUTPUT')
            LIMIT ?""", ("wallet:" + str(address), limit)).fetchall()
        return [r[0] for r in rows]

    # ==================================================== analytics helpers
    def num_nodes(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    def edge_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """(src_id, dst_id) of every edge as int64 arrays (0-based), streamed in batches."""
        n_edges = int(self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        src = np.empty(n_edges, dtype=np.int64)
        dst = np.empty(n_edges, dtype=np.int64)
        cur = self.conn.execute("SELECT src_id, dst_id FROM edges ORDER BY id")
        pos = 0
        while True:
            batch = cur.fetchmany(500_000)
            if not batch:
                break
            arr = np.asarray(batch, dtype=np.int64)
            src[pos:pos + len(arr)] = arr[:, 0] - 1
            dst[pos:pos + len(arr)] = arr[:, 1] - 1
            pos += len(arr)
        return src, dst

    def node_ids_for_keys(self, keys: Iterable[str]) -> Dict[str, int]:
        """0-based node ids for the given keys (unknown keys omitted)."""
        cur = self.conn.cursor()
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS want(k TEXT PRIMARY KEY)")
        cur.execute("DELETE FROM want")
        cur.executemany("INSERT OR IGNORE INTO want(k) VALUES (?)", [(k,) for k in keys])
        out = {k: i - 1 for k, i in cur.execute(
            "SELECT n.node_key, n.id FROM want JOIN nodes n ON n.node_key = want.k")}
        cur.execute("DELETE FROM want")
        return out

    def count_edges_into_type(self, node_type: str) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM edges e JOIN nodes n ON n.id = e.dst_id WHERE n.type = ?",
            (node_type,)).fetchone()[0])


_TYPE_PREFIX = {"IP": "ip:", "Transaction": "tx:", "Wallet": "wallet:", "ASN": "asn:", "Country": "country:"}
