"""
Graph layer: builds a NetworkX MultiDiGraph of IP / Wallet / Transaction / ASN / Country
nodes from the correlation edge list, and exports interactive neighborhood visualizations
with PyVis.

Node types are stored as a 'type' attribute on each node so the dashboard's Graph Explorer
can color/filter by type and traverse investigation paths like IP -> TX -> Wallet -> TX -> Wallet.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd

from src.correlation.correlate import Edge

NODE_COLORS = {
    "IP": "#4C9AFF",
    "Wallet": "#F5A623",
    "Transaction": "#7ED321",
    "ASN": "#BD10E0",
    "Country": "#D0021B",
}


def build_entity_graph(edges: List[Edge]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    for e in edges:
        if e.src not in g:
            g.add_node(e.src, type=e.src_type, label=e.src.split(":", 1)[-1])
        if e.dst not in g:
            g.add_node(e.dst, type=e.dst_type, label=e.dst.split(":", 1)[-1])
        g.add_edge(e.src, e.dst, relation=e.relation, **e.attrs)

    return g


def get_neighborhood(g: nx.MultiDiGraph, node_id: str, hops: int = 2) -> nx.MultiDiGraph:
    """Return the induced subgraph of all nodes within `hops` of node_id (undirected reachability),
    used to power investigation paths such as IP -> TX -> Wallet -> TX -> Wallet."""
    if node_id not in g:
        return nx.MultiDiGraph()
    undirected = g.to_undirected(as_view=True)
    nodes_in_range = set([node_id])
    frontier = {node_id}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(undirected.neighbors(n))
        nodes_in_range.update(next_frontier)
        frontier = next_frontier
    return g.subgraph(nodes_in_range).copy()


_EDGE_ATTRS_BY_RELATION = {
    "OBSERVED": ("timestamp", "src_port", "dst_port"),
    "INPUT": ("amount", "timestamp"),
    "OUTPUT": ("amount", "timestamp"),
}


def _clean_attr(v):
    return None if v is None or v is pd.NA or (isinstance(v, float) and v != v) else v


def build_entity_graph_from_frame(edges_df: pd.DataFrame) -> nx.MultiDiGraph:
    """Same graph as build_entity_graph(build_edges(df)), built from the columnar edge
    frame (src/correlation/correlate.py:build_edges_frame) in the same edge order."""
    g = nx.MultiDiGraph()
    cols = [edges_df[c].tolist() for c in ("src", "dst", "relation", "src_type", "dst_type",
                                            "timestamp", "src_port", "dst_port", "amount")]
    for src, dst, rel, st, dt, ts, sp, dp, amt in zip(*cols):
        if src not in g:
            g.add_node(src, type=st, label=src.split(":", 1)[-1])
        if dst not in g:
            g.add_node(dst, type=dt, label=dst.split(":", 1)[-1])
        values = {"timestamp": ts, "src_port": _clean_attr(sp), "dst_port": _clean_attr(dp), "amount": _clean_attr(amt)}
        g.add_edge(src, dst, relation=rel, **{k: values[k] for k in _EDGE_ATTRS_BY_RELATION.get(rel, ())})
    return g


# PyVis' template always references Bootstrap from a public CDN, even with
# cdn_resources="in_line". The graph does not need it, and the app must work with
# no Internet access, so those tags are removed from the generated HTML.
_EXTERNAL_TAG_RE = re.compile(
    r"<link\b[^>]*\bhref\s*=\s*[\"']https?://[^\"']*[\"'][^>]*>"
    r"|<script\b[^>]*\bsrc\s*=\s*[\"']https?://[^\"']*[\"'][^>]*>\s*</script>",
    re.IGNORECASE,
)


def strip_external_resources(html: str) -> str:
    return _EXTERNAL_TAG_RE.sub("", html)


def export_pyvis_html(g: nx.MultiDiGraph, out_path: Path, height: str = "700px", physics: bool = True) -> Path:
    """Render a graph (or subgraph) to a self-contained, fully offline HTML file via PyVis."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut()
    if not physics:
        net.toggle_physics(False)

    for node, data in g.nodes(data=True):
        ntype = data.get("type", "Unknown")
        net.add_node(
            node,
            label=str(data.get("label", node))[:20],
            title=f"{ntype}: {node}",
            color=NODE_COLORS.get(ntype, "#999999"),
            shape="dot",
            size=15 if ntype != "Transaction" else 10,
        )

    for u, v, data in g.edges(data=True):
        relation = data.get("relation", "")
        net.add_edge(u, v, title=relation, label=relation)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # generate_html + explicit UTF-8 write: PyVis' own write_html uses the platform
    # default encoding and fails on non-UTF-8 locales (the inlined JS is not ASCII).
    html = strip_external_resources(net.generate_html(notebook=False))
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path


def graph_summary(g: nx.MultiDiGraph) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    for _, data in g.nodes(data=True):
        t = data.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "node_type_counts": type_counts,
    }
