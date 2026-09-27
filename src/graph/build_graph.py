"""
Graph layer: builds a NetworkX MultiDiGraph of IP / Wallet / Transaction / ASN / Country
nodes from the correlation edge list, and exports interactive neighborhood visualizations
with PyVis.

Node types are stored as a 'type' attribute on each node so the dashboard's Graph Explorer
can color/filter by type and traverse investigation paths like IP -> TX -> Wallet -> TX -> Wallet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

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


def export_pyvis_html(
    g: nx.MultiDiGraph,
    out_path: Path,
    height: str = "700px",
    physics: bool = True
) -> Path:
    """Render a graph (or subgraph) to a self-contained interactive HTML file via PyVis."""
    from pyvis.network import Network

    net = Network(
        height=height,
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line"
    )

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
        net.add_edge(
            u,
            v,
            title=relation,
            label=relation
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    net.write_html(
        str(out_path),
        open_browser=False,
        notebook=False
    )

    # Add a color legend to the generated HTML
    html = out_path.read_text(encoding="utf-8")

    legend = """
    <div style="
        position: fixed;
        top: 15px;
        left: 15px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <div style="
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 14px;
        ">
            Graph Legend
        </div>

        <div style="margin: 5px 0;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                border-radius:50%;
                background:#4C9AFF;
                margin-right:7px;
            "></span>
            <b>Blue</b> → IP Address
        </div>

        <div style="margin: 5px 0;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                border-radius:50%;
                background:#F5A623;
                margin-right:7px;
            "></span>
            <b>Orange</b> → Wallet
        </div>

        <div style="margin: 5px 0;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                border-radius:50%;
                background:#7ED321;
                margin-right:7px;
            "></span>
            <b>Green</b> → Transaction
        </div>

        <div style="margin: 5px 0;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                border-radius:50%;
                background:#BD10E0;
                margin-right:7px;
            "></span>
            <b>Purple</b> → ASN
        </div>

        <div style="margin: 5px 0;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                border-radius:50%;
                background:#D0021B;
                margin-right:7px;
            "></span>
            <b>Red</b> → Country
        </div>
    </div>
    """

    html = html.replace("<body>", "<body>" + legend)

    out_path.write_text(html, encoding="utf-8")

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
