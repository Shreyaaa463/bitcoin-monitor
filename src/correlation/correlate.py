"""
Correlation layer: derives entity relationships from cleaned transaction records.

Produces a list of typed edges (source, target, relation, metadata) that the graph
module consumes to build the NetworkX MultiDiGraph. Keeping this as an intermediate
representation (rather than building the graph directly) makes it easy to inspect,
test, and reuse for both the graph and the correlation-based features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    src_type: str
    dst_type: str
    attrs: Dict[str, Any]


def build_edges(df: pd.DataFrame) -> List[Edge]:
    edges: List[Edge] = []

    for _, row in df.iterrows():
        tx_id = f"tx:{row['txid']}"
        ip_id = f"ip:{row['src_ip']}"
        asn_id = f"asn:{row['asn']}" if pd.notna(row.get("asn")) else None
        country_id = f"country:{row['geo_country']}" if pd.notna(row.get("geo_country")) else None
        ts = row.get("timestamp_parsed", row.get("timestamp"))

        # IP --OBSERVED--> TX
        edges.append(Edge(
            src=ip_id, dst=tx_id, relation="OBSERVED",
            src_type="IP", dst_type="Transaction",
            attrs={"timestamp": str(ts), "src_port": row.get("src_port"), "dst_port": row.get("dst_port")},
        ))

        # TX --INPUT--> Wallet (for each input address)
        input_addrs = row.get("input_addresses") or []
        input_amts = row.get("input_amounts") or []
        for i, addr in enumerate(input_addrs):
            if not addr:
                continue
            amt = input_amts[i] if i < len(input_amts) else None
            edges.append(Edge(
                src=tx_id, dst=f"wallet:{addr}", relation="INPUT",
                src_type="Transaction", dst_type="Wallet",
                attrs={"amount": amt, "timestamp": str(ts)},
            ))

        # TX --OUTPUT--> Wallet (for each output address)
        output_addrs = row.get("output_addresses") or []
        output_amts = row.get("output_amounts") or []
        for i, addr in enumerate(output_addrs):
            if not addr:
                continue
            amt = output_amts[i] if i < len(output_amts) else None
            edges.append(Edge(
                src=tx_id, dst=f"wallet:{addr}", relation="OUTPUT",
                src_type="Transaction", dst_type="Wallet",
                attrs={"amount": amt, "timestamp": str(ts)},
            ))

        # IP --BELONGS_TO--> ASN
        if asn_id:
            edges.append(Edge(
                src=ip_id, dst=asn_id, relation="BELONGS_TO",
                src_type="IP", dst_type="ASN", attrs={},
            ))

        # IP --LOCATED_IN--> Country
        if country_id:
            edges.append(Edge(
                src=ip_id, dst=country_id, relation="LOCATED_IN",
                src_type="IP", dst_type="Country", attrs={},
            ))

    return edges


def edges_to_dataframe(edges: List[Edge]) -> pd.DataFrame:
    """Flat tabular view of edges, useful for debugging / CSV export."""
    return pd.DataFrame([
        {
            "src": e.src, "dst": e.dst, "relation": e.relation,
            "src_type": e.src_type, "dst_type": e.dst_type, **e.attrs,
        }
        for e in edges
    ])
