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


EDGE_COLUMNS = ["src", "dst", "relation", "src_type", "dst_type",
                "timestamp", "src_port", "dst_port", "amount", "_row_no", "_seq"]


def build_edges_frame(df: pd.DataFrame, row_col: str = "_row_no") -> pd.DataFrame:
    """Columnar equivalent of build_edges(): the same typed edges, in the same order
    (row order, then OBSERVED, INPUT..., OUTPUT..., BELONGS_TO, LOCATED_IN), as a
    DataFrame. Used for partition-by-partition correlation of large datasets."""
    n = len(df)
    row_no = df[row_col].to_numpy() if row_col in df.columns else range(n)
    tx_ids = ["tx:" + str(t) for t in df["txid"]]
    ts_col = df["timestamp_parsed"] if "timestamp_parsed" in df.columns else df["timestamp"]
    ts_str = [str(t) for t in ts_col]

    cols = {c: [] for c in EDGE_COLUMNS}

    def add(src, dst, rel, st, dt, ts, sp, dp, amt, r, seq):
        cols["src"].append(src); cols["dst"].append(dst); cols["relation"].append(rel)
        cols["src_type"].append(st); cols["dst_type"].append(dt); cols["timestamp"].append(ts)
        cols["src_port"].append(sp); cols["dst_port"].append(dp); cols["amount"].append(amt)
        cols["_row_no"].append(r); cols["_seq"].append(seq)

    src_ips, asns, geos = df["src_ip"].tolist(), df["asn"].tolist(), df["geo_country"].tolist()
    sports, dports = df["src_port"].tolist(), df["dst_port"].tolist()
    ins, outs = df["input_addresses"].tolist(), df["output_addresses"].tolist()
    in_amts, out_amts = df["input_amounts"].tolist(), df["output_amounts"].tolist()

    for k in range(n):
        r, tx, ts, ip_id = row_no[k], tx_ids[k], ts_str[k], f"ip:{src_ips[k]}"
        seq = 0
        add(ip_id, tx, "OBSERVED", "IP", "Transaction", ts, sports[k], dports[k], None, r, seq)
        for rel, addrs, amts in (("INPUT", ins[k], in_amts[k]), ("OUTPUT", outs[k], out_amts[k])):
            addrs = addrs if addrs is not None else []
            amts = amts if amts is not None else []
            for i, addr in enumerate(addrs):
                if not addr:
                    continue
                seq += 1
                add(tx, f"wallet:{addr}", rel, "Transaction", "Wallet", ts, None, None,
                    amts[i] if i < len(amts) else None, r, seq)
        if pd.notna(asns[k]):
            seq += 1
            add(ip_id, f"asn:{asns[k]}", "BELONGS_TO", "IP", "ASN", None, None, None, None, r, seq)
        if pd.notna(geos[k]):
            seq += 1
            add(ip_id, f"country:{geos[k]}", "LOCATED_IN", "IP", "Country", None, None, None, None, r, seq)

    out = pd.DataFrame(cols, columns=EDGE_COLUMNS)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    for c in ("src_port", "dst_port"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def edges_to_dataframe(edges: List[Edge]) -> pd.DataFrame:
    """Flat tabular view of edges, useful for debugging / CSV export."""
    return pd.DataFrame([
        {
            "src": e.src, "dst": e.dst, "relation": e.relation,
            "src_type": e.src_type, "dst_type": e.dst_type, **e.attrs,
        }
        for e in edges
    ])
