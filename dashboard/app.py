"""
TRACE-X Bitcoin Forensic Intelligence Dashboard.

Reads ONLY the cached artifacts produced by `python run.py` (outputs/, models/,
data/processed/). It never retrains anything on interaction — all filtering/
sorting happens in-memory over the cached alerts table, and graph neighborhoods
are computed on demand from the cached NetworkX graph (fast — no retraining).

Run with:  streamlit run dashboard/app.py   (from the project root)
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_paths
from src.graph.investigator_view import (
    FOCUS_MODES,
    build_investigator_view,
    export_investigator_pyvis_html,
    get_entity_details,
)
from src.forensics.evidence import EvidenceStore
from src.forensics.blockchain import Blockchain
from src.forensics.verification import verify_case, generate_proof_for_txid

st.set_page_config(page_title="TRACE-X Bitcoin Forensic Dashboard", layout="wide")

CFG = load_config()
PATHS = get_paths()
DASH_CFG = CFG.get("dashboard", {})


# --------------------------------------------------------------------------- #
# Cached loaders — these read files from disk once per session, not per click.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_alerts() -> pd.DataFrame:
    p = PATHS["outputs_dir"] / "alerts.parquet"
    if not p.exists():
        p = p.with_suffix(".csv")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    p = PATHS["outputs_dir"] / name
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cases() -> list:
    p = PATHS["outputs_dir"] / "cases.json"
    if not p.exists():
        return []
    with open(p, "r") as f:
        return json.load(f)


def get_forensic_paths():
    forensics_cfg = CFG.get("forensics", {})
    evidence_store_path = PATHS["outputs_dir"] / forensics_cfg.get(
        "evidence_store_filename", "forensics/evidence_store.json")
    blockchain_path = PATHS["outputs_dir"] / forensics_cfg.get(
        "blockchain_filename", "forensics/blockchain_ledger.json")
    return evidence_store_path, blockchain_path


def forensics_available() -> bool:
    evidence_store_path, blockchain_path = get_forensic_paths()
    return evidence_store_path.exists() and blockchain_path.exists()


@st.cache_resource(show_spinner=False)
def load_graph() -> nx.MultiDiGraph:
    p = PATHS["outputs_dir"] / "entity_graph.pickle"
    with open(p, "rb") as f:
        return pickle.load(f)


def to_list(value) -> list:
    """Parquet round-trips list columns as numpy arrays; normalize to plain python
    lists everywhere so truthiness checks and 'in' tests behave predictably."""
    if value is None:
        return []
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, list):
        return value
    return list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else []


def outputs_exist() -> bool:
    return (PATHS["outputs_dir"] / "alerts.parquet").exists() or (PATHS["outputs_dir"] / "alerts.csv").exists()


# --------------------------------------------------------------------------- #
# Guard: nothing to show until `python run.py` has been executed at least once.
# --------------------------------------------------------------------------- #
if not outputs_exist():
    st.title(DASH_CFG.get("title", "TRACE-X Bitcoin Forensic Intelligence Dashboard"))
    st.error(
        "No cached pipeline outputs found. Run the pipeline first from the project root:\n\n"
        "`python run.py`\n\nThen restart this dashboard."
    )
    st.stop()

alerts = load_alerts()
df=alerts.copy()
data_quality = load_json("data_quality.json")
graph_summary = load_json("graph_summary.json")
cluster_summary = load_json("cluster_summary.json")
eval_metrics = load_json("evaluation_metrics.json")
run_meta = load_json("run_metadata.json")

st.title(DASH_CFG.get("title", "TRACE-X Bitcoin Forensic Intelligence Dashboard"))
st.caption(
    "Offline forensic intelligence prototype over a **synthetic** Bitcoin transaction-traffic dataset. "
    "All scores are model-generated investigative signals, not proof of real-world criminal activity, "
    "and this dataset contains no real Bitcoin network attribution."
)

tabs = st.tabs([
    "Overview",
    "Charts",
    "Alerts",
    "Investigation",
    "Graph Explorer",
    "ML / Evaluation",
    "Data Quality",
    "Forensic Blockchain",
])

# =========================================================================== #
# TAB 1 — OVERVIEW
# =========================================================================== #
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analyzed", f"{len(alerts):,}")
    c2.metric("Unique wallets (graph)", f"{graph_summary.get('node_type_counts', {}).get('Wallet', 0):,}")
    c3.metric("Unique source IPs", f"{graph_summary.get('node_type_counts', {}).get('IP', 0):,}")
    c4.metric("Flagged anomalies (ML)", f"{int(alerts['is_anomaly_prediction'].sum()):,}")

    c5, c6, c7, c8 = st.columns(4)
    priority_counts = alerts["priority"].value_counts()
    c5.metric("HIGH priority", f"{int(priority_counts.get('HIGH', 0)):,}")
    c6.metric("MEDIUM priority", f"{int(priority_counts.get('MEDIUM', 0)):,}")
    c7.metric("LOW priority", f"{int(priority_counts.get('LOW', 0)):,}")
    c8.metric("Behavioral clusters found", f"{cluster_summary.get('n_clusters', 0):,}")

    st.info(
        "**Interpretation guidance:** the priority score and anomaly score below are "
        "**investigative signals produced by an unsupervised model**, not calibrated probabilities "
        "and not proof of criminal activity. They are meant to help an analyst triage where to look first."
    )

    st.subheader("Run metadata")
    st.json(run_meta, expanded=False)

# =========================================================================== #
# TAB 2 — CHARTS
# =========================================================================== #
with tabs[1]:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Normal vs. Anomalous (synthetic label)")

        if "synthetic_label" in df.columns:
            label_counts = (
                df["synthetic_label"]
                .astype("string")
                .str.strip()
                .str.lower()
                .map({
                    "normal": "Normal",
                    "anomalous": "Anomalous",
                    "0": "Normal",
                    "1": "Anomalous",
                })
                .value_counts()
                .reindex(["Normal", "Anomalous"], fill_value=0)
            )

            fig = px.bar(
                x=label_counts.index,
                y=label_counts.values,
                labels={
                    "x": "Label",
                    "y": "Transactions",
                },
                title="Normal vs. Anomalous (synthetic label)",
                text=label_counts.values,
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                showlegend=False,
                height=400,
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="synthetic_label_chart",
            )
        else:
            st.warning("synthetic_label column is not available.")

    with colB:
        st.subheader("Risk priority distribution")
        prio_counts = alerts["priority"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
        fig = px.bar(x=prio_counts.index, y=prio_counts.values,
                     color=prio_counts.index,
                     color_discrete_map={"HIGH": "#D0021B", "MEDIUM": "#F5A623", "LOW": "#7ED321"},
                     labels={"x": "Priority", "y": "Count"})
        st.plotly_chart(fig, width='stretch')

    colC, colD = st.columns(2)
    with colC:
        st.subheader("Anomaly type breakdown")
        atype = alerts["anomaly_type"].dropna()
        if len(atype) > 0:
            fig = px.bar(atype.value_counts())
            st.plotly_chart(fig, width='stretch')
        else:
            st.write("No anomaly_type values present in this dataset.")

    with colD:
        st.subheader("Top countries by transaction volume")
        fig = px.bar(alerts["geo_country"].value_counts().head(10))
        st.plotly_chart(fig, width='stretch')

    st.subheader("Transaction volume over time")
    ts = pd.to_datetime(alerts["timestamp_parsed"], errors="coerce", utc=True)
    daily = ts.dt.date.value_counts().sort_index()
    fig = px.line(x=daily.index.astype(str), y=daily.values, labels={"x": "Date", "y": "Transactions"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Risk score distribution")
    fig = px.histogram(alerts, x="risk_score", nbins=40, color="priority",
                        color_discrete_map={"HIGH": "#D0021B", "MEDIUM": "#F5A623", "LOW": "#7ED321"})
    st.plotly_chart(fig, width='stretch')

# =========================================================================== #
# TAB 3 — ALERTS
# =========================================================================== #
with tabs[2]:
    st.subheader("Ranked, filterable alert queue")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        priority_filter = st.multiselect("Priority", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    with fcol2:
        min_score = st.slider("Min risk score", 0, 100, 0)
    with fcol3:
        countries = sorted(alerts["geo_country"].dropna().unique().tolist())
        country_filter = st.multiselect("Country", countries, default=[])
    with fcol4:
        cluster_filter = st.text_input("Cluster label (exact, optional)", value="")

    filtered = alerts.copy()
    if priority_filter:
        filtered = filtered[filtered["priority"].isin(priority_filter)]
    filtered = filtered[filtered["risk_score"] >= min_score]
    if country_filter:
        filtered = filtered[filtered["geo_country"].isin(country_filter)]
    if cluster_filter.strip() != "":
        try:
            cl = int(cluster_filter.strip())
            filtered = filtered[filtered["cluster_label"] == cl]
        except ValueError:
            st.warning("Cluster label must be an integer.")

    filtered = filtered.sort_values("risk_score", ascending=False)

    st.caption(f"{len(filtered):,} alerts match current filters (of {len(alerts):,} total).")

    display_cols = [
        "entity", "txid", "priority", "risk_score", "anomaly_score_0_100",
        "cluster_label", "anomaly_type", "geo_country", "asn", "timestamp_str", "reasons",
    ]
    page_size = DASH_CFG.get("alerts_page_size", 50)
    st.dataframe(
        filtered[display_cols].head(max(page_size, 1)),
        width='stretch',
        height=500,
    )
    st.download_button(
        "Download filtered alerts as CSV",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_alerts.csv",
        mime="text/csv",
    )

# =========================================================================== #
# TAB 4 — INVESTIGATION
# =========================================================================== #
with tabs[3]:
    st.subheader("Entity investigation")
    entity_type = st.radio("Look up by", ["Source IP", "Wallet address", "Transaction ID (txid)"], horizontal=True)

    if entity_type == "Source IP":
        options = sorted(alerts["src_ip"].dropna().unique().tolist())
        selected = st.selectbox("Select source IP", options)
        subset = alerts[alerts["src_ip"] == selected]
        st.write(f"**{len(subset)}** transactions observed from `{selected}`.")
        st.dataframe(subset[["txid", "timestamp_str", "priority", "risk_score", "anomaly_score_0_100",
                              "geo_country", "asn", "reasons"]], width='stretch')

    elif entity_type == "Wallet address":
        in_lists = alerts["input_addresses"].apply(to_list)
        out_lists = alerts["output_addresses"].apply(to_list)
        all_wallets = pd.Series(sum(in_lists.tolist(), []) + sum(out_lists.tolist(), [])).dropna().unique()
        selected = st.selectbox("Select wallet address", sorted(all_wallets.tolist())[:5000])
        mask = in_lists.apply(lambda a: selected in a) | out_lists.apply(lambda a: selected in a)
        subset = alerts[mask]
        st.write(f"**{len(subset)}** transactions involve wallet `{selected}`.")
        st.dataframe(subset[["txid", "src_ip", "timestamp_str", "priority", "risk_score",
                              "geo_country", "reasons"]], width='stretch')

    else:
        selected = st.selectbox("Select transaction ID", sorted(alerts["txid"].unique().tolist())[:5000])
        subset = alerts[alerts["txid"] == selected]
        if len(subset) > 0:
            row = subset.iloc[0]
            colL, colR = st.columns(2)
            with colL:
                st.markdown(f"**TXID:** `{row['txid']}`")
                st.markdown(f"**Source IP:** `{row['src_ip']}`  \n**Dest IP:** `{row['dst_ip']}`")
                st.markdown(f"**Timestamp:** {row['timestamp_str']}")
                st.markdown(f"**Country / ASN:** {row['geo_country']} / {row['asn']}")
                st.markdown(f"**Input addresses:** {row['input_addresses']}")
                st.markdown(f"**Output addresses:** {row['output_addresses']}")
                st.markdown(f"**Input amounts:** {row['input_amounts']}")
                st.markdown(f"**Output amounts:** {row['output_amounts']}")
            with colR:
                st.metric("Risk score", f"{row['risk_score']:.1f}", row["priority"])
                st.metric("ML anomaly score", f"{row['anomaly_score_0_100']:.1f}")
                st.metric("Cluster", str(row["cluster_label"]))
                st.markdown("**Reasons:**")
                for r in str(row["reasons"]).split(" | "):
                    st.write(f"- {r}")

# =========================================================================== #
# TAB 5 — GRAPH EXPLORER (investigator view over analytical entity graph)
# =========================================================================== #
with tabs[4]:
    st.markdown(
        """
        <style>
        .gx-wrap {
            background: #14171c;
            border: 1px solid #2a2f38;
            border-radius: 6px;
            padding: 14px 16px 8px 16px;
            margin-bottom: 12px;
        }
        .gx-title {
            font-family: 'Segoe UI', 'IBM Plex Sans', sans-serif;
            font-size: 1.15rem;
            font-weight: 600;
            color: #e8eaed;
            letter-spacing: 0.02em;
            margin: 0 0 4px 0;
        }
        .gx-sub {
            color: #8b919c;
            font-size: 0.85rem;
            margin: 0 0 10px 0;
        }
        .gx-panel {
            background: #1a1d23;
            border: 1px solid #2a2f38;
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }
        .gx-panel h4 {
            margin: 0 0 8px 0;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #8b919c;
            font-weight: 600;
        }
        .gx-kv {
            font-size: 0.88rem;
            color: #c8ccd4;
            margin: 3px 0;
            line-height: 1.45;
        }
        .gx-kv b { color: #e8eaed; font-weight: 600; }
        .gx-muted { color: #8b919c; font-size: 0.8rem; }
        .gx-stat {
            display: inline-block;
            margin-right: 14px;
            color: #c8ccd4;
            font-size: 0.82rem;
        }
        .gx-stat strong { color: #e8eaed; }
        .gx-why {
            background: #1e2229;
            border-left: 3px solid #4C9AFF;
            padding: 8px 10px;
            margin: 4px 0;
            font-size: 0.84rem;
            color: #c8ccd4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="gx-wrap">'
        '<p class="gx-title">Graph Explorer — Investigation Workspace</p>'
        '<p class="gx-sub">Selective investigator view derived from the analytical entity graph. '
        'The stored graph is not modified; this panel ranks and budgets a neighborhood for investigation.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    g = load_graph()
    cases_for_graph = load_cases()

    node_types: dict = {}
    for n, data in g.nodes(data=True):
        node_types.setdefault(data.get("type", "Unknown"), []).append(n)

    default_hops = int(DASH_CFG.get("max_graph_neighborhood_hops", 2))
    default_max_nodes = int(DASH_CFG.get("graph_max_nodes", 75))
    default_max_edges = int(DASH_CFG.get("graph_max_edges", 150))
    default_max_tx = int(DASH_CFG.get("graph_max_transactions", 40))

    if "gx_view_html" not in st.session_state:
        st.session_state["gx_view_html"] = None
    if "gx_view_stats" not in st.session_state:
        st.session_state["gx_view_stats"] = None
    if "gx_view_explanations" not in st.session_state:
        st.session_state["gx_view_explanations"] = {}
    if "gx_detail_node" not in st.session_state:
        st.session_state["gx_detail_node"] = None

    ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.6, 1.1])
    with ctrl1:
        selected_type = st.selectbox(
            "Entity type",
            sorted(node_types.keys()),
            key="gx_entity_type",
        )
    with ctrl2:
        candidates = sorted(node_types.get(selected_type, []))[:5000]
        selected_node = st.selectbox("Entity", candidates, key="gx_entity")
    with ctrl3:
        focus_mode = st.selectbox("Focus", list(FOCUS_MODES), key="gx_focus")

    ctrl4, ctrl5, ctrl6 = st.columns(3)
    with ctrl4:
        hops = st.slider(
            "Neighborhood hops",
            min_value=1,
            max_value=max(default_hops + 1, 2),
            value=min(2, default_hops),
            key="gx_hops",
        )
    with ctrl5:
        max_nodes = st.slider(
            "Max nodes",
            min_value=20,
            max_value=200,
            value=default_max_nodes,
            key="gx_max_nodes",
        )
    with ctrl6:
        max_edges = st.slider(
            "Max edges",
            min_value=30,
            max_value=400,
            value=default_max_edges,
            key="gx_max_edges",
        )

    b1, b2, _ = st.columns([1, 1, 2])
    with b1:
        build_clicked = st.button("Build Investigation View", type="primary", key="gx_build")
    with b2:
        reset_clicked = st.button("Reset View", key="gx_reset")

    if reset_clicked:
        st.session_state["gx_view_html"] = None
        st.session_state["gx_view_stats"] = None
        st.session_state["gx_view_explanations"] = {}
        st.session_state["gx_detail_node"] = None
        st.session_state["gx_built_node"] = None

    if build_clicked and selected_node:
        with st.spinner("Building ranked investigator neighborhood..."):
            view = build_investigator_view(
                g,
                alerts,
                selected_node,
                hops=hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
                max_transactions=default_max_tx,
                focus=focus_mode,
                cases=cases_for_graph,
            )
            st.session_state["gx_built_node"] = selected_node
            st.session_state["gx_detail_node"] = selected_node
            st.session_state["gx_view_stats"] = view.stats
            st.session_state["gx_view_explanations"] = view.explanations
            st.session_state["gx_view_relevance"] = view.relevance

            if view.empty:
                st.session_state["gx_view_html"] = None
                st.warning("Selected entity was not found in the analytical graph.")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_html = Path(tmpdir) / "investigator_view.html"
                    export_investigator_pyvis_html(
                        view.graph,
                        out_html,
                        selected_node=selected_node,
                        relevance=view.relevance,
                        explanations=view.explanations,
                    )
                    st.session_state["gx_view_html"] = out_html.read_text(encoding="utf-8")

    # Keep detail panel in sync with current entity selector when a view exists
    if st.session_state.get("gx_view_html") and selected_node:
        if st.session_state.get("gx_detail_node") != selected_node:
            # Allow inspecting another entity's details without rebuilding unless user clicks Build
            st.session_state["gx_detail_node"] = selected_node

    graph_col, detail_col = st.columns([1.65, 1.0])

    with graph_col:
        stats = st.session_state.get("gx_view_stats")
        if stats:
            st.markdown(
                f'<div class="gx-panel"><h4>View summary</h4>'
                f'<span class="gx-stat">Candidate <strong>{stats.get("candidate_nodes", 0)}</strong> nodes / '
                f'<strong>{stats.get("candidate_edges", 0)}</strong> edges</span>'
                f'<span class="gx-stat">Retained <strong>{stats.get("retained_nodes", 0)}</strong> nodes / '
                f'<strong>{stats.get("retained_edges", 0)}</strong> edges</span>'
                f'<span class="gx-stat">Focus <strong>{stats.get("focus", "")}</strong></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        html_content = st.session_state.get("gx_view_html")
        if html_content:
            st.iframe(html_content, height=660)
        else:
            st.markdown(
                '<div class="gx-panel"><p class="gx-muted">'
                "No investigation view built yet. Select an entity and click "
                "<b>Build Investigation View</b>."
                "</p></div>",
                unsafe_allow_html=True,
            )

    with detail_col:
        detail_node = st.session_state.get("gx_detail_node") or selected_node
        details = get_entity_details(
            g, detail_node, alerts=alerts, cases=cases_for_graph
        ) if detail_node else {"available": False}

        st.markdown('<div class="gx-panel"><h4>Selected entity details</h4>', unsafe_allow_html=True)
        if not details.get("available"):
            st.markdown('<p class="gx-muted">No entity selected.</p></div>', unsafe_allow_html=True)
        else:
            ntype = details.get("type")
            lines = [f'<div class="gx-kv"><b>Type</b> — {ntype}</div>']

            if ntype == "Transaction":
                def _fmt(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return "—"
                    if isinstance(v, float):
                        return f"{v:.1f}"
                    return str(v)

                lines.extend([
                    f'<div class="gx-kv"><b>TXID</b> — <code>{details.get("txid")}</code></div>',
                    f'<div class="gx-kv"><b>Priority</b> — {_fmt(details.get("priority"))}</div>',
                    f'<div class="gx-kv"><b>Risk score</b> — {_fmt(details.get("risk_score"))}</div>',
                    f'<div class="gx-kv"><b>Anomaly score</b> — {_fmt(details.get("anomaly_score_0_100"))}</div>',
                    f'<div class="gx-kv"><b>Cluster</b> — {_fmt(details.get("cluster_label"))}</div>',
                    f'<div class="gx-kv"><b>Source IP</b> — {_fmt(details.get("src_ip"))}</div>',
                    f'<div class="gx-kv"><b>Dest IP</b> — {_fmt(details.get("dst_ip"))}</div>',
                    f'<div class="gx-kv"><b>Country</b> — {_fmt(details.get("geo_country"))}</div>',
                    f'<div class="gx-kv"><b>ASN</b> — {_fmt(details.get("asn"))}</div>',
                    f'<div class="gx-kv"><b>Input wallets</b> — {_fmt(details.get("input_addresses"))}</div>',
                    f'<div class="gx-kv"><b>Output wallets</b> — {_fmt(details.get("output_addresses"))}</div>',
                ])
                reasons = details.get("reasons")
                if reasons is not None and str(reasons) not in ("", "nan", "None"):
                    lines.append(f'<div class="gx-kv"><b>Reasons</b> — {reasons}</div>')

            elif ntype == "IP":
                lines.extend([
                    f'<div class="gx-kv"><b>IP</b> — <code>{details.get("ip")}</code></div>',
                    f'<div class="gx-kv"><b>Connected transactions</b> — {details.get("connected_transaction_count", 0)}</div>',
                    f'<div class="gx-kv"><b>Suspicious transactions</b> — {details.get("suspicious_transaction_count", 0)}</div>',
                    f'<div class="gx-kv"><b>Maximum risk</b> — {details.get("maximum_risk", 0):.1f}</div>',
                    f'<div class="gx-kv"><b>Countries</b> — {", ".join(details.get("countries") or []) or "—"}</div>',
                    f'<div class="gx-kv"><b>ASN</b> — {", ".join(details.get("asns") or []) or "—"}</div>',
                ])

            elif ntype == "Wallet":
                lines.extend([
                    f'<div class="gx-kv"><b>Wallet</b> — <code>{details.get("wallet")}</code></div>',
                    f'<div class="gx-kv"><b>Connected transactions</b> — {details.get("connected_transaction_count", 0)}</div>',
                    f'<div class="gx-kv"><b>Suspicious transactions</b> — {details.get("suspicious_transaction_count", 0)}</div>',
                    f'<div class="gx-kv"><b>Maximum risk</b> — {details.get("maximum_risk", 0):.1f}</div>',
                    f'<div class="gx-kv"><b>Related IPs</b> — {", ".join(details.get("related_ips") or []) or "—"}</div>',
                ])

            elif ntype == "ASN":
                lines.extend([
                    f'<div class="gx-kv"><b>ASN</b> — <code>{details.get("asn")}</code></div>',
                    f'<div class="gx-kv"><b>Connected IPs</b> — {len(details.get("connected_ips") or [])}</div>',
                    f'<div class="gx-kv"><b>Countries</b> — {", ".join(details.get("countries") or []) or "—"}</div>',
                    f'<div class="gx-kv"><b>Relevant transactions</b> — {len(details.get("relevant_transactions") or [])}</div>',
                ])
                with st.expander("Connected IPs"):
                    st.write(details.get("connected_ips") or [])

            elif ntype == "Country":
                lines.extend([
                    f'<div class="gx-kv"><b>Country</b> — <code>{details.get("country")}</code></div>',
                    f'<div class="gx-kv"><b>Connected IPs</b> — {len(details.get("connected_ips") or [])}</div>',
                    f'<div class="gx-kv"><b>Relevant transactions</b> — {len(details.get("relevant_transactions") or [])}</div>',
                ])
                with st.expander("Connected IPs"):
                    st.write(details.get("connected_ips") or [])

            case_ids = details.get("case_ids") or []
            if case_ids:
                shown = ", ".join(case_ids[:8])
                more = f" (+{len(case_ids) - 8} more)" if len(case_ids) > 8 else ""
                lines.append(f'<div class="gx-kv"><b>Case IDs</b> — {shown}{more}</div>')

            st.markdown("\n".join(lines) + "</div>", unsafe_allow_html=True)

        # Why is this node shown?
        st.markdown('<div class="gx-panel"><h4>Why is this node shown?</h4>', unsafe_allow_html=True)
        exps = st.session_state.get("gx_view_explanations") or {}
        node_exps = exps.get(detail_node, []) if detail_node else []
        if node_exps:
            why_html = "".join(f'<div class="gx-why">{e}</div>' for e in node_exps[:6])
            st.markdown(why_html + "</div>", unsafe_allow_html=True)
        elif st.session_state.get("gx_view_html"):
            st.markdown(
                '<p class="gx-muted">No ranked explanation for this entity in the current view. '
                "Build the view with this entity selected to see retention reasons.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="gx-muted">Build an investigation view to see retention reasons.</p></div>',
                unsafe_allow_html=True,
            )

        # Quick list of retained high-relevance nodes
        rel = st.session_state.get("gx_view_relevance") or {}
        if rel:
            st.markdown('<div class="gx-panel"><h4>Top retained by relevance</h4>', unsafe_allow_html=True)
            top = sorted(rel.items(), key=lambda kv: kv[1], reverse=True)[:8]
            rows = []
            for nid, score in top:
                ntype = g.nodes[nid].get("type", "?") if nid in g else "?"
                label = g.nodes[nid].get("label", nid) if nid in g else nid
                rows.append(
                    f'<div class="gx-kv"><b>{ntype}</b> {str(label)[:28]} '
                    f'<span class="gx-muted">({score:.1f})</span></div>'
                )
            st.markdown("\n".join(rows) + "</div>", unsafe_allow_html=True)

# =========================================================================== #
# TAB 6 — ML / EVALUATION
# =========================================================================== #
with tabs[5]:
    st.subheader("Model details")
    st.markdown(
        "- **Anomaly detection:** Isolation Forest (unsupervised), trained on the standardized "
        "behavioral + graph feature matrix. `synthetic_label` was **not** used for training.\n"
        "- **Clustering:** DBSCAN over the same standardized feature space.\n"
        "- **Risk score:** transparent weighted combination of the ML anomaly score, behavioral, "
        "graph, and network/geographic components (see `config.yaml`)."
    )

    feature_cols = run_meta.get("feature_columns", [])
    st.markdown(f"**Feature matrix columns ({len(feature_cols)}):**")
    st.code(", ".join(feature_cols), language="text")

    st.subheader("Evaluation against synthetic_label (evaluation-only)")
    st.caption(
        "These metrics compare the unsupervised model's predictions to the dataset's synthetic ground-truth "
        "label. They are computed at runtime from this exact run — never hard-coded."
    )
    if eval_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{eval_metrics['precision']:.3f}")
        m2.metric("Recall", f"{eval_metrics['recall']:.3f}")
        m3.metric("F1", f"{eval_metrics['f1']:.3f}")
        roc = eval_metrics.get("roc_auc")
        m4.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "N/A")

        pr_auc = eval_metrics.get("pr_auc")
        st.metric("PR-AUC", f"{pr_auc:.3f}" if pr_auc is not None else "N/A")

        cm = eval_metrics.get("confusion_matrix", {})
        st.subheader("Confusion matrix")
        cm_df = pd.DataFrame(
            [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]],
            index=["Actual: Normal", "Actual: Anomalous"],
            columns=["Predicted: Normal", "Predicted: Anomalous"],
        )
        st.dataframe(cm_df, width='stretch')
    else:
        st.write("No evaluation metrics found — re-run `python run.py`.")

    st.subheader("Cluster summary")
    st.json(cluster_summary, expanded=False)

# =========================================================================== #
# TAB 7 — DATA QUALITY
# =========================================================================== #
with tabs[6]:
    st.subheader("Ingestion data-quality report")
    if data_quality:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total records", f"{data_quality.get('total_records', 0):,}")
        d2.metric("Valid records", f"{data_quality.get('valid_records', 0):,}")
        d3.metric("Invalid records", f"{data_quality.get('invalid_records', 0):,}")
        d4.metric("Duplicate records", f"{data_quality.get('duplicate_records', 0):,}")

        st.subheader("Invalid record reasons")
        reasons = data_quality.get("invalid_reasons", {})
        if reasons:
            st.bar_chart(pd.Series(reasons))
        else:
            st.write("No invalid records found.")

        st.subheader("Missing value counts by field")
        missing = data_quality.get("missing_value_counts", {})
        if missing:
            st.dataframe(pd.Series(missing, name="missing_count").sort_values(ascending=False),
                         width='stretch')
    else:
        st.write("No data-quality report found — re-run `python run.py`.")

# =========================================================================== #
# TAB 8 — FORENSIC BLOCKCHAIN / EVIDENCE INTEGRITY
# =========================================================================== #
with tabs[7]:
    st.subheader("Forensic blockchain and evidence integrity")
    st.caption(
        "Related suspicious transactions are grouped into investigation cases (not simply "
        "equated with a DBSCAN cluster). Each case's evidence is hashed into a Merkle tree, "
        "and the Merkle root is anchored into a local, private, append-only blockchain ledger. "
        "Everything on this tab runs fully offline; no external chain or network is involved."
    )

    if not forensics_available():
        st.warning(
            "No forensic ledger or evidence store found yet. Run `python run.py` first "
            "to build investigation cases and anchor their evidence."
        )
        st.stop()

    cases_list = load_cases()
    evidence_store_path, blockchain_path = get_forensic_paths()

    # Fresh reads every render -- never cached -- so tampering made outside this
    # process (e.g. via scripts/tamper_demo.py, or hand-editing the JSON file) is
    # always reflected immediately, and Verify never shows a stale "clean" result.
    live_chain = Blockchain(blockchain_path)
    live_store = EvidenceStore(evidence_store_path)
    live_chain_validation = live_chain.validate_chain()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Investigation cases", f"{len(cases_list):,}")
    m2.metric("Blocks in ledger", f"{len(live_chain.blocks):,}")
    m3.metric("Ledger integrity", "VALID" if live_chain_validation["valid"] else "INVALID")
    m4.metric("Ledger errors", f"{len(live_chain_validation['errors']):,}")

    if not live_chain_validation["valid"]:
        st.error(
            "The forensic ledger itself failed internal validation (hash chain or Merkle "
            "root mismatch). Details:\n\n" + "\n".join(f"- {e}" for e in live_chain_validation["errors"])
        )

    if not cases_list:
        st.info("No investigation cases were built on the most recent run "
                 "(no transactions met the seed-priority / anomaly criteria).")
        st.stop()

    st.markdown("### Case selection")
    case_options = {
        f"{c['case_id']}  [{c['case_priority']}, {c['num_transactions']} tx]": c["case_id"]
        for c in cases_list
    }
    selected_label = st.selectbox("Select an investigation case", list(case_options.keys()))
    selected_case_id = case_options[selected_label]
    selected_case = next(c for c in cases_list if c["case_id"] == selected_case_id)

    st.markdown("#### Case summary")
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Priority", selected_case["case_priority"])
    cc2.metric("Linked transactions", selected_case["num_transactions"])
    cc3.metric("Max risk score", f"{selected_case['max_risk_score']:.1f}")
    cc4.metric("Avg risk score", f"{selected_case['avg_risk_score']:.1f}")

    st.write(f"**Time range:** {selected_case['time_start']} to {selected_case['time_end']}")
    st.write(f"**Clusters involved:** {selected_case['clusters_involved']} "
             "(cluster membership alone never creates a case link -- see link types below)")
    st.write(f"**Link type counts:** {selected_case['link_type_counts']}")
    st.write(f"**Source IPs:** {', '.join(selected_case['source_ips']) or 'none'}")
    st.write(f"**Countries:** {', '.join(selected_case['countries']) or 'none'}")

    with st.expander(f"Wallets involved ({len(selected_case['wallets'])})"):
        st.write(selected_case["wallets"])
    with st.expander(f"Alert reasons across this case ({len(selected_case['alert_reasons'])})"):
        for r in selected_case["alert_reasons"]:
            st.write(f"- {r}")
    with st.expander(f"Graph-derived link evidence ({len(selected_case['link_edges'])} edge(s))"):
        if selected_case["link_edges"]:
            st.dataframe(pd.DataFrame(selected_case["link_edges"]), width='stretch')
        else:
            st.write("No pairwise links (single-transaction case).")

    st.markdown("### Case -> transactions -> entities -> evidence")
    linked_txids = selected_case["linked_txids"]
    selected_txid = st.selectbox("Select a transaction within this case", linked_txids)

    case_evidence_records = live_store.get_case_evidence(selected_case_id)
    matching_records = [r for r in case_evidence_records if r.get("txid") == selected_txid]
    if matching_records:
        st.markdown("**Current evidence record (as currently stored on disk):**")
        st.json(matching_records[0], expanded=False)
    else:
        st.warning("This transaction has no evidence record in the current store "
                    "(the store may be out of sync with the case summary -- re-run `python run.py`).")

    st.markdown("### Verify evidence integrity")
    st.caption(
        "Verification only reads what is currently on disk and compares it against the "
        "hashes frozen in the blockchain when this case was anchored. It never rewrites "
        "or 're-fixes' evidence, even if a mismatch is found."
    )

    if st.button("Verify Evidence", type="primary", key="verify_case_btn"):
        result = verify_case(selected_case_id, live_store, live_chain)
        st.session_state["last_verify_result"] = result

    result = st.session_state.get("last_verify_result")
    if result and result.get("case_id") == selected_case_id:
        status = result["status"]
        if status == "VERIFIED":
            st.success(f"Status: VERIFIED -- {result['message']}")
        elif status == "NOT_ANCHORED":
            st.warning(f"Status: NOT ANCHORED -- {result['message']}")
        else:
            st.error(f"Status: FAILED -- {result['message']}")

        if status != "NOT_ANCHORED":
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Anchored in block", result["block_index"])
            vc2.metric("Merkle root match", "YES" if result["merkle_root_match"] else "NO")
            vc3.metric("Evidence count match", "YES" if result["evidence_count_match"] else "NO")

            st.markdown("**Anchored Merkle root:**")
            st.code(result["anchored_merkle_root"], language="text")
            st.markdown("**Recomputed Merkle root (from current on-disk evidence):**")
            st.code(result["recomputed_merkle_root"], language="text")

            st.markdown("**Per-transaction hash comparison:**")
            per_txid_df = pd.DataFrame(result["per_txid_results"])
            if not per_txid_df.empty:
                per_txid_df["match"] = per_txid_df["match"].map({True: "MATCH", False: "MISMATCH"})
                st.dataframe(per_txid_df, width='stretch')

            if result["chain_errors"]:
                st.markdown("**Blockchain ledger validation errors:**")
                for e in result["chain_errors"]:
                    st.write(f"- {e}")

    st.markdown("### Merkle proof drill-down")
    st.caption(
        "Full trace for one transaction: current evidence hash -> Merkle proof (sibling "
        "hashes) -> whether replaying that proof reconstructs the anchored Merkle root -> "
        "which blockchain block anchors it."
    )
    if st.button("Generate Merkle Proof", key="proof_btn"):
        proof_result = generate_proof_for_txid(selected_case_id, selected_txid, live_store, live_chain)
        st.session_state["last_proof_result"] = proof_result

    proof_result = st.session_state.get("last_proof_result")
    if proof_result and proof_result.get("txid") == selected_txid and "error" not in proof_result:
        pc1, pc2 = st.columns(2)
        pc1.metric("Hash match vs anchored", "YES" if proof_result["hash_match"] else "NO")
        pc2.metric("Proof reconstructs anchored root", "YES" if proof_result["proof_reconstructs_anchored_root"] else "NO")

        st.markdown("**Current evidence hash:**")
        st.code(proof_result["current_hash"] or "N/A", language="text")
        st.markdown("**Hash anchored in the blockchain at anchoring time:**")
        st.code(proof_result["anchored_hash"] or "N/A", language="text")
        st.markdown(f"**Merkle proof ({len(proof_result['proof'])} step(s), leaf index {proof_result['leaf_index']}):**")
        if proof_result["proof"]:
            st.dataframe(pd.DataFrame(proof_result["proof"]), width='stretch')
        else:
            st.write("No proof steps needed (single-evidence case; the leaf hash is the root).")
    elif proof_result and "error" in proof_result:
        st.error(proof_result["error"])

    st.markdown("### Blockchain ledger")
    st.caption(
        "Only hashes, references, and small metadata are stored in each block -- never raw "
        "transaction data, wallet addresses, or IPs. Full evidence content lives only in the "
        "local evidence store shown above."
    )
    ledger_rows = []
    for b in live_chain.blocks:
        ledger_rows.append({
            "index": b.index,
            "timestamp": b.timestamp,
            "case_id": b.case_id,
            "evidence_count": b.evidence_count,
            "merkle_root": b.merkle_root[:16] + "...",
            "previous_hash": b.previous_hash[:16] + "...",
            "hash": b.hash[:16] + "...",
        })
    st.dataframe(pd.DataFrame(ledger_rows), width='stretch', height=350)

    st.download_button(
        "Download full blockchain ledger (JSON)",
        data=json.dumps(live_chain.to_list_of_dicts(), indent=2).encode("utf-8"),
        file_name="forensic_blockchain_ledger.json",
        mime="application/json",
    )

