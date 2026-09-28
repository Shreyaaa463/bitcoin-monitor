"""The on-disk graph index answers exactly what the in-memory NetworkX prototype answers,
and the edge-table feature backend matches the NetworkX feature backend."""
import random

import numpy as np
import pandas as pd
import pytest

from tests.helpers import generate_dataset
from src.correlation.correlate import build_edges, build_edges_frame, edges_to_dataframe
from src.graph.build_graph import (build_entity_graph, build_entity_graph_from_frame, export_pyvis_html,
                                   get_neighborhood, graph_summary)
from src.graph.store import GraphStore
from src.features import engineer as fe
from src.ingestion.loader import load_raw_data
from src.preprocessing.clean import clean_dataframe


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("graph")
    df, _ = clean_dataframe(load_raw_data(generate_dataset(tmp / "d.json", n=400, seed=11)),
                            ["timestamp", "src_ip", "txid"], True, ["txid"])
    df["_row_no"] = np.arange(len(df))
    ef = build_edges_frame(df)
    g = build_entity_graph(build_edges(df))
    GraphStore.build(tmp / "g.sqlite", [ef.iloc[:500], ef.iloc[500:]])
    return df, ef, g, GraphStore(tmp / "g.sqlite")


def test_edge_frame_and_graph_match_prototype(world):
    df, ef, g, _ = world
    old = edges_to_dataframe(build_edges(df))
    for c in ["src", "dst", "relation", "src_type", "dst_type"]:
        assert ef[c].tolist() == old[c].tolist()
    g2 = build_entity_graph_from_frame(ef)
    assert list(g2.nodes(data=True)) == list(g.nodes(data=True))
    assert [(u, v, d["relation"]) for u, v, d in g2.edges(data=True)] == \
           [(u, v, d["relation"]) for u, v, d in g.edges(data=True)]


def test_summary_and_listing(world):
    _, _, g, store = world
    assert store.summary() == graph_summary(g)
    wallets = sorted(n for n in g.nodes if n.startswith("wallet:"))
    assert store.list_nodes("Wallet", limit=50) == wallets[:50]
    pref = wallets[3].split(":", 1)[1][:3]
    assert store.list_nodes("Wallet", prefix=pref) == [w for w in wallets if w.split(":", 1)[1].startswith(pref)]


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_neighborhoods_match_networkx(world, hops):
    _, _, g, store = world
    random.seed(hops)
    for n in random.sample(list(g.nodes), 40):
        ref = get_neighborhood(g, n, hops)
        sub, info = store.neighborhood(n, hops, max_nodes=10 ** 9, max_neighbors_per_node=10 ** 9)
        assert not info["truncated"]
        assert set(sub.nodes) == set(ref.nodes)
        assert sorted((u, v, d["relation"]) for u, v, d in sub.edges(data=True)) == \
               sorted((u, v, d["relation"]) for u, v, d in ref.edges(data=True))


def test_neighborhood_caps_bound_hub_nodes(world):
    _, _, g, store = world
    country = next(n for n in g.nodes if n.startswith("country:"))
    sub, info = store.neighborhood(country, 2, max_nodes=50, max_neighbors_per_node=10)
    assert info["truncated"] and sub.number_of_nodes() <= 50


def test_wallet_lookup(world):
    df, _, _, store = world
    w = df["output_addresses"].iloc[0][0]
    expected = sorted(t for t, a, b in zip(df.txid, df.input_addresses, df.output_addresses) if w in a or w in b)
    assert sorted(store.txids_for_wallet(w)) == expected


def test_edge_table_graph_features_match_networkx(world):
    df, _, g, store = world
    ref = fe._ip_graph_features(df, g, True, 500)
    new = fe._ip_graph_features_from_store(df, store, True)
    for c in ["ip_degree", "ip_in_degree", "ip_out_degree", "ip_betweenness"]:
        assert (new[c].to_numpy() == ref[c].to_numpy()).all(), c
    np.testing.assert_allclose(new["ip_pagerank"].to_numpy(), ref["ip_pagerank"].to_numpy(), rtol=1e-6)


def test_vectorized_ip_features_match_reference_loops(world):
    df, *_ = world
    new = fe._ip_behavioral_features(df, 1.0)
    for ip, sub in list(df.groupby("src_ip"))[:200]:
        times = sub["timestamp_parsed"].dropna().sort_values().reset_index(drop=True)
        best, left = 0, 0
        for right in range(len(times)):
            while times[right] - times[left] > pd.Timedelta(hours=1):
                left += 1
            best = max(best, right - left + 1)
        row = new.loc[sub.index[0]]
        assert row["ip_burst_count"] == best
        assert row["ip_tx_count"] == len(sub)


def test_pyvis_html_has_no_external_resources(tmp_path, world):
    _, _, g, store = world
    sub, _ = store.neighborhood(next(iter(g.nodes)), 1)
    html = export_pyvis_html(sub, tmp_path / "n.html").read_text(encoding="utf-8")
    import re
    assert not re.search(r"<(script|link)\b[^>]*(src|href)\s*=\s*[\"']https?://", html, re.I)
    assert "vis-network" in html.lower() or "vis.network" in html.lower()
