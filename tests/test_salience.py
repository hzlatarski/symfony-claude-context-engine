"""Tests for structural salience ranking over the unified graph.

Toy graphs with hand-verifiable structure: a barbell (two cliques joined by
one edge) makes the broker and bridge definitions checkable by eye, and a
star makes the hub definition checkable.
"""
import json

from scripts import salience


def _graph(nodes, edges, *, kinds=None):
    """Build a unified-graph-shaped dict. ``kinds`` maps node id -> kind."""
    kinds = kinds or {}
    return {
        "nodes": {
            nid: {"kind": kinds.get(nid, "article"), "label": nid, "type": "fact"}
            for nid in nodes
        },
        "edges": [{"from": a, "to": b, "kind": "wikilink"} for a, b in edges],
    }


def _barbell():
    """Two 4-cliques (A*, B*) joined by the single edge A1–B1.

    A1 and B1 are the only route between the halves, so they are the brokers
    and A1–B1 is the only bridge.
    """
    a = ["A1", "A2", "A3", "A4"]
    b = ["B1", "B2", "B3", "B4"]
    edges = []
    for group in (a, b):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append((group[i], group[j]))
    edges.append(("A1", "B1"))
    return _graph(a + b, edges)


# ── hubs ──────────────────────────────────────────────────────────────


def test_hubs_rank_the_star_centre_first():
    graph = _graph(["H", "L1", "L2", "L3", "L4"],
                   [("H", "L1"), ("H", "L2"), ("H", "L3"), ("H", "L4")])
    report = salience.compute(graph)
    assert report["hubs"]["article"][0]["id"] == "H"
    assert report["hubs"]["article"][0]["degree"] == 4


def test_hubs_exclude_isolated_nodes():
    graph = _graph(["A", "B", "LONELY"], [("A", "B")])
    report = salience.compute(graph)
    assert "LONELY" not in [r["id"] for r in report["hubs"]["article"]]


def test_hubs_are_bucketed_by_scope_not_filtered_from_the_graph():
    """A code node must not be able to displace articles from the article table."""
    graph = _graph(
        ["ART", "A2", "SYM", "S2", "S3", "S4"],
        [("ART", "A2"), ("SYM", "S2"), ("SYM", "S3"), ("SYM", "S4"), ("SYM", "ART")],
        kinds={"SYM": "symbol", "S2": "symbol", "S3": "symbol", "S4": "symbol"},
    )
    report = salience.compute(graph)
    assert [r["id"] for r in report["hubs"]["code"]][0] == "SYM"
    assert all(r["kind"] == "article" for r in report["hubs"]["article"])


# ── brokers ───────────────────────────────────────────────────────────


def test_brokers_are_the_barbell_joint():
    report = salience.compute(_barbell())
    top_two = {r["id"] for r in report["brokers"]["article"][:2]}
    assert top_two == {"A1", "B1"}


def test_brokers_exclude_zero_betweenness_nodes():
    """A clique has no brokers — every pair is already adjacent."""
    graph = _graph(["A", "B", "C"], [("A", "B"), ("B", "C"), ("A", "C")])
    report = salience.compute(graph)
    assert report["brokers"]["article"] == []


def test_brokers_and_hubs_are_genuinely_different_orderings():
    """On the 5-path P1–P2–P3–P4–P5 the two measures disagree by hand.

    Degrees: P1=1, P2=P3=P4=2, P5=1 — so the degree ranking ties across the
    middle three and resolves to P2 on the id tiebreak. Betweenness: P3=4,
    P2=P4=3 — the midpoint carries strictly more traffic. Top hub P2, top
    broker P3. If brokers were silently ranked by degree this test fails.
    """
    graph = _graph(["P1", "P2", "P3", "P4", "P5"],
                   [("P1", "P2"), ("P2", "P3"), ("P3", "P4"), ("P4", "P5")])
    report = salience.compute(graph)
    assert report["hubs"]["article"][0]["id"] == "P2"
    assert report["brokers"]["article"][0]["id"] == "P3"
    assert report["brokers"]["article"][0]["betweenness"] == 4.0


# ── bridges (surprising connections) ──────────────────────────────────


def _all_bridges(report):
    return [b for bucket in report["bridges"].values() for b in bucket]


def test_bridge_is_the_single_cross_community_edge():
    report = salience.compute(_barbell())
    bridges = _all_bridges(report)
    assert len(bridges) == 1
    bridge = bridges[0]
    assert {bridge["from"], bridge["to"]} == {"A1", "B1"}
    assert bridge["from_community"] != bridge["to_community"]
    assert bridge["pair_edges"] == 1
    assert bridge["joins"] == "4<->4"


def test_triangle_closing_edge_is_not_a_bridge():
    """Endpoints sharing a neighbour are corroborated, not surprising."""
    pairs = {frozenset((b["from"], b["to"])) for b in _all_bridges(salience.compute(_barbell()))}
    assert frozenset(("A1", "A2")) not in pairs


def test_pendant_edge_is_not_a_bridge():
    """A leaf's only edge is trivially a cut — that is not a surprise."""
    graph = _barbell()
    graph["nodes"]["LEAF"] = {"kind": "article", "label": "LEAF", "type": "fact"}
    graph["edges"].append({"from": "A2", "to": "LEAF", "kind": "wikilink"})
    report = salience.compute(graph)
    assert all("LEAF" not in (b["from"], b["to"]) for b in _all_bridges(report))


def test_universal_sink_is_not_a_bridge():
    """The defect this filter exists for.

    ``SINK`` is called by every cluster, shares no neighbour with any caller,
    and sits in its own community — so on raw cross-community edge betweenness
    every call to it ranked as a "surprising connection". They are the least
    surprising edges in the graph. Two clusters wired to SINK by 4 edges each
    exceed ``max_pair_edges``, so none of them qualifies.
    """
    a = ["A1", "A2", "A3", "A4"]
    b = ["B1", "B2", "B3", "B4"]
    edges = []
    for group in (a, b):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append((group[i], group[j]))
    edges += [(n, "SINK") for n in a + b]
    report = salience.compute(_graph(a + b + ["SINK"], edges))
    assert all("SINK" not in (br["from"], br["to"]) for br in _all_bridges(report))


def test_max_pair_edges_admits_weaker_bridges_when_raised():
    """Two clusters joined by exactly 2 edges: excluded at 1, included at 2."""
    a, b = ["A1", "A2", "A3"], ["B1", "B2", "B3"]
    edges = [("A1", "A2"), ("A2", "A3"), ("A1", "A3"),
             ("B1", "B2"), ("B2", "B3"), ("B1", "B3"),
             ("A1", "B1"), ("A2", "B2")]
    graph = _graph(a + b, edges)
    assert _all_bridges(salience.compute(graph, max_pair_edges=1)) == []
    assert len(_all_bridges(salience.compute(graph, max_pair_edges=2))) == 2


def test_default_thresholds_differ_between_article_and_code():
    """A single threshold cannot serve both graphs — measured on the real KB.

    At 3 the code graph readmits EntityManagerInterface::flush; at 1 the
    article graph (which crosslink.py deliberately densifies) has no bridges
    at all. If these are ever collapsed to one value, one of the two buckets
    silently becomes useless.
    """
    assert salience.DEFAULT_MAX_PAIR_EDGES["code"] == 1
    assert salience.DEFAULT_MAX_PAIR_EDGES["article"] > 1


def test_resolve_max_pair_edges_accepts_none_int_and_partial_dict():
    assert salience.resolve_max_pair_edges(None) == salience.DEFAULT_MAX_PAIR_EDGES
    assert salience.resolve_max_pair_edges(7) == {"article": 7, "code": 7, "mixed": 7}
    merged = salience.resolve_max_pair_edges({"code": 9})
    assert merged["code"] == 9
    assert merged["article"] == salience.DEFAULT_MAX_PAIR_EDGES["article"]


def test_thresholds_are_applied_per_bucket_not_globally():
    """Two clusters joined by 2 edges qualify as articles but not as code."""
    def _two_edge_pair(kind):
        a, b = ["A1", "A2", "A3"], ["B1", "B2", "B3"]
        edges = [("A1", "A2"), ("A2", "A3"), ("A1", "A3"),
                 ("B1", "B2"), ("B2", "B3"), ("B1", "B3"),
                 ("A1", "B1"), ("A2", "B2")]
        return _graph(a + b, edges, kinds={n: kind for n in a + b})

    article_report = salience.compute(_two_edge_pair("article"))
    code_report = salience.compute(_two_edge_pair("symbol"))
    assert article_report["bridges"]["article"], "article threshold is >1, should qualify"
    assert code_report["bridges"]["code"] == [], "code threshold is 1, should not qualify"


def test_report_records_the_thresholds_it_used():
    report = salience.compute(_barbell(), max_pair_edges={"article": 5})
    assert report["max_pair_edges"]["article"] == 5
    assert "≤5" in salience.render(report)


def test_bridges_are_bucketed_so_dense_code_cannot_starve_articles():
    """A global cap would drop the article bridge behind the code ones."""
    graph = _barbell()
    for nid in ("C1", "C2", "C3", "C4"):
        graph["nodes"][nid] = {"kind": "symbol", "label": nid}
    graph["edges"] += [
        {"from": "C1", "to": "C2", "kind": "call"},
        {"from": "C2", "to": "C3", "kind": "call"},
        {"from": "C1", "to": "C3", "kind": "call"},
        {"from": "C1", "to": "A2", "kind": "call"},
    ]
    report = salience.compute(graph, cap=1)
    assert len(report["bridges"]["article"]) == 1
    assert {report["bridges"]["article"][0]["from"],
            report["bridges"]["article"][0]["to"]} == {"A1", "B1"}
    assert report["bridges"]["mixed"], "article<->code bridge should be its own bucket"


# ── orphans ───────────────────────────────────────────────────────────


def test_orphans_list_isolated_and_single_link_articles():
    graph = _graph(["A", "B", "C", "ISOLATED", "PENDANT"],
                   [("A", "B"), ("B", "C"), ("A", "C"), ("A", "PENDANT")])
    report = salience.compute(graph)
    found = {o["id"]: o["degree"] for o in report["orphans"]}
    assert found == {"ISOLATED": 0, "PENDANT": 1}


def test_orphans_ignore_code_nodes():
    """Only articles are actionable — an unreferenced vendor symbol is normal."""
    graph = _graph(["A", "B", "STRAY"], [("A", "B")],
                   kinds={"STRAY": "symbol"})
    report = salience.compute(graph)
    assert "STRAY" not in [o["id"] for o in report["orphans"]]


# ── determinism, edge cases, caching ──────────────────────────────────


def test_empty_graph_returns_empty_report():
    report = salience.compute({"nodes": {}, "edges": []})
    assert report["totals"]["nodes"] == 0
    assert _all_bridges(report) == []
    assert report["orphans"] == []


def test_edges_referencing_unknown_nodes_are_ignored():
    graph = _graph(["A", "B"], [("A", "B")])
    graph["edges"].append({"from": "A", "to": "GHOST", "kind": "wikilink"})
    report = salience.compute(graph)
    assert report["totals"]["nodes"] == 2
    assert report["totals"]["edges"] == 1


def test_parallel_edges_and_self_loops_do_not_inflate_degree():
    graph = _graph(["A", "B"], [("A", "B"), ("A", "B"), ("A", "A")])
    report = salience.compute(graph)
    assert report["hubs"]["article"][0]["degree"] == 1


def test_compute_is_deterministic():
    graph = _barbell()
    assert salience.compute(graph) == salience.compute(graph)


def test_cap_limits_every_ranking():
    graph = _graph([f"N{i}" for i in range(30)],
                   [(f"N{i}", f"N{j}") for i in range(30) for j in range(i + 1, 30)])
    report = salience.compute(graph, cap=5)
    assert len(report["hubs"]["article"]) == 5


def test_cache_distinguishes_different_thresholds(tmp_path):
    """A threshold change must not be served from the previous run's cache."""
    a, b = ["A1", "A2", "A3"], ["B1", "B2", "B3"]
    edges = [("A1", "A2"), ("A2", "A3"), ("A1", "A3"),
             ("B1", "B2"), ("B2", "B3"), ("B1", "B3"),
             ("A1", "B1"), ("A2", "B2")]
    graph = _graph(a + b, edges, kinds={n: "symbol" for n in a + b})
    cache = tmp_path / "salience.json"

    assert salience.load_or_compute(graph, cache_path=cache)["bridges"]["code"] == []
    loosened = salience.load_or_compute(graph, cache_path=cache, max_pair_edges={"code": 2})
    assert len(loosened["bridges"]["code"]) == 2


def test_cache_treats_none_and_the_equivalent_dict_as_one_request(tmp_path):
    graph = _barbell()
    cache = tmp_path / "salience.json"
    salience.load_or_compute(graph, cache_path=cache)
    stamp = cache.stat().st_mtime_ns
    salience.load_or_compute(
        graph, cache_path=cache, max_pair_edges=dict(salience.DEFAULT_MAX_PAIR_EDGES)
    )
    assert cache.stat().st_mtime_ns == stamp, "equivalent thresholds rewrote the cache"


def test_load_or_compute_reuses_cache_on_unchanged_graph(tmp_path):
    graph = _barbell()
    cache = tmp_path / "salience.json"
    first = salience.load_or_compute(graph, cache_path=cache)
    assert cache.exists()

    # Corrupt the stored report but keep the signature: a cache hit must
    # return the stored value, proving it did not silently recompute.
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["report"]["bridges"] = []
    cache.write_text(json.dumps(data), encoding="utf-8")

    assert salience.load_or_compute(graph, cache_path=cache)["bridges"] == []
    assert first["bridges"]  # the real computation did find one


def test_load_or_compute_recomputes_when_graph_changes(tmp_path):
    graph = _barbell()
    cache = tmp_path / "salience.json"
    salience.load_or_compute(graph, cache_path=cache)

    graph["nodes"]["NEW"] = {"kind": "article", "label": "NEW", "type": "fact"}
    graph["edges"].append({"from": "A2", "to": "NEW", "kind": "wikilink"})
    report = salience.load_or_compute(graph, cache_path=cache)
    assert "NEW" in [o["id"] for o in report["orphans"]]


def test_load_or_compute_survives_a_corrupt_cache(tmp_path):
    cache = tmp_path / "salience.json"
    cache.write_text("{not json", encoding="utf-8")
    report = salience.load_or_compute(_barbell(), cache_path=cache)
    assert len(_all_bridges(report)) == 1


def test_cache_version_bump_invalidates_an_existing_cache(tmp_path, monkeypatch):
    graph = _barbell()
    cache = tmp_path / "salience.json"
    salience.load_or_compute(graph, cache_path=cache)
    stale = json.loads(cache.read_text(encoding="utf-8"))["signature"]

    monkeypatch.setattr(salience, "CACHE_VERSION", salience.CACHE_VERSION + 1)
    salience.load_or_compute(graph, cache_path=cache)
    assert json.loads(cache.read_text(encoding="utf-8"))["signature"] != stale


# ── rendering ─────────────────────────────────────────────────────────


def test_render_includes_every_section():
    out = salience.render(salience.compute(_barbell()))
    for heading in ("## Hubs", "## Brokers", "## Bridges", "## Orphans"):
        assert heading in out


def test_render_escapes_pipes_so_tables_survive_node_ids():
    graph = _graph(["A|B", "C"], [("A|B", "C")])
    out = salience.render(salience.compute(graph))
    assert "A\\|B" in out
    assert "| A|B |" not in out


def test_render_scope_filters_presentation_only():
    graph = _graph(["ART", "A2", "SYM"], [("ART", "A2"), ("ART", "SYM")],
                   kinds={"SYM": "symbol"})
    report = salience.compute(graph)
    out = salience.render(report, scope="article")
    assert "### Code" not in out
    assert "### Articles" in out
