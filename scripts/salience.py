"""Structural salience over the unified knowledge graph.

Answers the questions the existing graph tools can't: *which* nodes carry
the knowledge base, and *which* links are the non-obvious ones.
``get_unified_neighbors`` and ``find_community`` both need you to already
know a node id. This module ranks the graph so you can find out what to
ask about.

Four rankings, all pure graph structure — zero LLM cost:

* **Hubs** — highest *degree*. The most-connected nodes: "what does this
  knowledge base mostly talk about?"
* **Brokers** — highest *betweenness centrality*. The load-bearing nodes:
  removing one disconnects parts of the graph from each other. A node can
  be a broker without being a hub (few links, but they are the only links).
* **Bridges** — the *surprising connections*. Edges whose endpoints sit in
  **different Leiden communities** that have almost no other link, and which
  share **no common neighbours**. Ranked by ``surprise`` — edge betweenness
  divided by ``deg(u)·deg(v)``, the configuration-model expectation — so
  that a utility everything calls cannot crowd out the real bridges. These
  are the links joining two otherwise-separate areas of the knowledge base:
  worth reading because nothing else in the graph implies them.
* **Orphans** — articles with degree ≤ 1. The compiler wrote them and
  (almost) nothing links to them, so neighbour- and community-based
  retrieval will never surface them. The actionable inverse of hubs.

Centrality is computed over the **whole** graph and only *presented* per
node kind. Filtering the graph to articles first would change every number
— an article's brokerage often runs through the code nodes it cites.
Filtering the presentation does not.

Public surface:

    compute(graph, *, seed=42, cap=50) -> dict
    load_or_compute(graph, *, cache_path, seed=42, cap=50) -> dict
    render(report, *, top_n=10, scope="all") -> str

Usage:
    uv run python scripts/salience.py              # write knowledge/SALIENCE.md
    uv run python scripts/salience.py --dry-run    # print, don't write
    uv run python scripts/salience.py --top-n 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Bump when the ranking logic changes shape or meaning. It is mixed into the
# cache signature, so an old cache computed by older logic can never be
# served as if this version had produced it.
# 2: bridges gained the max_pair_edges filter (v1 ranked universal sinks first).
# 3: bridges became a per-scope dict (v2 was a flat list; render() now indexes it).
# 4: max_pair_edges became per-scope (one global value could not serve both).
# 5: bridges rank by degree-normalized "surprise", not raw edge betweenness.
CACHE_VERSION = 5

# How many edges may join two communities before an edge between them stops
# counting as a "bridge". These differ because the two graphs have genuinely
# different densities, measured on this repo's own graph (8.5k nodes):
#
# * ``code`` must stay at 1. At 3, ``EntityManagerInterface::flush`` and
#   ``LoggerInterface::error`` reappear — 9 sink edges leak back into the
#   ranking, which is the exact noise this filter exists to remove.
# * ``article`` must be above 1. ``crosslink.py`` deliberately densifies the
#   article graph, so *no* two article communities are joined by a single
#   edge and the article bucket is empty at 1 — the most useful scope,
#   structurally guaranteed to return nothing.
#
# Sinks cannot leak into ``mixed``: that bucket requires an article endpoint,
# and articles do not call flush.
DEFAULT_MAX_PAIR_EDGES = {"article": 3, "code": 1, "mixed": 3}

# Everything that is not an article is "code" for presentation purposes.
_ARTICLE_KINDS = {"article"}

# A pendant edge (one endpoint has a single link) is trivially a "bridge" —
# it is the only way to reach a leaf. That is not a surprising connection,
# it is a leaf. Both endpoints must have at least this much degree.
_MIN_BRIDGE_DEGREE = 2


def _communities():
    """Import ``communities`` under either import regime.

    Entrypoints set up ``sys.path`` two different ways — the MCP server
    imports ``scripts.communities``, the CLI scripts import ``communities``.
    Same dual-path handling as ``unified_graph.build_for_project``.
    """
    try:
        from scripts import communities
    except ImportError:
        import communities
    return communities


def _build_igraph(graph: dict):
    """Return ``(igraph.Graph, node_ids)`` for the undirected simple projection.

    Self-loops and parallel edges are collapsed: both distort degree and
    betweenness, and neither carries meaning here (two wikilinks between the
    same pair of articles is one relationship). Edge direction is dropped for
    the same reason ``communities.detect`` drops it — salience is about
    connectivity, not flow.
    """
    import igraph as ig

    node_ids = list(graph["nodes"].keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    pairs = []
    for e in graph["edges"]:
        src = id_to_idx.get(e["from"])
        dst = id_to_idx.get(e["to"])
        if src is None or dst is None or src == dst:
            continue
        pairs.append((src, dst))

    g = ig.Graph(n=len(node_ids), edges=pairs, directed=False)
    g.simplify(multiple=True, loops=True)
    return g, node_ids


def _scope_of(kind: str) -> str:
    return "article" if kind in _ARTICLE_KINDS else "code"


def resolve_max_pair_edges(value: int | dict | None) -> dict[str, int]:
    """Normalize the ``max_pair_edges`` argument to a per-bucket dict.

    Accepts ``None`` (defaults), an ``int`` (same threshold everywhere — the
    "I know what I'm doing" override), or a partial dict merged over the
    defaults.
    """
    if value is None:
        return dict(DEFAULT_MAX_PAIR_EDGES)
    if isinstance(value, int):
        return {k: value for k in DEFAULT_MAX_PAIR_EDGES}
    return {**DEFAULT_MAX_PAIR_EDGES, **value}


def compute(
    graph: dict,
    *,
    seed: int = 42,
    cap: int = 50,
    max_pair_edges: int | dict | None = None,
) -> dict:
    """Rank the graph. Returns hubs / brokers / bridges / orphans.

    Args:
        graph: ``{nodes, edges}`` from ``unified_graph.build``.
        seed: Passed through to Leiden so the community assignment that
            drives the bridge filter is reproducible.
        cap: Maximum entries retained per ranking. The cache stores ``cap``
            rows and ``render`` slices to ``top_n``, so changing how many
            rows you display never invalidates the cache.
        max_pair_edges: An edge only counts as a bridge if its two
            communities are joined by at most this many edges in total.
            ``None`` uses the per-bucket ``DEFAULT_MAX_PAIR_EDGES``; pass an
            int to force one threshold everywhere, or a partial dict to
            override a single bucket. Raising it admits weaker bridges and
            eventually lets universal sinks back in.

    Returns:
        ``{"totals": {...}, "hubs": {"article": [...], "code": [...]},
           "brokers": {...}, "bridges": [...], "orphans": [...]}``
    """
    _detect = _communities().detect
    thresholds = resolve_max_pair_edges(max_pair_edges)

    nodes = graph["nodes"]
    if not nodes:
        return {
            "totals": {"nodes": 0, "edges": 0, "components": 0},
            "max_pair_edges": thresholds,
            "hubs": {"article": [], "code": []},
            "brokers": {"article": [], "code": []},
            "bridges": {"article": [], "code": [], "mixed": []},
            "orphans": [],
        }

    g, node_ids = _build_igraph(graph)
    degrees = g.degree()
    node_bt = g.betweenness()

    def _row(idx: int) -> dict:
        nid = node_ids[idx]
        meta = nodes[nid]
        return {
            "id": nid,
            "label": meta.get("label", nid),
            "kind": meta.get("kind", "?"),
            "degree": degrees[idx],
            "betweenness": round(node_bt[idx], 2),
        }

    # ── Hubs and brokers, bucketed by scope after ranking globally ──────
    hubs: dict[str, list] = {"article": [], "code": []}
    brokers: dict[str, list] = {"article": [], "code": []}

    by_degree = sorted(
        range(len(node_ids)),
        key=lambda i: (-degrees[i], node_ids[i]),
    )
    by_betweenness = sorted(
        range(len(node_ids)),
        key=lambda i: (-node_bt[i], node_ids[i]),
    )

    for idx in by_degree:
        bucket = hubs[_scope_of(nodes[node_ids[idx]].get("kind", ""))]
        if len(bucket) < cap and degrees[idx] > 0:
            bucket.append(_row(idx))
        if all(len(b) >= cap for b in hubs.values()):
            break

    for idx in by_betweenness:
        if node_bt[idx] <= 0:
            break  # sorted descending — everything after this is also zero
        bucket = brokers[_scope_of(nodes[node_ids[idx]].get("kind", ""))]
        if len(bucket) < cap:
            bucket.append(_row(idx))
        if all(len(b) >= cap for b in brokers.values()):
            break

    # ── Bridges: the surprising connections ────────────────────────────
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    community_of: dict[int, int] = {}
    community_size: dict[int, int] = {}
    for community in _detect(graph, seed=seed, min_size=2):
        community_size[community["community_id"]] = community["size"]
        for member in community["members"]:
            member_idx = idx_of.get(member)
            if member_idx is not None:
                community_of[member_idx] = community["community_id"]

    neighbours = [set(g.neighbors(i)) for i in range(len(node_ids))]
    edge_bt = g.edge_betweenness()

    # How many edges join each ordered-by-id pair of communities. This is the
    # filter that makes "surprising" mean anything. Without it the ranking is
    # swamped by universal sinks: EntityManagerInterface::flush has degree 277,
    # its callers never call each other, and Leiden puts each in its own
    # cluster — so every single call to flush scores as a high-betweenness,
    # zero-shared-neighbour, cross-community edge. Those are the *least*
    # surprising links in the graph. A connection is only surprising if the two
    # communities are not already wired together some other way.
    pair_edges: dict[tuple[int, int], int] = {}
    for edge in g.es:
        cu, cv = community_of.get(edge.tuple[0]), community_of.get(edge.tuple[1])
        if cu is None or cv is None or cu == cv:
            continue
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges[key] = pair_edges.get(key, 0) + 1

    collected: list[dict] = []
    for edge, score in zip(g.es, edge_bt):
        u, v = edge.tuple
        if degrees[u] < _MIN_BRIDGE_DEGREE or degrees[v] < _MIN_BRIDGE_DEGREE:
            continue
        cu, cv = community_of.get(u), community_of.get(v)
        # A node with no community (its cluster fell below min_size) gives us
        # no evidence that the edge crosses anything — skip rather than guess.
        if cu is None or cv is None or cu == cv:
            continue
        if neighbours[u] & neighbours[v]:
            continue  # a triangle-closing edge is corroborated, not surprising
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_count = pair_edges[key]
        su = _scope_of(nodes[node_ids[u]].get("kind", ""))
        sv = _scope_of(nodes[node_ids[v]].get("kind", ""))
        bucket_name = su if su == sv else "mixed"
        if pair_count > thresholds[bucket_name]:
            continue
        collected.append({
            "from": node_ids[u],
            "to": node_ids[v],
            "from_label": nodes[node_ids[u]].get("label", ""),
            "to_label": nodes[node_ids[v]].get("label", ""),
            "from_community": cu,
            "to_community": cv,
            "joins": f"{community_size.get(cu, 0)}<->{community_size.get(cv, 0)}",
            "pair_edges": pair_count,
            "edge_betweenness": round(score, 2),
            # Configuration-model normalization. Under a null model that
            # preserves degrees, the expected number of edges between u and v
            # is proportional to deg(u)·deg(v) — so raw betweenness is
            # systematically inflated for any edge touching a high-degree
            # node, and a universal sink's every edge outranks every genuine
            # bridge. Dividing by that expectation asks the right question:
            # how much traffic does this edge carry *relative to how
            # unremarkable its existence is*. Scale-free, no tuned constant.
            "surprise": round(score / (degrees[u] * degrees[v]), 3),
            "_bucket": bucket_name,
        })

    collected.sort(key=lambda b: (-b["surprise"], b["from"], b["to"]))
    # Cap *per bucket*, not globally. The code graph is an order of magnitude
    # denser than the article graph, so one global cap starves article↔article
    # bridges out of the report entirely — and those are the ones a knowledge
    # base is actually asked about. ``mixed`` is article↔code: "this article is
    # the only thing linking that cluster of code to the rest of the graph".
    bridges: dict[str, list] = {"article": [], "code": [], "mixed": []}
    for row in collected:
        bucket = bridges[row.pop("_bucket")]
        if len(bucket) < cap:
            bucket.append(row)

    # ── Orphans: articles nothing (much) links to ──────────────────────
    orphans = [
        {
            "id": node_ids[i],
            "label": nodes[node_ids[i]].get("label", ""),
            "type": nodes[node_ids[i]].get("type", "?"),
            "degree": degrees[i],
        }
        for i in range(len(node_ids))
        if _scope_of(nodes[node_ids[i]].get("kind", "")) == "article" and degrees[i] <= 1
    ]
    orphans.sort(key=lambda o: (o["degree"], o["id"]))
    orphans = orphans[:cap]

    return {
        "totals": {
            "nodes": len(node_ids),
            "edges": g.ecount(),
            "components": len(g.connected_components()),
            "articles": sum(
                1 for n in nodes.values() if _scope_of(n.get("kind", "")) == "article"
            ),
        },
        "max_pair_edges": thresholds,
        "hubs": hubs,
        "brokers": brokers,
        "bridges": bridges,
        "orphans": orphans,
    }


def signature(graph: dict, *, seed: int) -> str:
    """Cache key: graph shape + rendered node metadata + seed + logic version.

    ``communities.signature`` covers topology only — correct for Leiden,
    which reads nothing else. It is *not* sufficient here: this report
    buckets by ``kind``, displays ``label``, and shows an article's ``type``,
    none of which change the graph's shape. Renaming an article's title or
    correcting its type would otherwise reuse a cache that still shows the
    old value, indefinitely, because the topology hash never moved.

    ``CACHE_VERSION`` is in the key so a change to the ranking logic can
    never be served from a cache the old logic wrote.
    """
    import hashlib

    h = hashlib.sha1()
    for nid in sorted(graph["nodes"]):
        meta = graph["nodes"][nid]
        h.update(
            f"{nid}|{meta.get('kind', '')}|{meta.get('label', '')}"
            f"|{meta.get('type', '')}\n".encode()
        )
    topology = _communities().signature(graph, seed=seed)
    return f"{CACHE_VERSION}:{topology}:{h.hexdigest()}"


def _is_valid_report(report: object) -> bool:
    """True if ``report`` has the shape ``render`` and callers index into."""
    if not isinstance(report, dict):
        return False
    for key in ("totals", "hubs", "brokers", "bridges"):
        if not isinstance(report.get(key), dict):
            return False
    return isinstance(report.get("orphans"), list)


def load_or_compute(
    graph: dict,
    *,
    cache_path: Path,
    seed: int = 42,
    cap: int = 50,
    max_pair_edges: int | dict | None = None,
) -> dict:
    """Return the salience report, reusing ``cache_path`` when it still applies.

    Same contract as ``communities.load_or_compute``: a corrupt or stale
    cache is silently recomputed rather than raised, because a report is
    never worth failing a caller over. Every argument that changes the
    *content* of the report is part of the cache key — thresholds are
    compared in resolved form so ``None`` and an equivalent explicit dict
    are correctly treated as the same request.
    """
    cache_path = Path(cache_path)
    sig = signature(graph, seed=seed)
    thresholds = resolve_max_pair_edges(max_pair_edges)

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                data.get("signature") == sig
                and data.get("cap") == cap
                and data.get("max_pair_edges") == thresholds
            ):
                report = data["report"]
                # Well-formed JSON is not a well-formed report. A file whose
                # "report" is null or a list still parses and still matches
                # the signature, and would be handed to render(), which
                # indexes it — taking down get_salience with an
                # AttributeError instead of quietly recomputing.
                if _is_valid_report(report):
                    return report
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            pass  # corrupt cache — recompute

    report = compute(graph, seed=seed, cap=cap, max_pair_edges=thresholds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "signature": sig,
                "cap": cap,
                "max_pair_edges": thresholds,
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    """Render ``rows`` as a markdown table. ``columns`` is [(header, key)]."""
    if not rows:
        return ["_None._", ""]
    out = [
        "| " + " | ".join(h for h, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            cells.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def render(report: dict, *, top_n: int = 10, scope: str = "all") -> str:
    """Render the report as markdown. ``scope`` ∈ {all, article, code}."""
    totals = report.get("totals", {})
    lines = [
        "# Salience — what carries this knowledge graph",
        "",
        f"> {totals.get('nodes', 0)} nodes · {totals.get('edges', 0)} edges · "
        f"{totals.get('components', 0)} connected components · "
        f"{totals.get('articles', 0)} articles. "
        "Pure graph structure, zero LLM cost. Regenerated on every run.",
        "",
    ]

    scopes = ["article", "code"] if scope == "all" else [scope]
    titles = {"article": "Articles", "code": "Code"}

    lines += ["## Hubs — most connected", "",
              "_Degree centrality. What the knowledge base mostly talks about._", ""]
    for sc in scopes:
        rows = report.get("hubs", {}).get(sc, [])[:top_n]
        lines.append(f"### {titles.get(sc, sc)}")
        lines.append("")
        lines += _table(rows, [("Node", "id"), ("Label", "label"),
                               ("Degree", "degree"), ("Betweenness", "betweenness")])

    lines += ["## Brokers — most load-bearing", "",
              "_Betweenness centrality. Removing one of these disconnects parts of "
              "the graph from each other. A broker need not be a hub: few links, "
              "but they are the only links._", ""]
    for sc in scopes:
        rows = report.get("brokers", {}).get(sc, [])[:top_n]
        lines.append(f"### {titles.get(sc, sc)}")
        lines.append("")
        lines += _table(rows, [("Node", "id"), ("Label", "label"),
                               ("Betweenness", "betweenness"), ("Degree", "degree")])

    thresholds = report.get("max_pair_edges", DEFAULT_MAX_PAIR_EDGES)
    lines += ["## Bridges — surprising connections", "",
              "_An edge joining two communities that have almost no other link, "
              "whose endpoints share no common neighbour. Cut it and the two "
              "clusters are nearly severed — which is what makes the link worth "
              "reading. Ranked by **surprise** = edge betweenness ÷ "
              "deg(u)·deg(v): traffic carried relative to how expected the edge "
              "is, so a utility that everything calls does not crowd out the "
              "real bridges. `Joins` gives the two community sizes; the "
              "per-section threshold caps how many edges may join them, and "
              "differs by section because the code graph is far denser than the "
              "article graph._", ""]
    bridge_titles = {"article": "Article ↔ Article", "code": "Code ↔ Code",
                     "mixed": "Article ↔ Code"}
    # "mixed" is shown under either single scope: an article↔code bridge is
    # equally an article finding and a code finding.
    bridge_buckets = ["article", "code", "mixed"] if scope == "all" else [scope, "mixed"]
    for bucket in bridge_buckets:
        rows = report.get("bridges", {}).get(bucket, [])[:top_n]
        lines.append(f"### {bridge_titles.get(bucket, bucket)} "
                     f"(≤{thresholds.get(bucket, '?')} edges between communities)")
        lines.append("")
        lines += _table(rows, [("From", "from"), ("To", "to"),
                               ("Joins", "joins"), ("Surprise", "surprise"),
                               ("Edge betweenness", "edge_betweenness")])

    orphans = report.get("orphans", [])[:top_n]
    lines += [f"## Orphans — written but unreachable ({len(orphans)} shown)", "",
              "_Articles with degree ≤ 1. Neighbour- and community-based retrieval "
              "will never surface these; only direct search will. Add a "
              "`[[wikilink]]` or a `[src:]` anchor to connect them._", ""]
    lines += _table(orphans, [("Article", "id"), ("Label", "label"),
                              ("Type", "type"), ("Degree", "degree")])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank the unified knowledge graph: hubs, brokers, bridges, orphans."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the report instead of writing SALIENCE.md.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Rows per table (default 10).")
    parser.add_argument("--scope", choices=("all", "article", "code"), default="all",
                        help="Which node kinds to present (default all).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute even if the cache is valid.")
    parser.add_argument(
        "--max-pair-edges", type=int, default=None,
        help="Force one bridge threshold for every section, overriding the "
             f"per-section defaults {DEFAULT_MAX_PAIR_EDGES}. Higher admits "
             "weaker bridges; on a dense code graph it readmits universal "
             "sinks like EntityManagerInterface::flush.",
    )
    args = parser.parse_args()

    from config import KNOWLEDGE_DIR, PROJECT_ROOT
    from utils import make_stdout_unicode_safe
    import unified_graph

    make_stdout_unicode_safe()

    graph = unified_graph.build_for_project(PROJECT_ROOT, KNOWLEDGE_DIR)
    cache_path = KNOWLEDGE_DIR / "salience.json"
    if args.no_cache:
        report = compute(graph, max_pair_edges=args.max_pair_edges)
    else:
        report = load_or_compute(
            graph, cache_path=cache_path, max_pair_edges=args.max_pair_edges
        )

    content = render(report, top_n=args.top_n, scope=args.scope)

    if args.dry_run:
        print(content)
        return

    out = KNOWLEDGE_DIR / "SALIENCE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    bridges = report.get("bridges", {})
    print(f"Wrote {out}")
    print(f"  Bridges: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(bridges.items()))
          + f" | Orphan articles: {len(report.get('orphans', []))}")


if __name__ == "__main__":
    main()
