"""Tests for entity nodes in the unified graph (Slice 2).

``build()`` lifts recurring named things out of article PROSE and joins the
articles that mention them via ``mentions`` edges. Prose ``path``/``service``
mentions fold into existing ``file:``/``class:`` code nodes; every other type
mints a standalone ``entity:`` node. No article file is ever written. Tests
use synthetic tmp_path knowledge dirs and minimal call-graph dicts.
"""
from scripts import entities, unified_graph

_EMPTY_CG = {"symbols": {}, "edges": [], "classes": {}}


def _mentions(result):
    return {(e["from"], e["to"]) for e in result["edges"] if e["kind"] == "mentions"}


def test_shared_host_becomes_one_node_with_two_mentions(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nProd host 65.21.4.203 runs it.", encoding="utf-8"
    )
    (concepts / "b.md").write_text(
        "---\ntitle: B\n---\nSSH into 65.21.4.203 for logs.", encoding="utf-8"
    )
    result = unified_graph.build(call_graph=_EMPTY_CG, knowledge_root=tmp_path)

    node = result["nodes"]["entity:host/65.21.4.203"]
    assert node["kind"] == "entity"
    assert node["entity_type"] == "host"
    assert _mentions(result) == {
        ("article:concepts/a", "entity:host/65.21.4.203"),
        ("article:concepts/b", "entity:host/65.21.4.203"),
    }


def test_role_and_envvar_mint_entity_nodes(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nRequires ROLE_ADMIN and the ANTHROPIC_API_KEY var.",
        encoding="utf-8",
    )
    result = unified_graph.build(call_graph=_EMPTY_CG, knowledge_root=tmp_path)

    assert result["nodes"]["entity:role/ROLE_ADMIN"]["entity_type"] == "role"
    assert result["nodes"]["entity:envvar/ANTHROPIC_API_KEY"]["entity_type"] == "envvar"
    assert ("article:concepts/a", "entity:role/ROLE_ADMIN") in _mentions(result)
    assert ("article:concepts/a", "entity:envvar/ANTHROPIC_API_KEY") in _mentions(result)


def test_service_folds_into_existing_class_node(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nThe ChatService handles it.", encoding="utf-8"
    )
    call_graph = {
        "symbols": {},
        "edges": [],
        "classes": {"App\\Service\\ChatService": {"file": "src/Service/ChatService.php"}},
    }
    result = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)

    # Mention points at the existing class node, NOT a parallel entity node.
    assert ("article:concepts/a", "class:App\\Service\\ChatService") in _mentions(result)
    assert "entity:service/ChatService" not in result["nodes"]


def test_path_folds_into_existing_file_node(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nEdit src/Service/ChatService.php now.", encoding="utf-8"
    )
    call_graph = {
        "symbols": {},
        "edges": [],
        "classes": {"App\\Service\\ChatService": {"file": "src/Service/ChatService.php"}},
    }
    result = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)

    assert ("article:concepts/a", "file:src/Service/ChatService.php") in _mentions(result)
    assert "entity:path/src/Service/ChatService.php" not in result["nodes"]


def test_unmatched_service_mints_entity_node(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nThe GhostlyMissingService is only named in prose.",
        encoding="utf-8",
    )
    result = unified_graph.build(call_graph=_EMPTY_CG, knowledge_root=tmp_path)

    node = result["nodes"]["entity:service/GhostlyMissingService"]
    assert node["kind"] == "entity"
    assert node["entity_type"] == "service"


def test_prose_path_already_cited_is_not_duplicated(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    # The article both cites the file via [src:] AND names it in prose.
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nFact [src:src/Foo.php]. Also edit src/Foo.php by hand.",
        encoding="utf-8",
    )
    result = unified_graph.build(call_graph=_EMPTY_CG, knowledge_root=tmp_path)

    file_edges = [
        (e["from"], e["to"], e["kind"]) for e in result["edges"]
        if e["to"] == "file:src/Foo.php"
    ]
    # Exactly one edge to the file — the cites edge; no redundant mention.
    assert file_edges == [("article:concepts/a", "file:src/Foo.php", "cites")]


def test_no_entities_when_prose_has_none(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nJust ordinary words, nothing named.", encoding="utf-8"
    )
    result = unified_graph.build(call_graph=_EMPTY_CG, knowledge_root=tmp_path)
    assert _mentions(result) == set()
    assert not [n for n in result["nodes"] if n.startswith("entity:")]


# ── find_entity_matches (Slice 3 search surface) ────────────────────────

def _graph_two_hosts(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nProd host 65.21.4.203 runs it.", encoding="utf-8"
    )
    (concepts / "b.md").write_text(
        "---\ntitle: B\n---\nSSH into 65.21.4.203; also needs ROLE_ADMIN.",
        encoding="utf-8",
    )
    return unified_graph.build(call_graph=_EMPTY_CG, knowledge_root=tmp_path)


def test_find_entity_matches_lists_mentioning_notes(tmp_path):
    graph = _graph_two_hosts(tmp_path)
    hits = entities.find_entity_matches(graph, "65.21")
    assert len(hits) == 1
    assert hits[0]["target"] == "entity:host/65.21.4.203"
    assert hits[0]["mentioners"] == ["article:concepts/a", "article:concepts/b"]


def test_find_entity_matches_type_filter(tmp_path):
    graph = _graph_two_hosts(tmp_path)
    assert [h["target"] for h in entities.find_entity_matches(graph, "ROLE", "role")] == [
        "entity:role/ROLE_ADMIN"
    ]
    # A host is not a role → filtered out.
    assert entities.find_entity_matches(graph, "65.21", "role") == []


def test_find_entity_matches_empty_query_returns_nothing(tmp_path):
    graph = _graph_two_hosts(tmp_path)
    assert entities.find_entity_matches(graph, "   ") == []


def test_find_entity_matches_finds_folded_code_node(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text(
        "---\ntitle: A\n---\nThe ChatService handles it.", encoding="utf-8"
    )
    call_graph = {
        "symbols": {},
        "edges": [],
        "classes": {"App\\Service\\ChatService": {"file": "src/Service/ChatService.php"}},
    }
    graph = unified_graph.build(call_graph=call_graph, knowledge_root=tmp_path)
    # Untyped search finds the folded class node.
    hits = entities.find_entity_matches(graph, "chatservice")
    assert hits[0]["target"] == "class:App\\Service\\ChatService"
    assert hits[0]["kind"] == "class"
    # A type filter excludes folded code nodes (they carry no entity_type).
    assert entities.find_entity_matches(graph, "chatservice", "service") == []
