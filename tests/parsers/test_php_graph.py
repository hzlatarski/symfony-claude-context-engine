"""Tests for PHP dependency graph parser.

Uses the real Symfony codebase as fixture — asserts on high-level
invariants so tests survive routine refactors.
"""
from scripts.parsers import PROJECT_ROOT, php_graph


def test_parse_returns_expected_shape():
    result = php_graph.parse(PROJECT_ROOT)
    assert "nodes" in result
    assert "edges" in result
    assert "stats" in result
    assert "total_files" in result["stats"]
    assert "by_type" in result["stats"]


def test_parse_finds_controllers():
    result = php_graph.parse(PROJECT_ROOT)
    by_type = result["stats"]["by_type"]
    # Sanity: the app has dozens of controllers
    assert by_type.get("controller", 0) >= 20, f"Expected >=20 controllers, got {by_type.get('controller', 0)}"


def test_parse_finds_entities():
    result = php_graph.parse(PROJECT_ROOT)
    by_type = result["stats"]["by_type"]
    assert by_type.get("entity", 0) >= 20, f"Expected >=20 entities, got {by_type.get('entity', 0)}"


def test_user_entity_exists_and_is_classified():
    result = php_graph.parse(PROJECT_ROOT)
    user_node = result["nodes"].get("src/Entity/User.php")
    assert user_node is not None, "src/Entity/User.php should be in nodes"
    assert user_node["type"] == "entity"
    assert user_node["class"] == "User"
    assert user_node["namespace"] == "App\\Entity"


def test_edges_link_to_real_files():
    """Every edge target must exist in nodes (no dangling edges)."""
    result = php_graph.parse(PROJECT_ROOT)
    node_paths = set(result["nodes"].keys())
    for edge in result["edges"]:
        assert edge["from"] in node_paths, f"Edge from missing node: {edge['from']}"
        # Target may be external (vendor) — only check internal targets
        if edge["to"].startswith("src/"):
            assert edge["to"] in node_paths, f"Edge to missing internal node: {edge['to']}"


def test_in_degree_matches_edges():
    """A node's in_degree should equal the count of edges pointing to it."""
    result = php_graph.parse(PROJECT_ROOT)
    incoming_counts: dict[str, int] = {}
    for edge in result["edges"]:
        incoming_counts[edge["to"]] = incoming_counts.get(edge["to"], 0) + 1
    for path, node in result["nodes"].items():
        assert node["in_degree"] == incoming_counts.get(path, 0), f"in_degree mismatch for {path}"


def test_summary_returns_short_string():
    s = php_graph.summary(PROJECT_ROOT)
    assert isinstance(s, str)
    assert len(s) < 500
    assert "files" in s.lower()
