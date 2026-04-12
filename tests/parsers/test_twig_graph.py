"""Tests for Twig template graph parser."""
from scripts.parsers import PROJECT_ROOT, twig_graph


def test_parse_returns_expected_shape():
    result = twig_graph.parse(PROJECT_ROOT)
    assert "templates" in result
    assert "inheritance_chains" in result
    assert "stats" in result
    assert "total_templates" in result["stats"]


def test_finds_base_template_with_children():
    """base.html.twig should have many children (everything extends it)."""
    result = twig_graph.parse(PROJECT_ROOT)
    chains = result["inheritance_chains"]
    # Any template whose children list is non-trivial
    base_candidates = [k for k, v in chains.items() if "base" in k and len(v) >= 3]
    assert base_candidates, f"Expected at least one base template with 3+ children, found: {list(chains.keys())[:5]}"


def test_template_has_required_fields():
    result = twig_graph.parse(PROJECT_ROOT)
    assert result["templates"], "Expected at least one template"
    path, info = next(iter(result["templates"].items()))
    assert "extends" in info
    assert "includes" in info
    assert "included_by" in info
    assert "stimulus_controllers" in info
    assert isinstance(info["includes"], list)
    assert isinstance(info["stimulus_controllers"], list)


def test_extends_parent_is_resolved_to_real_file():
    """If template A extends 'base.html.twig', the extends field should point
    at the resolved templates/base.html.twig path."""
    result = twig_graph.parse(PROJECT_ROOT)
    for info in result["templates"].values():
        if info["extends"]:
            assert info["extends"].startswith("templates/"), f"extends should be rel path, got: {info['extends']}"


def test_summary_returns_short_string():
    s = twig_graph.summary(PROJECT_ROOT)
    assert isinstance(s, str)
    assert len(s) < 500
