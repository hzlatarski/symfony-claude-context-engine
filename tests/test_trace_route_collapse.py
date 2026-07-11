"""Tests for entity-accessor collapsing in trace_route (usability fix)."""
from __future__ import annotations

from scripts import mcp_server


def _leaf(symbol: str) -> dict:
    return {"symbol": symbol, "kind": "call", "confidence": 0.7, "evidence": "", "children": []}


def test_is_entity_accessor_true_for_entity_setter():
    assert mcp_server._is_entity_accessor(_leaf("App\\Entity\\User::setEmail"))
    assert mcp_server._is_entity_accessor(_leaf("App\\Entity\\User::getId"))
    assert mcp_server._is_entity_accessor(_leaf("App\\Entity\\User::isActive"))


def test_is_entity_accessor_false_for_non_accessor_method():
    assert not mcp_server._is_entity_accessor(_leaf("App\\Entity\\User::revokeVoiceConsent"))


def test_is_entity_accessor_false_for_non_entity_class():
    assert not mcp_server._is_entity_accessor(_leaf("App\\Controller\\FooController::getUser"))


def test_is_entity_accessor_false_when_node_has_children():
    node = _leaf("App\\Entity\\User::setThing")
    node["children"] = [_leaf("App\\Service\\Bar::baz")]
    assert not mcp_server._is_entity_accessor(node)


def _render(tree: dict, collapse: bool) -> str:
    lines: list[str] = []
    mcp_server._render_trace_node(tree, lines, indent=0, is_root=True, collapse_accessors=collapse)
    return "\n".join(lines)


def _sample_tree() -> dict:
    return {
        "symbol": "App\\Service\\Gdpr\\GdprAccountDeletionService::softDeleteUser",
        "children": [
            _leaf("Psr\\Log\\LoggerInterface::info"),
            _leaf("App\\Entity\\User::setDeletedAt"),
            _leaf("App\\Entity\\User::setEmail"),
            _leaf("App\\Entity\\User::setFirstName"),
            _leaf("App\\Entity\\User::revokeVoiceConsent"),
            _leaf("Doctrine\\ORM\\EntityManagerInterface::flush"),
        ],
    }


def test_collapse_hides_individual_entity_setters():
    out = _render(_sample_tree(), collapse=True)
    assert "setDeletedAt" not in out.split("collapsed")[0]  # not rendered as its own line
    assert "entity accessor call(s) collapsed" in out
    # Meaningful calls are still shown.
    assert "revokeVoiceConsent" in out
    assert "flush" in out
    assert "LoggerInterface::info" in out


def test_collapse_summary_counts_the_accessors():
    out = _render(_sample_tree(), collapse=True)
    assert "[3 entity accessor call(s) collapsed]" in out


def test_no_collapse_when_disabled_shows_every_setter():
    out = _render(_sample_tree(), collapse=False)
    assert "setDeletedAt" in out
    assert "setEmail" in out
    assert "collapsed" not in out


def test_cycle_truncated_accessor_keeps_its_marker():
    """A cycle-truncated entity accessor must NOT be collapsed — otherwise the
    _(cycle)_ marker (why the subtree ends) is silently lost."""
    node = _leaf("App\\Entity\\User::getParent")
    node["truncated"] = "cycle"
    tree = {"symbol": "App\\Service\\Foo::bar", "children": [node]}
    out = _render(tree, collapse=True)
    assert "getParent" in out
    assert "(cycle)" in out
    assert "collapsed" not in out
