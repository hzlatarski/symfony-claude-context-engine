"""Tests for Symfony route map parser."""
from scripts.parsers import PROJECT_ROOT, route_map


def test_parse_returns_expected_shape():
    result = route_map.parse(PROJECT_ROOT)
    assert "routes" in result
    assert "stats" in result
    assert "total_routes" in result["stats"]
    assert "by_prefix" in result["stats"]


def test_finds_api_routes():
    result = route_map.parse(PROJECT_ROOT)
    api_routes = [path for path in result["routes"].keys() if path.startswith("/api")]
    assert len(api_routes) >= 5, f"Expected >=5 API routes, got {len(api_routes)}"


def test_route_has_required_fields():
    result = route_map.parse(PROJECT_ROOT)
    assert result["routes"], "Expected at least one route"
    path, route = next(iter(result["routes"].items()))
    assert "methods" in route
    assert "controller" in route
    assert "action" in route
    assert "file" in route
    assert "services" in route
    assert isinstance(route["methods"], list)
    assert isinstance(route["services"], list)


def test_class_prefix_is_combined_with_method_path():
    """If a class has #[Route('/api/foo')] and a method has #[Route('/bar')],
    the combined path should be /api/foo/bar."""
    result = route_map.parse(PROJECT_ROOT)
    # ConsentApiController has class-level #[Route('/api/consent')]
    consent_routes = [p for p in result["routes"] if p.startswith("/api/consent")]
    assert len(consent_routes) >= 1, f"Expected /api/consent routes, got {consent_routes}"


def test_summary_returns_short_string():
    s = route_map.summary(PROJECT_ROOT)
    assert isinstance(s, str)
    assert len(s) < 500
