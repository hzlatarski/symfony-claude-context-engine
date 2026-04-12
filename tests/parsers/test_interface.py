"""Smoke test: scripts.parsers package exists and exposes expected modules."""
import importlib


def test_parsers_package_importable():
    mod = importlib.import_module("scripts.parsers")
    assert mod is not None


def test_project_root_constant():
    """The package exports PROJECT_ROOT pointing at the Symfony repo."""
    from scripts.parsers import PROJECT_ROOT
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "templates").is_dir()
