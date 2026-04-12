"""Smoke test to confirm pytest runs and PROJECT_ROOT resolves."""
from pathlib import Path


def test_project_root_is_symfony_repo():
    """PROJECT_ROOT should point to the outer Symfony project with src/ and templates/."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    assert (root / "src").is_dir(), f"Expected {root}/src to exist"
    assert (root / "templates").is_dir(), f"Expected {root}/templates to exist"
    assert (root / "assets" / "controllers").is_dir(), f"Expected {root}/assets/controllers to exist"
