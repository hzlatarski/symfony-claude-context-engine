"""Tests for the PDF source handler.

The extractor libraries are optional, so the page-extraction backend is
monkeypatched throughout: these tests cover the handler's own contract —
selection, budget, truncation, empty-document guard, provenance — without
requiring pymupdf4llm or pypdf to be installed.
"""
import pytest

from scripts.source_handlers import pdf as pdf_handler
from scripts import source_handlers


@pytest.fixture
def fake_pdf(tmp_path):
    """A file that exists (the handler stats it) but is never really parsed."""
    path = tmp_path / "handbook.pdf"
    path.write_bytes(b"%PDF-1.4 not a real pdf")
    return path


def _use_pages(monkeypatch, pages, name="pymupdf4llm"):
    monkeypatch.setattr(pdf_handler, "_select_extractor", lambda: (name, lambda _p: pages))


# ── registration ──────────────────────────────────────────────────────


def test_pdf_type_is_registered():
    assert "pdf" in source_handlers.available_types()
    assert source_handlers.get_handler("pdf") is pdf_handler.extract


def test_registry_is_populated_under_both_import_regimes():
    """Regression: absolute self-imports in __init__ built two empty registries.

    ``ingest.py`` imports ``source_handlers``; tests and the MCP server import
    ``scripts.source_handlers``. When __init__ registered submodules via an
    absolute import, only that one spelling's ``_HANDLERS`` got filled and the
    other raised "No handler registered for source type 'markdown'" — the
    handler existed, the registry it landed in was simply the wrong object.
    """
    import importlib

    bare = importlib.import_module("source_handlers")
    scoped = importlib.import_module("scripts.source_handlers")
    for registry in (bare, scoped):
        assert "markdown" in registry.available_types()
        assert "pdf" in registry.available_types()


# ── extraction ────────────────────────────────────────────────────────


def test_extract_joins_pages_with_page_markers(monkeypatch, fake_pdf):
    _use_pages(monkeypatch, ["First page text.", "Second page text."])
    doc = pdf_handler.extract(fake_pdf)
    assert "[page 1]" in doc.content
    assert "[page 2]" in doc.content
    assert "First page text." in doc.content
    assert "Second page text." in doc.content


def test_extract_records_provenance_in_frontmatter(monkeypatch, fake_pdf):
    _use_pages(monkeypatch, ["a", "b", "c"])
    doc = pdf_handler.extract(fake_pdf)
    assert doc.frontmatter["source_format"] == "pdf"
    assert doc.frontmatter["extractor"] == "pymupdf4llm"
    assert doc.frontmatter["pages"] == 3
    assert doc.frontmatter["pages_extracted"] == 3
    assert doc.frontmatter["truncated"] is False
    assert doc.path == fake_pdf
    assert doc.mtime > 0


def test_blank_pages_are_skipped_but_still_counted(monkeypatch, fake_pdf):
    _use_pages(monkeypatch, ["real text", "   ", "", "more text"])
    doc = pdf_handler.extract(fake_pdf)
    assert doc.frontmatter["pages"] == 4
    assert doc.frontmatter["pages_extracted"] == 2
    assert doc.frontmatter["truncated"] is False
    assert "[page 4]" in doc.content  # numbering follows the real page number


# ── empty-document guard ──────────────────────────────────────────────


def test_scanned_pdf_with_no_text_raises_rather_than_ingesting_nothing(monkeypatch, fake_pdf):
    _use_pages(monkeypatch, ["", "   ", "\n"])
    with pytest.raises(ValueError, match="No extractable text"):
        pdf_handler.extract(fake_pdf)


def test_pdf_with_zero_pages_raises(monkeypatch, fake_pdf):
    _use_pages(monkeypatch, [])
    with pytest.raises(ValueError, match="No extractable text"):
        pdf_handler.extract(fake_pdf)


def test_parser_failure_is_reported_with_the_backend_name(monkeypatch, fake_pdf):
    def _boom(_path):
        raise RuntimeError("encrypted")

    monkeypatch.setattr(pdf_handler, "_select_extractor", lambda: ("pypdf", _boom))
    with pytest.raises(ValueError, match="Could not read PDF handbook.pdf via pypdf"):
        pdf_handler.extract(fake_pdf)


# ── character budget ──────────────────────────────────────────────────


def test_budget_truncates_on_a_page_boundary(monkeypatch, fake_pdf):
    monkeypatch.setenv("MEMORY_COMPILER_PDF_MAX_CHARS", "60")
    _use_pages(monkeypatch, ["A" * 40, "B" * 40, "C" * 40])
    doc = pdf_handler.extract(fake_pdf)
    assert doc.frontmatter["pages_extracted"] == 1
    assert doc.frontmatter["truncated"] is True
    assert "B" * 40 not in doc.content  # no half-fed page
    assert "[TRUNCATED]" in doc.content


def test_truncation_notice_names_the_env_var(monkeypatch, fake_pdf):
    monkeypatch.setenv("MEMORY_COMPILER_PDF_MAX_CHARS", "60")
    _use_pages(monkeypatch, ["A" * 40, "B" * 40])
    doc = pdf_handler.extract(fake_pdf)
    assert "MEMORY_COMPILER_PDF_MAX_CHARS" in doc.content
    assert "first 1 of 2 pages" in doc.content


def test_single_oversized_page_is_hard_cut_rather_than_dropped(monkeypatch, fake_pdf):
    """Returning nothing here would trip the empty guard and lose the document."""
    monkeypatch.setenv("MEMORY_COMPILER_PDF_MAX_CHARS", "50")
    _use_pages(monkeypatch, ["X" * 500])
    doc = pdf_handler.extract(fake_pdf)
    assert doc.frontmatter["truncated"] is True
    assert doc.frontmatter["pages_extracted"] == 1
    assert "X" in doc.content


def test_content_stays_within_budget_plus_notice(monkeypatch, fake_pdf):
    monkeypatch.setenv("MEMORY_COMPILER_PDF_MAX_CHARS", "100")
    _use_pages(monkeypatch, ["A" * 80, "B" * 80, "C" * 80])
    doc = pdf_handler.extract(fake_pdf)
    body = doc.content.split("[TRUNCATED]")[0]
    assert len(body) <= 100


def test_default_budget_applies_when_env_var_is_absent(monkeypatch):
    monkeypatch.delenv("MEMORY_COMPILER_PDF_MAX_CHARS", raising=False)
    assert pdf_handler._max_chars() == pdf_handler.DEFAULT_MAX_CHARS


@pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "0", "-5"])
def test_invalid_budget_falls_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv("MEMORY_COMPILER_PDF_MAX_CHARS", bad)
    assert pdf_handler._max_chars() == pdf_handler.DEFAULT_MAX_CHARS


# ── extractor selection ───────────────────────────────────────────────


def test_selection_prefers_pymupdf4llm_when_both_are_available(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    name, _fn = pdf_handler._select_extractor()
    assert name == "pymupdf4llm"


def test_selection_falls_back_to_pypdf(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "pypdf" else None,
    )
    name, _fn = pdf_handler._select_extractor()
    assert name == "pypdf"


def test_no_extractor_installed_raises_with_both_install_commands(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        pdf_handler._select_extractor()
    message = str(excinfo.value)
    assert "pymupdf4llm" in message
    assert "pypdf" in message
