"""Tests for the check_memory_types lint check."""
from __future__ import annotations


class TestCheckMemoryTypes:
    def test_valid_type_passes(self, tmp_path, monkeypatch):
        import config
        import lint
        import utils

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        monkeypatch.setattr(config, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(config, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(config, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(lint, "KNOWLEDGE_DIR", tmp_path)
        # list_wiki_articles reads the frozen top-level bindings in utils
        monkeypatch.setattr(utils, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(utils, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(utils, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)

        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "good.md").write_text(
            "---\ntitle: Good\ntype: fact\n---\n\n## Truth\n\nvalid\n"
        )

        issues = lint.check_memory_types()
        assert issues == []

    def test_invalid_type_flagged_as_error(self, tmp_path, monkeypatch):
        import config
        import lint
        import utils

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        monkeypatch.setattr(config, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(config, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(config, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(lint, "KNOWLEDGE_DIR", tmp_path)
        # list_wiki_articles reads the frozen top-level bindings in utils
        monkeypatch.setattr(utils, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(utils, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(utils, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)

        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "bad.md").write_text(
            "---\ntitle: Bad\ntype: banana\n---\n\n## Truth\n\ncontent\n"
        )

        issues = lint.check_memory_types()
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["check"] == "invalid_memory_type"
        assert "banana" in issues[0]["detail"]

    def test_missing_type_flagged_as_suggestion(self, tmp_path, monkeypatch):
        import config
        import lint
        import utils

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        monkeypatch.setattr(config, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(config, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(config, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(lint, "KNOWLEDGE_DIR", tmp_path)
        # list_wiki_articles reads the frozen top-level bindings in utils
        monkeypatch.setattr(utils, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(utils, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(utils, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)

        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "untyped.md").write_text(
            "---\ntitle: Untyped\n---\n\n## Truth\n\ncontent\n"
        )

        issues = lint.check_memory_types()
        assert len(issues) == 1
        assert issues[0]["severity"] == "suggestion"
        assert issues[0]["check"] == "missing_memory_type"

    def test_all_six_canonical_types_accepted(self, tmp_path, monkeypatch):
        import config
        import lint
        import utils

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        monkeypatch.setattr(config, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(config, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(config, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(lint, "KNOWLEDGE_DIR", tmp_path)
        # list_wiki_articles reads the frozen top-level bindings in utils
        monkeypatch.setattr(utils, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(utils, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(utils, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)

        (tmp_path / "concepts").mkdir()
        for i, t in enumerate(["fact", "event", "discovery", "preference", "advice", "decision"]):
            (tmp_path / "concepts" / f"article-{i}.md").write_text(
                f"---\ntitle: Article {i}\ntype: {t}\n---\n\n## Truth\n\ncontent\n"
            )

        issues = lint.check_memory_types()
        assert issues == []

    def test_mixed_articles_report_only_problems(self, tmp_path, monkeypatch):
        import config
        import lint
        import utils

        monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path)
        monkeypatch.setattr(config, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(config, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(config, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(lint, "KNOWLEDGE_DIR", tmp_path)
        # list_wiki_articles reads the frozen top-level bindings in utils
        monkeypatch.setattr(utils, "CONCEPTS_DIR", tmp_path / "concepts")
        monkeypatch.setattr(utils, "CONNECTIONS_DIR", tmp_path / "connections")
        monkeypatch.setattr(utils, "QA_DIR", tmp_path / "qa")
        monkeypatch.setattr(utils, "KNOWLEDGE_DIR", tmp_path)

        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "good.md").write_text(
            "---\ntitle: Good\ntype: fact\n---\n\ncontent\n"
        )
        (tmp_path / "concepts" / "bad.md").write_text(
            "---\ntitle: Bad\ntype: wrong\n---\n\ncontent\n"
        )
        (tmp_path / "concepts" / "untyped.md").write_text(
            "---\ntitle: Untyped\n---\n\ncontent\n"
        )

        issues = lint.check_memory_types()
        assert len(issues) == 2
        severities = sorted(i["severity"] for i in issues)
        assert severities == ["error", "suggestion"]
