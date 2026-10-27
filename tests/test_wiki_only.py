"""The installed call wiki works without the full developer dependency set."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

import utils


ROOT = Path(__file__).resolve().parents[1]


def test_absolute_include_glob(tmp_path, monkeypatch):
    sources = tmp_path / "José Müller" / "sources"
    sources.mkdir(parents=True)
    call = sources / "call.md"
    call.write_text("A recorded call.", encoding="utf-8")
    (sources / "ignore.md").write_text("Not a call.", encoding="utf-8")
    config = tmp_path / "sources.yaml"
    config.write_text(yaml.safe_dump({"version": 1, "sources": [{
        "id": "calls", "type": "markdown", "include": [sources.as_posix() + "/*.md"],
        "exclude": ["ignore.md"],
    }]}), encoding="utf-8")
    monkeypatch.setattr(utils, "SOURCES_FILE", config)
    group, = utils.load_sources_config()
    assert utils.resolve_source_files(group, root=tmp_path / "unrelated-code") == [call.resolve()]


@pytest.mark.slow
def test_wiki_modules_import_with_wiki_requirements(tmp_path):
    uv = shutil.which("uv")
    assert uv, "uv is required to verify the wiki-only environment"
    venv = tmp_path / "wiki-env"
    subprocess.run([uv, "venv", "--python", sys.executable, str(venv)],
                   check=True, capture_output=True, timeout=60)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    install = subprocess.run([uv, "pip", "install", "--python", str(python),
                              "-r", str(ROOT / "requirements-wiki.txt")],
                             capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert install.returncode == 0, install.stderr
    env = dict(os.environ, PYTHONPATH=str(ROOT / "scripts"),
               MEMORY_COMPILER_HOME=str(tmp_path / "user wiki"),
               MC_EMBEDDER_CACHE=str(tmp_path / "cache"))
    imported = subprocess.run([str(python), "-c",
        "import ingest, reindex, hybrid_search, vector_store, compile_truth, pypdf"],
        env=env, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert imported.returncode == 0, imported.stderr
