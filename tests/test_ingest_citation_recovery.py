"""Duplicate imports must learn provenance automatically without weakening success checks."""
import asyncio
import subprocess

import pytest
import ingest
import utils


def run_case(tmp_path, monkeypatch, outcome, first_credit='metadata'):
    knowledge = tmp_path / 'knowledge'
    concepts = knowledge / 'concepts'
    concepts.mkdir(parents=True)
    article = concepts / 'existing.md'
    article.write_text('A supported fact [src:sources/calls/original.md]\n', encoding='utf-8')
    source = tmp_path / 'duplicate.md'
    source.write_text('A supported fact', encoding='utf-8')
    schema = tmp_path / 'AGENTS.md'
    schema.write_text('Facts require source anchors.', encoding='utf-8')
    for name, path in [('CONCEPTS_DIR', concepts), ('CONNECTIONS_DIR', knowledge/'connections'),
                       ('KNOWLEDGE_DIR', knowledge), ('AGENTS_FILE', schema),
                       ('COMPILED_TRUTH_FILE', knowledge/'compiled-truth.md')]:
        monkeypatch.setattr(ingest, name, path)
    monkeypatch.setattr(ingest, 'read_wiki_index', lambda **kw: 'existing article')
    monkeypatch.setattr(ingest.dedup, 'similar_to_text', lambda *a, **k: [])
    monkeypatch.setattr(ingest.dedup, 'format_preflight_block', lambda *a, **k: '')
    saved = {'ingested_sources': {}}
    monkeypatch.setattr(ingest, 'update_state', lambda fn: fn(saved))
    calls = []
    state = {'ingested_sources': {}}
    expected = 'sources/calls/duplicate.md'

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if outcome == 'cli_error':
            return subprocess.CompletedProcess(argv, 1, '', 'authentication error')
        if len(calls) == 1:
            # The observed failure: metadata credits the new ID but the claim does not.
            article.write_text('---\nsources: ["'+expected+'"]\n---\n'
                               'A supported fact [src:sources/calls/original.md]\n', encoding='utf-8')
            if first_credit != 'metadata':
                truth = '## Truth\n\n### Observed\n- A supported fact [src:sources/calls/original.md]\n'
                misplaced = '[src:'+expected+']'
                if first_credit == 'timeline':
                    content = truth + '\n## Timeline\n- Imported duplicate '+misplaced+'\n'
                elif first_credit == 'synthesized':
                    content = truth + '\n### Synthesized\n- An inference '+misplaced+'\n'
                else:
                    content = '---\nnotes: "'+misplaced+'"\n---\n'+truth
                article.write_text(content, encoding='utf-8')
            assert not state['ingested_sources']
        elif outcome == 'repair':
            assert 'duplicate' in kwargs['input'].lower()
            assert '[src:'+expected+']' in kwargs['input']
            assert 'A supported fact' in kwargs['input']
            article.write_text(article.read_text(encoding='utf-8').replace(
                '[src:sources/calls/original.md]',
                '[src:sources/calls/original.md] [src:'+expected+']'), encoding='utf-8')
        elif outcome == 'timeout':
            raise subprocess.TimeoutExpired(argv, 600)
        return subprocess.CompletedProcess(argv, 0, 'done', '')

    monkeypatch.setattr(ingest.subprocess, 'run', fake_run)
    group = utils.SourceGroup(id='calls', type='markdown', include=[], exclude=[],
                              category='calls', description='recorded calls')
    _, ok = asyncio.run(ingest.ingest_source_file(group, source, state))
    return ok, state, saved, calls, article


def test_duplicate_source_metadata_only_is_repaired_automatically(tmp_path, monkeypatch):
    ok, state, saved, calls, article = run_case(tmp_path, monkeypatch, 'repair')
    assert ok
    assert len(calls) == 2
    assert list(state['ingested_sources']) == ['calls/duplicate.md']
    assert saved['ingested_sources'] == state['ingested_sources']
    assert '[src:sources/calls/original.md]' in article.read_text(encoding='utf-8')
    assert article.read_text(encoding='utf-8').count('A supported fact') == 1
    assert calls[0][0] == calls[1][0], 'Repair must retain the same restricted tools and settings'
    assert calls[0][1]['cwd'] == calls[1][1]['cwd'] == str(article.parent.parent)


@pytest.mark.parametrize('outcome', ['no_repair', 'timeout'])
def test_recovery_is_bounded_and_never_stamps_uncredited_source(tmp_path, monkeypatch, outcome):
    ok, state, saved, calls, _ = run_case(tmp_path, monkeypatch, outcome)
    assert not ok
    assert len(calls) == 2
    assert not state['ingested_sources'] and not saved['ingested_sources']


def test_cli_authentication_failure_is_not_retried_as_citation_repair(tmp_path, monkeypatch):
    ok, state, _, calls, _ = run_case(tmp_path, monkeypatch, 'cli_error')
    assert not ok and len(calls) == 1
    assert not state['ingested_sources']


@pytest.mark.parametrize('first_credit', ['timeline', 'synthesized', 'frontmatter'])
@pytest.mark.parametrize('outcome', ['repair', 'no_repair'])
def test_invalid_provenance_needs_observed_claim_repair(tmp_path, monkeypatch, first_credit, outcome):
    ok, state, saved, calls, _ = run_case(tmp_path, monkeypatch, outcome, first_credit)
    assert len(calls) == 2, 'non-observed citations must trigger the one repair attempt'
    assert ok == (outcome == 'repair')
    assert bool(state['ingested_sources']) == ok
    assert bool(saved['ingested_sources']) == ok
