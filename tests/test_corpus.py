# Contrato: o placeholder resolve cada passagem uma vez, no offset do arquivo.

import json
from pathlib import Path

import pytest

from chunking_study.chunkers import FixedSizeChunker
from chunking_study.corpus import load_corpus, load_queries
from chunking_study.errors import ConfigError
from chunking_study.metrics import spans_contained

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "placeholder"


def test_placeholder_offsets_match_the_passage_text():
    documents = load_corpus(CORPUS)
    by_id = {document.doc_id: document for document in documents}
    assert set(by_id) == {"biblioteca", "horta", "oficina"}
    queries = load_queries(CORPUS / "queries.jsonl", documents)
    assert [query.query_id for query in queries] == ["q01", "q02", "q03", "q04"]
    raw_lines = [
        json.loads(line)
        for line in (CORPUS / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for query, raw in zip(queries, raw_lines, strict=True):
        passage = raw["passages"][0]
        document = by_id[passage["doc_id"]]
        span = query.gold_spans[0]
        assert document.text[span.start : span.end] == passage["passage"]
        assert document.text.count(passage["passage"]) == 1
        assert query.question == raw["question"]


def test_gold_span_maps_into_a_chunk_only_when_the_window_can_cover_it():
    documents = load_corpus(CORPUS)
    queries = load_queries(CORPUS / "queries.jsonl", documents)
    wide = []
    narrow = []
    for document in documents:
        wide.extend(FixedSizeChunker(size=10_000, overlap=0).chunk(document))
        narrow.extend(FixedSizeChunker(size=20, overlap=0).chunk(document))
    long_span = next(
        span for query in queries for span in query.gold_spans if span.end - span.start > 20
    )
    assert spans_contained(wide, [long_span]) is True
    assert spans_contained(narrow, [long_span]) is False


def test_loader_rejects_ambiguous_missing_and_duplicate_inputs(tmp_path: Path):
    (tmp_path / "nota.txt").write_text("alpha beta alpha", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "id": "q",
                "question": "alpha?",
                "passages": [{"doc_id": "nota", "passage": "alpha"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    documents = load_corpus(tmp_path)
    with pytest.raises(ConfigError, match="ambígua"):
        load_queries(queries, documents)

    queries.write_text(
        json.dumps(
            {
                "id": "q",
                "question": "gamma?",
                "passages": [{"doc_id": "nota", "passage": "gamma"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="ausente"):
        load_queries(queries, documents)

    queries.write_text("{não é json}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        load_queries(queries, documents)

    (tmp_path / "a.md").write_text("outro", encoding="utf-8")
    (tmp_path / "a.txt").write_text("outro", encoding="utf-8")
    with pytest.raises(ConfigError, match="repetido"):
        load_corpus(tmp_path)


def test_missing_corpus_path_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="CORPUS_PATH"):
        load_corpus(tmp_path / "ausente")
    with pytest.raises(ConfigError, match="QUERIES_PATH"):
        load_queries(tmp_path / "ausente.jsonl", [])
