# Contrato: o runner com embedder fake grava estratégia, chunks, ids e métricas.
# O acerto abaixo é do fixture, não um resultado do estudo.

import json
from pathlib import Path

import pytest

from chunking_study.config import ENV_KEYS, load_settings
from chunking_study.corpus import load_corpus, load_queries
from chunking_study.embeddings import HashingEmbedder
from chunking_study.runner import run_record_to_dict, run_retrieval
from chunking_study.types import Document, GoldSpan, Query

ROOT = Path(__file__).resolve().parents[1]


def _settings(**overrides: str):
    base = {
        "EMBEDDING_PROVIDER": "fake",
        "CHUNK_STRATEGY": "fixed",
        "CHUNK_UNIT": "chars",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "0",
        "TOP_K": "1",
        "EMBEDDING_DIMENSION": "256",
    }
    base.update(overrides)
    return load_settings(base)


def test_runner_with_fake_embedder_records_a_hit_and_a_miss():
    horta = "A horta comunitária colhe alface e couve no sábado de manhã."
    oficina = "A oficina de bicicletas troca pneu e corrente na quarta à tarde."
    documents = [
        Document("horta", horta, "horta.txt"),
        Document("oficina", oficina, "oficina.txt"),
    ]
    hit = Query(
        "hit",
        "horta comunitária alface couve",
        (GoldSpan("horta", 0, len(horta)),),
    )
    miss = Query(
        "miss",
        "oficina bicicletas pneu corrente",
        (GoldSpan("horta", 0, len(horta)),),
    )
    records = run_retrieval(documents, [hit, miss], _settings())
    assert [record.query_id for record in records] == ["hit", "miss"]
    assert {record.strategy for record in records} == {"fixed"}
    assert {record.chunk_count for record in records} == {2}
    assert {record.embedded_texts for record in records} == {4}
    assert records[0].retrieved_ids == ("horta:0",)
    assert records[0].k == 1
    assert records[0].recall_at_k == 1.0
    assert records[0].reciprocal_rank == 1.0
    assert records[0].span_contained is True
    assert records[1].retrieved_ids == ("oficina:0",)
    assert records[1].recall_at_k == 0.0
    assert records[1].reciprocal_rank == 0.0
    assert records[1].span_contained is False

    ranked = run_retrieval(documents, [miss], _settings(TOP_K="2"))
    assert ranked[0].retrieved_ids == ("oficina:0", "horta:0")
    assert ranked[0].recall_at_k == 1.0
    assert ranked[0].reciprocal_rank == 0.5
    assert ranked[0].span_contained is True


def test_semantic_strategy_counts_breakpoint_embeddings():
    document = Document("horta", "A horta comunitária colhe alface.", "horta.txt")
    query = Query("q", "alface", (GoldSpan("horta", 0, len(document.text)),))
    fixed = run_retrieval([document], [query], _settings())
    semantic = run_retrieval(
        [document],
        [query],
        _settings(CHUNK_STRATEGY="semantic", SEMANTIC_SIMILARITY_THRESHOLD="-1"),
    )
    assert fixed[0].embedded_texts == 2
    assert semantic[0].embedded_texts == fixed[0].embedded_texts + 1
    assert semantic[0].strategy == "semantic"
    assert semantic[0].chunk_count == 1


def test_empty_corpus_still_records_a_miss():
    query = Query("q", "alface", (GoldSpan("horta", 0, 6),))
    records = run_retrieval([], [query], _settings())
    assert records[0].chunk_count == 0
    assert records[0].retrieved_ids == ()
    assert records[0].recall_at_k == 0.0
    assert records[0].reciprocal_rank == 0.0
    assert records[0].span_contained is False
    assert records[0].embedded_texts == 1


def test_record_dict_keeps_the_public_fields():
    document = Document("horta", "alface", "horta.txt")
    query = Query("q", "alface", (GoldSpan("horta", 0, 6),))
    record = run_retrieval([document], [query], _settings(), embedder=HashingEmbedder(32))[0]
    assert list(run_record_to_dict(record)) == [
        "strategy",
        "chunk_count",
        "embedded_texts",
        "query_id",
        "retrieved_ids",
        "k",
        "recall_at_k",
        "reciprocal_rank",
        "span_contained",
    ]


def test_placeholder_smoke_shape_is_stable():
    documents = load_corpus(ROOT / "data" / "placeholder")
    queries = load_queries(ROOT / "data" / "placeholder" / "queries.jsonl", documents)
    records = run_retrieval(documents, queries, _settings(TOP_K="3"))
    chunk_ids = {record.retrieved_ids for record in records}
    assert len(records) == 4
    assert {record.chunk_count for record in records} == {records[0].chunk_count}
    assert records[0].chunk_count >= len(documents)
    for record in records:
        assert record.strategy == "fixed"
        assert 1 <= len(record.retrieved_ids) <= 3
        assert record.recall_at_k is not None
        assert 0.0 <= record.recall_at_k <= 1.0
        assert record.reciprocal_rank is not None
        assert 0.0 <= record.reciprocal_rank <= 1.0
        assert record.span_contained in (True, False)
    assert chunk_ids


def test_main_prints_one_json_line_per_placeholder_query(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    from chunking_study.__main__ import main

    monkeypatch.chdir(ROOT)
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    main()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    assert [row["query_id"] for row in rows] == ["q01", "q02", "q03", "q04"]
    assert {row["strategy"] for row in rows} == {"fixed"}


def test_main_exits_on_a_missing_corpus(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    from chunking_study.__main__ import main

    monkeypatch.chdir(ROOT)
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CORPUS_PATH", "data/does-not-exist")
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1
    assert "CORPUS_PATH" in capsys.readouterr().err
