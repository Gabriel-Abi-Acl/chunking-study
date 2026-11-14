# Contrato: a agregação ignora nulo e conserva o zero. O hash confere o
# pipeline. Provedor real sem --allow-live ou sem chave sai com código 2
# e não embeda. Nenhum número daqui é achado sobre chunking.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chunking_study.chunkers import FixedSizeChunker, sentence_spans
from chunking_study.config import CHUNK_STRATEGIES, EMBEDDING_PROVIDERS
from chunking_study.corpus import EVAL_QUERY_CATEGORIES, load_eval_corpus, load_eval_queries
from chunking_study.embeddings import HashingEmbedder
from chunking_study.errors import ConfigError, ProviderError
from chunking_study.evaluate import (
    AGGREGATE_KEYS,
    RECORD_KEYS,
    RECORDS_FILENAME,
    RETRIEVAL_DEPTH,
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    EvalConfig,
    EvalRecord,
    aggregate_records,
    eval_record_from_dict,
    evaluate_config,
    expand_strategy,
    iter_configs,
    main,
    parse_args,
    record_to_dict,
)
from chunking_study.types import Document, GoldSpan, Query

ROOT = Path(__file__).resolve().parents[1]
SECRET = "super-secret-value"
OTHER_SECRET = "another-secret-value"


def _row(**overrides: object) -> EvalRecord:
    base: dict[str, object] = {
        "strategy": "fixed",
        "chunk_size": 120,
        "chunk_overlap": 0,
        "chunk_unit": "chars",
        "semantic_similarity_threshold": None,
        "query_id": "q",
        "category": "single_sentence",
        "chunk_count": 10,
        "mean_chunk_length": 12.5,
        "embedded_texts": 30,
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
        "reciprocal_rank": None,
        "span_contained": None,
        "span_contained_in_index": None,
        "retrieved_ids": (),
        "error": None,
    }
    base.update(overrides)
    return EvalRecord(**base)  # type: ignore[arg-type]


def _scored(
    query_id: str,
    category: str,
    recall_at_1: float | None,
    reciprocal_rank: float | None,
    span_contained: bool | None,
    span_contained_in_index: bool | None,
    **overrides: object,
) -> EvalRecord:
    return _row(
        query_id=query_id,
        category=category,
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_1,
        recall_at_5=recall_at_1,
        reciprocal_rank=reciprocal_rank,
        span_contained=span_contained,
        span_contained_in_index=span_contained_in_index,
        **overrides,
    )


def test_aggregate_keeps_zero_and_drops_null():
    records = [
        _scored("q1", "single_sentence", 0.0, 0.0, False, False),
        _scored(
            "q2", "single_sentence", None, None, None, None, error="falhou", retrieved_ids=None
        ),
    ]
    [block] = aggregate_records(records)
    assert block["query_count"] == 2
    assert block["error_count"] == 1
    assert block["recall_at_1"] == 0.0
    assert block["recall_at_3"] == 0.0
    assert block["recall_at_5"] == 0.0
    assert block["mrr"] == 0.0
    assert block["span_containment_rate"] == 0.0
    assert block["index_span_containment_rate"] == 0.0
    assert block["chunk_count"] == 10
    assert block["mean_chunk_length"] == 12.5
    assert block["embedded_texts"] == 30
    assert set(AGGREGATE_KEYS) <= set(block)


def test_aggregate_is_null_when_every_value_is_null():
    records = [
        _scored(
            "q1",
            "paraphrase",
            None,
            None,
            None,
            None,
            error="falhou",
            chunk_count=None,
            mean_chunk_length=None,
            embedded_texts=4,
            retrieved_ids=None,
        ),
        _scored(
            "q2",
            "paraphrase",
            None,
            None,
            None,
            None,
            error="falhou",
            chunk_count=None,
            mean_chunk_length=None,
            embedded_texts=4,
            retrieved_ids=None,
        ),
    ]
    [block] = aggregate_records(records)
    assert block["recall_at_1"] is None
    assert block["recall_at_5"] is None
    assert block["mrr"] is None
    assert block["span_containment_rate"] is None
    assert block["index_span_containment_rate"] is None
    assert block["chunk_count"] is None
    assert block["mean_chunk_length"] is None
    assert block["embedded_texts"] == 4
    assert block["error_count"] == 2
    assert block["query_count"] == 2


def test_aggregate_pools_queries_and_splits_categories():
    # Média das perguntas, não média das categorias. 1, 0 e 1, com um nulo
    # de fora, dá 2/3. A média das categorias daria 0,75.
    records = [
        _scored("a", "single_sentence", 1.0, 0.0, True, True),
        _scored("b", "single_sentence", 0.0, 1.0, False, True),
        _scored("c", "paraphrase", 1.0, 0.5, True, True),
        _scored("d", "paraphrase", None, None, None, None, error="transporte", retrieved_ids=None),
    ]
    [block] = aggregate_records(records)
    assert block["recall_at_1"] == pytest.approx(2 / 3)
    assert block["mrr"] == pytest.approx(0.5)
    assert block["span_containment_rate"] == pytest.approx(2 / 3)
    assert block["index_span_containment_rate"] == 1.0
    assert block["error_count"] == 1
    by_category = {item["category"]: item for item in block["by_category"]}
    assert list(by_category) == ["single_sentence", "paraphrase"]
    assert by_category["single_sentence"]["recall_at_1"] == 0.5
    assert by_category["single_sentence"]["mrr"] == 0.5
    assert by_category["single_sentence"]["span_containment_rate"] == 0.5
    assert by_category["single_sentence"]["query_count"] == 2
    assert by_category["single_sentence"]["error_count"] == 0
    assert by_category["single_sentence"]["chunk_count"] == 10
    assert by_category["paraphrase"]["recall_at_1"] == 1.0
    assert by_category["paraphrase"]["mrr"] == 0.5
    assert by_category["paraphrase"]["error_count"] == 1
    assert by_category["paraphrase"]["query_count"] == 2
    assert by_category["paraphrase"]["embedded_texts"] == 30


def test_containment_rate_is_not_the_mean_recall():
    records = [
        _row(
            query_id="q1",
            recall_at_1=0.5,
            recall_at_3=0.5,
            recall_at_5=0.5,
            reciprocal_rank=1.0,
            span_contained=False,
            span_contained_in_index=True,
        )
    ]
    [block] = aggregate_records(records)
    assert block["recall_at_5"] == 0.5
    assert block["span_containment_rate"] == 0.0
    assert block["index_span_containment_rate"] == 1.0
    assert block["mrr"] == 1.0


def test_aggregate_orders_strategies_and_does_not_mix_configs():
    records = [
        _scored(
            "s",
            "semantic",
            0.0,
            0.0,
            False,
            False,
            strategy="semantic",
            semantic_similarity_threshold=0.9,
        ),
        _scored("f", "single_sentence", 1.0, 1.0, True, True, strategy="fixed"),
        _scored(
            "t",
            "single_sentence",
            0.0,
            0.0,
            False,
            False,
            strategy="semantic",
            semantic_similarity_threshold=0.2,
        ),
    ]
    blocks = aggregate_records(records)
    assert [(block["strategy"], block["semantic_similarity_threshold"]) for block in blocks] == [
        ("fixed", None),
        ("semantic", 0.2),
        ("semantic", 0.9),
    ]
    assert blocks[0]["recall_at_1"] == 1.0
    assert blocks[1]["recall_at_1"] == 0.0
    assert blocks[2]["recall_at_1"] == 0.0


def test_aggregate_drops_disagreeing_index_stats():
    records = [
        _scored("q1", "single_sentence", 1.0, 1.0, True, True, chunk_count=4),
        _scored("q2", "single_sentence", 0.0, 0.0, False, False, chunk_count=5),
    ]
    [block] = aggregate_records(records)
    assert block["chunk_count"] is None
    assert block["recall_at_1"] == 0.5


def test_default_grid_is_one_config_per_strategy():
    configs = iter_configs(CHUNK_STRATEGIES, (120,), None, (0.5,), "chars")
    assert [
        (item.strategy, item.chunk_size, item.chunk_overlap, item.semantic_similarity_threshold)
        for item in configs
    ] == [
        ("fixed", 120, 0, None),
        ("fixed_overlap", 120, 20, None),
        ("sentence_paragraph", 120, 0, None),
        ("semantic", 120, 0, 0.5),
    ]
    assert expand_strategy("ALL") == CHUNK_STRATEGIES
    assert expand_strategy("Fixed") == ("fixed",)


def test_grid_filters_overlaps_and_crosses_thresholds():
    configs = iter_configs(
        CHUNK_STRATEGIES,
        (50, 100),
        (0, 10, 50),
        (0.5, 0.25),
        "chars",
    )
    identities = [
        (item.strategy, item.chunk_size, item.chunk_overlap, item.semantic_similarity_threshold)
        for item in configs
    ]
    assert ("fixed", 50, 0, None) in identities
    assert ("fixed", 100, 10, None) not in identities
    assert ("fixed_overlap", 50, 0, None) not in identities
    assert ("fixed_overlap", 50, 10, None) in identities
    assert ("fixed_overlap", 100, 50, None) in identities
    assert ("fixed_overlap", 50, 50, None) not in identities
    assert ("sentence_paragraph", 100, 0, None) in identities
    assert ("sentence_paragraph", 50, 50, None) not in identities
    assert ("semantic", 100, 10, 0.25) in identities
    assert ("semantic", 100, 10, 0.5) in identities
    assert ("semantic", 50, 0, 0.25) in identities
    assert identities.index(("semantic", 100, 10, 0.25)) < identities.index(
        ("semantic", 100, 10, 0.5)
    )


def test_grid_rejects_empty_fixed_overlap_tokens_and_unknown_strategy():
    with pytest.raises(ConfigError, match="fixed_overlap"):
        iter_configs(("fixed_overlap",), (120,), (0,), (0.5,), "chars")
    with pytest.raises(ConfigError, match="tokens"):
        iter_configs(("semantic",), (120,), None, (0.5,), "tokens")
    with pytest.raises(ConfigError, match="CHUNK_STRATEGY"):
        expand_strategy("recursive")


def test_index_embed_error_keeps_chunk_stats_and_nulls_metrics():
    text = "a" * 250
    documents = [Document("d", text, "d.txt")]
    queries = [Query("q", "aaaa", (GoldSpan("d", 0, 10),), "single_sentence")]

    class Boom:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ProviderError(f"Gemini devolveu embedding vazio. token={SECRET}")

    config = EvalConfig("fixed", 100, 0, "chars", None)
    [record] = evaluate_config(documents, queries, config, Boom(), secrets=(SECRET,))
    chunks = FixedSizeChunker(100, 0).chunk(documents[0])
    assert record.chunk_count == len(chunks) == 3
    assert record.mean_chunk_length == pytest.approx(250 / 3)
    assert record.embedded_texts == 3
    assert record.recall_at_1 is None
    assert record.recall_at_5 is None
    assert record.reciprocal_rank is None
    assert record.span_contained is None
    assert record.span_contained_in_index is None
    assert record.retrieved_ids is None
    assert record.error is not None
    assert SECRET not in record.error
    assert "[redacted]" in record.error
    [block] = aggregate_records([record])
    assert block["recall_at_1"] is None
    assert block["mrr"] is None
    assert block["chunk_count"] == 3


def test_semantic_transport_error_nulls_chunk_stats_and_hides_the_exception_text():
    text = "A horta comunitária colhe alface."
    documents = [Document("d", text, "d.txt")]
    queries = [Query("q", "alface", (GoldSpan("d", 0, 6),), "single_sentence")]

    class Boom:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise TimeoutError(f"timed out {SECRET}")

    config = EvalConfig("semantic", 80, 0, "chars", 0.5)
    [record] = evaluate_config(documents, queries, config, Boom(), secrets=(SECRET,))
    assert record.chunk_count is None
    assert record.mean_chunk_length is None
    assert record.embedded_texts == len(sentence_spans(text))
    assert record.recall_at_1 is None
    assert record.error == "Falha de transporte ao embedar (TimeoutError)."
    assert SECRET not in record.error


def test_query_transport_error_is_excluded_from_the_mean():
    text = "A horta comunitária colhe alface no sábado."
    documents = [Document("d", text, "d.txt")]
    start = text.index("colhe alface")
    good = Query(
        "good",
        "alface horta",
        (GoldSpan("d", start, start + len("colhe alface")),),
        "single_sentence",
    )
    bad = Query(
        "bad", "BOOM rede", (GoldSpan("d", start, start + len("colhe alface")),), "paraphrase"
    )

    class Selective:
        def __init__(self) -> None:
            self._inner = HashingEmbedder(16)

        def embed(self, texts: list[str]) -> list[list[float]]:
            if any(text.startswith("BOOM") for text in texts):
                raise ConnectionError(f"reset {SECRET}")
            return self._inner.embed(texts)

    config = EvalConfig("fixed", 80, 0, "chars", None)
    records = evaluate_config(documents, [good, bad], config, Selective(), secrets=(SECRET,))
    assert [record.query_id for record in records] == ["good", "bad"]
    assert records[0].error is None
    assert records[0].recall_at_1 is not None
    assert records[0].chunk_count is not None and records[0].chunk_count >= 1
    assert records[1].error == "Falha de transporte ao embedar (ConnectionError)."
    assert records[1].recall_at_1 is None
    assert records[1].reciprocal_rank is None
    assert records[1].span_contained is None
    assert records[1].retrieved_ids is None
    assert records[0].embedded_texts == records[1].embedded_texts
    [block] = aggregate_records(records)
    assert block["recall_at_1"] == records[0].recall_at_1
    assert block["mrr"] == records[0].reciprocal_rank
    assert block["error_count"] == 1
    assert block["query_count"] == 2
    paraphrase = block["by_category"][1]
    assert paraphrase["category"] == "paraphrase"
    assert paraphrase["recall_at_1"] is None
    assert paraphrase["error_count"] == 1


def test_unexpected_embed_error_propagates():
    documents = [Document("d", "alface", "d.txt")]
    queries = [Query("q", "alface", (GoldSpan("d", 0, 6),), "single_sentence")]

    class Bad:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        evaluate_config(documents, queries, EvalConfig("fixed", 20, 0, "chars", None), Bad())


def test_semantic_count_includes_breakpoint_texts_and_fixed_does_not():
    text = "A horta comunitária colhe alface. A oficina troca pneu."
    documents = [Document("d", text, "d.txt")]
    end = text.index(".")
    queries = [Query("q", "alface", (GoldSpan("d", 0, end),), "single_sentence")]
    embedder = HashingEmbedder(32)
    fixed = evaluate_config(
        documents, queries, EvalConfig("fixed", 200, 0, "chars", None), embedder
    )[0]
    semantic = evaluate_config(
        documents,
        queries,
        EvalConfig("semantic", 200, 0, "chars", -1.0),
        embedder,
    )[0]
    chunks = FixedSizeChunker(200, 0).chunk(documents[0])
    assert fixed.error is None
    assert semantic.error is None
    assert fixed.chunk_count == len(chunks)
    assert fixed.mean_chunk_length == pytest.approx(
        sum(len(chunk.text) for chunk in chunks) / len(chunks)
    )
    assert fixed.embedded_texts == fixed.chunk_count + len(queries)
    assert semantic.embedded_texts == len(sentence_spans(text)) + semantic.chunk_count + len(
        queries
    )
    if fixed.span_contained:
        assert fixed.span_contained_in_index is True


def test_query_without_gold_span_is_null_not_zero():
    documents = [Document("d", "alface e couve.", "d.txt")]
    queries = [Query("q", "alface", (), "single_sentence")]
    [record] = evaluate_config(
        documents,
        queries,
        EvalConfig("fixed", 50, 0, "chars", None),
        HashingEmbedder(16),
    )
    assert record.error is None
    assert record.recall_at_1 is None
    assert record.recall_at_5 is None
    assert record.reciprocal_rank is None
    assert record.span_contained is None
    assert record.span_contained_in_index is None
    [block] = aggregate_records([record])
    assert block["recall_at_1"] is None
    assert block["mrr"] is None
    assert block["span_containment_rate"] is None
    assert block["error_count"] == 0
    assert block["query_count"] == 1


def test_record_schema_round_trip():
    record = _scored("q1", "two_sentences", 1.0, 0.5, True, False, retrieved_ids=("d:0",))
    raw = record_to_dict(record)
    assert tuple(raw) == RECORD_KEYS
    assert raw["schema_version"] == SCHEMA_VERSION
    restored = eval_record_from_dict(raw)
    assert restored == record


def _write_tiny(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "A horta comunitária colhe alface no sábado.\n\nA oficina troca pneu na quarta."
    (corpus / "nota.txt").write_text(text, encoding="utf-8")
    passage = "colhe alface"
    start = text.index(passage)
    query = {
        "id": "q1",
        "question": "Quando a horta colhe alface?",
        "category": "single_sentence",
        "passages": [
            {
                "doc_id": "nota",
                "passage": passage,
                "start": start,
                "end": start + len(passage),
            }
        ],
    }
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps(query, ensure_ascii=False) + "\n", encoding="utf-8")
    return corpus, queries


def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch, cwd: Path | None = None) -> None:
    if cwd is not None:
        monkeypatch.chdir(cwd)
    main(argv)


def test_parser_defaults_to_fake_v1_and_local_output():
    args = parse_args([])
    assert args.provider == "fake"
    assert args.strategy == "all"
    assert args.allow_live is False
    assert args.corpus == Path("data/eval/v1")
    assert args.queries == Path("data/eval/v1/queries.jsonl")
    assert args.output == Path("results/eval")
    assert args.chunk_size is None
    assert args.overlap is None
    assert args.threshold is None
    assert args.dimension == 64


def test_cli_fake_v1_schema_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", OTHER_SECRET)
    first = tmp_path / "first"
    second = tmp_path / "second"
    argv = [
        "--strategy",
        "fixed",
        "--chunk-size",
        "200",
        "--overlap",
        "20",
        "--output",
        str(first),
    ]
    _run(argv, monkeypatch)
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    _run([*argv[:-1], str(second)], monkeypatch)
    records_name = RECORDS_FILENAME
    summary_name = SUMMARY_FILENAME
    assert (first / records_name).read_bytes() == (second / records_name).read_bytes()
    assert (first / summary_name).read_bytes() == (second / summary_name).read_bytes()

    documents = load_eval_corpus(ROOT / "data" / "eval" / "v1")
    queries = load_eval_queries(ROOT / "data" / "eval" / "v1" / "queries.jsonl", documents)
    rows = [
        json.loads(line) for line in (first / records_name).read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((first / summary_name).read_text(encoding="utf-8"))
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["embedding_provider"] == "fake"
    assert summary["embedding_model"] is None
    assert summary["embedding_dimension"] == 64
    assert summary["document_count"] == len(documents)
    assert summary["query_count"] == len(queries)
    assert summary["recall_ks"] == [1, 3, 5]
    assert summary["retrieval_depth"] == RETRIEVAL_DEPTH
    assert summary["corpus_path"] == "data/eval/v1"
    assert [row["query_id"] for row in rows] == [query.query_id for query in queries]
    assert {row["category"] for row in rows} == set(EVAL_QUERY_CATEGORIES)
    [config] = summary["configs"]
    assert config["strategy"] == "fixed"
    assert config["chunk_size"] == 200
    assert config["chunk_overlap"] == 0
    assert config["semantic_similarity_threshold"] is None
    assert config["query_count"] == len(queries)
    assert config["error_count"] == 0
    assert config["chunk_count"] > 0
    assert config["mean_chunk_length"] > 0
    assert config["embedded_texts"] == config["chunk_count"] + len(queries)
    assert {item["category"] for item in config["by_category"]} == set(EVAL_QUERY_CATEGORIES)
    assert sum(item["query_count"] for item in config["by_category"]) == len(queries)
    for row in rows:
        assert set(row) == set(RECORD_KEYS)
        assert row["error"] is None
        assert row["schema_version"] == SCHEMA_VERSION
        assert SECRET not in json.dumps(row)
        for key in ("recall_at_1", "recall_at_3", "recall_at_5", "reciprocal_rank"):
            assert 0.0 <= row[key] <= 1.0
        assert row["recall_at_1"] <= row["recall_at_3"] <= row["recall_at_5"]
        assert row["span_contained"] is (row["recall_at_5"] == 1.0)
        if row["span_contained"]:
            assert row["span_contained_in_index"] is True
        assert len(row["retrieved_ids"]) == min(RETRIEVAL_DEPTH, row["chunk_count"])
    restored = [eval_record_from_dict(row) for row in rows]
    assert aggregate_records(restored) == summary["configs"]
    blob = (first / summary_name).read_text(encoding="utf-8")
    assert SECRET not in blob
    assert OTHER_SECRET not in blob


def test_cli_all_strategies_uses_the_default_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(ROOT)
    output = tmp_path / "all"
    main(["--output", str(output), "--strategy", "all"])
    summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output / RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    documents = load_eval_corpus(ROOT / "data" / "eval" / "v1")
    queries = load_eval_queries(ROOT / "data" / "eval" / "v1" / "queries.jsonl", documents)
    by_strategy = {config["strategy"]: config for config in summary["configs"]}
    assert list(by_strategy) == list(CHUNK_STRATEGIES)
    assert len(rows) == len(queries) * len(CHUNK_STRATEGIES)
    assert by_strategy["fixed"]["chunk_overlap"] == 0
    assert by_strategy["fixed"]["chunk_size"] == 120
    assert by_strategy["fixed_overlap"]["chunk_overlap"] == 20
    assert by_strategy["fixed_overlap"]["semantic_similarity_threshold"] is None
    assert by_strategy["sentence_paragraph"]["chunk_overlap"] == 0
    assert by_strategy["semantic"]["chunk_overlap"] == 0
    assert by_strategy["semantic"]["semantic_similarity_threshold"] == 0.5
    for strategy in ("fixed", "fixed_overlap", "sentence_paragraph"):
        config = by_strategy[strategy]
        assert config["embedded_texts"] == config["chunk_count"] + len(queries)
        assert config["error_count"] == 0
    semantic = by_strategy["semantic"]
    assert semantic["embedded_texts"] >= semantic["chunk_count"] + len(documents) + len(queries)
    assert semantic["error_count"] == 0


def test_cli_custom_grid_on_a_tiny_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus, queries = _write_tiny(tmp_path)
    output = tmp_path / "grid"
    monkeypatch.chdir(tmp_path)
    main(
        [
            "--strategy",
            "all",
            "--chunk-size",
            "30,80",
            "--overlap",
            "0,10",
            "--threshold",
            "0.2",
            "--corpus",
            str(corpus),
            "--queries",
            str(queries),
            "--output",
            str(output),
        ]
    )
    summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    identities = [
        (
            config["strategy"],
            config["chunk_size"],
            config["chunk_overlap"],
            config["semantic_similarity_threshold"],
        )
        for config in summary["configs"]
    ]
    assert identities == [
        ("fixed", 30, 0, None),
        ("fixed", 80, 0, None),
        ("fixed_overlap", 30, 10, None),
        ("fixed_overlap", 80, 10, None),
        ("sentence_paragraph", 30, 0, None),
        ("sentence_paragraph", 30, 10, None),
        ("sentence_paragraph", 80, 0, None),
        ("sentence_paragraph", 80, 10, None),
        ("semantic", 30, 0, 0.2),
        ("semantic", 30, 10, 0.2),
        ("semantic", 80, 0, 0.2),
        ("semantic", 80, 10, 0.2),
    ]
    assert summary["embedding_provider"] == "fake"
    assert all(config["error_count"] == 0 for config in summary["configs"])


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_live_provider_without_flag_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider: str,
):
    monkeypatch.chdir(tmp_path)
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    monkeypatch.setenv(key_name, SECRET)

    def boom(settings: object) -> object:
        raise AssertionError(f"embedder construído: {settings!r}")

    monkeypatch.setattr("chunking_study.evaluate.build_embedder", boom)
    output = tmp_path / "nope"
    with pytest.raises(SystemExit) as caught:
        main(["--provider", provider, "--output", str(output)])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "allow-live" in err
    assert "ausente" not in err
    assert SECRET not in err
    assert not output.exists()


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_live_provider_without_key_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider: str,
    key_name: str,
):
    monkeypatch.chdir(tmp_path)
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    def boom(settings: object) -> object:
        raise AssertionError("embedder construído")

    monkeypatch.setattr("chunking_study.evaluate.build_embedder", boom)
    output = tmp_path / "nope"
    with pytest.raises(SystemExit) as caught:
        main(["--provider", provider, "--allow-live", "--output", str(output)])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert key_name in err
    assert "Nenhuma chamada foi feita." in err
    assert not output.exists()


def test_allow_live_with_key_uses_the_provider_and_hides_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, queries = _write_tiny(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    seen: dict[str, object] = {}

    def fake_build(settings):
        seen["provider"] = settings.embedding_provider
        seen["key"] = settings.gemini_api_key
        seen["model"] = settings.gemini_embedding_model
        return HashingEmbedder(8)

    monkeypatch.setattr("chunking_study.evaluate.build_embedder", fake_build)
    output = tmp_path / "live"
    main(
        [
            "--provider",
            "gemini",
            "--allow-live",
            "--strategy",
            "fixed",
            "--chunk-size",
            "80",
            "--corpus",
            str(corpus),
            "--queries",
            str(queries),
            "--output",
            str(output),
        ]
    )
    assert seen["provider"] == "gemini"
    assert seen["key"] == SECRET
    summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    blob = (output / SUMMARY_FILENAME).read_text(encoding="utf-8") + (
        output / RECORDS_FILENAME
    ).read_text(encoding="utf-8")
    assert summary["embedding_provider"] == "gemini"
    assert summary["embedding_model"] == "gemini-embedding-001"
    assert summary["embedding_dimension"] is None
    assert SECRET not in blob
    assert "google" not in sys.modules
    assert "openai" not in sys.modules


def test_allow_live_reads_the_key_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus, queries = _write_tiny(tmp_path)
    monkeypatch.chdir(tmp_path)
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        f"GEMINI_API_KEY={SECRET}\nEMBEDDING_PROVIDER=gemini\nCHUNK_SIZE=nao-e-inteiro\n",
        encoding="utf-8",
    )
    seen: dict[str, str] = {}

    def fake_build(settings):
        seen["key"] = settings.gemini_api_key
        return HashingEmbedder(8)

    monkeypatch.setattr("chunking_study.evaluate.build_embedder", fake_build)
    output = tmp_path / "from-dotenv"
    main(
        [
            "--provider",
            "Gemini",
            "--allow-live",
            "--strategy",
            "fixed",
            "--corpus",
            str(corpus),
            "--queries",
            str(queries),
            "--output",
            str(output),
        ]
    )
    assert seen["key"] == SECRET
    summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["embedding_provider"] == "gemini"
    assert SECRET not in (output / SUMMARY_FILENAME).read_text(encoding="utf-8")


def test_invalid_strategy_and_overlap_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit) as caught:
        main(["--strategy", "recursive", "--output", str(tmp_path / "out")])
    assert caught.value.code == 1
    assert "CHUNK_STRATEGY" in capsys.readouterr().err

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--strategy",
                "fixed_overlap",
                "--overlap",
                "0",
                "--output",
                str(tmp_path / "out2"),
            ]
        )
    assert caught.value.code == 1
    assert "fixed_overlap" in capsys.readouterr().err


def test_module_and_console_script_refuse_a_live_provider(tmp_path: Path):
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env["GEMINI_API_KEY"] = SECRET
    module = subprocess.run(
        [
            sys.executable,
            "-m",
            "chunking_study.evaluate",
            "--provider",
            "gemini",
            "--output",
            str(tmp_path / "mod"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert module.returncode == 2, module.stderr
    assert "allow-live" in module.stderr
    assert SECRET not in module.stderr
    assert not (tmp_path / "mod").exists()

    script = Path(sys.executable).with_name("evaluate-chunking")
    assert script.is_file(), f"console script ausente em {script}"
    env["OPENAI_API_KEY"] = OTHER_SECRET
    env.pop("GEMINI_API_KEY", None)
    console = subprocess.run(
        [str(script), "--provider", "openai", "--output", str(tmp_path / "cli")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert console.returncode == 2, console.stderr
    assert "allow-live" in console.stderr
    assert OTHER_SECRET not in console.stderr
    assert not (tmp_path / "cli").exists()


def test_readme_documents_the_harness_and_withholds_hash_numbers():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "evaluate-chunking" in readme
    assert "python -m chunking_study.evaluate" in readme
    assert "--allow-live" in readme
    assert "código 2" in readme
    for key in (
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "span_containment_rate",
        "index_span_containment_rate",
        "chunk_count",
        "mean_chunk_length",
        "embedded_texts",
    ):
        assert key in readme
    assert "não achado sobre chunking" in readme
    assert "Este README não publica recall, MRR nem contenção." in readme
    assert "fake" in EMBEDDING_PROVIDERS
