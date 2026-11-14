"""Avaliação offline da grade de chunking.

Grava um JSONL por pergunta e um agregado por configuração e por
categoria. O embedder padrão é o hash: confere o pipeline, não a
qualidade do corte. Provedor real só segue com `--allow-live` e chave.
"""

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from chunking_study.chunkers import build_chunker
from chunking_study.config import (
    CHUNK_STRATEGIES,
    CHUNK_UNITS,
    EMBEDDING_PROVIDERS,
    Settings,
    load_settings,
)
from chunking_study.corpus import EVAL_QUERY_CATEGORIES, load_eval_corpus, load_eval_queries
from chunking_study.embeddings import Embedder, HashingEmbedder, build_embedder
from chunking_study.errors import ConfigError, ProviderError
from chunking_study.index import InMemoryVectorIndex
from chunking_study.metrics import mean_defined, recall_at_k, reciprocal_rank, spans_contained
from chunking_study.runner import CountingEmbedder
from chunking_study.types import Chunk, Document, Query

SCHEMA_VERSION = 1
RECALL_KS = (1, 3, 5)
RETRIEVAL_DEPTH = max(RECALL_KS)
DEFAULT_CHUNK_SIZES = (120,)
DEFAULT_FIXED_OVERLAP = 20
DEFAULT_THRESHOLDS = (0.5,)
RECORDS_FILENAME = "records.jsonl"
SUMMARY_FILENAME = "summary.json"

RECORD_KEYS = (
    "schema_version",
    "strategy",
    "chunk_size",
    "chunk_overlap",
    "chunk_unit",
    "semantic_similarity_threshold",
    "query_id",
    "category",
    "chunk_count",
    "mean_chunk_length",
    "embedded_texts",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "reciprocal_rank",
    "span_contained",
    "span_contained_in_index",
    "retrieved_ids",
    "error",
)

AGGREGATE_KEYS = (
    "query_count",
    "error_count",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "span_containment_rate",
    "index_span_containment_rate",
    "chunk_count",
    "mean_chunk_length",
    "embedded_texts",
)

_TRANSPORT_ROOTS = frozenset(
    {
        "aiohttp",
        "anyio",
        "google",
        "grpc",
        "h2",
        "httpcore",
        "httpx",
        "openai",
        "requests",
        "urllib",
        "urllib3",
    }
)

_CREDENTIAL_KEYS = (
    "GEMINI_API_KEY",
    "GEMINI_EMBEDDING_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
)


@dataclass(frozen=True)
class EvalConfig:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    chunk_unit: str
    semantic_similarity_threshold: float | None


@dataclass(frozen=True)
class EvalRecord:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    chunk_unit: str
    semantic_similarity_threshold: float | None
    query_id: str
    category: str
    chunk_count: int | None
    mean_chunk_length: float | None
    embedded_texts: int | None
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    reciprocal_rank: float | None
    span_contained: bool | None
    span_contained_in_index: bool | None
    retrieved_ids: tuple[str, ...] | None
    error: str | None


def expand_strategy(name: str) -> tuple[str, ...]:
    """`all` vira as quatro estratégias, na ordem estável do projeto."""
    normalized = name.strip().lower()
    if normalized == "all":
        return CHUNK_STRATEGIES
    if normalized not in CHUNK_STRATEGIES:
        options = ", ".join((*CHUNK_STRATEGIES, "all"))
        raise ConfigError(f"CHUNK_STRATEGY inválida: {name}. Use: {options}.")
    return (normalized,)


def iter_configs(
    strategies: Sequence[str],
    sizes: Sequence[int],
    overlaps: Sequence[int] | None,
    thresholds: Sequence[float],
    unit: str,
) -> tuple[EvalConfig, ...]:
    """Grade pequena. `overlaps is None` é o padrão, não uma lista vazia.

    `fixed` ignora overlap e grava 0. Sem a flag, `fixed_overlap` usa
    `DEFAULT_FIXED_OVERLAP` para a estratégia ser válida. Esse 20 não é
    resultado de experimento.
    """
    if unit not in CHUNK_UNITS:
        options = ", ".join(CHUNK_UNITS)
        raise ConfigError(f"CHUNK_UNIT inválida: {unit}. Use: {options}.")
    blocked = [name for name in strategies if name in ("sentence_paragraph", "semantic")]
    if unit == "tokens" and blocked:
        raise ConfigError("CHUNK_UNIT=tokens só vale para fixed e fixed_overlap.")
    if not sizes or not thresholds:
        raise ConfigError("A grade precisa de tamanho e de limiar.")
    ordered_sizes = tuple(sorted(set(sizes)))
    ordered_thresholds = tuple(sorted(set(thresholds)))
    ordered_overlaps = None if overlaps is None else tuple(sorted(set(overlaps)))
    for size in ordered_sizes:
        if size < 1:
            raise ConfigError("--chunk-size precisa ser >= 1.")
    for threshold in ordered_thresholds:
        if math.isnan(threshold) or threshold < -1.0 or threshold > 1.0:
            raise ConfigError("SEMANTIC_SIMILARITY_THRESHOLD precisa estar entre -1 e 1.")

    configs: list[EvalConfig] = []
    empty: list[str] = []
    for strategy in strategies:
        produced = 0
        for size in ordered_sizes:
            for overlap in _overlap_values(strategy, ordered_overlaps, size):
                if strategy == "semantic":
                    for threshold in ordered_thresholds:
                        configs.append(EvalConfig(strategy, size, overlap, unit, threshold))
                        produced += 1
                    continue
                configs.append(EvalConfig(strategy, size, overlap, unit, None))
                produced += 1
        if produced == 0:
            empty.append(strategy)
    if empty:
        names = ", ".join(empty)
        raise ConfigError(
            f"Nenhuma configuração válida para: {names}. "
            "fixed usa overlap 0. fixed_overlap precisa de overlap >= 1 e menor que o tamanho. "
            "Os outros cortes usam overlap >= 0 e menor que o tamanho."
        )
    return tuple(configs)


def is_embedding_error(exc: BaseException) -> bool:
    """Erro de provedor ou de transporte. Bug de código não entra aqui."""
    if isinstance(exc, (ProviderError, TimeoutError, ConnectionError)):
        return True
    root = type(exc).__module__.split(".", 1)[0]
    return root in _TRANSPORT_ROOTS


def evaluate_config(
    documents: Sequence[Document],
    queries: Sequence[Query],
    config: EvalConfig,
    embedder: Embedder,
    *,
    secrets: Sequence[str] = (),
) -> list[EvalRecord]:
    """Um corte, um índice, uma linha por pergunta.

    Falha ao embedar anula a métrica da pergunta (ou da configuração, se
    o índice não ficou pronto). O zero continua significando acerto nenhum.
    """
    counter = CountingEmbedder(embedder)
    chunker = build_chunker(
        config.strategy,
        size=config.chunk_size,
        overlap=config.chunk_overlap,
        unit=config.chunk_unit,
        embedder=counter,
        similarity_threshold=(
            0.5
            if config.semantic_similarity_threshold is None
            else config.semantic_similarity_threshold
        ),
    )
    chunks: list[Chunk] = []
    try:
        for document in documents:
            chunks.extend(chunker.chunk(document))
    except Exception as exc:
        if not is_embedding_error(exc):
            raise
        return _error_records(
            queries,
            config,
            embedded_texts=counter.texts,
            chunk_count=None,
            mean_length=None,
            exc=exc,
            secrets=secrets,
        )

    try:
        vectors = counter.embed([chunk.text for chunk in chunks]) if chunks else []
        if len(vectors) != len(chunks):
            raise ProviderError("O embedder devolveu uma quantidade inesperada de vetores.")
        index = InMemoryVectorIndex()
        if chunks:
            index.add(chunks, vectors)
    except Exception as exc:
        if not is_embedding_error(exc):
            raise
        return _error_records(
            queries,
            config,
            embedded_texts=counter.texts,
            chunk_count=len(chunks),
            mean_length=_mean_length(chunks),
            exc=exc,
            secrets=secrets,
        )

    outcomes: list[tuple[Query, dict[str, object]]] = []
    for query in queries:
        try:
            query_vectors = counter.embed([query.question])
            if len(query_vectors) != 1:
                raise ProviderError("O embedder devolveu uma quantidade inesperada de vetores.")
            ranked_hits = index.query(query_vectors[0], RETRIEVAL_DEPTH) if chunks else []
            ranked = [hit.chunk for hit in ranked_hits]
            outcomes.append((query, _success_fields(query, ranked, chunks)))
        except Exception as exc:
            if not is_embedding_error(exc):
                raise
            outcomes.append((query, _failure_fields(exc, secrets)))

    return [
        _record(
            config,
            query,
            fields,
            chunk_count=len(chunks),
            mean_length=_mean_length(chunks),
            embedded_texts=counter.texts,
        )
        for query, fields in outcomes
    ]


def aggregate_records(records: Sequence[EvalRecord]) -> list[dict[str, object]]:
    """Uma entrada por configuração, com bloco por categoria.

    A média ignora nulo e conserva o zero. `chunk_count`, comprimento
    médio e `embedded_texts` são do índice: repetem em cada categoria e
    viram nulo se as linhas discordarem.
    """
    groups: dict[tuple[object, ...], list[EvalRecord]] = {}
    for record in records:
        groups.setdefault(_config_key(record), []).append(record)
    blocks: list[dict[str, object]] = []
    for key in sorted(groups, key=_config_sort):
        group = groups[key]
        block = _identity(group[0])
        block.update(_metrics(group))
        block["by_category"] = _categories(group)
        blocks.append(block)
    return blocks


def build_summary(
    records: Sequence[EvalRecord],
    *,
    corpus_path: Path,
    queries_path: Path,
    embedding_provider: str,
    embedding_model: str | None,
    embedding_dimension: int | None,
    document_count: int,
    query_count: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_path": corpus_path.as_posix(),
        "queries_path": queries_path.as_posix(),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "document_count": document_count,
        "query_count": query_count,
        "recall_ks": list(RECALL_KS),
        "retrieval_depth": RETRIEVAL_DEPTH,
        "configs": aggregate_records(records),
    }


def record_to_dict(record: EvalRecord) -> dict[str, object]:
    raw: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "strategy": record.strategy,
        "chunk_size": record.chunk_size,
        "chunk_overlap": record.chunk_overlap,
        "chunk_unit": record.chunk_unit,
        "semantic_similarity_threshold": record.semantic_similarity_threshold,
        "query_id": record.query_id,
        "category": record.category,
        "chunk_count": record.chunk_count,
        "mean_chunk_length": record.mean_chunk_length,
        "embedded_texts": record.embedded_texts,
        "recall_at_1": record.recall_at_1,
        "recall_at_3": record.recall_at_3,
        "recall_at_5": record.recall_at_5,
        "reciprocal_rank": record.reciprocal_rank,
        "span_contained": record.span_contained,
        "span_contained_in_index": record.span_contained_in_index,
        "retrieved_ids": None if record.retrieved_ids is None else list(record.retrieved_ids),
        "error": record.error,
    }
    if tuple(raw) != RECORD_KEYS:
        missing = set(RECORD_KEYS) - set(raw)
        raise RuntimeError(f"schema do registro divergiu: {sorted(missing)}.")
    return raw


def eval_record_from_dict(raw: Mapping[str, object]) -> EvalRecord:
    missing = [key for key in RECORD_KEYS if key not in raw]
    if missing:
        names = ", ".join(missing)
        raise ConfigError(f"Registro sem campo: {names}.")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("schema_version do registro não é a desta versão.")
    return EvalRecord(
        strategy=_text(raw["strategy"], "strategy"),
        chunk_size=_required_int(raw["chunk_size"], "chunk_size"),
        chunk_overlap=_required_int(raw["chunk_overlap"], "chunk_overlap"),
        chunk_unit=_text(raw["chunk_unit"], "chunk_unit"),
        semantic_similarity_threshold=_optional_float(
            raw["semantic_similarity_threshold"], "semantic_similarity_threshold"
        ),
        query_id=_text(raw["query_id"], "query_id"),
        category=_text(raw["category"], "category"),
        chunk_count=_optional_int(raw["chunk_count"], "chunk_count"),
        mean_chunk_length=_optional_float(raw["mean_chunk_length"], "mean_chunk_length"),
        embedded_texts=_optional_int(raw["embedded_texts"], "embedded_texts"),
        recall_at_1=_optional_float(raw["recall_at_1"], "recall_at_1"),
        recall_at_3=_optional_float(raw["recall_at_3"], "recall_at_3"),
        recall_at_5=_optional_float(raw["recall_at_5"], "recall_at_5"),
        reciprocal_rank=_optional_float(raw["reciprocal_rank"], "reciprocal_rank"),
        span_contained=_optional_bool(raw["span_contained"], "span_contained"),
        span_contained_in_index=_optional_bool(
            raw["span_contained_in_index"], "span_contained_in_index"
        ),
        retrieved_ids=_ids(raw["retrieved_ids"]),
        error=None if raw["error"] is None else _text(raw["error"], "error"),
    )


def write_evaluation(
    output: Path, records: Sequence[EvalRecord], summary: Mapping[str, object]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [record_to_dict(record) for record in records]
    (output / RECORDS_FILENAME).write_text(_dumps_jsonl(rows), encoding="utf-8")
    (output / SUMMARY_FILENAME).write_text(_dumps_json(summary), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate-chunking",
        description=(
            "Avalia a grade de chunking no conjunto v1 e grava JSONL e um agregado. "
            "O embedder padrão é o hash, só para conferir o pipeline."
        ),
    )
    parser.add_argument(
        "--strategy",
        default="all",
        help="fixed, fixed_overlap, sentence_paragraph, semantic ou all.",
    )
    parser.add_argument(
        "--chunk-size",
        action="append",
        dest="chunk_size",
        help="Tamanho da janela. Repete ou separa por vírgula. Padrão: 120.",
    )
    parser.add_argument(
        "--overlap",
        action="append",
        help="Overlap da grade. Sem esta flag, fixed_overlap usa 20 e o resto usa 0.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        help="Limiar do corte semântico, entre -1 e 1. Padrão: 0.5.",
    )
    parser.add_argument("--unit", default="chars", help="chars ou tokens. tokens só no corte fixo.")
    parser.add_argument("--provider", default="fake", help="fake, gemini ou openai. Padrão: fake.")
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Autoriza chamada de rede para gemini ou openai. Sem isto o comando sai com código 2.",
    )
    parser.add_argument(
        "--output", default="results/eval", help="Diretório de records.jsonl e summary.json."
    )
    parser.add_argument("--corpus", default="data/eval/v1", help="Diretório de documentos do eval.")
    parser.add_argument(
        "--queries", default="data/eval/v1/queries.jsonl", help="JSONL de perguntas do eval."
    )
    parser.add_argument("--dimension", type=int, default=64, help="Dimensão do embedder hash.")
    namespace = parser.parse_args(None if argv is None else list(argv))
    namespace.output = Path(namespace.output)
    namespace.corpus = Path(namespace.corpus)
    namespace.queries = Path(namespace.queries)
    return namespace


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        provider = args.provider.strip().lower()
        if provider not in EMBEDDING_PROVIDERS:
            options = ", ".join(EMBEDDING_PROVIDERS)
            raise ConfigError(f"EMBEDDING_PROVIDER inválida: {args.provider}. Use: {options}.")
        if args.dimension < 1:
            raise ConfigError("EMBEDDING_DIMENSION precisa ser >= 1.")
        credentials = _credential_settings() if provider != "fake" else None
        if provider != "fake":
            assert credentials is not None
            _require_live(provider, args.allow_live, credentials)
        configs = iter_configs(
            expand_strategy(args.strategy),
            _parse_ints(
                args.chunk_size, default=DEFAULT_CHUNK_SIZES, name="--chunk-size", minimum=1
            ),
            None
            if args.overlap is None
            else _parse_ints(args.overlap, default=(), name="--overlap", minimum=0),
            _parse_floats(args.threshold, default=DEFAULT_THRESHOLDS, name="--threshold"),
            args.unit.strip().lower(),
        )
        documents = load_eval_corpus(args.corpus)
        queries = load_eval_queries(args.queries, documents)
        embedder = _build_embedder(provider, args.dimension, credentials)
        records: list[EvalRecord] = []
        for config in configs:
            records.extend(
                evaluate_config(
                    documents,
                    queries,
                    config,
                    embedder,
                    secrets=_secrets(credentials),
                )
            )
        summary = build_summary(
            records,
            corpus_path=args.corpus,
            queries_path=args.queries,
            embedding_provider=provider,
            embedding_model=_model_name(provider, credentials),
            embedding_dimension=args.dimension if provider == "fake" else None,
            document_count=len(documents),
            query_count=len(queries),
        )
        write_evaluation(args.output, records, summary)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Registros: {len(records)}. Diretório: {args.output.as_posix()}")


def _overlap_values(
    strategy: str,
    overlaps: Sequence[int] | None,
    size: int,
) -> tuple[int, ...]:
    if strategy == "fixed":
        return (0,)
    if overlaps is None:
        candidates = (DEFAULT_FIXED_OVERLAP,) if strategy == "fixed_overlap" else (0,)
    else:
        candidates = tuple(overlaps)
    values: list[int] = []
    for overlap in candidates:
        if overlap < 0 or overlap >= size:
            continue
        if strategy == "fixed_overlap" and overlap < 1:
            continue
        values.append(overlap)
    return tuple(values)


def _success_fields(
    query: Query, ranked: Sequence[Chunk], chunks: Sequence[Chunk]
) -> dict[str, object]:
    return {
        "recall_at_1": recall_at_k(ranked, query.gold_spans, 1),
        "recall_at_3": recall_at_k(ranked, query.gold_spans, 3),
        "recall_at_5": recall_at_k(ranked, query.gold_spans, 5),
        "reciprocal_rank": reciprocal_rank(ranked, query.gold_spans),
        "span_contained": spans_contained(ranked, query.gold_spans),
        "span_contained_in_index": spans_contained(chunks, query.gold_spans),
        "retrieved_ids": tuple(chunk.chunk_id for chunk in ranked),
        "error": None,
    }


def _failure_fields(exc: BaseException, secrets: Sequence[str]) -> dict[str, object]:
    return {
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
        "reciprocal_rank": None,
        "span_contained": None,
        "span_contained_in_index": None,
        "retrieved_ids": None,
        "error": _public_error(exc, secrets),
    }


def _error_records(
    queries: Sequence[Query],
    config: EvalConfig,
    *,
    embedded_texts: int,
    chunk_count: int | None,
    mean_length: float | None,
    exc: BaseException,
    secrets: Sequence[str],
) -> list[EvalRecord]:
    fields = _failure_fields(exc, secrets)
    return [
        _record(
            config,
            query,
            fields,
            chunk_count=chunk_count,
            mean_length=mean_length,
            embedded_texts=embedded_texts,
        )
        for query in queries
    ]


def _record(
    config: EvalConfig,
    query: Query,
    fields: Mapping[str, object],
    *,
    chunk_count: int | None,
    mean_length: float | None,
    embedded_texts: int,
) -> EvalRecord:
    return EvalRecord(
        strategy=config.strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        chunk_unit=config.chunk_unit,
        semantic_similarity_threshold=config.semantic_similarity_threshold,
        query_id=query.query_id,
        category=query.category,
        chunk_count=chunk_count,
        mean_chunk_length=mean_length,
        embedded_texts=embedded_texts,
        recall_at_1=_as_optional_float(fields["recall_at_1"]),
        recall_at_3=_as_optional_float(fields["recall_at_3"]),
        recall_at_5=_as_optional_float(fields["recall_at_5"]),
        reciprocal_rank=_as_optional_float(fields["reciprocal_rank"]),
        span_contained=_as_optional_bool(fields["span_contained"]),
        span_contained_in_index=_as_optional_bool(fields["span_contained_in_index"]),
        retrieved_ids=_ids(fields["retrieved_ids"]),
        error=None if fields["error"] is None else str(fields["error"]),
    )


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("métrica com tipo inesperado.")
    return float(value)


def _as_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("contenção com tipo inesperado.")
    return value


def _mean_length(chunks: Sequence[Chunk]) -> float | None:
    if not chunks:
        return None
    return sum(len(chunk.text) for chunk in chunks) / len(chunks)


def _config_key(record: EvalRecord) -> tuple[object, ...]:
    return (
        record.strategy,
        record.chunk_size,
        record.chunk_overlap,
        record.chunk_unit,
        record.semantic_similarity_threshold,
    )


def _config_sort(key: tuple[object, ...]) -> tuple[object, ...]:
    strategy, size, overlap, unit, threshold = key
    if strategy in CHUNK_STRATEGIES:
        order = CHUNK_STRATEGIES.index(str(strategy))
    else:
        order = len(CHUNK_STRATEGIES)
    missing = threshold is None
    value = 0.0 if not isinstance(threshold, (int, float)) else float(threshold)
    return (order, size, overlap, unit, missing, value)


def _identity(record: EvalRecord) -> dict[str, object]:
    return {
        "strategy": record.strategy,
        "chunk_size": record.chunk_size,
        "chunk_overlap": record.chunk_overlap,
        "chunk_unit": record.chunk_unit,
        "semantic_similarity_threshold": record.semantic_similarity_threshold,
    }


def _metrics(records: Sequence[EvalRecord]) -> dict[str, object]:
    block: dict[str, object] = {
        "query_count": len(records),
        "error_count": sum(1 for record in records if record.error is not None),
        "mrr": mean_defined([record.reciprocal_rank for record in records]),
        "span_containment_rate": _rate([record.span_contained for record in records]),
        "index_span_containment_rate": _rate(
            [record.span_contained_in_index for record in records]
        ),
        "chunk_count": _shared([record.chunk_count for record in records]),
        "mean_chunk_length": _shared([record.mean_chunk_length for record in records]),
        "embedded_texts": _shared([record.embedded_texts for record in records]),
    }
    for k in RECALL_KS:
        block[f"recall_at_{k}"] = mean_defined(
            [getattr(record, f"recall_at_{k}") for record in records]
        )
    return block


def _categories(records: Sequence[EvalRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)
    names = [name for name in EVAL_QUERY_CATEGORIES if name in grouped]
    names.extend(sorted(name for name in grouped if name not in names))
    blocks: list[dict[str, object]] = []
    for name in names:
        block: dict[str, object] = {"category": name}
        block.update(_metrics(grouped[name]))
        blocks.append(block)
    return blocks


def _rate(flags: Sequence[bool | None]) -> float | None:
    return mean_defined([None if flag is None else float(flag) for flag in flags])


def _shared(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    first = defined[0]
    for value in defined[1:]:
        if value != first:
            return None
    return first


def _public_error(exc: BaseException, secrets: Sequence[str]) -> str:
    if isinstance(exc, ProviderError):
        text = str(exc)
    else:
        text = f"Falha de transporte ao embedar ({type(exc).__name__})."
    return _redact(text, secrets)


def _redact(text: str, secrets: Sequence[str]) -> str:
    redacted = text
    for secret in secrets:
        if len(secret) >= 8:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _credential_settings() -> Settings:
    """Lê só chave e modelo. A grade vem da CLI, não de CHUNK_* no ambiente."""
    ambient = _ambient_values()
    picked = {
        key: ambient[key]
        for key in _CREDENTIAL_KEYS
        if key in ambient and ambient[key].strip() != ""
    }
    picked["EMBEDDING_PROVIDER"] = "fake"
    return load_settings(picked)


def _ambient_values() -> dict[str, str]:
    merged: dict[str, str] = {}
    path = Path(".env")
    if path.is_file():
        for key, value in dotenv_values(path).items():
            if key is None or value is None:
                continue
            merged[key] = value
    for key, value in os.environ.items():
        merged[key] = value
    return merged


def _require_live(provider: str, allow_live: bool, credentials: Settings) -> None:
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    key = credentials.gemini_api_key if provider == "gemini" else credentials.openai_api_key
    if not allow_live:
        print(
            f"Provedor {provider} faz chamada de rede. "
            "Passe --allow-live para autorizar. Sem essa flag o comando não embeda.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not key.strip():
        print(
            f"{key_name} ausente. O provedor {provider} precisa da chave no ambiente. "
            "Nenhuma chamada foi feita.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _build_embedder(provider: str, dimension: int, credentials: Settings | None) -> Embedder:
    if provider == "fake":
        return HashingEmbedder(dimension=dimension)
    assert credentials is not None
    settings = load_settings(
        {
            "EMBEDDING_PROVIDER": provider,
            "GEMINI_API_KEY": credentials.gemini_api_key,
            "GEMINI_EMBEDDING_MODEL": credentials.gemini_embedding_model,
            "OPENAI_API_KEY": credentials.openai_api_key,
            "OPENAI_EMBEDDING_MODEL": credentials.openai_embedding_model,
            "EMBEDDING_DIMENSION": str(dimension),
        }
    )
    return build_embedder(settings)


def _model_name(provider: str, credentials: Settings | None) -> str | None:
    if credentials is None:
        return None
    if provider == "gemini":
        return credentials.gemini_embedding_model
    if provider == "openai":
        return credentials.openai_embedding_model
    return None


def _secrets(credentials: Settings | None) -> tuple[str, ...]:
    if credentials is None:
        return ()
    return tuple(
        secret
        for secret in (credentials.gemini_api_key, credentials.openai_api_key)
        if len(secret) >= 8
    )


def _parse_ints(
    raw: list[str] | None,
    *,
    default: tuple[int, ...],
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    if raw is None:
        return default
    values: list[int] = []
    for item in raw:
        for part in item.split(","):
            text = part.strip()
            if text == "":
                continue
            try:
                value = int(text)
            except ValueError as exc:
                raise ConfigError(f"{name} precisa ser um inteiro.") from exc
            if value < minimum:
                raise ConfigError(f"{name} precisa ser >= {minimum}.")
            values.append(value)
    if not values:
        raise ConfigError(f"{name} precisa de ao menos um valor.")
    return tuple(sorted(set(values)))


def _parse_floats(
    raw: list[str] | None,
    *,
    default: tuple[float, ...],
    name: str,
) -> tuple[float, ...]:
    if raw is None:
        return default
    values: list[float] = []
    for item in raw:
        for part in item.split(","):
            text = part.strip()
            if text == "":
                continue
            try:
                value = float(text)
            except ValueError as exc:
                raise ConfigError(f"{name} precisa ser um número.") from exc
            if math.isnan(value) or value < -1.0 or value > 1.0:
                raise ConfigError(f"{name} precisa estar entre -1 e 1.")
            values.append(value)
    if not values:
        raise ConfigError(f"{name} precisa de ao menos um valor.")
    return tuple(sorted(set(values)))


def _dumps_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _dumps_jsonl(rows: Sequence[Mapping[str, object]]) -> str:
    lines = (
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    return "".join(lines)


def _text(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return value


def _required_int(value: object, key: str) -> int:
    parsed = _optional_int(value, key)
    if parsed is None:
        raise ConfigError(f"Campo {key} inválido no registro.")
    return parsed


def _optional_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return value


def _optional_float(value: object, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return float(value)


def _optional_bool(value: object, key: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return value


def _ids(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError("Campo retrieved_ids inválido no registro.")
    if not all(isinstance(item, str) for item in value):
        raise ConfigError("Campo retrieved_ids inválido no registro.")
    return tuple(value)


if __name__ == "__main__":
    main()
