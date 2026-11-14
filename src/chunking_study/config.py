"""Configuração lida de variáveis de ambiente. Chave nunca entra no repr.

`load_settings(env)` ignora o processo e o arquivo .env de propósito: o
teste passa um mapa e não pode vazar credencial do ambiente.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from chunking_study.errors import ConfigError

EMBEDDING_PROVIDERS = ("fake", "gemini", "openai")
CHUNK_STRATEGIES = ("fixed", "fixed_overlap", "sentence_paragraph", "semantic")
CHUNK_UNITS = ("chars", "tokens")

ENV_KEYS = (
    "EMBEDDING_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_EMBEDDING_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "CHUNK_STRATEGY",
    "CHUNK_UNIT",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "SEMANTIC_SIMILARITY_THRESHOLD",
    "TOP_K",
    "EMBEDDING_DIMENSION",
    "CORPUS_PATH",
    "QUERIES_PATH",
)


@dataclass(frozen=True)
class Settings:
    embedding_provider: str
    gemini_api_key: str = field(repr=False)
    gemini_embedding_model: str
    openai_api_key: str = field(repr=False)
    openai_embedding_model: str
    chunk_strategy: str
    chunk_unit: str
    chunk_size: int
    chunk_overlap: int
    semantic_similarity_threshold: float
    top_k: int
    embedding_dimension: int
    corpus_path: Path
    queries_path: Path


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> Settings:
    """Monta Settings. Com `env` explícito, o .env não entra no resultado."""
    source = _resolve_source(env, dotenv_path)
    chunk_size = _integer(source, "CHUNK_SIZE", default=120, minimum=1)
    chunk_overlap = _integer(source, "CHUNK_OVERLAP", default=0, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP precisa ser >= 0 e menor que CHUNK_SIZE.")
    strategy = _choice(source, "CHUNK_STRATEGY", "fixed", CHUNK_STRATEGIES)
    if strategy == "fixed_overlap" and chunk_overlap < 1:
        raise ConfigError("CHUNK_OVERLAP precisa ser >= 1 na estratégia fixed_overlap.")
    unit = _choice(source, "CHUNK_UNIT", "chars", CHUNK_UNITS)
    if strategy in ("sentence_paragraph", "semantic") and unit != "chars":
        raise ConfigError("CHUNK_UNIT=tokens só vale para fixed e fixed_overlap.")
    return Settings(
        embedding_provider=_choice(source, "EMBEDDING_PROVIDER", "fake", EMBEDDING_PROVIDERS),
        gemini_api_key=_text(source, "GEMINI_API_KEY", ""),
        gemini_embedding_model=_text(source, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        openai_api_key=_text(source, "OPENAI_API_KEY", ""),
        openai_embedding_model=_text(source, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_strategy=strategy,
        chunk_unit=unit,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        semantic_similarity_threshold=_number(
            source,
            "SEMANTIC_SIMILARITY_THRESHOLD",
            default=0.5,
            minimum=-1.0,
            maximum=1.0,
        ),
        top_k=_integer(source, "TOP_K", default=3, minimum=1),
        embedding_dimension=_integer(source, "EMBEDDING_DIMENSION", default=64, minimum=1),
        corpus_path=Path(_text(source, "CORPUS_PATH", "data/placeholder")),
        queries_path=Path(_text(source, "QUERIES_PATH", "data/placeholder/queries.jsonl")),
    )


def _resolve_source(
    env: Mapping[str, str] | None,
    dotenv_path: Path | None,
) -> Mapping[str, str]:
    if env is not None:
        return env
    merged = dict(os.environ)
    path = Path(".env") if dotenv_path is None else dotenv_path
    if path.is_file():
        # Valor já exportado no processo ganha do arquivo.
        for key, value in dotenv_values(path).items():
            if value is not None:
                merged.setdefault(key, value)
    return merged


def _text(source: Mapping[str, str], key: str, default: str) -> str:
    raw = source.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def _choice(
    source: Mapping[str, str],
    key: str,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    value = _text(source, key, default).lower()
    if value not in allowed:
        options = ", ".join(allowed)
        raise ConfigError(f"{key} inválida: {value}. Use: {options}.")
    return value


def _number(
    source: Mapping[str, str],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = source.get(key)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = float(str(raw).strip())
        except ValueError as exc:
            raise ConfigError(f"{key} precisa ser um número.") from exc
    if value < minimum or value > maximum:
        raise ConfigError(f"{key} precisa estar entre {minimum} e {maximum}.")
    return value


def _integer(
    source: Mapping[str, str],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    raw = source.get(key)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise ConfigError(f"{key} precisa ser um inteiro.") from exc
    if value < minimum:
        raise ConfigError(f"{key} precisa ser >= {minimum}.")
    return value
