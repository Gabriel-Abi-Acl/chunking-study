"""Tipos do estudo. O vão de ouro é offset no documento, não um id solto.

`start` é inclusivo e `end` é exclusivo, nos dois casos em caracteres
(índice de `str`). Sem isso, contenção e overlap não têm o mesmo eixo.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    start: int
    end: int


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Query:
    query_id: str
    question: str
    gold_spans: tuple[GoldSpan, ...]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RunRecord:
    strategy: str
    chunk_count: int
    embedded_texts: int
    query_id: str
    retrieved_ids: tuple[str, ...]
    k: int
    recall_at_k: float | None
    reciprocal_rank: float | None
    span_contained: bool | None
