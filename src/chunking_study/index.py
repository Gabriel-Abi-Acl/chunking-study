"""Índice em memória com cosseno. O corpus deste PR cabe em RAM.

A mesma função de cosseno decide o top-k e a quebra semântica, para o
limiar e o ranking não usarem geometrias diferentes. Empate desempata por
índice e id: a ordem não depende da inserção.
"""

import math
from collections.abc import Sequence

from chunking_study.types import Chunk, ScoredChunk


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, tuple[float, ...]]] = []
        self._dimension: int | None = None

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks e vetores precisam ter o mesmo tamanho.")
        for chunk, vector in zip(chunks, vectors, strict=True):
            values = tuple(float(value) for value in vector)
            if self._dimension is None:
                self._dimension = len(values)
            if len(values) != self._dimension:
                raise ValueError("Todos os vetores do índice precisam ter a mesma dimensão.")
            self._items.append((chunk, values))

    def query(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        if top_k < 1:
            raise ValueError("top_k precisa ser >= 1.")
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError("A consulta tem dimensão diferente do índice.")
        scored = [
            ScoredChunk(chunk=chunk, score=cosine_similarity(stored, vector))
            for chunk, stored in self._items
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk.index, item.chunk.chunk_id))
        return scored[:top_k]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vetores com dimensões diferentes.")
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        norm_left += left_value * left_value
        norm_right += right_value * right_value
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))
