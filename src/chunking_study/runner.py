"""Runner de recuperação. Grava estratégia, tamanho do índice e métricas.

`embedded_texts` soma o que o corte semântico embeda para achar a quebra
e o que o índice embeda de novo, mais as perguntas. Não é preço em dólar.
O registro é por pergunta; o MRR do run é a média de `reciprocal_rank`.
"""

from collections.abc import Sequence

from chunking_study.chunkers import build_chunker
from chunking_study.config import Settings
from chunking_study.embeddings import Embedder, build_embedder
from chunking_study.errors import ProviderError
from chunking_study.index import InMemoryVectorIndex
from chunking_study.metrics import recall_at_k, reciprocal_rank, spans_contained
from chunking_study.types import Chunk, Document, Query, RunRecord


class CountingEmbedder:
    """Conta textos enviados ao embedder interno, sem alterar o vetor."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.texts = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts += len(texts)
        return self._inner.embed(texts)


def run_retrieval(
    documents: Sequence[Document],
    queries: Sequence[Query],
    settings: Settings,
    embedder: Embedder | None = None,
) -> list[RunRecord]:
    counter = CountingEmbedder(embedder if embedder is not None else build_embedder(settings))
    chunker = build_chunker(
        settings.chunk_strategy,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        unit=settings.chunk_unit,
        embedder=counter,
        similarity_threshold=settings.semantic_similarity_threshold,
    )
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    vectors = counter.embed([chunk.text for chunk in chunks]) if chunks else []
    if len(vectors) != len(chunks):
        raise ProviderError("O embedder devolveu uma quantidade inesperada de vetores.")
    index = InMemoryVectorIndex()
    if chunks:
        index.add(chunks, vectors)

    pending: list[tuple[str, tuple[str, ...], float | None, float | None, bool | None]] = []
    for query in queries:
        query_vectors = counter.embed([query.question])
        if len(query_vectors) != 1:
            raise ProviderError("O embedder devolveu uma quantidade inesperada de vetores.")
        hits = index.query(query_vectors[0], settings.top_k) if chunks else []
        retrieved = [hit.chunk for hit in hits]
        pending.append(
            (
                query.query_id,
                tuple(chunk.chunk_id for chunk in retrieved),
                recall_at_k(retrieved, query.gold_spans, settings.top_k),
                reciprocal_rank(retrieved, query.gold_spans),
                spans_contained(retrieved, query.gold_spans),
            )
        )

    embedded_texts = counter.texts
    return [
        RunRecord(
            strategy=settings.chunk_strategy,
            chunk_count=len(chunks),
            embedded_texts=embedded_texts,
            query_id=query_id,
            retrieved_ids=retrieved_ids,
            k=settings.top_k,
            recall_at_k=recall,
            reciprocal_rank=rank,
            span_contained=contained,
        )
        for query_id, retrieved_ids, recall, rank, contained in pending
    ]


def run_record_to_dict(record: RunRecord) -> dict[str, object]:
    return {
        "strategy": record.strategy,
        "chunk_count": record.chunk_count,
        "embedded_texts": record.embedded_texts,
        "query_id": record.query_id,
        "retrieved_ids": list(record.retrieved_ids),
        "k": record.k,
        "recall_at_k": record.recall_at_k,
        "reciprocal_rank": record.reciprocal_rank,
        "span_contained": record.span_contained,
    }
