"""Métricas de recuperação. Sem vão de ouro o valor é nulo, não zero.

Recall e MRR usam o mesmo critério de contenção: o chunk cobre o vão
inteiro no documento de origem. Match aproximado de string não conta.
"""

from collections.abc import Sequence

from chunking_study.types import Chunk, GoldSpan


def span_is_contained(chunk: Chunk, span: GoldSpan) -> bool:
    return (
        chunk.doc_id == span.doc_id
        and span.start < span.end
        and chunk.start <= span.start
        and chunk.end >= span.end
    )


def chunks_containing_span(chunks: Sequence[Chunk], span: GoldSpan) -> list[Chunk]:
    return [chunk for chunk in chunks if span_is_contained(chunk, span)]


def recall_at_k(
    chunks: Sequence[Chunk],
    spans: Sequence[GoldSpan],
    k: int,
) -> float | None:
    """Fração dos vãos cobertos por algum chunk dos `k` primeiros.

    Sem vão, devolve None. k menor que 1 é erro de chamada, não métrica.
    """
    if k < 1:
        raise ValueError("k precisa ser >= 1.")
    if not spans:
        return None
    top = list(chunks)[:k]
    hits = sum(1 for span in spans if any(span_is_contained(chunk, span) for chunk in top))
    return hits / len(spans)


def reciprocal_rank(chunks: Sequence[Chunk], spans: Sequence[GoldSpan]) -> float | None:
    """1/posição do primeiro chunk que cobre algum vão. Nenhum hit vale 0.

    O MRR do run é a média destes valores. Sem vão, devolve None.
    """
    if not spans:
        return None
    for rank, chunk in enumerate(chunks, start=1):
        if any(span_is_contained(chunk, span) for span in spans):
            return 1.0 / rank
    return 0.0


def spans_contained(chunks: Sequence[Chunk], spans: Sequence[GoldSpan]) -> bool | None:
    """Verdadeiro só se cada vão cabe inteiro em algum chunk da lista."""
    if not spans:
        return None
    return all(any(span_is_contained(chunk, span) for chunk in chunks) for span in spans)


def mean_defined(values: Sequence[float | None]) -> float | None:
    """Média que ignora nulo e conserva o zero. Lista vazia devolve None."""
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)
