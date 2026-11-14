# Contrato: recall@k, MRR e contenção usam o vão inteiro, no doc_id certo.
# Sem vão o valor é nulo; zero só significa que nenhum chunk cobriu o ouro.

import pytest

from chunking_study.metrics import (
    chunks_containing_span,
    mean_defined,
    recall_at_k,
    reciprocal_rank,
    span_is_contained,
    spans_contained,
)
from chunking_study.types import Chunk, GoldSpan


def _chunk(chunk_id: str, doc_id: str, start: int, end: int, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="x" * (end - start),
        index=index,
        start=start,
        end=end,
    )


def test_span_mapping_requires_the_same_document_and_full_cover():
    host = _chunk("d:0", "d", 0, 10, index=0)
    other = _chunk("o:0", "o", 0, 10, index=0)
    span = GoldSpan("d", 2, 8)
    assert span_is_contained(host, span) is True
    assert span_is_contained(other, span) is False
    assert span_is_contained(host, GoldSpan("d", 8, 12)) is False
    assert span_is_contained(host, GoldSpan("d", 4, 4)) is False
    assert [chunk.chunk_id for chunk in chunks_containing_span([other, host], span)] == ["d:0"]


def test_recall_mrr_and_containment_on_a_ranked_list():
    first = _chunk("d:1", "d", 20, 30, index=1)
    second = _chunk("d:0", "d", 0, 10, index=0)
    span_a = GoldSpan("d", 0, 4)
    span_b = GoldSpan("d", 20, 24)
    ranked = [first, second]

    assert recall_at_k(ranked, [span_a, span_b], k=1) == 0.5
    assert recall_at_k(ranked, [span_a, span_b], k=2) == 1.0
    assert reciprocal_rank(ranked, [span_a, span_b]) == 1.0
    assert spans_contained(ranked[:1], [span_a, span_b]) is False
    assert spans_contained(ranked, [span_a, span_b]) is True

    missed = [_chunk("d:2", "d", 40, 50, index=2), first]
    assert recall_at_k(missed, [span_b], k=1) == 0.0
    assert recall_at_k(missed, [span_b], k=2) == 1.0
    assert reciprocal_rank(missed, [span_b]) == 0.5
    assert spans_contained([], [span_b]) is False
    assert reciprocal_rank([_chunk("d:9", "d", 0, 1)], [span_b]) == 0.0


def test_metrics_are_null_without_a_gold_span_and_reject_bad_k():
    chunk = _chunk("d:0", "d", 0, 5)
    assert recall_at_k([chunk], [], k=1) is None
    assert reciprocal_rank([chunk], []) is None
    assert spans_contained([chunk], []) is None
    with pytest.raises(ValueError, match="k precisa"):
        recall_at_k([chunk], [GoldSpan("d", 0, 2)], k=0)


def test_mean_keeps_zero_and_skips_null():
    assert mean_defined([1.0, 0.5, 0.0]) == 0.5
    assert mean_defined([0.0, None]) == 0.0
    assert mean_defined([None, 1.0]) == 1.0
    assert mean_defined([None, None]) is None
    assert mean_defined([]) is None
