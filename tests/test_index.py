# Contrato: top-k por cosseno. Empate desempata por índice e depois por id.

import pytest

from chunking_study.index import InMemoryVectorIndex, cosine_similarity
from chunking_study.types import Chunk


def _chunk(chunk_id: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split(":")[0], text=chunk_id, index=index, start=0, end=1
    )


def test_cosine_of_identical_orthogonal_and_zero_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="dimens"):
        cosine_similarity([1.0], [1.0, 0.0])


def test_query_ranks_by_cosine_and_breaks_ties_stably():
    index = InMemoryVectorIndex()
    later = _chunk("b:1", 1)
    earlier = _chunk("b:0", 0)
    index.add([later, earlier], [[1.0, 0.0], [1.0, 0.0]])
    same = index.query([1.0, 0.0], top_k=2)
    assert [hit.chunk.chunk_id for hit in same] == ["b:0", "b:1"]
    assert same[0].score == pytest.approx(1.0)

    ranked = InMemoryVectorIndex()
    ranked.add(
        [_chunk("a:0", 0), _chunk("a:1", 1)],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    hits = ranked.query([1.0, 0.0], top_k=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["a:0", "a:1"]
    assert hits[0].score > hits[1].score


def test_same_index_tie_uses_chunk_id():
    index = InMemoryVectorIndex()
    index.add([_chunk("b:0", 0), _chunk("a:0", 0)], [[1.0, 0.0], [1.0, 0.0]])
    hits = index.query([1.0, 0.0], top_k=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["a:0", "b:0"]


def test_top_k_limits_the_result_and_rejects_bad_input():
    index = InMemoryVectorIndex()
    index.add([_chunk("a:0", 0), _chunk("a:1", 1)], [[1.0, 0.0], [0.6, 0.8]])
    assert len(index.query([1.0, 0.0], top_k=5)) == 2
    assert [hit.chunk.index for hit in index.query([1.0, 0.0], top_k=1)] == [0]
    assert InMemoryVectorIndex().query([1.0, 0.0], top_k=3) == []
    with pytest.raises(ValueError, match="top_k"):
        index.query([1.0, 0.0], top_k=0)
    with pytest.raises(ValueError, match="dimensão"):
        index.query([1.0], top_k=1)
    with pytest.raises(ValueError, match="dimensão"):
        index.add([_chunk("a:2", 2)], [[1.0]])
    with pytest.raises(ValueError, match="mesmo tamanho"):
        InMemoryVectorIndex().add([_chunk("a:0", 0)], [])


def test_add_copies_the_vector():
    index = InMemoryVectorIndex()
    vector = [1.0, 0.0]
    index.add([_chunk("a:0", 0)], [vector])
    vector[0] = 0.0
    assert index.query([1.0, 0.0], top_k=1)[0].score == pytest.approx(1.0)
