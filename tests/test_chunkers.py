# Contrato: cada estratégia corta no eixo de caracteres do documento.
# Overlap 0 cobre o texto; overlap não pode deixar o cursor parado.

from pathlib import Path

import pytest

from chunking_study.chunkers import (
    STRATEGIES,
    FixedSizeChunker,
    SemanticChunker,
    SentenceParagraphChunker,
    build_chunker,
    sentence_spans,
)
from chunking_study.config import CHUNK_STRATEGIES, CHUNK_UNITS
from chunking_study.corpus import load_corpus
from chunking_study.embeddings import HashingEmbedder
from chunking_study.errors import ConfigError
from chunking_study.index import cosine_similarity
from chunking_study.metrics import chunks_containing_span, spans_contained
from chunking_study.types import Document, GoldSpan

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _doc(text: str, doc_id: str = "d") -> Document:
    return Document(doc_id=doc_id, text=text, source=f"{doc_id}.txt")


def test_strategy_names_match_config():
    assert STRATEGIES == CHUNK_STRATEGIES
    from chunking_study.chunkers import UNITS

    assert UNITS == CHUNK_UNITS


@pytest.mark.parametrize(
    "text",
    [
        "",
        "sem ponto final",
        "Aa. Bb. Cc.",
        "Aa. Bb.\n\nCc.\n",
        "àéí. ó.\n",
        "a" * 25,
        "linha\núnica sem ponto",
        "\n\nAa.",
        "Sério?!",
    ],
)
def test_sentence_spans_partition_the_text(text: str):
    spans = sentence_spans(text)
    assert "".join(text[start:end] for start, end in spans) == text
    cursor = 0
    for start, end in spans:
        assert start == cursor
        assert end > start
        cursor = end


def test_fixed_chars_boundaries_and_exact_overlap():
    document = _doc(ALPHABET)
    plain = FixedSizeChunker(size=10, overlap=0, unit="chars").chunk(document)
    assert [chunk.text for chunk in plain] == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert [chunk.start for chunk in plain] == [0, 10, 20]
    assert "".join(chunk.text for chunk in plain) == ALPHABET

    overlapped = FixedSizeChunker(size=10, overlap=4, unit="chars").chunk(document)
    assert [chunk.text for chunk in overlapped] == [
        "abcdefghij",
        "ghijklmnop",
        "mnopqrstuv",
        "stuvwxyz",
    ]
    assert [chunk.start for chunk in overlapped] == [0, 6, 12, 18]
    assert overlapped[1].text[:4] == overlapped[0].text[-4:]


def test_span_across_a_fixed_boundary_needs_overlap_to_be_contained():
    document = _doc(ALPHABET)
    span = GoldSpan(doc_id="d", start=8, end=14)
    plain = FixedSizeChunker(size=10, overlap=0).chunk(document)
    overlapped = FixedSizeChunker(size=10, overlap=4).chunk(document)
    assert spans_contained(plain, [span]) is False
    assert [chunk.chunk_id for chunk in chunks_containing_span(overlapped, span)] == ["d:1"]
    assert overlapped[1].text[
        span.start - overlapped[1].start : span.end - overlapped[1].start
    ] == ("ijklmn")


def test_fixed_token_windows_keep_character_offsets():
    text = "um dois tres quatro cinco"
    document = _doc(text)
    plain = FixedSizeChunker(size=2, overlap=0, unit="tokens").chunk(document)
    assert [chunk.text for chunk in plain] == ["um dois ", "tres quatro ", "cinco"]
    assert "".join(chunk.text for chunk in plain) == text

    overlapped = FixedSizeChunker(size=3, overlap=1, unit="tokens").chunk(document)
    assert [chunk.text for chunk in overlapped] == [
        "um dois tres ",
        "tres quatro cinco",
    ]
    assert "tres" in overlapped[0].text and overlapped[1].text.startswith("tres")

    crossing = GoldSpan(doc_id="d", start=3, end=12)
    assert spans_contained(plain, [crossing]) is False
    assert spans_contained(overlapped, [crossing]) is True


def test_fixed_chars_counts_code_points_and_short_tail():
    chunks = FixedSizeChunker(size=3, overlap=0).chunk(_doc("ààààà"))
    assert [chunk.text for chunk in chunks] == ["ààà", "àà"]
    assert "".join(chunk.text for chunk in chunks) == "ààààà"


def test_overlap_near_the_window_still_advances():
    chunks = FixedSizeChunker(size=10, overlap=9).chunk(_doc("a" * 1000))
    assert len(chunks) == 991
    assert chunks[1].start == 1
    assert chunks[-1].end == 1000


def test_empty_and_whitespace_inputs():
    empty = _doc("")
    assert FixedSizeChunker(10, 0).chunk(empty) == []
    assert FixedSizeChunker(10, 0, unit="tokens").chunk(empty) == []
    assert SentenceParagraphChunker(10, 0).chunk(empty) == []
    spaces = FixedSizeChunker(3, 0, unit="tokens").chunk(_doc("   "))
    assert [chunk.text for chunk in spaces] == ["   "]


def test_sentence_boundaries_keep_the_paragraph_break_with_the_paragraph():
    text = "Aa. Bb.\n\nCc.\n"
    chunks = SentenceParagraphChunker(size=9, overlap=0).chunk(_doc(text))
    assert [chunk.text for chunk in chunks] == ["Aa. Bb.\n\n", "Cc.\n"]
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)

    packed = SentenceParagraphChunker(size=100, overlap=0).chunk(_doc("Aa. Bb. Cc."))
    assert [chunk.text for chunk in packed] == ["Aa. Bb. Cc."]


def test_sentence_overlap_repeats_a_whole_sentence():
    text = "Aa. Bb. Cc."
    chunks = SentenceParagraphChunker(size=8, overlap=4).chunk(_doc(text))
    assert [chunk.text for chunk in chunks] == ["Aa. Bb. ", "Bb. Cc."]
    assert chunks[1].start < chunks[0].end
    assert chunks[1].end > chunks[0].end


def test_sentence_longer_than_size_falls_back_to_character_windows():
    text = "a" * 25
    plain = SentenceParagraphChunker(size=10, overlap=0).chunk(_doc(text))
    assert [chunk.text for chunk in plain] == ["a" * 10, "a" * 10, "a" * 5]
    overlapped = SentenceParagraphChunker(size=10, overlap=4).chunk(_doc(text))
    assert [chunk.start for chunk in overlapped] == [0, 6, 12, 18]
    assert overlapped[-1].end == 25


def test_overlap_zero_partitions_the_placeholder_corpus():
    documents = load_corpus(ROOT / "data" / "placeholder")
    sentence = SentenceParagraphChunker(size=80, overlap=0)
    semantic = SemanticChunker(HashingEmbedder(64), size=80, overlap=0, similarity_threshold=0.0)
    fixed = FixedSizeChunker(size=80, overlap=0, unit="chars")
    tokens = FixedSizeChunker(size=12, overlap=0, unit="tokens")
    for document in documents:
        for chunker in (sentence, semantic, fixed, tokens):
            chunks = chunker.chunk(document)
            assert chunks
            assert "".join(chunk.text for chunk in chunks) == document.text
            assert all(chunk.text == document.text[chunk.start : chunk.end] for chunk in chunks)


def test_semantic_splits_only_where_similarity_drops():
    text = (
        "O rio transborda na vila durante a chuva. "
        "O rio transborda na vila com a chuva forte. "
        "A oficina de bicicletas fecha as quartas."
    )
    spans = sentence_spans(text)
    assert len(spans) == 3
    embedder = HashingEmbedder(dimension=256)
    vectors = embedder.embed([text[start:end] for start, end in spans])
    sim_near = cosine_similarity(vectors[0], vectors[1])
    sim_far = cosine_similarity(vectors[1], vectors[2])
    assert sim_near > sim_far, (sim_near, sim_far)
    assert sim_near < 1.0
    threshold = (sim_near + sim_far) / 2
    document = _doc(text)
    chunks = SemanticChunker(
        embedder,
        size=10_000,
        overlap=0,
        similarity_threshold=threshold,
    ).chunk(document)
    assert [chunk.text for chunk in chunks] == [
        text[spans[0][0] : spans[1][1]],
        text[spans[2][0] : spans[2][1]],
    ]

    kept = SemanticChunker(embedder, size=10_000, overlap=0, similarity_threshold=-1.0).chunk(
        document
    )
    assert [chunk.text for chunk in kept] == [text]
    split = SemanticChunker(embedder, size=10_000, overlap=0, similarity_threshold=1.0).chunk(
        document
    )
    assert len(split) == 3


def test_semantic_overlap_repeats_the_boundary_sentence():
    text = (
        "O rio transborda na vila durante a chuva. "
        "O rio transborda na vila com a chuva forte. "
        "A oficina de bicicletas fecha as quartas."
    )
    spans = sentence_spans(text)
    embedder = HashingEmbedder(dimension=256)
    vectors = embedder.embed([text[start:end] for start, end in spans])
    threshold = (
        cosine_similarity(vectors[0], vectors[1]) + cosine_similarity(vectors[1], vectors[2])
    ) / 2
    second = spans[1][1] - spans[1][0]
    chunks = SemanticChunker(
        embedder,
        size=len(text),
        overlap=second,
        similarity_threshold=threshold,
    ).chunk(_doc(text))
    assert len(chunks) == 2
    assert chunks[1].text.startswith(text[spans[1][0] : spans[1][1]])
    assert text[spans[2][0] : spans[2][1]] in chunks[1].text


def test_semantic_embeds_each_sentence_unit():
    text = "Aa. Bb."
    seen: list[list[str]] = []

    class RecordingEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            seen.append(list(texts))
            return HashingEmbedder(32).embed(texts)

    SemanticChunker(RecordingEmbedder(), size=100, overlap=0, similarity_threshold=-1.0).chunk(
        _doc(text)
    )
    assert seen == [[text[start:end] for start, end in sentence_spans(text)]]


def test_semantic_does_not_embed_an_empty_document():
    class Boom:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError(texts)

    assert SemanticChunker(Boom(), size=10).chunk(_doc("")) == []


def test_fixed_strategy_ignores_overlap_and_names_are_explicit():
    chunker = build_chunker("fixed", size=10, overlap=4, unit="chars")
    chunks = chunker.chunk(_doc(ALPHABET))
    assert [chunk.start for chunk in chunks] == [0, 10, 20]

    tokens = build_chunker("fixed_overlap", size=3, overlap=1, unit="tokens")
    assert [chunk.text for chunk in tokens.chunk(_doc("um dois tres quatro cinco"))] == [
        "um dois tres ",
        "tres quatro cinco",
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"size": 0, "overlap": 0, "unit": "chars"}, "CHUNK_SIZE"),
        ({"size": 10, "overlap": 10, "unit": "chars"}, "CHUNK_OVERLAP"),
        ({"size": 10, "overlap": -1, "unit": "chars"}, "CHUNK_OVERLAP"),
        ({"strategy": "fixed_overlap", "size": 10, "overlap": 0, "unit": "chars"}, "fixed_overlap"),
        (
            {"strategy": "sentence_paragraph", "size": 10, "overlap": 0, "unit": "tokens"},
            "tokens",
        ),
        ({"strategy": "semantic", "size": 10, "overlap": 0, "unit": "chars"}, "embedder"),
        ({"strategy": "nope", "size": 10, "overlap": 0, "unit": "chars"}, "CHUNK_STRATEGY"),
    ],
)
def test_build_chunker_rejects_bad_windows(kwargs: dict[str, object], match: str):
    payload = {"strategy": "fixed", "embedder": None, "similarity_threshold": 0.5}
    payload.update(kwargs)
    with pytest.raises(ConfigError, match=match):
        build_chunker(**payload)  # type: ignore[arg-type]


def test_chunk_ids_follow_document_order():
    chunks = FixedSizeChunker(size=4, overlap=0).chunk(_doc("abcdefgh", doc_id="nota"))
    assert [chunk.chunk_id for chunk in chunks] == ["nota:0", "nota:1"]
    assert [chunk.index for chunk in chunks] == [0, 1]
