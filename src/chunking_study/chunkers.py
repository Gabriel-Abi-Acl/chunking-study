"""Quatro estratégias de corte. O offset é sempre caractere do documento.

`fixed` e `fixed_overlap` são a mesma janela; o nome da estratégia deixa
explícito se há overlap. Frase e semântico não cortam unidade que caiba
no tamanho. Unidade maior que o tamanho cai na janela de caracteres.
Abreviatura com ponto (Sr.) também fecha unidade: não é um parser.
"""

import re
from typing import Protocol

from chunking_study.embeddings import Embedder
from chunking_study.errors import ConfigError, ProviderError
from chunking_study.index import cosine_similarity
from chunking_study.types import Chunk, Document

STRATEGIES = ("fixed", "fixed_overlap", "sentence_paragraph", "semantic")
UNITS = ("chars", "tokens")

_TOKEN = re.compile(r"\S+")


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


class _PieceGroup:
    def __init__(self, pieces: list[tuple[int, int]], *, skip_overlap: bool) -> None:
        self.pieces = pieces
        self.skip_overlap = skip_overlap


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Unidades de frase ou parágrafo. Concatenadas, reconstroem o texto.

    O ponto, a interrogação e a exclamação levam o espaço ou a quebra de
    parágrafo seguinte. Quebra de parágrafo sem pontuação também fecha.
    """
    if text == "":
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in ".!?":
            end = _consume_sentence_end(text, index)
            spans.append((start, end))
            start = end
            index = end
            continue
        if char == "\n":
            end = index
            while end < length and text[end] == "\n":
                end += 1
            if end - index >= 2:
                spans.append((start, end))
                start = end
                index = end
                continue
        index += 1
    if start < length:
        spans.append((start, length))
    return spans


class FixedSizeChunker:
    """Janela fixa em caracteres ou tokens, com passo `size - overlap`.

    Token conta `\\S+`. O espaço entre janelas fica no chunk anterior, para
    overlap 0 ainda cobrir o texto inteiro. O tamanho é em tokens; o
    offset gravado continua em caracteres.
    """

    def __init__(self, size: int, overlap: int = 0, unit: str = "chars") -> None:
        _validate_window(size, overlap)
        if unit not in UNITS:
            options = ", ".join(UNITS)
            raise ConfigError(f"CHUNK_UNIT inválida: {unit}. Use: {options}.")
        self.size = size
        self.overlap = overlap
        self.unit = unit

    def chunk(self, document: Document) -> list[Chunk]:
        if self.unit == "chars":
            spans = _character_spans(0, len(document.text), self.size, self.overlap)
        else:
            spans = _token_spans(document.text, self.size, self.overlap)
        return _chunks_from_spans(document, spans)


class SentenceParagraphChunker:
    """Empilha frases até `size` caracteres. Overlap repete frases inteiras.

    Não corta frase para completar o overlap. Se a frase anterior mais a
    nova não cabem em `size`, o overlap dessa fronteira fica de fora.
    """

    def __init__(self, size: int, overlap: int = 0) -> None:
        _validate_window(size, overlap)
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        spans = _pack(sentence_spans(document.text), self.size, self.overlap, break_before=set())
        return _chunks_from_spans(document, spans)


class SemanticChunker:
    """Quebra onde o cosseno entre frases vizinhas cai abaixo do limiar.

    A comparação é estrita (`similaridade < limiar`). As frases são
    embeddadas uma a uma, sem janela de contexto. O `size` ainda limita o
    chunk; overlap, se couber, repete a frase da fronteira nos dois lados.
    """

    def __init__(
        self,
        embedder: Embedder,
        size: int,
        overlap: int = 0,
        similarity_threshold: float = 0.5,
    ) -> None:
        _validate_window(size, overlap)
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ConfigError("SEMANTIC_SIMILARITY_THRESHOLD precisa estar entre -1 e 1.")
        self.embedder = embedder
        self.size = size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:
        units = sentence_spans(document.text)
        if not units:
            return []
        texts = [document.text[start:end] for start, end in units]
        vectors = self.embedder.embed(texts)
        if len(vectors) != len(texts):
            raise ProviderError("O embedder devolveu uma quantidade inesperada de vetores.")
        breaks: set[int] = set()
        for index in range(1, len(units)):
            similarity = cosine_similarity(vectors[index - 1], vectors[index])
            if similarity < self.similarity_threshold:
                breaks.add(index)
        spans = _pack(units, self.size, self.overlap, break_before=breaks)
        return _chunks_from_spans(document, spans)


def build_chunker(
    strategy: str,
    *,
    size: int,
    overlap: int,
    unit: str,
    embedder: Embedder | None = None,
    similarity_threshold: float = 0.5,
) -> Chunker:
    """`fixed` força overlap 0. As outras estratégias usam o overlap recebido."""
    if strategy not in STRATEGIES:
        options = ", ".join(STRATEGIES)
        raise ConfigError(f"CHUNK_STRATEGY inválida: {strategy}. Use: {options}.")
    _validate_window(size, overlap)
    if strategy in ("sentence_paragraph", "semantic") and unit != "chars":
        raise ConfigError("CHUNK_UNIT=tokens só vale para fixed e fixed_overlap.")
    if strategy == "fixed":
        return FixedSizeChunker(size=size, overlap=0, unit=unit)
    if strategy == "fixed_overlap":
        if overlap < 1:
            raise ConfigError("CHUNK_OVERLAP precisa ser >= 1 na estratégia fixed_overlap.")
        return FixedSizeChunker(size=size, overlap=overlap, unit=unit)
    if strategy == "sentence_paragraph":
        return SentenceParagraphChunker(size=size, overlap=overlap)
    if embedder is None:
        raise ConfigError("A estratégia semantic precisa de um embedder.")
    return SemanticChunker(
        embedder=embedder,
        size=size,
        overlap=overlap,
        similarity_threshold=similarity_threshold,
    )


def _validate_window(size: int, overlap: int) -> None:
    if size < 1:
        raise ConfigError("CHUNK_SIZE precisa ser >= 1.")
    if overlap < 0 or overlap >= size:
        raise ConfigError("CHUNK_OVERLAP precisa ser >= 0 e menor que CHUNK_SIZE.")


def _consume_sentence_end(text: str, index: int) -> int:
    end = index + 1
    length = len(text)
    while end < length and text[end] in ".!?":
        end += 1
    while end < length and text[end] in " \t":
        end += 1
    if end < length and text[end] == "\n":
        while end < length and text[end] == "\n":
            end += 1
    return end


def _character_spans(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if start >= end:
        return []
    spans: list[tuple[int, int]] = []
    cursor = start
    step = size - overlap
    while cursor < end:
        window_end = min(cursor + size, end)
        spans.append((cursor, window_end))
        if window_end == end:
            break
        cursor += step
    return spans


def _token_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    if text == "":
        return []
    matches = list(_TOKEN.finditer(text))
    if not matches:
        return [(0, len(text))]
    step = size - overlap
    spans: list[tuple[int, int]] = []
    start_index = 0
    count = len(matches)
    while start_index < count:
        end_index = min(start_index + size, count)
        start = 0 if start_index == 0 else matches[start_index].start()
        end = len(text) if end_index == count else matches[end_index].start()
        spans.append((start, end))
        if end_index == count:
            break
        start_index += step
    return spans


def _pack(
    units: list[tuple[int, int]],
    size: int,
    overlap: int,
    *,
    break_before: set[int],
) -> list[tuple[int, int]]:
    groups: list[_PieceGroup] = []
    current: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(_PieceGroup(current, skip_overlap=False))
            current = []

    for index, unit in enumerate(units):
        if index in break_before and current:
            flush()
        unit_length = unit[1] - unit[0]
        if unit_length > size:
            flush()
            for window in _character_spans(unit[0], unit[1], size, overlap):
                groups.append(_PieceGroup([window], skip_overlap=True))
            continue
        if not current:
            current = [unit]
            continue
        if unit[1] - current[0][0] <= size:
            current.append(unit)
            continue
        flush()
        current = [unit]
    flush()
    return _apply_overlap(groups, size, overlap)


def _apply_overlap(
    groups: list[_PieceGroup],
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if not groups:
        return []
    originals = [group.pieces for group in groups]
    emitted: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        pieces = group.pieces
        if index == 0 or overlap <= 0 or group.skip_overlap:
            emitted.append((pieces[0][0], pieces[-1][1]))
            continue
        tail = _tail_within(originals[index - 1], overlap)
        merged = tail + pieces
        while len(merged) > len(pieces) and merged[-1][1] - merged[0][0] > size:
            merged.pop(0)
        start, end = merged[0][0], merged[-1][1]
        if emitted and end <= emitted[-1][1]:
            raise RuntimeError("o chunker não avançou no documento.")
        emitted.append((start, end))
    return emitted


def _tail_within(spans: list[tuple[int, int]], overlap: int) -> list[tuple[int, int]]:
    if overlap <= 0 or not spans:
        return []
    tail: list[tuple[int, int]] = []
    for span in reversed(spans):
        candidate = [span, *tail]
        if candidate[-1][1] - candidate[0][0] > overlap:
            break
        tail = candidate
    return tail


def _chunks_from_spans(document: Document, spans: list[tuple[int, int]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        if end <= start:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{index}",
                doc_id=document.doc_id,
                text=document.text[start:end],
                index=index,
                start=start,
                end=end,
            )
        )
    return chunks
