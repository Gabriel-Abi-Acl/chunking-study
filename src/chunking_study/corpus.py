"""Documentos `.txt`/`.md` e JSONL de perguntas. O doc_id é o nome do arquivo.

O placeholder busca a passagem e recusa ocorrência ambígua. O eval v1
exige `start`/`end` e confere o recorte com o texto do documento.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from chunking_study.errors import ConfigError
from chunking_study.types import Document, GoldSpan, Query


def load_corpus(path: Path) -> list[Document]:
    if not path.is_dir():
        raise ConfigError(f"CORPUS_PATH não é um diretório: {path}")
    files = sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".txt", ".md"}
    )
    documents: list[Document] = []
    seen: set[str] = set()
    for file in files:
        doc_id = file.stem
        if doc_id in seen:
            raise ConfigError(f"doc_id repetido: {doc_id}")
        seen.add(doc_id)
        documents.append(
            Document(
                doc_id=doc_id,
                text=file.read_text(encoding="utf-8"),
                source=str(file),
            )
        )
    return documents


def load_queries(path: Path, documents: Sequence[Document]) -> list[Query]:
    if not path.is_file():
        raise ConfigError(f"QUERIES_PATH não é um arquivo: {path}")
    by_id = {document.doc_id: document for document in documents}
    queries: list[Query] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON inválido na linha {line_number} de perguntas.") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"A linha {line_number} de perguntas precisa ser um objeto.")
        query_id = _text_field(raw, "id", line_number)
        if query_id in seen:
            raise ConfigError(f"id de pergunta repetido: {query_id}")
        seen.add(query_id)
        question = _text_field(raw, "question", line_number)
        passages = raw.get("passages")
        if not isinstance(passages, list) or not passages:
            raise ConfigError(f"passages vazio na linha {line_number}.")
        spans: list[GoldSpan] = []
        for passage in passages:
            spans.append(_span_from_passage(passage, by_id, line_number))
        queries.append(Query(query_id=query_id, question=question, gold_spans=tuple(spans)))
    return queries


def _span_from_passage(passage: object, by_id: dict[str, Document], line_number: int) -> GoldSpan:
    if not isinstance(passage, dict):
        raise ConfigError(f"passagem inválida na linha {line_number}.")
    doc_id = passage.get("doc_id")
    text = passage.get("passage")
    if not isinstance(doc_id, str) or doc_id.strip() == "":
        raise ConfigError(f"doc_id ausente na linha {line_number}.")
    if not isinstance(text, str) or text == "":
        raise ConfigError(f"passage vazia na linha {line_number}.")
    document = by_id.get(doc_id)
    if document is None:
        raise ConfigError(f"doc_id desconhecido na linha {line_number}: {doc_id}.")
    start = document.text.find(text)
    if start < 0:
        raise ConfigError(f"passagem ausente em {doc_id} (linha {line_number}).")
    if document.text.find(text, start + 1) != -1:
        raise ConfigError(f"passagem ambígua em {doc_id} (linha {line_number}).")
    return GoldSpan(doc_id=doc_id, start=start, end=start + len(text))


def _text_field(raw: dict[str, object], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ConfigError(f"Campo {key} ausente na linha {line_number} de perguntas.")
    return value.strip()


# Rótulos do gabarito v1. O teste exige as quatro categorias, equilibradas.
EVAL_QUERY_CATEGORIES = (
    "single_sentence",
    "two_sentences",
    "paragraph_boundary",
    "paraphrase",
)


# Corpus versionado da avaliação. Documento vazio não serve de gabarito.
# Reusa a leitura do placeholder e acrescenta a recusa de texto em branco.
def load_eval_corpus(path: Path) -> list[Document]:
    documents = load_corpus(path)
    if not documents:
        raise ConfigError(f"CORPUS_PATH não tem documentos: {path}")
    for document in documents:
        if document.text.strip() == "":
            raise ConfigError(f"documento vazio: {document.doc_id}")
    return documents


# Perguntas do eval: o offset vem no JSONL e tem de bater com o texto.
# Vão vazio, fora do documento ou diferente da passagem recusa o arquivo.
def load_eval_queries(path: Path, documents: Sequence[Document]) -> list[Query]:
    if not path.is_file():
        raise ConfigError(f"QUERIES_PATH não é um arquivo: {path}")
    by_id = {document.doc_id: document for document in documents}
    queries: list[Query] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON inválido na linha {line_number} de perguntas.") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"A linha {line_number} de perguntas precisa ser um objeto.")
        query_id = _text_field(raw, "id", line_number)
        if query_id in seen:
            raise ConfigError(f"id de pergunta repetido: {query_id}")
        seen.add(query_id)
        question = _text_field(raw, "question", line_number)
        category = _text_field(raw, "category", line_number)
        if category not in EVAL_QUERY_CATEGORIES:
            allowed = ", ".join(EVAL_QUERY_CATEGORIES)
            raise ConfigError(
                f"categoria inválida na linha {line_number}: {category}. Use: {allowed}."
            )
        passages = raw.get("passages")
        if not isinstance(passages, list) or not passages:
            raise ConfigError(f"passages vazio na linha {line_number}.")
        spans: list[GoldSpan] = []
        seen_spans: set[tuple[str, int, int]] = set()
        for passage in passages:
            span = _eval_span(passage, by_id, line_number)
            key = (span.doc_id, span.start, span.end)
            if key in seen_spans:
                raise ConfigError(f"vão repetido na linha {line_number}.")
            seen_spans.add(key)
            spans.append(span)
        queries.append(
            Query(
                query_id=query_id,
                question=question,
                gold_spans=tuple(spans),
                category=category,
            )
        )
    if not queries:
        raise ConfigError(f"QUERIES_PATH não tem perguntas: {path}")
    return queries


def _eval_span(passage: object, by_id: dict[str, Document], line_number: int) -> GoldSpan:
    if not isinstance(passage, dict):
        raise ConfigError(f"passagem inválida na linha {line_number}.")
    doc_id = passage.get("doc_id")
    text = passage.get("passage")
    if not isinstance(doc_id, str) or doc_id.strip() == "":
        raise ConfigError(f"doc_id ausente na linha {line_number}.")
    if not isinstance(text, str) or text.strip() == "":
        raise ConfigError(f"passage vazia na linha {line_number}.")
    document = by_id.get(doc_id)
    if document is None:
        raise ConfigError(f"doc_id desconhecido na linha {line_number}: {doc_id}.")
    start = _offset(passage.get("start"), "start", line_number)
    end = _offset(passage.get("end"), "end", line_number)
    if start < 0 or end > len(document.text) or start >= end:
        raise ConfigError(f"vão fora do documento {doc_id} na linha {line_number}.")
    if document.text[start:end] != text:
        raise ConfigError(f"o vão da linha {line_number} não coincide com o texto de {doc_id}.")
    return GoldSpan(doc_id=doc_id, start=start, end=end)


def _offset(value: object, key: str, line_number: int) -> int:
    # bool é int em Python; True no JSON não pode virar offset 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} inválido na linha {line_number}.")
    return value
