# Contrato: o eval v1 confere offset com o texto e separa as quatro categorias.
# Contagem e forma do vão são invariantes do arquivo, não métrica de recuperação.

import json
import re
from pathlib import Path

import pytest

from chunking_study.chunkers import FixedSizeChunker, sentence_spans
from chunking_study.corpus import (
    EVAL_QUERY_CATEGORIES,
    load_corpus,
    load_eval_corpus,
    load_eval_queries,
    load_queries,
)
from chunking_study.errors import ConfigError
from chunking_study.metrics import spans_contained

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval" / "v1"

# Funcionais de superfície. A razão usa só o que sobra na pergunta.
STOPWORDS = frozenset(
    {
        "para",
        "como",
        "pela",
        "pelo",
        "pelas",
        "pelos",
        "sobre",
        "entre",
        "desde",
        "após",
        "antes",
        "depois",
        "durante",
        "quando",
        "onde",
        "qual",
        "quais",
        "quem",
        "quanto",
        "quantos",
        "quantas",
        "porque",
        "porquê",
        "cada",
        "mais",
        "menos",
        "muito",
        "pouco",
        "também",
        "ainda",
        "apenas",
        "somente",
        "outro",
        "outra",
        "outros",
        "outras",
        "mesmo",
        "mesma",
        "esse",
        "essa",
        "este",
        "esta",
        "isso",
        "isto",
        "aquele",
        "aquela",
        "seus",
        "suas",
        "seu",
        "sua",
        "numa",
        "nas",
        "nos",
        "das",
        "dos",
        "uma",
        "uns",
        "umas",
        "são",
        "está",
        "estão",
        "foi",
        "foram",
        "tem",
        "têm",
        "pode",
        "podem",
        "ser",
        "ter",
        "há",
        "não",
        "sim",
        "sem",
        "sob",
        "até",
        "com",
        "todo",
        "toda",
        "todos",
        "todas",
        "já",
        "então",
    }
)


_TOKEN = re.compile(r"[0-9a-zà-ÿ]+")


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) >= 4 and token not in STOPWORDS
    }


def _overlap_ratio(question: str, passage: str) -> float:
    question_tokens = _content_tokens(question)
    if not question_tokens:
        return 1.0
    return len(question_tokens & _content_tokens(passage)) / len(question_tokens)


def _content_bounds(text: str) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for start, end in sentence_spans(text):
        content_end = start + len(text[start:end].rstrip(" \t\n"))
        if content_end > start:
            bounds.append((start, content_end))
    return bounds


def _terminators(text: str) -> int:
    return sum(text.count(char) for char in ".!?")


def _write_doc(tmp_path: Path, text: str, name: str = "nota.txt") -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def _queries(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "queries.jsonl"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _payload(text: str, **passage_overrides: object) -> dict[str, object]:
    passage = {
        "doc_id": "nota",
        "start": 0,
        "end": len(text),
        "passage": text,
    }
    passage.update(passage_overrides)
    return {
        "id": "q",
        "category": "single_sentence",
        "question": "Qual é a frase do documento?",
        "passages": [passage],
    }


def test_eval_dataset_counts_offsets_and_balance():
    documents = load_eval_corpus(EVAL)
    by_id = {document.doc_id: document for document in documents}
    assert list(by_id) == [
        "acervo",
        "agua",
        "assembleia",
        "biblioteca-faq",
        "cozinha",
        "estatuto",
        "faq-associacao",
        "horta-turno",
        "mutirao",
        "oficina-precos",
        "plantio",
        "transporte",
    ]
    assert all("\r" not in document.text and document.text.strip() for document in documents)
    assert sum(len(document.text) for document in documents) == 19795

    queries = load_eval_queries(EVAL / "queries.jsonl", documents)
    assert [query.query_id for query in queries] == [f"v1-q{number:02d}" for number in range(1, 49)]
    assert len({query.question for query in queries}) == len(queries)
    counts = {category: 0 for category in EVAL_QUERY_CATEGORIES}
    per_doc: dict[str, list[str]] = {document.doc_id: [] for document in documents}
    intervals: dict[str, list[tuple[int, int]]] = {document.doc_id: [] for document in documents}
    multi = [query for query in queries if len(query.gold_spans) != 1]
    assert len(multi) == 5
    assert {query.category for query in multi} == {"single_sentence"}
    assert all(len(query.gold_spans) == 2 for query in multi)

    raw_lines = [
        json.loads(line)
        for line in (EVAL / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "\r" not in (EVAL / "queries.jsonl").read_text(encoding="utf-8")
    for query, raw in zip(queries, raw_lines, strict=True):
        assert query.query_id == raw["id"]
        assert query.category == raw["category"]
        assert query.question == raw["question"]
        counts[query.category] += 1
        doc_ids = set()
        for span, passage in zip(query.gold_spans, raw["passages"], strict=True):
            document = by_id[span.doc_id]
            assert span.start == passage["start"]
            assert span.end == passage["end"]
            assert span.start < span.end
            snippet = document.text[span.start : span.end]
            assert snippet == passage["passage"]
            assert snippet.strip() != ""
            doc_ids.add(span.doc_id)
            intervals[span.doc_id].append((span.start, span.end))
        assert doc_ids == {query.gold_spans[0].doc_id}
        per_doc[query.gold_spans[0].doc_id].append(query.category)
    assert counts == {category: 12 for category in EVAL_QUERY_CATEGORIES}
    assert all(
        sorted(categories) == sorted(EVAL_QUERY_CATEGORIES) for categories in per_doc.values()
    )
    for ranges in intervals.values():
        ordered = sorted(ranges)
        for (_, end), (start, _) in zip(ordered, ordered[1:], strict=False):
            assert end <= start
    names = sorted(path.name for path in EVAL.iterdir())
    assert names == sorted([*(f"{doc_id}.txt" for doc_id in by_id), "queries.jsonl"])


def test_eval_span_shapes_follow_the_category():
    documents = load_eval_corpus(EVAL)
    by_id = {document.doc_id: document for document in documents}
    queries = load_eval_queries(EVAL / "queries.jsonl", documents)
    for query in queries:
        document = by_id[query.gold_spans[0].doc_id]
        bounds = _content_bounds(document.text)
        for span in query.gold_spans:
            snippet = document.text[span.start : span.end]
            if query.category in {"single_sentence", "paraphrase"}:
                assert (span.start, span.end) in bounds
                assert "\n" not in snippet
                assert snippet[-1] in ".!?"
                assert _terminators(snippet) == 1
                assert span.end - span.start < 120
                continue
            assert len(query.gold_spans) == 1
            pair = next(
                (left, right)
                for left, right in zip(bounds, bounds[1:], strict=False)
                if left[0] == span.start and right[1] == span.end
            )
            gap = document.text[pair[0][1] : pair[1][0]]
            assert span.end - span.start > 120
            if query.category == "two_sentences":
                assert gap == " "
                assert "\n\n" not in snippet
            else:
                assert query.category == "paragraph_boundary"
                assert "\n\n" in gap and gap.strip() == ""
                before, after = snippet.split("\n\n", maxsplit=1)
                assert before.strip() and after.strip()


def test_eval_lexical_overlap_separates_paraphrase():
    documents = load_eval_corpus(EVAL)
    by_id = {document.doc_id: document for document in documents}
    queries = load_eval_queries(EVAL / "queries.jsonl", documents)
    for query in queries:
        document = by_id[query.gold_spans[0].doc_id]
        passage = " ".join(document.text[span.start : span.end] for span in query.gold_spans)
        assert len(_content_tokens(query.question)) >= 4
        ratio = _overlap_ratio(query.question, passage)
        if query.category == "paraphrase":
            assert ratio <= 0.25
        else:
            assert ratio >= 0.50


def test_eval_documents_vary_in_structure_and_length():
    documents = load_eval_corpus(EVAL)
    texts = {document.doc_id: document.text for document in documents}
    assert "Serviço | Valor | Prazo" in texts["oficina-precos"]
    assert "Trecho | Ida | Volta" in texts["transporte"]
    assert "Passo 1." in texts["horta-turno"]
    assert "Passo 1." in texts["cozinha"]
    assert "Pergunta." in texts["faq-associacao"]
    assert "Pergunta." in texts["biblioteca-faq"]
    assert "\n- " in texts["mutirao"]
    assert "\n- " in texts["agua"]
    long_paragraphs = sum(
        1 for text in texts.values() if any(len(part) > 400 for part in text.split("\n\n"))
    )
    assert long_paragraphs >= 3
    lengths = [len(text) for text in texts.values()]
    assert min(lengths) >= 1000
    assert max(lengths) >= 2 * min(lengths)
    accented = False
    queries = load_eval_queries(EVAL / "queries.jsonl", documents)
    for query in queries:
        for span in query.gold_spans:
            snippet = texts[span.doc_id][span.start : span.end]
            if len(snippet.encode("utf-8")) > len(snippet):
                accented = True
    assert accented


def test_eval_long_spans_do_not_fit_in_the_default_fixed_window():
    documents = load_eval_corpus(EVAL)
    queries = load_eval_queries(EVAL / "queries.jsonl", documents)
    narrow = []
    wide = []
    for document in documents:
        narrow.extend(FixedSizeChunker(size=120, overlap=0).chunk(document))
        wide.extend(FixedSizeChunker(size=20_000, overlap=0).chunk(document))
    spans = [span for query in queries for span in query.gold_spans]
    long_spans = [span for span in spans if span.end - span.start > 120]
    short_spans = [span for span in spans if span.end - span.start < 120]
    assert long_spans and short_spans
    assert spans_contained(narrow, long_spans) is False
    assert spans_contained(wide, spans) is True


def test_eval_loader_rejects_bad_corpus(tmp_path: Path):
    with pytest.raises(ConfigError, match="CORPUS_PATH"):
        load_eval_corpus(tmp_path / "ausente")
    with pytest.raises(ConfigError, match="não tem documentos"):
        load_eval_corpus(tmp_path)
    _write_doc(tmp_path, " \n\n ")
    with pytest.raises(ConfigError, match="documento vazio"):
        load_eval_corpus(tmp_path)
    _write_doc(tmp_path, "outro", "nota.md")
    with pytest.raises(ConfigError, match="repetido"):
        load_eval_corpus(tmp_path)


def test_eval_loader_rejects_bad_queries(tmp_path: Path):
    text = "Frase com ação única."
    _write_doc(tmp_path, text)
    documents = load_eval_corpus(tmp_path)
    missing = tmp_path / "ausente.jsonl"
    with pytest.raises(ConfigError, match="QUERIES_PATH"):
        load_eval_queries(missing, documents)

    path = _queries(tmp_path, "\n\n")
    with pytest.raises(ConfigError, match="não tem perguntas"):
        load_eval_queries(path, documents)

    path = _queries(tmp_path, "{não é json}\n")
    with pytest.raises(ConfigError, match="JSON"):
        load_eval_queries(path, documents)

    path = _queries(tmp_path, [1])
    with pytest.raises(ConfigError, match="objeto"):
        load_eval_queries(path, documents)

    payload = _payload(text)
    payload["category"] = "quase"
    path = _queries(tmp_path, payload)
    with pytest.raises(ConfigError, match="categoria inválida"):
        load_eval_queries(path, documents)

    payload = _payload(text)
    payload["passages"] = []
    path = _queries(tmp_path, payload)
    with pytest.raises(ConfigError, match="passages vazio"):
        load_eval_queries(path, documents)

    payload = _payload(text)
    payload["passages"] = ["texto"]
    path = _queries(tmp_path, payload)
    with pytest.raises(ConfigError, match="passagem inválida"):
        load_eval_queries(path, documents)

    first = _payload(text)
    second = _payload(text)
    second["id"] = "q"
    path = _queries(tmp_path, json.dumps(first) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ConfigError, match="id de pergunta repetido"):
        load_eval_queries(path, documents)


def test_eval_loader_rejects_bad_spans(tmp_path: Path):
    text = "Frase com ação única."
    _write_doc(tmp_path, text)
    documents = load_eval_corpus(tmp_path)

    cases: list[tuple[dict[str, object], str]] = [
        (_payload(text, passage=""), "vazia"),
        (_payload(text, passage="   ", start=0, end=3), "vazia"),
        (_payload(text, start=True, end=4, passage=text[1:4]), "inválido"),
        (_payload(text, start="0", end=4, passage=text[:4]), "inválido"),
        (_payload(text, start=0, end=len(text) + 5, passage=text), "fora do documento"),
        (_payload(text, start=-1, end=len(text), passage=text[-1:]), "fora do documento"),
        (_payload(text, start=4, end=2, passage="x"), "fora do documento"),
        (_payload(text, start=0, end=0, passage="x"), "fora do documento"),
        (_payload(text, start=0, end=5, passage="xxxxx"), "não coincide"),
        (_payload(text, doc_id="outro"), "desconhecido"),
    ]
    for payload, match in cases:
        path = _queries(tmp_path, payload)
        with pytest.raises(ConfigError, match=match):
            load_eval_queries(path, documents)

    repeated = _payload(text)
    repeated["passages"] = [repeated["passages"][0], dict(repeated["passages"][0])]
    path = _queries(tmp_path, repeated)
    with pytest.raises(ConfigError, match="vão repetido"):
        load_eval_queries(path, documents)


def test_eval_loader_accepts_repeated_text_at_explicit_offsets(tmp_path: Path):
    sentence = "alpha único."
    text = f"{sentence} {sentence}"
    _write_doc(tmp_path, text)
    _write_doc(tmp_path, "Outro arquivo markdown.", "extra.md")
    documents = load_eval_corpus(tmp_path)
    assert [document.doc_id for document in documents] == ["extra", "nota"]
    second = text.rfind(sentence)
    payload = {
        "id": "q",
        "category": "single_sentence",
        "question": "  Onde aparece alpha único?  ",
        "passages": [
            {"doc_id": "nota", "start": 0, "end": len(sentence), "passage": sentence},
            {
                "doc_id": "nota",
                "start": second,
                "end": second + len(sentence),
                "passage": sentence,
            },
        ],
    }
    path = _queries(tmp_path, payload)
    queries = load_eval_queries(path, documents)
    assert queries[0].question == "Onde aparece alpha único?"
    assert queries[0].category == "single_sentence"
    assert [span.start for span in queries[0].gold_spans] == [0, second]
    assert len(sentence.encode("utf-8")) > len(sentence)
    with pytest.raises(ConfigError, match="ambígua"):
        load_queries(path, documents)
    assert load_corpus(tmp_path)[0].doc_id == "extra"
