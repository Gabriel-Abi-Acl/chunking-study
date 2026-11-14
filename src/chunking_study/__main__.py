"""Fumaça offline: lê o placeholder e imprime um JSON por pergunta.

Não é experimento. O número que sair aqui não entra no README.
"""

import json
import sys

from chunking_study.config import load_settings
from chunking_study.corpus import load_corpus, load_queries
from chunking_study.errors import ConfigError
from chunking_study.runner import run_record_to_dict, run_retrieval


def main() -> None:
    try:
        settings = load_settings()
        documents = load_corpus(settings.corpus_path)
        queries = load_queries(settings.queries_path, documents)
        records = run_retrieval(documents, queries, settings)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    for record in records:
        print(json.dumps(run_record_to_dict(record), ensure_ascii=False))


if __name__ == "__main__":
    main()
