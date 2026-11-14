# Contrato: o mapa explícito não lê processo nem .env, e a chave não aparece no repr.

from pathlib import Path

import pytest

from chunking_study.config import ENV_KEYS, load_settings
from chunking_study.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_when_the_map_is_empty():
    settings = load_settings({})
    assert settings.embedding_provider == "fake"
    assert settings.chunk_strategy == "fixed"
    assert settings.chunk_unit == "chars"
    assert settings.chunk_size == 120
    assert settings.chunk_overlap == 0
    assert settings.semantic_similarity_threshold == 0.5
    assert settings.top_k == 3
    assert settings.embedding_dimension == 64
    assert settings.gemini_api_key == ""
    assert settings.openai_api_key == ""
    assert settings.gemini_embedding_model == "gemini-embedding-001"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.corpus_path == Path("data/placeholder")
    assert settings.queries_path == Path("data/placeholder/queries.jsonl")


def test_explicit_env_ignores_process_and_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=from-file\nCHUNK_SIZE=80\n", encoding="utf-8")
    settings = load_settings({"CHUNK_SIZE": "30"}, dotenv_path=dotenv)
    assert settings.chunk_size == 30
    assert settings.gemini_api_key == ""
    rendered = repr(settings)
    assert "from-env" not in rendered
    assert "from-file" not in rendered


def test_dotenv_fills_only_keys_missing_from_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CHUNK_SIZE", "50")
    dotenv = tmp_path / ".env"
    dotenv.write_text("CHUNK_SIZE=80\nTOP_K=4\n", encoding="utf-8")
    settings = load_settings(dotenv_path=dotenv)
    assert settings.chunk_size == 50
    assert settings.top_k == 4


def test_repr_hides_the_api_key():
    settings = load_settings({"EMBEDDING_PROVIDER": "gemini", "GEMINI_API_KEY": "super-secret"})
    assert settings.gemini_api_key == "super-secret"
    assert "super-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"CHUNK_STRATEGY": "recursive"}, "CHUNK_STRATEGY"),
        ({"CHUNK_SIZE": "0"}, "CHUNK_SIZE"),
        ({"CHUNK_SIZE": "10", "CHUNK_OVERLAP": "10"}, "CHUNK_OVERLAP"),
        ({"CHUNK_STRATEGY": "fixed_overlap", "CHUNK_OVERLAP": "0"}, "fixed_overlap"),
        ({"CHUNK_STRATEGY": "semantic", "CHUNK_UNIT": "tokens"}, "tokens"),
        ({"CHUNK_SIZE": "12.5"}, "inteiro"),
        ({"SEMANTIC_SIMILARITY_THRESHOLD": "1.5"}, "SEMANTIC_SIMILARITY_THRESHOLD"),
        ({"EMBEDDING_PROVIDER": "azure"}, "EMBEDDING_PROVIDER"),
        ({"TOP_K": "0"}, "TOP_K"),
    ],
)
def test_invalid_values_raise(env: dict[str, str], match: str):
    with pytest.raises(ConfigError, match=match):
        load_settings(env)


def test_choices_are_case_insensitive():
    settings = load_settings(
        {"EMBEDDING_PROVIDER": "Fake", "CHUNK_STRATEGY": "Fixed_Overlap", "CHUNK_OVERLAP": "2"}
    )
    assert settings.embedding_provider == "fake"
    assert settings.chunk_strategy == "fixed_overlap"
    assert settings.chunk_overlap == 2


def test_env_example_lists_the_same_keys_and_empty_secrets():
    keys: list[str] = []
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        keys.append(key)
        if key in {"GEMINI_API_KEY", "OPENAI_API_KEY"}:
            assert value == ""
    assert tuple(keys) == ENV_KEYS
