# Contrato: o hash é determinístico e os SDKs reais só importam na chamada.

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chunking_study.config import load_settings
from chunking_study.embeddings import (
    GeminiEmbedder,
    HashingEmbedder,
    OpenAIEmbedder,
    build_embedder,
)
from chunking_study.errors import ConfigError
from chunking_study.index import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dimension=64)
    again = HashingEmbedder(dimension=64)
    first, second = embedder.embed(["Alface couve", "bicicleta"])
    assert first == again.embed(["Alface couve"])[0]
    assert first != second
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)
    assert embedder.embed([""]) == [[0.0] * 64]
    assert embedder.embed([]) == []

    shared = embedder.embed(["alface couve horta alface"])[0]
    near = embedder.embed(["alface couve horta"])[0]
    far = embedder.embed(["bicicleta camara oficina"])[0]
    assert cosine_similarity(shared, near) > cosine_similarity(shared, far)


def test_import_does_not_load_sdks():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys, chunking_study.embeddings; "
        "assert 'openai' not in sys.modules; "
        "assert 'google' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("cls", "match"),
    [
        (GeminiEmbedder, "GEMINI_API_KEY"),
        (OpenAIEmbedder, "OPENAI_API_KEY"),
    ],
)
def test_missing_key_does_not_import_the_sdk(monkeypatch: pytest.MonkeyPatch, cls, match: str):
    seen: list[str] = []
    real_import = __import__

    def tracking(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "openai"
            or name.startswith("openai.")
            or name == "google"
            or name.startswith("google.")
        ):
            seen.append(name)
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", tracking)
    with pytest.raises(ConfigError, match=match):
        cls(api_key="  ", model="model")
    assert seen == []


@pytest.mark.parametrize(
    ("cls", "match"),
    [
        (GeminiEmbedder, "google-genai"),
        (OpenAIEmbedder, "Pacote openai"),
    ],
)
def test_sdk_import_happens_on_the_first_embed(monkeypatch: pytest.MonkeyPatch, cls, match: str):
    seen: list[str] = []
    real_import = __import__

    def tracking(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "openai"
            or name.startswith("openai.")
            or name == "google"
            or name.startswith("google.")
        ):
            seen.append(name)
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", tracking)
    embedder = cls(api_key="secret-key", model="model")
    assert seen == []
    with pytest.raises(ConfigError, match=match) as caught:
        embedder.embed(["oi"])
    assert seen
    assert "secret-key" not in str(caught.value)


def test_build_embedder_fake_does_not_need_a_key_and_live_without_key_fails():
    fake = build_embedder(load_settings({"EMBEDDING_DIMENSION": "8"}))
    assert isinstance(fake, HashingEmbedder)
    assert fake.embed(["um"])[0]
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_embedder(load_settings({"EMBEDDING_PROVIDER": "gemini"}))
    with pytest.raises(ConfigError, match="EMBEDDING_DIMENSION"):
        HashingEmbedder(dimension=0)
