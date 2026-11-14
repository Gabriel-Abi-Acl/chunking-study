"""Embeddings. O hash existe para teste e fumaça offline.

Não é baseline de qualidade e não entra na comparação do PR4. Gemini e
OpenAI só importam o SDK dentro de `embed`, na primeira chamada.
"""

import hashlib
import math
import re
from typing import Protocol

from chunking_study.config import Settings
from chunking_study.errors import ConfigError, ProviderError

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Saco de palavras com hash. Determinístico, sem rede.

    Cada token cai num eixo via SHA-256, com sinal. A norma L2 é 1 quando
    há token. Vetor vazio fica zerado e o cosseno com ele é 0.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 1:
            raise ConfigError("EMBEDDING_DIMENSION precisa ser >= 1.")
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(text, self.dimension) for text in texts]


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ConfigError("GEMINI_API_KEY é obrigatória quando o provedor é gemini.")
        self._api_key = api_key
        self._model = model
        self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client_or_create()
        vectors: list[list[float]] = []
        for text in texts:
            # Um texto por chamada: há modelo Gemini que agrega a lista num vetor só.
            response = client.models.embed_content(model=self._model, contents=text)
            embeddings = getattr(response, "embeddings", None) or []
            if not embeddings or embeddings[0].values is None:
                raise ProviderError("Gemini devolveu embedding vazio.")
            vectors.append([float(value) for value in embeddings[0].values])
        return vectors

    def _client_or_create(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ConfigError(
                    "Pacote google-genai ausente. Instale com: uv sync --extra gemini"
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ConfigError("OPENAI_API_KEY é obrigatória quando o provedor é openai.")
        self._api_key = api_key
        self._model = model
        self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client_or_create()
        response = client.embeddings.create(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise ProviderError("OpenAI devolveu uma quantidade inesperada de embeddings.")
        return [[float(value) for value in item.embedding] for item in ordered]

    def _client_or_create(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigError(
                    "Pacote openai ausente. Instale com: uv sync --extra openai"
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "fake":
        return HashingEmbedder(dimension=settings.embedding_dimension)
    if settings.embedding_provider == "gemini":
        return GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
        )
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    raise ConfigError(f"EMBEDDING_PROVIDER inválida: {settings.embedding_provider}.")


def _hash_embed(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
