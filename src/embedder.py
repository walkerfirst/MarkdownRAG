from __future__ import annotations

import hashlib
import math
import re
import sys
from typing import Any


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


class _HashingEncoder:
    def __init__(self, vector_size: int = 256):
        self.vector_size = vector_size

    def _tokenize(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(text.lower())
        if len(tokens) < 2:
            return tokens
        bigrams = [f"{tokens[index]}_{tokens[index + 1]}" for index in range(len(tokens) - 1)]
        return tokens + bigrams

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        tokens = self._tokenize(text or " ")
        if not tokens:
            tokens = ["empty"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.vector_size
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class LocalEmbedder:
    def __init__(
        self,
        model_name: str,
        model: Any | None = None,
        use_fallback: bool = True,
        query_instruction: str = "",
    ):
        self.model_name = model_name
        self._model = model
        self.backend = "sentence-transformers"
        self.query_instruction = query_instruction or ""

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
            except Exception as exc:  # pragma: no cover - 依赖和模型下载不稳定
                if not use_fallback:
                    raise RuntimeError(f"加载 embedding 模型失败: {model_name}") from exc
                self._model = _HashingEncoder()
                self.backend = "hash-fallback"
                print(
                    f"[warn] sentence-transformers 不可用，已切换到本地哈希向量: {exc}",
                    file=sys.stderr,
                )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        normalized = [text.strip() if text and text.strip() else " " for text in texts]
        if self.backend == "sentence-transformers":
            embeddings = self._model.encode(normalized, normalize_embeddings=True)
        else:
            embeddings = self._model.encode(normalized)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(vector) for vector in embeddings]

    def embed_query(self, query: str) -> list[float]:
        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        return self.embed_texts([text])[0]
