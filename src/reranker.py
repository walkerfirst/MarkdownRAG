from __future__ import annotations

import math
import sys
from typing import Any


class LocalReranker:
    """交叉编码器重排,镜像 LocalEmbedder 的兜底风格:模型不可用则跳过、不中断检索。

    重排分经 sigmoid 归一到 0~1(相关概率),与现有 cosine 分同量纲,
    下游 _filter_results 的 min_score / relative_score_ratio 仍然适用。
    """

    def __init__(self, model_name: str, model: Any | None = None, use_fallback: bool = True):
        self.model_name = model_name
        self._model = model
        self.backend = "cross-encoder"

        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name)
            except Exception as exc:  # pragma: no cover - 模型下载/加载不稳定
                if not use_fallback:
                    raise RuntimeError(f"加载 reranker 模型失败: {model_name}") from exc
                self._model = None
                self.backend = "skip"
                print(f"[warn] reranker 不可用，已跳过重排: {exc}", file=sys.stderr)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates or self.backend == "skip":
            return candidates

        pairs = [(query, candidate["content"]) for candidate in candidates]
        logits = self._model.predict(pairs)
        for candidate, logit in zip(candidates, logits):
            candidate["score"] = 1.0 / (1.0 + math.exp(-float(logit)))
        return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)
