from src.reranker import LocalReranker


class FakeCrossEncoder:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.logits


def test_reranker_reorders_by_score() -> None:
    candidates = [{"content": "a", "score": 0.9}, {"content": "b", "score": 0.1}]
    # 交叉编码器给 b 更高 logit,应把 b 排到前面
    reranker = LocalReranker("x", model=FakeCrossEncoder([0.0, 5.0]))

    out = reranker.rerank("q", candidates)

    assert [c["content"] for c in out] == ["b", "a"]
    assert out[0]["score"] > out[1]["score"]
    assert 0.0 <= out[1]["score"] <= 1.0  # sigmoid 归一到 0~1


def test_reranker_empty_candidates() -> None:
    reranker = LocalReranker("x", model=FakeCrossEncoder([]))
    assert reranker.rerank("q", []) == []


def test_reranker_skip_returns_unchanged(monkeypatch) -> None:
    # 离线 + 不存在的本地路径 → 加载失败走兜底跳过,不中断
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    reranker = LocalReranker("models/does-not-exist", use_fallback=True)

    assert reranker.backend == "skip"
    candidates = [{"content": "a", "score": 0.1}, {"content": "b", "score": 0.9}]
    assert reranker.rerank("q", candidates) == candidates
