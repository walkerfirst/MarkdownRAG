from pathlib import Path

import yaml

from src.ingest import ingest_project


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self.records[chunk["chunk_id"]] = {
                "file_path": chunk["file_path"],
                "content": chunk["content"],
                "embedding": embedding,
            }

    def delete_by_file_path(self, file_path: str) -> None:
        stale_ids = [chunk_id for chunk_id, item in self.records.items() if item["file_path"] == file_path]
        for chunk_id in stale_ids:
            self.records.pop(chunk_id, None)


def write_config(root: Path) -> Path:
    config = {
        "project": {"name": "test-rag"},
        "paths": {
            "notes_dir": "notes",
            "chroma_dir": "data/chroma",
            "processed_dir": "data/processed",
            "metadata_db": "data/metadata.sqlite",
            "eval_queries": "eval/queries.yaml",
        },
        "embedding": {"provider": "fake", "model_name": "fake"},
        "chunking": {
            "target_chars": 80,
            "max_chars": 160,
            "min_chars": 40,
            "overlap_chars": 20,
        },
        "search": {"top_k": 5, "preview_chars": 120},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


def test_incremental_ingest_flow(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "a.md").write_text("# A\n\n第一段内容。\n\n## 渠道\n\n渠道分析内容。", encoding="utf-8")
    (notes_dir / "b.md").write_text("# B\n\n第二个文件内容。", encoding="utf-8")

    config_path = write_config(tmp_path)
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    first = ingest_project(
        config_path=str(config_path),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert first["new_files"] == 2
    assert first["updated_files"] == 0
    assert first["skipped_files"] == 0
    assert first["deleted_files"] == 0
    assert first["generated_chunks"] > 0

    second = ingest_project(
        config_path=str(config_path),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert second["new_files"] == 0
    assert second["updated_files"] == 0
    assert second["skipped_files"] == 2
    assert second["deleted_files"] == 0
    assert second["generated_chunks"] == 0
    assert second["embedded_chunks"] == 0

    (notes_dir / "a.md").write_text("# A\n\n第一段内容。\n\n## 渠道\n\n渠道分析内容，新增一段。", encoding="utf-8")
    third = ingest_project(
        config_path=str(config_path),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert third["updated_files"] == 1
    assert third["skipped_files"] == 1
    a_records = [item for item in vector_store.records.values() if item["file_path"] == "notes/a.md"]
    assert a_records
    assert any("新增一段" in item["content"] for item in a_records)

    (notes_dir / "b.md").unlink()
    fourth = ingest_project(
        config_path=str(config_path),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert fourth["deleted_files"] == 1
    assert all(item["file_path"] != "notes/b.md" for item in vector_store.records.values())
