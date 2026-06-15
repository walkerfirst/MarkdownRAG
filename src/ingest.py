from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.chunker import chunk_markdown
from src.cleaner import clean_markdown
from src.config import load_config, resolve_path
from src.embedder import LocalEmbedder
from src.hash_utils import compute_sha256
from src.loader import load_markdown_files
from src.metadata_store import MetadataStore
from src.sqlite_metadata_store import SQLiteMetadataStore
from src.vector_store import ChromaVectorStore

app = typer.Typer(add_completion=False)
console = Console()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_cache(config: dict) -> None:
    chroma_dir = resolve_path(config["root_dir"], config["paths"]["chroma_dir"])
    metadata_db = resolve_path(config["root_dir"], config["paths"]["metadata_db"])

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    if metadata_db.exists():
        metadata_db.unlink()


def ingest_project(
    config_path: str = "config.yaml",
    reset: bool = False,
    embedder: LocalEmbedder | None = None,
    vector_store: ChromaVectorStore | None = None,
    metadata_store: MetadataStore | None = None,
) -> dict:
    config = load_config(config_path)
    if reset:
        _reset_cache(config)
        config = load_config(config_path)

    notes_dir = resolve_path(config["root_dir"], config["paths"]["notes_dir"])
    chroma_dir = resolve_path(config["root_dir"], config["paths"]["chroma_dir"])
    processed_dir = resolve_path(config["root_dir"], config["paths"]["processed_dir"])
    metadata_db = resolve_path(config["root_dir"], config["paths"]["metadata_db"])

    metadata_store = metadata_store or SQLiteMetadataStore(str(metadata_db))
    metadata_store.init_schema()
    vector_store = vector_store or ChromaVectorStore(str(chroma_dir))
    lazy_embedder = embedder

    files = load_markdown_files(str(notes_dir))
    existing_documents = {
        document["file_path"]: document for document in metadata_store.list_documents()
    }
    current_paths = {item["file_path"] for item in files}

    stats = {
        "scanned_files": len(files),
        "new_files": 0,
        "updated_files": 0,
        "skipped_files": 0,
        "deleted_files": 0,
        "generated_chunks": 0,
        "embedded_chunks": 0,
        "saved_to": str(chroma_dir),
    }

    for file_path in sorted(set(existing_documents) - current_paths):
        vector_store.delete_by_file_path(file_path)
        metadata_store.delete_document(file_path)
        stats["deleted_files"] += 1

    chunking = config["chunking"]

    for item in files:
        content_hash = compute_sha256(item["content"])
        existing = existing_documents.get(item["file_path"])
        if existing and existing["content_hash"] == content_hash:
            stats["skipped_files"] += 1
            continue

        if existing:
            vector_store.delete_by_file_path(item["file_path"])
            metadata_store.delete_chunks_by_file_path(item["file_path"])
            stats["updated_files"] += 1
        else:
            stats["new_files"] += 1

        cleaned = clean_markdown(item["content"])
        chunks = chunk_markdown(
            cleaned,
            item["file_path"],
            target_chars=chunking["target_chars"],
            max_chars=chunking["max_chars"],
            min_chars=chunking["min_chars"],
            overlap_chars=chunking["overlap_chars"],
        )

        for chunk in chunks:
            chunk["created_at"] = _utc_now()

        if chunks:
            if lazy_embedder is None:
                lazy_embedder = LocalEmbedder(config["embedding"]["model_name"])
            embeddings = lazy_embedder.embed_texts([chunk["content"] for chunk in chunks])
            vector_store.upsert_chunks(chunks, embeddings)
            metadata_store.upsert_chunks(chunks)
            stats["generated_chunks"] += len(chunks)
            stats["embedded_chunks"] += len(chunks)

        metadata_store.upsert_document(
            {
                "file_path": item["file_path"],
                "file_name": item["file_name"],
                "content_hash": content_hash,
                "last_modified": item["last_modified"],
                "last_ingested_at": _utc_now(),
                "status": "indexed" if chunks else "empty",
                "chunk_count": len(chunks),
            }
        )

    processed_report = processed_dir / "last_ingest_stats.json"
    processed_report.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return stats


def print_stats(stats: dict) -> None:
    console.print(f"Scanned files: {stats['scanned_files']}")
    console.print(f"New files: {stats['new_files']}")
    console.print(f"Updated files: {stats['updated_files']}")
    console.print(f"Skipped files: {stats['skipped_files']}")
    console.print(f"Deleted files: {stats['deleted_files']}")
    console.print(f"Generated chunks: {stats['generated_chunks']}")
    console.print(f"Embedded chunks: {stats['embedded_chunks']}")
    console.print(f"Saved to: {stats['saved_to']}")


@app.command()
def main(reset: bool = typer.Option(False, "--reset", help="清空缓存后重建")) -> None:
    try:
        stats = ingest_project(reset=reset)
    except Exception as exc:  # pragma: no cover - CLI 错误出口
        console.print(f"[red]ingest 失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    print_stats(stats)
    console.print("Done.")


if __name__ == "__main__":
    app()
