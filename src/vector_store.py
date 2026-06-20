from __future__ import annotations

from pathlib import Path


class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "markdown_chunks"):
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        metadatas = [
            {
                "file_path": chunk["file_path"],
                "title_path": chunk["title_path"],
                "char_count": int(chunk["char_count"]),
                "chunk_index": int(chunk["chunk_index"]),
                "domain": chunk.get("domain", ""),
                "type": chunk.get("type", ""),
                "evidence_level": chunk.get("evidence_level", ""),
                "freshness": chunk.get("freshness", ""),
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=[chunk["content"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[dict] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance or 0.0))
            results.append(
                {
                    "chunk_id": chunk_id,
                    "file_path": metadata.get("file_path", ""),
                    "title_path": metadata.get("title_path", ""),
                    "char_count": metadata.get("char_count", 0),
                    "domain": metadata.get("domain", ""),
                    "type": metadata.get("type", ""),
                    "evidence_level": metadata.get("evidence_level", ""),
                    "freshness": metadata.get("freshness", ""),
                    "content": content,
                    "score": score,
                }
            )
        return results

    def delete_by_file_path(self, file_path: str) -> None:
        self.collection.delete(where={"file_path": file_path})

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
