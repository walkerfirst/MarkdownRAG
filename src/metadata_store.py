from __future__ import annotations

from abc import ABC, abstractmethod


class MetadataStore(ABC):
    @abstractmethod
    def init_schema(self) -> None:
        ...

    @abstractmethod
    def get_document_by_path(self, file_path: str) -> dict | None:
        ...

    @abstractmethod
    def upsert_document(self, document: dict) -> None:
        ...

    @abstractmethod
    def list_documents(self) -> list[dict]:
        ...

    @abstractmethod
    def delete_document(self, file_path: str) -> None:
        ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[dict]) -> None:
        ...

    @abstractmethod
    def delete_chunks_by_file_path(self, file_path: str) -> None:
        ...

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        ...

    @abstractmethod
    def search_chunks_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict]:
        ...
