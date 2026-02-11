from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from travel_assistant.core.config import Settings
from travel_assistant.models.schemas import DocumentChunk

import chromadb
from chromadb.config import Settings as ChromaSettings

import hashlib

Metadata = Dict[str, Any]

@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float # similarity score from the vector database

class EmbeddingServiceProtocol(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class VectorStoreService:
    """
    Encapsulates operations with the vector database (ChromaDB).

    Responsibilities:
    - Add embeddings with metadata
    - Query top-k similar embeddings
    - Return results as RetrievedDocument objects
    """

    def __init__(self, settings: Settings, embedder: EmbeddingServiceProtocol):
  
        """
        Initialize ChromaDB client and collection.

        Args:
            persist_dir: folder to persist database locally
        """
        self.settings = settings
        self.embedder = embedder

        self.persist_dir = Path(self.settings.vector_store_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._collection = self._init_collection()

        ### T0do: initialize Chroma client and collection here

    def _init_collection(self) -> Any:
        client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(
            name=self.settings.vector_store_collection
        )
    
    def _chunk_id(self, chunk: DocumentChunk) -> str:
        """
        Deterministic ID for idempotent upserts across runs.
        Same chunk -> same id, so upsert won't duplicate.
        """
        meta = chunk.metadata or {}
        raw = (
            f"{meta.get('source_file')}|{meta.get('page_number')}|{meta.get('chunk_id')}|"
            f"{hashlib.md5(chunk.content.encode('utf-8')).hexdigest()}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest() 
    
    def upsert_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """
        Upsert chunks into the vector store.
        - Filters empty chunks
        - Embeds in batches
        - Upserts documents + metadata + embeddings
        """
        valid = [c for c in chunks if c.text and c.text.strip()]
        if not valid:
            return 0

        total = 0
        for i in range(0, len(valid), batch_size):
            batch = valid[i:i + batch_size]

            texts = [c.text for c in batch]
            embeddings = self.embedder.embed_texts(texts)

            if len(embeddings) != len(batch):
                raise ValueError("Embedding count mismatch with chunk batch size.")

            ids = [self._chunk_id(c) for c in batch]
            metadatas = [c.metadata for c in batch]
            documents = texts

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            total += len(batch)

        return total

    
    def query(
            self,
            text: str,
            top_k: int = 5,
            where: Optional[Metadata] = None,
        ) -> List[SearchResult]:
            raise NotImplementedError

    def count(self, where: Optional[Metadata] = None) -> int:
            raise NotImplementedError

    def delete_by_source(self, source: str) -> int:
            raise NotImplementedError
        

