from typing import List, Optional
from travel_assistant.models.schemas import RetrievedDocument

class VectorStoreService:
    """
    Encapsulates operations with the vector database (ChromaDB).

    Responsibilities:
    - Add embeddings with metadata
    - Query top-k similar embeddings
    - Return results as RetrievedDocument objects
    """

    def __init__(self, persist_dir: str):
        """
        Initialize ChromaDB client and collection.

        Args:
            persist_dir: folder to persist database locally
        """
        self.persist_dir = persist_dir
        ### T0do: initialize Chroma client and collection here

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Add embeddings to the vector store.

        Args:
            texts: Original text chunks
            embeddings: List of embeddings corresponding to texts
            metadatas: Optional metadata per chunk
            ids: Optional unique IDs
        """
        # Placeholder: currently do nothing
        # Later: self.collection.add(...)
        pass

    def query(self, embedding: List[float], top_k: int = 4) -> List[RetrievedDocument]:
        """
        Query top-k most similar embeddings from the vector store.

        Args:
            embedding: embedding vector of the query
            top_k: number of results to return

        Returns:
            List of RetrievedDocument objects
        """
        # Placeholder: return dummy RetrievedDocument objects
        return [
            RetrievedDocument(content=f"Dummy content {i}", source=f"doc_{i}.txt", score=1.0)
            for i in range(top_k)
        ]