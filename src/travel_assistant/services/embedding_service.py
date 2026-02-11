import logging
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Converts text into vector embeddings using a chosen embedding model.

    Responsibilities:
    - Accept text (single or list)
    - Produce embeddings compatible with vector databases
    - Encapsulate the embedding model
    """

    def __init__(self, model_name: str, normalize: bool = True):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name or any embedding model identifier
        """
        self.model_name = model_name
        self.normalize = normalize

        logger.info("Initialized EmbeddingService with model: %s", model_name)

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device='cpu')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info("EmbeddingService init model=%s dim=%s", self.model_name, self.embedding_dim)


    

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Convert a single string to embedding (wrapper over embed_texts)

        Args:
            text: input string

        Returns:
            Embedding vector (list of floats)
        """

        vector = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vector.tolist()
    

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single string to embedding (wrapper over embed_texts)

        Args:
            text: input string

        Returns:
            Embedding vector (list of floats)
        """
        return self.embed_texts([text], batch_size=1)[0]