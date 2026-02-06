class EmbeddingService:
    """
    Converts text into vector embeddings using a chosen embedding model.

    Responsibilities:
    - Accept text (single or list)
    - Produce embeddings compatible with vector databases
    - Encapsulate the embedding model
    """

    def __init__(self, model_name: str):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name or any embedding model identifier
        """
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of texts into embeddings.

        Args:
            texts: List of input text strings

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        # Placeholder logic: return a dummy embedding for each text
        return [[0.0] * 10 for _ in texts]

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single string to embedding (wrapper over embed_texts)

        Args:
            text: input string

        Returns:
            Embedding vector (list of floats)
        """
        return self.embed_texts([text])[0]