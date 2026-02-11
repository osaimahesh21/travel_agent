from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central config for the whole app.

    Priority order:
      1) Environment variables (TA_*)
      2) .env file (optional)
      3) Defaults below
    """

    model_config = SettingsConfigDict(
        env_prefix="TA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (Ollama)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")

    # Embeddings & Vector Store
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_dir: str = Field(default=".chroma")

    # RAG parameters
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    top_k: int = Field(default=4, ge=1, le=20)

    # Vector store (Chroma)
    vector_store_dir: str = "vectorstore"
    vector_store_collection: str = "travel_docs"

    def validate_chunking(self) -> None:
        """
        Validate relationships between chunking parameters.

        Why this exists:
        - Some configuration values depend on each other
        - chunk_overlap must always be smaller than chunk_size
        - If this rule is broken, chunking logic can loop forever or produce duplicates

        We fail early here so the app never starts with a bad configuration.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

    @classmethod
    def load(cls) -> "Settings":
        """
        Create and return a VALIDATED Settings object.

        Think of this as the ONLY safe way to get configuration for the app.

        Step-by-step:
        1. cls refers to the Settings class itself
        2. cls() creates a Settings instance using env vars, .env, and defaults
        3. validate_chunking() enforces cross-field rules
        4. If validation passes, return the Settings object
        """

        # Step 1: Create a Settings instance
        # At this point, values are loaded but NOT yet guaranteed to be valid
        settings = cls()

        # Step 2: Validate important configuration rules
        # If something is wrong, this will raise an error immediately
        settings.validate_chunking()

        # Step 3: Return a SAFE, validated Settings object
        return settings