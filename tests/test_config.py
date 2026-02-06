import pytest
from travel_assistant.core.config import Settings


def test_settings_defaults_load():
    settings = Settings.load()
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "mistral"
    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.chroma_persist_dir == ".chroma"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.top_k == 4


def test_env_override(monkeypatch):
    monkeypatch.setenv("TA_OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("TA_TOP_K", "6")
    settings = Settings.load()
    assert settings.ollama_model == "llama3"
    assert settings.top_k == 6


def test_chunk_overlap_validation(monkeypatch):
    monkeypatch.setenv("TA_CHUNK_SIZE", "500")
    monkeypatch.setenv("TA_CHUNK_OVERLAP", "500")

    with pytest.raises(ValueError, match="chunk_overlap must be < chunk_size"):
        Settings.load()