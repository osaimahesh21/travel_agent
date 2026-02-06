from travel_assistant.services.embedding_service import EmbeddingService

emb = EmbeddingService("dummy-model")
print(emb.embed_texts(["Hello","World"]))
print(emb.embed_text("Hello"))