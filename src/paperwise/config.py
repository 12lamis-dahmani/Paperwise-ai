from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()