from typing import Any

import requests


class OllamaEmbeddings:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        embeddings: list[list[float]] = response.json().get("embeddings", [])
        if not embeddings:
            raise ValueError("Ollama returned no embeddings")
        return embeddings