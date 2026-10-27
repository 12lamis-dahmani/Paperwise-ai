from paperwise.models import Chunk
from .embeddings import OllamaEmbeddings
from .vector_store import InMemoryVectorStore


class Retriever:
    def __init__(self, embeddings: OllamaEmbeddings, top_k: int = 5) -> None:
        self.embeddings = embeddings
        self.top_k = top_k
        self.store = InMemoryVectorStore()

    def index(self, chunks: list[Chunk]) -> None:
        self.store.add(chunks, self.embeddings.embed([chunk.text for chunk in chunks]))

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        query_embedding = self.embeddings.embed([question])[0]
        return self.store.search(query_embedding, self.top_k)