from paperwise.models import Chunk


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")
        self._items = list(zip(chunks, embeddings))

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        ranked = sorted(
            ((chunk, cosine_similarity(embedding, query_embedding)) for chunk, embedding in self._items),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]