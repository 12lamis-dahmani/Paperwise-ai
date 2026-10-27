from paperwise.models import Chunk
from paperwise.retrieval.vector_store import InMemoryVectorStore, cosine_similarity


def test_cosine_similarity_ranks_matching_vector() -> None:
    store = InMemoryVectorStore()
    chunks = [Chunk("irrelevant", 1, 0), Chunk("relevant", 2, 1)]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([0.0, 1.0], top_k=1)

    assert results[0][0].text == "relevant"
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0