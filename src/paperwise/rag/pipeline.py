from paperwise.generation import OllamaClient
from paperwise.generation.prompts import build_question_prompt
from paperwise.ingestion import chunk_pages, load_pdf
from paperwise.models import AnswerResponse, Citation
from paperwise.retrieval import OllamaEmbeddings, Retriever


class RAGPipeline:
    def __init__(self, ollama_url: str, chat_model: str, embedding_model: str, top_k: int = 5) -> None:
        self.retriever = Retriever(OllamaEmbeddings(ollama_url, embedding_model), top_k)
        self.generator = OllamaClient(ollama_url, chat_model)

    def index_pdf(self, pdf_bytes: bytes, chunk_size: int = 1200, chunk_overlap: int = 200) -> int:
        chunks = chunk_pages(load_pdf(pdf_bytes), chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("No extractable text found in PDF")
        self.retriever.index(chunks)
        return len(chunks)

    def ask(self, question: str) -> AnswerResponse:
        retrieved = self.retriever.retrieve(question)
        answer = self.generator.answer(build_question_prompt(question, retrieved))
        citations = [
            Citation(page=chunk.page, text=chunk.text, score=score)
            for chunk, score in retrieved
        ]
        return AnswerResponse(answer=answer, citations=citations)