from fastapi import APIRouter, File, HTTPException, UploadFile

from paperwise.config import settings
from paperwise.models import AnswerResponse, QuestionRequest
from paperwise.rag import RAGPipeline

router = APIRouter()
pipeline = RAGPipeline(
    settings.ollama_url,
    settings.chat_model,
    settings.embedding_model,
    settings.top_k,
)
indexed = False


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, int | str]:
    global indexed
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")
    try:
        count = pipeline.index_pdf(
            await file.read(), settings.chunk_size, settings.chunk_overlap
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    indexed = True
    return {"file_name": file.filename or "paper.pdf", "chunks": count}


@router.post("/questions", response_model=AnswerResponse)
def ask_question(request: QuestionRequest) -> AnswerResponse:
    if not indexed:
        raise HTTPException(status_code=409, detail="Upload a PDF first")
    try:
        return pipeline.ask(request.question)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error