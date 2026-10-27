from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int


@dataclass(frozen=True)
class Citation:
    page: int
    text: str
    score: float


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]