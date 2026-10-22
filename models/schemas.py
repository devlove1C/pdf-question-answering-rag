from typing import Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1)


class SourceReference(BaseModel):
    filename: Optional[str] = None
    page_number: int
    chunk_id: Optional[str] = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]