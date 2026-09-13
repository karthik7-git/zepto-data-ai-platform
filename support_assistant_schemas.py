"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Customer support inquiry text")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Direct answer to customer inquiry")
    sources: list[str] = Field(default_factory=list, description="Source document IDs used for grounding")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")