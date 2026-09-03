from pydantic import BaseModel, Field
from typing import Dict


class CategoryPredictionRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=200
    )


class CategoryPredictionResponse(BaseModel):
    category: str
    confidence: float
    probabilities: Dict[str, float]