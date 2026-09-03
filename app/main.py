from fastapi import FastAPI, HTTPException

from app.model import predict_category
from app.schemas import (
    CategoryPredictionRequest,
    CategoryPredictionResponse,
)


app = FastAPI(
    title="SpendSmart ML Service",
    description="Machine learning service for SpendSmart",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "success": True,
        "message": "ML service is running"
    }


@app.post(
    "/predict-category",
    response_model=CategoryPredictionResponse
)
def predict(request: CategoryPredictionRequest):

    try:
        result = predict_category(request.title)

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )