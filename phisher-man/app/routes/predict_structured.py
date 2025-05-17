from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from app.services.predictor import predict
from app.models.prediction import EmailInput
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict")
async def predict_structured(email_input: EmailInput):
    try:
        result = predict(
            input.dict(),
        )
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        logger.error("Structured prediction failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to process structured prediction."
        )
