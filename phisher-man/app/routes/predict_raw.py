from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from app.services.predictor import predict
from app.models.prediction import PredictionOutput
import logging
from app.utils.parser import parse_email_raw

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict_raw", response_model=PredictionOutput)
async def predict_raw(
    input: str = Body(..., media_type="text/plain", max_length=30000),
):
    try:
        email = parse_email_raw(input)
        result = predict(
            email,
        )
        # email_text = f"{email_input.subject} {email_input.body}"
        # result = predict(email_text)
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        logger.error("Raw prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process raw prediction.")
