from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from app.models.prediction import EmailInput, PredictionOutput
from app.services.predictor import predict
from app.utils.parser import parse_email_raw

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
def classify_email(input: EmailInput):
    result = predict(
        input.dict(),
    )
    return JSONResponse(content=jsonable_encoder(result))


@router.post("/predict_raw", response_model=PredictionOutput)
def classify_email_raw(input: str = Body(..., media_type="text/plain")):
    email = parse_email_raw(input)
    result = predict(
        email,
    )
    return JSONResponse(content=jsonable_encoder(result))
