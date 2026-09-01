from fastapi import APIRouter, HTTPException, Request
import numpy as np
import uuid
from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import logger

router = APIRouter(prefix="/api/v1")

SPECIES = ["setosa", "versicolor", "virginica"]
MODEL_VERSION = "1.0.0"


class ModelNotLoadedError(Exception):
    pass


def get_ml_models():
    from app.main import ml_models
    return ml_models


@router.get("/health")
def health():
    ml_models = get_ml_models()
    model_loaded = "iris_model" in ml_models
    return {"status": "ok", "model_loaded": model_loaded}


@router.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput, request: Request):
    ml_models = get_ml_models()
    request_id = request.state.request_id

    if "iris_model" not in ml_models:
        raise ModelNotLoadedError()

    try:
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])

        model = ml_models["iris_model"]
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        species = SPECIES[prediction]

        logger.info(f"request_id={request_id} prediction={species} confidence={confidence:.4f}")

        return PredictionOutput(
            prediction=species,
            confidence=round(confidence, 4),
            model_version=MODEL_VERSION,
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"request_id={request_id} prediction_failed error={e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


# CHALLENGE NOTE (Task 10):
# If /api/v2/predict needs to return an extra field tomorrow (e.g. full
# probability distribution across all 3 classes, not just the top one),
# I would NOT modify PredictionOutput directly, since that would silently
# change v1's contract for existing clients. Instead I'd create a new
# PredictionOutputV2(PredictionOutput) schema (or a separate schema entirely)
# with the extra field, and a separate v2 router file (app/routers/v2.py)
# with its own /predict endpoint using that new schema. v1 stays byte-for-byte identical. 