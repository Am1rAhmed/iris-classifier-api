from fastapi import APIRouter, HTTPException, Request
import numpy as np
from app.models.schemas import PredictionInput, PredictionOutputV2
from app.logging_config import logger
from app.config import settings

router = APIRouter(prefix="/api/v2")

SPECIES = ["setosa", "versicolor", "virginica"]


def get_ml_models():
    from app.main import ml_models
    return ml_models


class ModelNotLoadedError(Exception):
    pass


@router.post("/predict", response_model=PredictionOutputV2)
def predict_v2(input_data: PredictionInput, request: Request):
    ml_models = get_ml_models()
    request_id = request.state.request_id

    if "iris_model" not in ml_models:
        raise ModelNotLoadedError

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

        species = SPECIES[prediction]
        prob_dict = {SPECIES[i]: round(float(p), 4) for i, p in enumerate(probabilities)}

        logger.info(f"request_id={request_id} v2_prediction={species} probabilities={prob_dict}")

        return PredictionOutputV2(
            prediction=species,
            probabilities=prob_dict,
            model_version=settings.MODEL_VERSION,
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"request_id={request_id} v2_prediction_failed error={e}")
        raise HTTPException(status_code=500, detail="Prediction failed")