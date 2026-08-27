from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import Request
from contextlib import asynccontextmanager
import joblib
import numpy as np
from app.models.schemas import PredictionInput, PredictionOutput
import uuid

ml_models = {}
MODEL_VERSION = "1.0.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["iris_model"] = joblib.load("ml/saved_model/model.joblib")
    print("Model loaded successfully")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class ModelNotLoadedError(Exception):
    """Raise when a prediction is attempted but the model isn't loaded."""
    pass

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Model is not currently loaded. Please try again shortly."}
    )

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input caused a processing error: {str(exc)}"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": err["loc"][-1], "message": err["msg"]} for err in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input", "errors": errors}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error: {exc}")  # log internally
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

SPECIES = ["setosa", "versicolor", "virginica"]

@app.get("/")
def root():
    return {"message": "ML API is alive"}

@app.get("/health")
def health():
    model_loaded = "iris_model" in ml_models
    return {"status": "ok", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if "iris_model" not in ml_models:
        raise ModelNotLoadedError

    try:
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])

        model = ml_models.get("iris_model")
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

        species = SPECIES[prediction]

        return PredictionOutput(
            prediction=species,
            confidence=confidence,
            model_version=MODEL_VERSION,
            request_id=str(uuid.uuid4())
        )

    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")