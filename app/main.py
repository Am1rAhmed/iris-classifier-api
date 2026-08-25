from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import numpy as np
from app.models.schemas import PredictionInput
import uuid

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["iris_model"] = joblib.load("ml/saved_model/model.joblib")
    print("Model loaded successfully")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

SPECIES = ["setosa", "versicolor", "virginica"]

@app.get("/")
def root():
    return {"message": "ML API is alive"}

@app.get("/health")
def health():
    model_loaded = "iris_model" in ml_models
    return {"stsatus": "ok", "model_loaded": model_loaded}

@app.post("/predict")
def predict(input_data: PredictionInput):

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

    return {"predicted_species": species,
            "confidence": confidence,
            "request_id": str(uuid.uuid4())
        }