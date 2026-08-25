from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import numpy as np

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

@app.post("/predict")
def predict():
    sample_input = {
        "sepal_length": 7.1,
        "sepal_width": 2.5,
        "petal_length": 5.4,
        "petal_width": 2.2
    }

    features = np.array([[
        sample_input["sepal_length"],
        sample_input["sepal_width"],
        sample_input["petal_length"],
        sample_input["petal_width"]
    ]])

    model = ml_models.get("iris_model")
    prediction = model.predict(features)[0]
    species = SPECIES[prediction]

    return {"predicted_species": species}