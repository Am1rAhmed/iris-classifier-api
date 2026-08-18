# Iris Classifier API

A REST API that predicts iris flower species from measurements, built to
practice production-style ML API engineering (not model complexity).

## Problem
Multi-class classification: given 4 flower measurements, predict the species
(setosa, versicolor, or virginica) using the classic Iris dataset.

## API Contract
`POST /predict` accepts sepal length, sepal width, petal length, and petal
width (cm, positive floats) as JSON, and returns the predicted species name
plus a confidence score (0-1).

## Request Flow

```
Client sends POST /predict
   { "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 }
        │
        ▼
1. VALIDATION
   - Are all 4 fields present?
   - Are they all numbers, and positive?
   - If invalid → return 422 with a clear error message, model never runs
        │
        ▼
2. MODEL
   - Load the saved model.joblib (once, at startup — not per-request)
   - Convert the 4 validated numbers into the shape the model expects
   - Run model.predict() and model.predict_proba()
        │
        ▼
3. RESPONSE
   - Map the model's numeric class output (0, 1, 2) back to species name
   - Package as JSON: { "species": "setosa", "confidence": 0.98 }
   - Return with status 200
```

## Status
🚧 In progress — following a 5-phase, 20-task build plan.
- Task 1: Project planning
- Task 2: Environment & structure
- Task 3: Train & save model
- Task 4: Bare-bones FastAPI app