from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    request_id: str