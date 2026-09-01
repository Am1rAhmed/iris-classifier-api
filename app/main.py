from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import Request
from contextlib import asynccontextmanager
import joblib
import numpy as np
from app.models.schemas import PredictionInput, PredictionOutput
import uuid
import time
from app.logging_config import logger
from app.routers.v1 import router as v1_router, ModelNotLoadedError

ml_models = {}
MODEL_VERSION = "1.0.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["iris_model"] = joblib.load("ml/saved_model/model.joblib")
    logger.info("Model loaded successfully")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)
app.include_router(v1_router)

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
    logger.info(f"Unhandled error: {exc}")  # log internally
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

SPECIES = ["setosa", "versicolor", "virginica"]

@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"request_id={request_id} method={request.method} path={request.url.path} "
        f"status={response.status_code} duration={duration:.4f}s"
    )
    return response