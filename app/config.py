from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "ml/saved_model/model.joblib"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 50
    API_TITLE: str = "Iris Classifier API"
    MODEL_VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"

settings = Settings()