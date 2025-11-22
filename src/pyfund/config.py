from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("./data"), env="PYFUND_DATA_DIR")
    cache_dir: Path = Field(default=Path("./cache"), env="PYFUND_CACHE_DIR")
    mlflow_tracking_uri: str = Field(default="file://./mlruns", env="MLFLOW_TRACKING_URI")
    broker: str = Field(default="alpaca", env="PYFUND_BROKER")  # alpaca, ibkr, etc.
    api_key: str = Field(..., env="PYFUND_API_KEY")
    api_secret: str = Field(..., env="PYFUND_API_SECRET")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
