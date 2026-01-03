from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Core directories
    data_dir: Path = Field(default=Path("./data"), env="PYFUND_DATA_DIR")
    cache_dir: Path = Field(default=Path("./cache"), env="PYFUND_CACHE_DIR")
    
    # ML & Tracking
    mlflow_tracking_uri: str = Field(default="file://./mlruns", env="MLFLOW_TRACKING_URI")
    
    # Broker & APIs
    broker: str = Field(default="alpaca", env="PYFUND_BROKER")
    api_key: Optional[str] = Field(default=None)
    api_secret: Optional[str] = Field(default=None)
    
    # Data Defaults (Polars refactor)
    default_source: str = Field(default="yfinance")
    use_polars: bool = Field(default=True)
    async_fetching: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
