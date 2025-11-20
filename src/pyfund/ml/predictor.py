# src/pyfundlib/ml/predictor.py
from __future__ import annotations

import mlflow
import mlflow.pyfunc
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .models.base_model import BaseMLModel
from .pipelines.feature_pipeline import FeaturePipeline
from .pipelines.training_pipeline import TrainingPipeline
from ..utils.logger import get_logger
from ..data.storage import DataStorage

logger = get_logger(__name__)


class MLPredictor:
    """
    The central ML brain of pyfundlib.
    Handles:
    - Per-ticker model training & versioning
    - Latest/best model loading
    - Real-time prediction
    - MLflow tracking
    - Model registry
    """

    def __init__(
        self,
        model_dir: str = "models",
        mlflow_tracking_uri: str = "file://./mlruns",
        experiment_name: str = "pyfundlib",
        registry_name: str = "pyfundlib_models",
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        try:
            mlflow.create_model_registry(registry_name)
        except:
            pass  # already exists

        self.registry = mlflow.tracking.MlflowClient()
        self.storage = DataStorage()

        logger.info(f"MLPredictor initialized | Tracking: {mlflow_tracking_uri} | Models: {self.model_dir}")

    def train(
        self,
        ticker: str,
        raw_data: pd.DataFrame,
        target: pd.Series,
        model_class: type[BaseMLModel],
        feature_pipeline: FeaturePipeline,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ) -> BaseMLModel:
        """Train a model for a specific ticker with full MLflow logging"""
        logger.info(f"Starting training for {ticker} using {model_class.__name__}")

        pipeline_config = pipeline_config or {}
        training_pipeline = TrainingPipeline(
            model_class=model_class,
            feature_pipeline=feature_pipeline,
            **pipeline_config,
        )

        with mlflow.start_run(run_name=f"{ticker}_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("model_type", model_class.__name__)
            mlflow.log_param("data_points", len(raw_data))

            result = training_pipeline.run(raw_data, target, project_name=f"{ticker}_training")

            best_model = result.best_model
            best_model.metadata.tags.append(ticker)
            best_model.metadata.tags.append("production-candidate")

            model_path = self.model_dir / f"{ticker}_{best_model.name}_v{best_model.version}.pkl"
            best_model.save(model_path)

            # Log to MLflow
            mlflow.log_artifact(str(model_path))
            mlflow.log_metrics(result.best_model.metadata.performance_metrics or {})
            mlflow.log_params(result.best_params)

            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, f"{ticker}_{best_model.name}")

            # Tag as latest
            self.registry.transition_model_version_stage(
                name=registered_model.name,
                version=registered_model.version,
                stage="Staging"
            )

            logger.info(f"Model trained & registered for {ticker} | Version: {registered_model.version}")
            return best_model

    def predict(
        self,
        ticker: str,
        raw_data: pd.DataFrame,
        model_name: Optional[str] = None,
        stage: str = "Production",
    ) -> np.ndarray:
        """Get latest prediction for a ticker"""
        model = self.load_latest(ticker, model_name or "xgboost", stage=stage)
        if model is None:
            logger.warning(f"No model found for {ticker}, returning neutral signal")
            return np.zeros(len(raw_data))

        # Assume last row is latest
        latest_features = raw_data.iloc[-1:]
        try:
            pred = model.predict(latest_features)
            logger.info(f"Prediction for {ticker}: {pred[-1]:.4f}")
            return pred
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return np.zeros(len(raw_data))

    def load_latest(
        self,
        ticker: str,
        model_name: str = "xgboost",
        stage: str = "Production",
    ) -> Optional[BaseMLModel]:
        """Load the latest production/staging model for a ticker"""
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(f"{ticker}_{model_name}", stages=[stage])

            if not model_versions:
                logger.warning(f"No {stage} model found for {ticker}_{model_name}")
                return None

            latest_version = model_versions[0]
            model_uri = f"models:/{latest_version.name}/{latest_version.version}"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            if isinstance(loaded_model, BaseMLModel):
                logger.info(f"Loaded {stage} model: {latest_version.name} v{latest_version.version}")
                return loaded_model
            else:
                # Fallback: load from disk if not pyfunc
                path = self.model_dir / f"{ticker}_{model_name}_latest.pkl"
                if path.exists():
                    return BaseMLModel.load(path)
        except Exception as e:
            logger.error(f"Failed to load model for {ticker}: {e}")

        return None

    def promote_to_production(self, ticker: str, model_name: str, version: int) -> None:
        """Promote a model version to Production"""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=f"{ticker}_{model_name}",
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Promoted {ticker}_{model_name} v{version} → Production")

    def list_models(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """List all registered models"""
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
        rows = []
        for model in models:
            for version in model.latest_versions:
                rows.append({
                    "name": model.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "ticker": model.name.split("_")[0] if "_" in model.name else "unknown",
                })
        df = pd.DataFrame(rows)
        if ticker:
            df = df[df["ticker"] == ticker]
        return df