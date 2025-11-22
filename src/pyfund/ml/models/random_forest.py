# src/pyfundlib/ml/models/random_forest.py
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from ...utils.logger import get_logger
from .base_model import BaseMLModel

logger = get_logger(__name__)


class RandomForestModel(BaseMLModel):
    """
    Fully-featured Random Forest with automatic classification/regression detection.
    100% compatible with BaseMLModel → rich metadata, cloudpickle, MLflow, etc.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | int = "sqrt",
        bootstrap: bool = True,
        task: Literal["classification", "regression"] = "classification",
        name: str = "random_forest",
        version: str = "1.0",
        **rf_kwargs,
    ):
        super().__init__(name=name, version=version, tags=["random-forest", "tree-ensemble", task])

        self.task = task
        self.hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            **rf_kwargs,
        }

        # Auto-select classifier or regressor
        model_class = RandomForestClassifier if task == "classification" else RandomForestRegressor
        self.model = model_class(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1,
            **rf_kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple | None = None,
    ) -> RandomForestModel:
        """
        Train the model with automatic metric tracking.
        """
        X = self._validate_input(X)
        y = pd.Series(y) if not isinstance(y, (pd.Series, np.ndarray)) else y

        self.model.fit(X, y, sample_weight=sample_weight)

        # Store training info
        self._is_fitted = True
        self.feature_names_in_ = list(X.columns)
        self.metadata.training_samples = len(y)
        self.metadata.features_used = self.feature_names_in_
        self.metadata.target_column = y.name if hasattr(y, "name") else None
        self.metadata.hyperparameters = self.hyperparams

        # Auto-evaluate on training data (or eval_set)
        X_eval, y_eval = (eval_set[0], eval_set[1]) if eval_set else (X, y)
        y_pred = self.model.predict(X_eval)
        y_prob = self.model.predict_proba(X_eval) if self.task == "classification" else None

        metrics = {}
        if self.task == "classification":
            metrics["accuracy"] = accuracy_score(y_eval, y_pred)
            if y_prob is not None and len(np.unique(y_eval)) == 2:
                metrics["auc"] = roc_auc_score(y_eval, y_prob[:, 1])
        else:
            metrics["rmse"] = mean_squared_error(y_eval, y_pred, squared=False)
            metrics["r2"] = self.model.score(X_eval, y_eval)

        self.metadata.performance_metrics = metrics
        self.metadata.status = "trained"

        logger.info(
            f"RandomForest ({self.task}) trained | "
            f"Samples: {len(y)} | Features: {X.shape[1]} | "
            f"{metrics}"
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = self._validate_input(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("predict_proba only available for classification")
        X = self._validate_input(X)
        return self.model.predict_proba(X)

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """Return sorted feature importance"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        importances = pd.Series(self.model.feature_importances_, index=self.feature_names_in_)
        return importances.sort_values(ascending=False).head(top_n)

    def plot_feature_importance(self, top_n: int = 20, save_path: str | None = None):
        """Quick importance plot"""
        import matplotlib.pyplot as plt

        imp = self.feature_importance(top_n)
        plt.figure(figsize=(10, 8))
        imp.plot(kind="barh", color="#2E86AB")
        plt.title(f"Top {top_n} Feature Importance - {self.name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
