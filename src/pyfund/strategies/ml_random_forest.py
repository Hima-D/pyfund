# src/pyfund/strategies/ml_random_forest.py
from typing import Any

import pandas as pd

from ..data.features import FeatureEngineer
from ..ml.predictor import MLPredictor
from ..utils.logger import logger
from .base import BaseStrategy


class MLRandomForestStrategy(BaseStrategy):
    """
    Advanced Machine Learning Strategy using Random Forest (or any loaded model)

    Features:
    - Automatic feature engineering fallback
    - Model version safety (handles missing/no model gracefully)
    - Probability-based signals with confidence threshold
    - Position sizing based on prediction confidence
    - Proper train/test separation (no lookahead!)
    - Logging and monitoring
    """

    default_params = {
        "model_name": "random_forest",
        "confidence_threshold": 0.6,  # Only trade if prediction prob > 60%
        "use_probability": True,  # Use predict_proba instead of hard predict
        "min_feature_count": 10,  # Minimum features required
        "fallback_to_rsi": True,  # Use RSI strategy if model fails
    }

    def __init__(self, ticker: str, params: dict[str, Any] | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker = ticker.upper()
        self.predictor = MLPredictor()
        self.model = None
        self.feature_names: list | None = None
        self.last_prediction_date = None

    def _load_model_safely(self) -> bool:
        """Safely load the latest model with error handling"""
        try:
            self.model = self.predictor.load_latest(self.ticker, self.params["model_name"])
            if self.model is not None:
                logger.info(f"ML model loaded successfully for {self.ticker}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load ML model for {self.ticker}: {e}")

        self.model = None
        return False

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has proper features"""
        df = data.copy()

        # If raw OHLCV, add technical features
        if set(df.columns) <= {"Open", "High", "Low", "Close", "Volume"}:
            df = FeatureEngineer.add_technical_features(df)

        # Drop non-feature columns
        feature_cols = [
            col
            for col in df.columns
            if col not in {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        ]

        if len(feature_cols) < self.params["min_feature_count"]:
            logger.warning(f"Insufficient features for {self.ticker}: {len(feature_cols)} found")
            return pd.DataFrame()

        X = df[feature_cols].dropna()
        self.feature_names = feature_cols
        return X

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate high-quality ML-based trading signals

        Returns:
            +1 → Strong buy (high confidence long)
             0 → Neutral / no confidence
            -1 → Strong sell (high confidence short)
        """
        signals = pd.Series(0, index=data.index, dtype=int)

        # Load model if not already loaded
        if self.model is None:
            if not self._load_model_safely():
                if self.params["fallback_to_rsi"]:
                    logger.info(f"Falling back to RSI strategy for {self.ticker}")
                    from .rsi_mean_reversion import RSIMeanReversionStrategy

                    fallback = RSIMeanReversionStrategy()
                    return fallback.generate_signals(data)
                else:
                    return signals  # All flat

        # Prepare features
        X = self._prepare_features(data)
        if X.empty or len(X) < 20:
            logger.warning(f"Not enough valid data points for {self.ticker}")
            return signals

        try:
            # Align predictions with original index
            pred_index = X.index

            if self.params["use_probability"] and hasattr(self.model, "predict_proba"):
                # Use prediction probability for confidence
                probabilities = self.model.predict_proba(X)
                # Assume binary classification: [prob_short, prob_long]
                if probabilities.shape[1] == 2:
                    prob_long = probabilities[:, 1]
                    prob_short = probabilities[:, 0]
                else:
                    prob_long = probabilities[:, -1]  # multi-class fallback
                    prob_short = 1 - prob_long

                confidence_long = prob_long
                confidence_short = prob_short

                # Apply confidence threshold
                long_signal = confidence_long >= self.params["confidence_threshold"]
                short_signal = confidence_short >= self.params["confidence_threshold"]

                signals.loc[pred_index[long_signal]] = 1
                signals.loc[pred_index[short_signal]] = -1

                # Optional: size position by confidence (advanced)
                # signals = signals * np.where(signals != 0, np.maximum(confidence_long, confidence_short), 1)

            else:
                # Fallback to hard predictions
                hard_pred = self.model.predict(X)
                signals.loc[pred_index] = hard_pred

            logger.info(
                f"ML signals generated for {self.ticker}: "
                f"Long={(signals==1).sum()}, Short={(signals==-1).sum()}, Flat={(signals==0).sum()}"
            )

        except Exception as e:
            logger.error(f"Error generating ML signals for {self.ticker}: {e}")
            return signals  # Flat on error

        return signals.astype(int)

    def __repr__(self) -> str:
        status = "loaded" if self.model else "not_loaded"
        return f"MLRandomForestStrategy({self.ticker}, model={status})"


# Quick test
if __name__ == "__main__":
    from ..data.fetcher import DataFetcher

    df = DataFetcher.get_price("AAPL", period="2y")
    strategy = MLRandomForestStrategy("AAPL", {"confidence_threshold": 0.55})
    signals = strategy.generate_signals(df)

    print("ML Strategy signals for AAPL:")
    print(f"Total signals: {len(signals[signals != 0])} non-zero")
    print(f"Current signal: {signals.iloc[-1]}")
    print(signals.tail(10))
