# src/pyfund/indicators/rsi.py
from __future__ import annotations

import pandas as pd


def rsi(
    close: pd.Series,
    *,
    window: int = 14,
    method: str = "wilder",
    fillna: bool = True,
) -> pd.Series:
    """
    Relative Strength Index (RSI) — exactly as implemented in TradingView, Thinkorswim, etc.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices
    window : int, default 14
        Lookback period
    method : str, default "wilder"
        - "wilder": Original Wilder smoothing (1/(window) decay) → matches TradingView
        - "sma": Simple moving average of gains/losses (used in early implementations)
    fillna : bool, default True
        Replace NaN values with neutral 50.0 for first `window` periods

    Returns
    -------
    pd.Series
        RSI values between 0 and 100
        Name: "RSI_{window}"

    Examples
    --------
    >>> df['rsi_14'] = rsi(df['Close'])
    >>> df['rsi_7'] = rsi(df['Close'], window=7)
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if method not in {"wilder", "sma"}:
        raise ValueError("method must be 'wilder' or 'sma'")

    delta = close.diff().dropna()

    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    if method == "wilder":
        # Wilder's smoothing (most accurate & widely used)
        roll_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        roll_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    else:
        # Simple moving average (legacy)
        roll_up = up.rolling(window=window, min_periods=window).mean()
        roll_down = down.rolling(window=window, min_periods=window).mean()

    rs = roll_up / roll_down
    rsi_series = 100 - (100 / (1 + rs))

    # Reindex to original series length
    rsi_series = rsi_series.reindex(close.index)

    if fillna:
        rsi_series = rsi_series.fillna(50.0)  # Neutral value before enough data

    rsi_series.name = f"RSI_{window}"
    return rsi_series


# Optional: Class version for strategy objects / optimization
class RSI:
    """Callable RSI class for hyperparameter tuning or stateful use."""

    def __init__(self, window: int = 14, method: str = "wilder"):
        self.window = window
        self.method = method

    def __call__(self, close: pd.Series) -> pd.Series:
        return rsi(close, window=self.window, method=self.method)

    def __repr__(self) -> str:
        return f"RSI(window={self.window}, method={self.method})"
