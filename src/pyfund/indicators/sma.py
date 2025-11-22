# src/pyfund/indicators/sma.py
from __future__ import annotations

import pandas as pd


def sma(
    series: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
    center: bool = False,
    fillna: float | None = None,
) -> pd.Series:
    """
    Simple Moving Average (SMA) — with smart defaults and flexibility.

    Parameters
    ----------
    series : pd.Series
        Input price series (Close, High, etc.)
    window : int
        Lookback period
    min_periods : int, optional
        Minimum number of observations in window required to have a value.
        Defaults to `window` (standard behavior), set to 1 for partial averages.
    center : bool, default False
        If True, computes centered (two-sided) moving average.
        Useful for smoothing without lag in offline analysis.
    fillna : float, optional
        Fill NaN values with a constant (e.g., first valid price, or 0)

    Returns
    -------
    pd.Series
        SMA values with name: "SMA_{window}"

    Examples
    --------
    >>> df['sma_20'] = sma(df['Close'], 20)
    >>> df['sma_50_partial'] = sma(df['Close'], 50, min_periods=1)  # ramps up early
    >>> df['sma_200_centered'] = sma(df['Close'], 200, center=True)
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if min_periods is None:
        min_periods = window

    if center:
        # Centered SMA: average of past/future values (no lag, but lookahead!)
        result = series.rolling(window=window * 2 + 1, min_periods=window + 1, center=True).mean()
        result.name = f"SMA_{window}_centered"
    else:
        result = series.rolling(window=window, min_periods=min_periods).mean()
        result.name = f"SMA_{window}"

    if fillna is not None:
        result = result.fillna(fillna)

    return result


# Optional: Class version for optimization / strategy reuse
class SMA:
    """
    Callable SMA class — perfect for hyperparameter tuning.
    """

    def __init__(
        self,
        window: int,
        min_periods: int | None = None,
        center: bool = False,
    ):
        self.window = window
        self.min_periods = min_periods or window
        self.center = center

    def __call__(self, series: pd.Series) -> pd.Series:
        return sma(series, self.window, min_periods=self.min_periods, center=self.center)

    def __repr__(self) -> str:
        return f"SMA({self.window}, min_periods={self.min_periods}, center={self.center})"
