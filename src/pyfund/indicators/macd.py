# src/pyfund/indicators/macd.py
from __future__ import annotations

import pandas as pd


def macd(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    adjust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    This is the standard MACD as used by TradingView, MetaTrader, and most platforms.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices (or any price series).
    fast : int, default 12
        Fast EMA period.
    slow : int, default 26
        Slow EMA period.
    signal : int, default 9
        Signal line EMA period.
    adjust : bool, default True
        If True, uses EMA with `adjust=True` (more accurate, matches TradingView).
        If False, uses simple iterative EMA (slightly different initial values).

    Returns
    -------
    macd_line : pd.Series
        MACD line (fast EMA - slow EMA)
    signal_line : pd.Series
        Signal line (EMA of MACD line)
    histogram : pd.Series
        Histogram (macd_line - signal_line)

    Examples
    --------
    >>> macd_line, signal_line, hist = macd(df['Close'])
    >>> macd_line, signal_line, hist = macd(df['Close'], fast=8, slow=17, signal=9)
    """
    if fast >= slow:
        raise ValueError("fast period must be less than slow period")
    if not all(isinstance(p, int) and p > 0 for p in [fast, slow, signal]):
        raise ValueError("Periods must be positive integers")

    # Use EMA with adjust=True → matches TradingView exactly
    ema_fast = close.ewm(span=fast, adjust=adjust).mean()
    ema_slow = close.ewm(span=slow, adjust=adjust).mean()

    macd_line = ema_fast - ema_slow
    macd_line.name = f"MACD_{fast}_{slow}"

    signal_line = macd_line.ewm(span=signal, adjust=adjust).mean()
    signal_line.name = f"MACD_signal_{signal}"

    histogram = macd_line - signal_line
    histogram.name = f"MACD_hist_{fast}_{slow}_{signal}"

    return macd_line, signal_line, histogram


# Optional: Add a convenience class version for consistency with other indicators
class MACD:
    """
    Class-based MACD indicator (useful for strategy objects or stateful use).
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, adjust: bool = True):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.adjust = adjust

    def __call__(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        return macd(close, fast=self.fast, slow=self.slow, signal=self.signal, adjust=self.adjust)

    def __repr__(self) -> str:
        return f"MACD(fast={self.fast}, slow={self.slow}, signal={self.signal})"
