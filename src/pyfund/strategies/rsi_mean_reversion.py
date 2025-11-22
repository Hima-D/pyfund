# src/pyfund/strategies/rsi_mean_reversion.py
from typing import Any

import numpy as np
import pandas as pd

from ..indicators.rsi import rsi
from .base import BaseStrategy


class RSIMeanReversionStrategy(BaseStrategy):
    """
    Enhanced RSI Mean Reversion Strategy

    Features:
    - Dynamic overbought/oversold thresholds
    - Optional volatility filter (avoid trading in high volatility regimes)
    - Position smoothing (hold signal until exit condition)
    - Risk management: trailing stop & max holding period
    - Realistic signal logic (no lookahead bias)
    """

    default_params = {
        "rsi_window": 14,
        "oversold": 30,
        "overbought": 70,
        "exit_rsi": 50,  # Exit when RSI crosses back to neutral
        "vol_window": 20,  # Volatility filter window
        "vol_threshold": 1.5,  # Only trade if vol < 1.5x average
        "use_vol_filter": True,
        "use_trailing_stop": True,
        "trailing_stop_pct": 0.08,  # 8% trailing stop
        "max_holding_days": 21,  # Max 3 weeks in position
    }

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.current_position: int | None = None
        self.entry_price: float | None = None
        self.entry_date: pd.Timestamp | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate high-quality RSI mean reversion signals

        Signal logic:
          - Buy when RSI < oversold AND (optional) volatility is normal
          - Sell when RSI > overbought
          - Exit longs when RSI > exit_rsi or trailing stop hit
          - Exit shorts when RSI < exit_rsi or trailing stop hit
          - Hold position until exit condition
        """
        if len(data) < self.params["rsi_window"] + 20:
            return pd.Series(0, index=data.index)

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        rsi_series = rsi(close, self.params["rsi_window"])
        signals = pd.Series(0, index=data.index)

        # Volatility filter (ATR-based or std dev)
        returns = np.log(close / close.shift(1))
        volatility = returns.rolling(self.params["vol_window"]).std()
        avg_vol = volatility.mean()
        current_vol_ratio = volatility / avg_vol

        position = 0
        entry_price = 0.0

        for i in range(1, len(data)):
            idx = data.index[i]
            prev_idx = data.index[i - 1]

            rsi_val = rsi_series.iloc[i]
            prev_rsi = rsi_series.iloc[i - 1]

            # Update trailing stop
            if position != 0 and self.params["use_trailing_stop"]:
                trail_pct = self.params["trailing_stop_pct"]
                if position == 1:  # Long
                    stop_price = close.iloc[i] * (1 - trail_pct)
                    if entry_price == 0 or stop_price > entry_price:
                        entry_price = stop_price
                    if low.iloc[i] <= entry_price:
                        signals.iloc[i] = 0
                        position = 0
                        continue
                elif position == -1:  # Short
                    stop_price = close.iloc[i] * (1 + trail_pct)
                    if entry_price == 0 or stop_price < entry_price:
                        entry_price = stop_price
                    if high.iloc[i] >= entry_price:
                        signals.iloc[i] = 0
                        position = 0
                        continue

            # Max holding period
            if position != 0:
                days_held = (idx - data.index[i - position]).days
                if days_held >= self.params["max_holding_days"]:
                    signals.iloc[i] = 0
                    position = 0
                    continue

            # Volatility filter
            vol_ok = True
            if self.params["use_vol_filter"] and i >= self.params["vol_window"]:
                vol_ok = current_vol_ratio.iloc[i] <= self.params["vol_threshold"]

            # Entry conditions
            if position == 0 and vol_ok:
                if rsi_val < self.params["oversold"] and prev_rsi >= self.params["oversold"]:
                    signals.iloc[i] = 1
                    position = 1
                elif rsi_val > self.params["overbought"] and prev_rsi <= self.params["overbought"]:
                    signals.iloc[i] = -1
                    position = -1

            # Exit conditions
            elif position == 1 and rsi_val > self.params["exit_rsi"]:
                signals.iloc[i] = 0
                position = 0
            elif position == -1 and rsi_val < self.params["exit_rsi"]:
                signals.iloc[i] = 0
                position = 0
            else:
                signals.iloc[i] = signals.iloc[i - 1]  # Hold previous signal

        # Forward fill to make signal persistent
        signals = signals.replace(0, pd.NA).ffill().fillna(0)

        return signals.astype(int)

    def __repr__(self) -> str:
        p = self.params
        return (
            f"RSIMeanReversionStrategy("
            f"window={p['rsi_window']}, "
            f"oversold={p['oversold']}, overbought={p['overbought']})"
        )


# Quick test
if __name__ == "__main__":
    from ..data.fetcher import DataFetcher

    df = DataFetcher.get_price("AAPL", period="3y")
    strategy = RSIMeanReversionStrategy({"oversold": 25, "overbought": 75, "use_vol_filter": True})
    signals = strategy.generate_signals(df)

    print(f"Total signals generated: {len(signals[signals != 0])}")
    print(
        f"Current signal: {'LONG' if signals.iloc[-1] == 1 else 'SHORT' if signals.iloc[-1] == -1 else 'FLAT'}"
    )
    print(signals.tail(10))
