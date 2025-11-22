# src/pyfund/strategies/hft_micro_reversion.py

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..utils.logger import logger
from .base import BaseStrategy


class HFTMicroReversionStrategy(BaseStrategy):
    """
    High-Frequency Micro-Price Mean Reversion Strategy

    Core idea:
    - Use MICRO-PRICE = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
    - When micro-price deviates significantly from mid → expect reversion
    - Amplified by order flow imbalance and volume spikes
    - Ultra-short holding period (seconds to minutes)

    Edge comes from:
    - Adverse selection avoidance
    - Inventory mean reversion of market makers
    - Liquidity provision dynamics
    """

    default_params = {
        "lookback_ticks": 50,  # Number of recent ticks for stats
        "z_entry": 2.2,  # Enter when |z| > 2.2
        "z_exit": 0.3,  # Exit when |z| < 0.3
        "max_position": 1000,  # Max contracts/shares per signal
        "holding_seconds": 30,  # Force exit after 30s
        "volume_filter": 1.5,  # Only trade if volume > 1.5x avg
        "imbalance_threshold": 0.6,  # Order flow imbalance trigger
    }

    def __init__(self, ticker: str = "ES", params: dict | None = None):
        super().__init__({**self.default_params, **(params or {})})
        self.ticker = ticker.upper()
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.tick_history = []

    def _calculate_micro_price(self, row: pd.Series) -> float:
        """Weighted mid price using bid/ask size"""
        bid_price, bid_size = row["bid_price"], row["bid_size"]
        ask_price, ask_size = row["ask_price"], row["ask_size"]
        total_size = bid_size + ask_size
        if total_size == 0:
            return (bid_price + ask_price) / 2
        return (bid_price * ask_size + ask_price * bid_size) / total_size

    def _order_flow_imbalance(self, df: pd.DataFrame) -> float:
        """OFI = sum(delta buy volume) - sum(delta sell volume)"""
        if len(df) < 2:
            return 0.0
        volume = df["volume"]
        price = df["close"]
        signed_vol = np.where(price.diff() > 0, volume, np.where(price.diff() < 0, -volume, 0))
        return signed_vol.tail(10).sum()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate HFT micro-reversion signals on tick data
        Expected columns: ['close', 'volume', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
        """
        if len(data) < self.params["lookback_ticks"]:
            return pd.Series(0, index=data.index)

        signals = pd.Series(0, index=data.index, dtype=int)

        # Calculate micro-price
        data = data.copy()
        data["micro_price"] = data.apply(self._calculate_micro_price, axis=1)
        data["mid"] = (data["bid_price"] + data["ask_price"]) / 2
        data["deviation"] = data["micro_price"] - data["mid"]

        # Rolling stats
        deviation_mean = data["deviation"].rolling(self.params["lookback_ticks"]).mean()
        deviation_std = data["deviation"].rolling(self.params["lookback_ticks"]).std()

        z_score = (data["deviation"] - deviation_mean) / deviation_std

        # Volume filter
        avg_volume = data["volume"].rolling(20).mean()
        volume_spike = data["volume"] / avg_volume

        # Order flow imbalance
        ofi = self._order_flow_imbalance(data)

        for i in range(self.params["lookback_ticks"], len(data)):
            idx = data.index[i]
            z = z_score.iloc[i]
            vol_ok = volume_spike.iloc[i] > self.params["volume_filter"] or np.isnan(
                volume_spike.iloc[i]
            )
            imbalance = abs(ofi) / (data["volume"].iloc[i - 10 : i].sum() + 1e-6)

            current_time = pd.Timestamp(idx)

            # Force exit on holding time
            if self.position != 0 and self.entry_time is not None:
                if (current_time - self.entry_time).total_seconds() > self.params[
                    "holding_seconds"
                ]:
                    signals.iloc[i] = 0
                    self.position = 0
                    continue

            # Exit conditions
            if self.position != 0 and abs(z) < self.params["z_exit"]:
                signals.iloc[i] = 0
                self.position = 0
                continue

            # Entry conditions
            if self.position == 0 and not np.isnan(z):
                if z > self.params["z_entry"] and vol_ok:
                    signals.iloc[i] = -1  # Short: micro-price too high → revert down
                    self.position = -1
                    self.entry_time = current_time
                elif z < -self.params["z_entry"] and vol_ok:
                    signals.iloc[i] = 1  # Long: micro-price too low → revert up
                    self.position = 1
                    self.entry_time = current_time

            # Hold current position
            elif self.position != 0:
                signals.iloc[i] = self.position

        logger.info(
            f"HFT Micro Reversion | Z-score: {z_score.iloc[-1]:.2f} | "
            f"Position: {self.position} | Signal: {signals.iloc[-1]}"
        )

        return signals

    def __repr__(self):
        return f"HFTMicroReversion({self.ticker}, pos={self.position})"


# Live test (requires tick data)
if __name__ == "__main__":
    # Simulate tick data for ES futures or BTC
    ticker = "BTC-USD"
    df = DataFetcher.get_price(ticker, period="1d", interval="1m")

    # Fake order book data
    df["bid_price"] = df["Close"] * (1 - np.random.uniform(0.0001, 0.0005, len(df)))
    df["ask_price"] = df["Close"] * (1 + np.random.uniform(0.0001, 0.0005, len(df)))
    df["bid_size"] = np.random.uniform(0.1, 10, len(df))
    df["ask_size"] = np.random.uniform(0.1, 10, len(df))

    strategy = HFTMicroReversionStrategy(ticker, {"z_entry": 2.0, "holding_seconds": 60})

    signals = strategy.generate_signals(df)
    trades = signals.diff().abs() > 0
    print(f"Total signals generated: {trades.sum()}")
    print(f"Final position: {signals.iloc[-1]}")
    print(signals.value_counts())
