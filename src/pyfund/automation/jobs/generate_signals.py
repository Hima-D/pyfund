# src/pyfundlib/automation/jobs/generate_signals.py
from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Dict, Any
import numpy as np

from ...ml.predictor import MLPredictor
from ...data.fetcher import DataFetcher
from ...data.storage import DataStorage
from ...utils.logger import get_logger
from ...reporting.perf_report import PerformanceReport  # optional for signal quality

logger = get_logger(__name__)

# Configurable watchlist — can be loaded from YAML/DB
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
    "SPY", "QQQ", "IWM", "GLD", "BTC-USD"
]

# Signal threshold (adjust per strategy)
SIGNAL_THRESHOLD = 0.6  # |signal| > 0.6 → strong buy/sell

def generate_signals_job() -> None:
    """
    Daily signal generation job — runs on scheduler (e.g., 4:30 AM ET)
    """
    logger.info("=== Starting Daily Signal Generation Job ===")
    start_time = datetime.now()

    predictor = MLPredictor()
    storage = DataStorage()
    signals_today: Dict[str, Dict[str, Any]] = {}

    for ticker in WATCHLIST:
        try:
            logger.info(f"Generating signal for {ticker}...")

            # 1. Fetch latest data
            df = DataFetcher.get_price(ticker, period="2y")  # or "max"

            if len(df) < 100:
                logger.warning(f"Not enough data for {ticker}")
                continue

            # 2. Get latest prediction
            raw_signal = predictor.predict(ticker, df.tail(252))  # use last year

            if raw_signal is None or len(raw_signal) == 0:
                signal = 0.0
            else:
                signal = float(raw_signal[-1])

            # 3. Convert to position (-1, 0, +1)
            if signal > SIGNAL_THRESHOLD:
                position = 1
                strength = "STRONG BUY"
            elif signal < -SIGNAL_THRESHOLD:
                position = -1
                strength = "STRONG SELL"
            elif abs(signal) > SIGNAL_THRESHOLD / 2:
                position = np.sign(signal)
                strength = "BUY" if position > 0 else "SELL"
            else:
                position = 0
                strength = "HOLD"

            # 4. Store signal
            signal_record = {
                "ticker": ticker,
                "date": datetime.now().date().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "raw_signal": round(signal, 4),
                "position": int(position),
                "strength": strength,
                "confidence": round(abs(signal), 3),
                "model_version": predictor.get_latest(ticker).version if predictor.get_latest(ticker) else "unknown",
            }

            signals_today[ticker] = signal_record

            # 5. Alert on strong signals
            if abs(position) == 1:
                send_alert(ticker, strength, signal_record["confidence"])

            logger.info(f"{ticker}: {strength} (signal={signal:+.3f})")

        except Exception as e:
            logger.error(f"Failed to generate signal for {ticker}: {e}")
            signals_today[ticker] = {
                "ticker": ticker,
                "error": str(e),
                "position": 0,
            }

    # 6. Save all signals
    signals_df = pd.DataFrame(signals_today.values())
    today_str = datetime.now().strftime("%Y-%m-%d")
    storage.save(signals_df, name=f"signals/daily_signals_{today_str}")

    # 7. Summary
    active = len([s for s in signals_today.values() if s.get("position", 0) != 0])
    logger.info(f"Signal generation complete | Active positions: {active}/{len(WATCHLIST)}")
    logger.info(f"Time taken: {(datetime.now() - start_time).seconds}s")

    # Optional: send summary
    send_daily_summary(signals_df)


def send_alert(ticker: str, direction: str, confidence: float) -> None:
    """Send real-time alert via Telegram, Discord, Email, etc."""
    message = f"🚨 {direction} SIGNAL\n{ticker}\nConfidence: {confidence:.1%}\nTime: {datetime.now().strftime('%H:%M:%S')}"
    
    # Replace with your real notifier
    try:
        import telegram  # pip install python-telegram-bot
        bot = telegram.Bot(token="YOUR_TOKEN")
        bot.send_message(chat_id="@your_channel", text=message)
    except:
        logger.debug(f"Alert (dry-run): {message}")


def send_daily_summary(df: pd.DataFrame) -> None:
    """End-of-day summary"""
    buys = len(df[df["position"] == 1])
    sells = len(df[df["position"] == -1])
    summary = f"Daily Signals Summary {datetime.now().date()}\n\n"
    summary += f"BUY: {buys} | SELL: {sells} | HOLD: {len(df) - buys - sells}\n\n"
    summary += "Top Signals:\n"
    top = df[df["position"] != 0].sort_values("confidence", key=abs, ascending=False).head(5)
    for _, row in top.iterrows():
        summary += f"{row['strength']} {row['ticker']} ({row['confidence']:.1%})\n"

    logger.info(summary)
    # send via email/slack/etc.