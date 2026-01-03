# scripts/demo_v2026.py
from __future__ import annotations

import asyncio
import os
import pandas as pd
import polars as pl

from pyfundlib.data.fetcher import DataFetcher
from pyfundlib.data.processor import DataProcessor
from pyfundlib.ml.interpretive import GenAIInterpreter
from pyfundlib.utils.logger import setup_logging, get_logger
from pyfundlib.utils.voice import VoiceInterface

# Initialize
setup_logging()
logger = get_logger("demo_2026")

async def run_demo():
    print("\n--- Phase 2: Polars-Backed High Performance Data ---")
    ticker = "AAPL"
    # Fetch as Polars
    df_pl = DataFetcher.get_price(ticker, period="1y")
    print(f"Fetched {ticker} using Polars. Type: {type(df_pl)}")
    print(df_pl.head(2))
    
    # Resample using Polars logic
    resampled = DataProcessor.resample(df_pl, rule="1w")
    print(f"Resampled to weekly. Rows: {len(resampled)}")
    
    print("\n--- Phase 2/3: Interpretive Risk Layer ---")
    # Mock some risk metrics
    risk_data = {
        "ticker": ticker,
        "weekly_vol": 0.045,
        "VaR_95": -0.021,
        "concentration": "High (100% in Tech)"
    }
    
    interpreter = GenAIInterpreter()
    explanation = interpreter.explain_risk(risk_data)
    print("GenAI Risk Explanation:")
    print(explanation)
    
    print("\n--- Phase 4: Voice-Driven Interactive Interface ---")
    voice = VoiceInterface(use_elevenlabs=False) # Use local fallback for demo
    print("Voice synthesis starting (local)...")
    voice.speak(f"Analysis complete for {ticker}. The Value at Risk is 2.1 percent.")

if __name__ == "__main__":
    asyncio.run(run_demo())
