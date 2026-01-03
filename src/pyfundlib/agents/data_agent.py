# src/pyfundlib/agents/data_agent.py
from __future__ import annotations

from typing import Any, Optional, Type

import pandas as pd
import polars as pl
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from pyfundlib.data.fetcher import DataFetcher
from pyfundlib.data.processor import DataProcessor
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


class FetchPriceSchema(BaseModel):
    """Input for FetchPriceTool"""
    ticker: str = Field(..., description="The stock ticker symbol (e.g., AAPL, TSLA)")
    period: str = Field(default="1y", description="Time period (1mo, 6mo, 1y, 5y, max)")
    interval: str = Field(default="1d", description="Data interval (1m, 5m, 1h, 1d)")


class FetchPriceTool(BaseTool):
    name: str = "fetch_price_data"
    description: str = "Fetch historical OHLCV price data for a given ticker."
    args_schema: Type[BaseModel] = FetchPriceSchema

    def _run(self, ticker: str, period: str = "1y", interval: str = "1d") -> str:
        try:
            logger.info("agent_fetching_data", ticker=ticker, period=period)
            df = DataFetcher.get_price(ticker=ticker, period=period, interval=interval, as_polars=False)
            # Return summary for the agent
            summary = (
                f"Successfully fetched {len(df)} rows for {ticker}.\n"
                f"Start: {df.index.min()}, End: {df.index.max()}\n"
                f"Columns: {list(df.columns)}\n"
                f"Last Close: {df['Close'].iloc[-1]:.2f}"
            )
            return summary
        except Exception as e:
            return f"Error fetching data: {str(e)}"


class ResampleDataSchema(BaseModel):
    """Input for ResampleDataTool"""
    ticker: str = Field(..., description="Ticker symbol")
    rule: str = Field(default="1w", description="Polars duration string (1w, 1mo, 1d)")


class ResampleDataTool(BaseTool):
    name: str = "resample_ohlcv"
    description: str = "Resample OHLCV data to a different frequency (e.g., daily to weekly)."
    args_schema: Type[BaseModel] = ResampleDataSchema

    def _run(self, ticker: str, rule: str = "1w") -> str:
        try:
            # Re-fetch or use state? For now, we fetch
            df = DataFetcher.get_price(ticker=ticker, period="2y", as_polars=True)
            resampled = DataProcessor.resample(df, rule=rule)
            
            return (
                f"Resampled {ticker} to {rule} frequency.\n"
                f"New row count: {len(resampled)}\n"
                f"Recent Close: {resampled['Close'][-1]:.2f}"
            )
        except Exception as e:
            return f"Error resampling: {str(e)}"
