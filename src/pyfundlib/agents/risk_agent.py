# src/pyfundlib/agents/risk_agent.py
from __future__ import annotations

from typing import Any, Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from pyfundlib.data.fetcher import DataFetcher
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


class ComputeVaRSchema(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    confidence: float = Field(default=0.95, description="Confidence level (0.95, 0.99)")


class ComputeVaRTool(BaseTool):
    name: str = "compute_var"
    description: str = "Compute the historical Value-at-Risk (VaR) for a ticker."
    args_schema: Type[BaseModel] = ComputeVaRSchema

    def _run(self, ticker: str, confidence: float = 0.95) -> str:
        try:
            df = DataFetcher.get_price(ticker, period="1y", as_polars=False)
            returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            
            var = np.percentile(returns, (1 - confidence) * 100)
            
            return (
                f"Historical VaR for {ticker} at {confidence:.0%} confidence: {var:.2%}.\n"
                f"This means there is a {(1-confidence):.0%} chance of a loss exceeding {abs(var):.2%} in a single day."
            )
        except Exception as e:
            return f"Error computing VaR: {str(e)}"
