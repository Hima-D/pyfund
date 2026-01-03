# src/pyfundlib/data/fetcher.py
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import pandas as pd
import polars as pl
import yfinance as yf

from pyfundlib.config import settings
from pyfundlib.core import broker_registry
from pyfundlib.utils.cache import cached_function
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


Interval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo"
]

Source = Literal["yfinance", "alpaca", "zerodha", "ibkr", "binance", "polygon"]


class DataFetcher:
    """
    Unified, smart data fetcher with Polars support and institutional logging.
    """

    @staticmethod
    @cached_function(
        dir_name="price_data",
        key_lambda=lambda *a, **kw: "_".join([
            kw.get("source", "yf"),
            a[0] if isinstance(a[0], str) else "-".join(sorted(a[0])),
            kw.get("interval", "1d"),
            kw.get("period", "max"),
            kw.get("start", "")[:10],
            kw.get("end", "")[:10],
        ]),
        expire_seconds=lambda **kw: (
            0 if kw.get("interval", "1d") in ("1m", "2m", "5m", "15m", "30m", "60m")
            else 7 * 24 * 60 * 60
        ),
    )
    def get_price(
        ticker: Union[str, Sequence[str]],
        *,
        period: Optional[str] = "max",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: Interval = "1d",
        source: Source = "yfinance",
        prepost: bool = False,
        auto_adjust: bool = True,
        keep_na: bool = False,
        cache: bool = True,
        as_polars: Optional[bool] = None,
        **source_kwargs: Any,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Fetch OHLCV data with intelligent caching and Polars support.
        """
        if as_polars is None:
            as_polars = settings.use_polars

        df = DataFetcher._fetch_and_clean(
            ticker, period, start, end, interval, source,
            prepost, auto_adjust, keep_na, source_kwargs
        )

        if as_polars:
            return pl.from_pandas(df.reset_index())
        return df

    @staticmethod
    def _fetch_and_clean(
        ticker: Union[str, Sequence[str]],
        period: Optional[str],
        start: Optional[str],
        end: Optional[str],
        interval: Interval,
        source: Source,
        prepost: bool,
        auto_adjust: bool,
        keep_na: bool,
        source_kwargs: dict,
    ) -> pd.DataFrame:
        fetch_func = broker_registry.get_data_fetcher(source)

        logger.info("fetching_data", ticker=ticker, source=source, interval=interval)
        
        df = fetch_func(
            ticker=ticker,
            period=period,
            start=start,
            end=end,
            interval=interval,
            prepost=prepost,
            auto_adjust=auto_adjust,
            **source_kwargs,
        )

        if df is None or df.empty:
            logger.error("data_fetch_failed", ticker=ticker, source=source)
            raise ValueError(f"No data for {ticker} from {source}")

        if isinstance(ticker, (list, tuple)) and isinstance(df.columns, pd.MultiIndex):
            df = df.swaplevel(axis=1).sort_index(axis=1)

        if auto_adjust and "Adj Close" in df.columns:
            ratio = df["Adj Close"] / df["Close"].replace(0, pd.NA)
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = df[col] * ratio
            df["Close"] = df["Adj Close"]

        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]

        if not keep_na:
            df = df.dropna()

        df.index.name = "date"
        
        logger.debug("data_fetched", rows=len(df), ticker=ticker)
        return df

    @staticmethod
    async def get_price_async(ticker: str, **kwargs) -> Union[pd.DataFrame, pl.DataFrame]:
        """Async wrapper for data fetching"""
        return await asyncio.to_thread(DataFetcher.get_price, ticker, **kwargs)

    @staticmethod
    async def get_multiple_async(tickers: Sequence[str], **kwargs) -> dict[str, Union[pd.DataFrame, pl.DataFrame]]:
        """Fetch multiple tickers concurrently"""
        tasks = [DataFetcher.get_price_async(t, **kwargs) for t in tickers]
        results = await asyncio.gather(*tasks)
        return dict(zip(tickers, results))


def _fetch_yfinance(
    ticker: Union[str, Sequence[str]],
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    prepost: bool = False,
    auto_adjust: bool = True,
    **kwargs,
) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period or "max",
        start=start,
        end=end,
        interval=interval,
        prepost=prepost,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
        **kwargs,
    )
    return df


broker_registry.register_data_fetcher("yfinance", _fetch_yfinance)
broker_registry.register_data_fetcher("yf", _fetch_yfinance)
broker_registry.register_data_fetcher("yahoo", _fetch_yfinance)