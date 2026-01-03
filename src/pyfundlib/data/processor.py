# src/pyfundlib/data/processor.py
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    High-performance data processing using Polars.
    """

    @staticmethod
    def resample(
        df: Union[pd.DataFrame, pl.DataFrame],
        rule: str = "1w",
        index_col: str = "date",
    ) -> pl.DataFrame:
        """
        Resample OHLCV data using Polars.
        rule: Polars duration string (e.g., '1w', '1d', '30m')
        """
        if isinstance(df, pd.DataFrame):
            # Ensure index is a column
            if df.index.name == index_col or df.index.name is not None:
                df = df.reset_index()
            else:
                df = df.reset_index(names=[index_col])
            ldf = pl.from_pandas(df).lazy()
        else:
            ldf = df.lazy()

        # Polars resample (upsample/group_by_dynamic)
        processed = (
            ldf.group_by_dynamic(index_col, every=rule)
            .agg([
                pl.col("Open").first(),
                pl.col("High").max(),
                pl.col("Low").min(),
                pl.col("Close").last(),
                pl.col("Volume").sum(),
            ])
            .collect()
        )
        
        logger.info("data_resampled", rule=rule, rows=len(processed))
        return processed

    @staticmethod
    def clean_and_fill(
        df: Union[pd.DataFrame, pl.DataFrame],
        method: Literal["ffill", "bfill", "both", "drop"] = "both",
        index_col: str = "date",
    ) -> pl.DataFrame:
        """
        Clean and fill holes using Polars.
        """
        if isinstance(df, pd.DataFrame):
            ldf = pl.from_pandas(df.reset_index() if df.index.name else df).lazy()
        else:
            ldf = df.lazy()

        if method == "drop":
            res = ldf.drop_nulls()
        elif method == "ffill":
            res = ldf.select(pl.all().forward_fill())
        elif method == "bfill":
            res = ldf.select(pl.all().backward_fill())
        elif method == "both":
            res = ldf.select(pl.all().forward_fill().backward_fill())
        else:
            res = ldf

        processed = res.collect()
        logger.info("data_cleaned", method=method, rows=len(processed))
        return processed

    @staticmethod
    def add_returns(
        df: Union[pd.DataFrame, pl.DataFrame],
        price_col: str = "Close",
        log: bool = True,
    ) -> pl.DataFrame:
        """Add arithmetic or log returns."""
        if isinstance(df, pd.DataFrame):
            ldf = pl.from_pandas(df.reset_index() if df.index.name else df).lazy()
        else:
            ldf = df.lazy()

        if log:
            res = ldf.with_columns(
                (pl.col(price_col) / pl.col(price_col).shift(1)).ln().alias("returns")
            )
        else:
            res = ldf.with_columns(
                (pl.col(price_col) / pl.col(price_col).shift(1) - 1).alias("returns")
            )
            
        return res.collect()

    @staticmethod
    def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
        """Back-compatibility helper"""
        pdf = df.to_pandas()
        if "date" in pdf.columns:
            pdf = pdf.set_index("date")
        return pdf
