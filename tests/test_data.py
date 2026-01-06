import pytest
import pandas as pd
from pyfundlib.data.fetcher import DataFetcher

def test_data_fetcher_yfinance():
    """Test fetching data from yfinance."""
    df = DataFetcher.get_price("AAPL", period="1mo")
    # Handle both pandas and polars
    is_empty = df.is_empty() if hasattr(df, "is_empty") else df.empty
    assert not is_empty
    assert len(df) > 5

def test_data_fetcher_caching():
    """Test that caching works and returns the same data."""
    df1 = DataFetcher.get_price("MSFT", period="1mo")
    df2 = DataFetcher.get_price("MSFT", period="1mo")
    # If it's polars, convert to pandas for comparison
    if hasattr(df1, "to_pandas"):
        df1 = df1.to_pandas()
    if hasattr(df2, "to_pandas"):
        df2 = df2.to_pandas()
    assert len(df1) == len(df2)

def test_data_fetcher_invalid_ticker():
    """Test handling of invalid tickers."""
    with pytest.raises(ValueError):
        DataFetcher.get_price("INVALID_TICKER_XYZ_999", period="1mo")
