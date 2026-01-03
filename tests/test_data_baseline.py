import pytest
import pandas as pd
import polars as pl
from pyfundlib.data.fetcher import DataFetcher
from pyfundlib.data.processor import DataProcessor

def test_data_fetcher_get_price_mock(mocker):
    # Mock yfinance download
    mock_df = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Volume": [1000, 1100]
    }, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
    
    mocker.patch("yfinance.download", return_value=mock_df)
    
    # By default it returns Polars now
    df = DataFetcher.get_price("AAPL", period="2d", cache=False, as_polars=True)
    
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 2
    assert "Close" in df.columns
    assert "date" in df.columns

def test_data_processor_resample():
    df = pl.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "Open": [100.0, 101.0, 102.0, 103.0],
        "High": [102.0, 103.0, 104.0, 105.0],
        "Low": [99.0, 100.0, 101.0, 102.0],
        "Close": [101.0, 102.0, 103.0, 104.0],
        "Volume": [1000, 1100, 1200, 1300]
    })
    
    # Resample to 2-day frequency ('2d' in Polars)
    resampled = DataProcessor.resample(df, rule="2d")
    
    assert len(resampled) == 3
    assert resampled["Close"][0] == 101.0
    assert resampled["High"].max() == 105.0

def test_data_processor_clean_and_fill():
    df = pl.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "Close": [100.0, None, 102.0, None]
    })
    
    filled = DataProcessor.clean_and_fill(df, method="ffill")
    
    # In Polars, ffill leave the last one None if it's the start, but here it's the end.
    # Wait, ffill will fill the 2nd row with 100, and 4th row with 102.
    assert filled["Close"].null_count() == 0
    assert filled["Close"][1] == 100.0
    
    filled_both = DataProcessor.clean_and_fill(df, method="both")
    assert filled_both["Close"].null_count() == 0
    assert filled_both["Close"][3] == 102.0
