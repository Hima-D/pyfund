import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from pyfund.core.broker_registry import register_broker


@register_broker("alpaca")
class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def get_price(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        from datetime import datetime, timedelta

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=730),
        )
        bars = self.data_client.get_stock_bars(request).df
        if ticker in bars.index.get_level_values(0):
            df = bars.xs(ticker, level=0)[["open", "high", "low", "close", "volume"]]
            df.index.name = "date"
            return df.rename(columns=str.capitalize)
        return pd.DataFrame()

    def place_order(self, ticker: str, qty: float, side: str, order_type: str = "market"):
        self.trading_client.submit_order(
            symbol=ticker, qty=abs(qty), side=side, type=order_type, time_in_force="gtc"
        )
