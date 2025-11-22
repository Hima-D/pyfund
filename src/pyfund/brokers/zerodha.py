import pandas as pd
from kiteconnect import KiteConnect

from pyfund.core.broker import Broker
from pyfund.core.broker_registry import register_broker


@register_broker("zerodha")
class ZerodhaBroker(Broker):
    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

    def get_price(self, ticker: str, period: str = "2y", interval: str = "day") -> pd.DataFrame:
        from datetime import datetime, timedelta

        to_date = datetime.today()
        from_date = to_date - timedelta(days=365 * 2)
        instrument = f"NSE:{ticker}" if ticker not in ["NIFTY", "BANKNIFTY"] else f"NFO:{ticker}"
        data = self.kite.historical_data(instrument, from_date, to_date, interval)
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[["open", "high", "low", "close", "volume"]].rename(columns=str.capitalize)

    def get_balance(self):
        return self.kite.margins()

    def place_order(
        self, ticker: str, qty: float, side: str, order_type: str = "market", price: float = None
    ):
        return self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange="NSE",
            tradingsymbol=ticker,
            transaction_type=(
                self.kite.TRANSACTION_TYPE_BUY if side == "buy" else self.kite.TRANSACTION_TYPE_SELL
            ),
            quantity=int(abs(qty)),
            product=self.kite.PRODUCT_CNC,
            order_type=(
                self.kite.ORDER_TYPE_MARKET
                if order_type == "market"
                else self.kite.ORDER_TYPE_LIMIT
            ),
            price=price,
            validity=self.kite.VALIDITY_DAY,
        )

    def get_positions(self):
        positions = self.kite.positions()["net"]
        return {p["tradingsymbol"]: p["quantity"] for p in positions}

    def cancel_all_orders(self):
        orders = self.kite.orders()
        for order in orders:
            if order["status"] not in ["COMPLETE", "REJECTED", "CANCELLED"]:
                self.kite.cancel_order(order["variety"], order["order_id"])
