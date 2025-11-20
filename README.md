# PyFund: Python for Modern Funds & Algo Trading

Build, backtest, optimize, and automate quantitative strategies with ML-powered predictions.

## Install
pip install -e .[dev]

## Quickstart
from pyfund import RSIMeanReversion, Predictor
from pyfund.data.fetcher import fetch_ohlcv

df = fetch_ohlcv('AAPL')
strategy = RSIMeanReversion()
signals = strategy.generate_signals(df)
predictor = Predictor()
pred = predictor.predict_next_day(df)

## CLI
pyfund backtest --strategy rsi --symbol AAPL
pyfund schedule --job retrain_ml --cron "0 0 * * *"