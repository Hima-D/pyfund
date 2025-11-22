"""pyfund - End-to-End Python Toolkit for Algo Trading."""

__version__ = "0.1.0"
__author__ = "Himanshu Dixit"

from .data import fetcher, processor, features
from .indicators import rsi, sma, macd
from .strategies import RSIMeanReversionStrategy, MLRandomForestStrategy
from .backtester import Backtester
from .ml.predictor import MLPredictor
from .portfolio.allocator import PortfolioAllocator
from .execution.live import LiveExecutor
from .automation.runner import AutomationRunner

__all__ = [
    "fetcher", "processor", "features",
    "rsi", "sma", "macd",
    "RSIMeanReversionStrategy", "MLRandomForestStrategy",
    "Backtester", "MLPredictor", "PortfolioAllocator",
    "LiveExecutor", "AutomationRunner",
]from .econometrics.core import describe_financial, deflated_sharpe_ratio, probabilistic_sharpe_ratio
