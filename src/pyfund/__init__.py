__version__ = "0.1.0"

# Easy imports
from .strategies import *
from .ml.predictor import Predictor
from .automation.scheduler import JobScheduler
from .data.fetcher import fetch_ohlcv  # Example