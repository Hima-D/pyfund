# PyFundLib ⚡

**The most powerful, modern, and production-ready Python framework for algorithmic trading & ML alpha**

From backtesting RSI mean-reversion to running 24/7 autonomous LSTM-powered live trading — all with a clean, type-annotated, zero-boilerplate API.

Built by a quant. For quants who refuse to compromise.

<p align="center">
  <a href="https://pypi.org/project/pyfundlib/"><img src="https://img.shields.io/pypi/v/pyfundlib?style=flat-square&color=blue" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/pyfundlib?style=flat-square" alt="Python">
  <a href="https://github.com/Hima-D/pyfundlib/stargazers"><img src="https://img.shields.io/github/stars/Hima-D/pyfundlib?style=social" alt="Stars"></a>
  <a href="https://github.com/Hima-D/pyfundlib/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Hima-D/pyfundlib?style=flat-square&color=green" alt="License"></a>
  <img src="https://img.shields.io/badge/status-Production%20Ready-success?style=flat-square" alt="Status">
</p>

<p align="center">
  <strong>Clean API • Institutional MLOps • Zero Boilerplate • Real Edge</strong>
</p>

## Why PyFundLib Dominates

| Feature                             | PyFundLib                                  | Backtrader • VectorBT • Others |
|-------------------------------------|--------------------------------------------|--------------------------------|
| Published on PyPI                   | Yes `pip install pyfundlib`                | No                             |
| Modern `src/` layout + full typing  | Yes                                        | Rarely                         |
| Built-in ML (LSTM • XGBoost • RF)   | Yes Full pipelines + MLflow + versioning   | No                             |
| Multi-broker live execution         | Alpaca • Zerodha • IBKR • Binance          | Partial                        |
| Zstd Parquet caching + metadata     | Yes Automatic                              | No                             |
| **Robust Statistical Validation**   | **Yes DSR, PBO, Walk-Forward Analysis**    | Manual/External                |
| **System Monitoring**               | **Yes Real-time CPU/RAM/Orders/Signals**   | No                             |
| One-liner equity curves & reports   | Yes Gorgeous out-of-the-box                | Manual                         |
| CLI + scheduler + auto-retrain      | Yes `pyfundlib live`                       | No                             |
| Dry-run by default                  | Yes No accidental nuclear launches         | Risky                          |

## Installation

```bash
# Core library
pip install pyfundlib

# Full suite including all ML, broker, and utility dependencies
pip install pyfundlib[full]
```

## Quick Start (Smoke Test)

To verify your installation and see the core components in action, run the smoke test:

```bash
# 1. Clone the repo (if you haven't already)
git clone https://github.com/Hima-D/pyfundlib.git
cd pyfundlib

# 2. Setup environment (ensure all dependencies are installed)
pip install -e .[dev]

# 3. Run the smoke test
python tests/x.py
```

A successful run will print: `PYFUNDLIB IS 100% ALIVE` and confirm that data fetching, ML training, backtesting, and reporting are all functional.

## Core Modules

| Module | Description | Key Features |
|---|---|---|
| `data` | Unified data access and management. | `DataFetcher` (yfinance, Alpaca, etc.), intelligent caching, Polars support. |
| `ml` | Institutional-grade Machine Learning pipelines. | `MLPredictor`, `XGBoostModel`, MLflow tracking, hyperparameter optimization (Optuna). |
| `backtester` | Fast, vectorized backtesting engine. | `Backtester` class, `PerformanceReport`, trade logging. |
| `strategies` | Collection of professional trading strategies. | SMA Crossover, RSI Mean Reversion, Pairs Trading, etc. |
| `utils` | Essential utilities for a production system. | **`Scheduler`**, **`SystemMonitor`**, **`StatisticalValidator`**, logging, caching. |

## Development & Testing

All tests are passing and can be run with `pytest`:

```bash
# Run all tests
pytest
```

---
*PyFundLib is a project by Himanshu Dixit. Licensed under MIT.*
