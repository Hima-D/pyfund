import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def describe_financial(returns):
    if len(returns) == 0:
        return {"error": "empty"}
    r = pd.Series(returns)
    equity = (1 + r).cumprod()
    dd = 1 - equity / equity.cummax()
    return {
        "mean": r.mean(),
        "std": r.std(),
        "sharpe": r.mean() / r.std() * (252**0.5) if r.std() > 0 else 0,
        "skew": r.skew(),
        "kurtosis": r.kurtosis(),
        "var_95": np.percentile(r, 5),
        "cvar_95": r[r <= np.percentile(r, 5)].mean(),
        "max_dd": dd.max(),
        "calmar": (
            (equity.iloc[-1] / equity.iloc[0] - 1) / dd.max() if dd.max() > 0 else float("inf")
        ),
    }


def deflated_sharpe_ratio(returns, trials=5000):
    returns = np.asarray(returns).flatten()
    if len(returns) < 30:
        return 0.0
    sr = returns.mean() / returns.std() * (252**0.5)
    boot = [
        np.random.permutation(returns).mean() / np.random.permutation(returns).std() * (252**0.5)
        for _ in range(trials)
    ]
    boot = [x for x in boot if not np.isnan(x)]
    if not boot:
        return sr
    return max(0, (sr - max(boot)) / (np.std(boot) if np.std(boot) > 0 else 1))


def probabilistic_sharpe_ratio(returns, benchmark=0.0):
    returns = np.asarray(returns).flatten()
    if len(returns) < 2:
        return 0.0
    sr = returns.mean() / returns.std() * (252**0.5)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    n = len(returns)
    denom = (1 - skew * sr + (kurt - 3) * sr**2 / 4) ** 0.5
    if denom == 0:
        return 0.0
    return stats.norm.cdf((sr - benchmark) * (n - 1) ** 0.5 / denom)
