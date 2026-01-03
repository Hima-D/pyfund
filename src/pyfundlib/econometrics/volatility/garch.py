import warnings

import numpy as np
from arch import arch_model

warnings.filterwarnings("ignore")


class VolatilityModels:
    @staticmethod
    def fit(returns, model="GARCH", p=1, q=1, dist="normal"):
        r = returns * 100
        if model == "EGARCH":
            m = arch_model(r, vol="EGarch", p=p, o=1, q=q, dist=dist)
        elif model == "GJR":
            m = arch_model(r, vol="Garch", p=p, o=1, q=q, dist=dist)
        elif model == "APARCH":
            m = arch_model(r, vol="APARCH", p=p, q=q, dist=dist)
        else:
            m = arch_model(r, vol="Garch", p=p, q=q, dist=dist)
        res = m.fit(disp="off")
        return {
            "model": model,
            "aic": res.aic,
            "params": res.params.to_dict(),
            "volatility": res.conditional_volatility / 100,
            "forecast": res.forecast(horizon=5).variance.iloc[-1] / 10000,
        }

    @staticmethod
    def realized(prices, window=22):
        return np.log(prices).diff().rolling(window).std() * np.sqrt(252)
