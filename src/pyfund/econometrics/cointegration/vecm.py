import warnings

import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

warnings.filterwarnings("ignore")


class Cointegration:
    @staticmethod
    def johansen(data, k=1):
        result = coint_johansen(data, 0, k)
        return {
            "rank": sum(result.lr1 > result.cvt[:, 1]),
            "trace_stat": result.lr1.tolist(),
            "critical_5%": result.cvt[:, 1].tolist(),
        }

    @staticmethod
    def vecm_fit(prices_df, rank=1):
        model = VECM(prices_df, k_ar_diff=1, coint_rank=rank)
        return model.fit()

    @staticmethod
    def half_life(spread):
        s = spread.dropna()
        lag = s.shift(1).dropna()
        ret = s.diff().dropna()
        if len(lag) < 10:
            return float("inf")
        beta = np.linalg.lstsq(lag[:-1].values.reshape(-1, 1), ret[1:], rcond=None)[0][0]
        return round(-np.log(2) / beta, 1) if beta < 0 else float("inf")
