from pyfund.econometrics.core import deflated_sharpe_ratio, describe_financial
from pyfund.econometrics.volatility.garch import VolatilityModels


def run(returns, prices=None, name="Strategy"):
    print("\n" + "═" * 80)
    print(f" PYFUNDLIB 2025 INSTITUTIONAL REPORT: {name} ".center(80))
    print("═" * 80)
    d = describe_financial(returns)
    print(f"Sharpe Ratio       : {d['sharpe']:.3f}")
    print(f"Deflated Sharpe    : {deflated_sharpe_ratio(returns.values):.3f}")
    print(f"Max Drawdown       : {d['max_dd']*100:6.2f}%")
    print(f"Calmar Ratio       : {d['calmar']:.2f}")
    if prices is not None:
        rv = VolatilityModels.realized(prices).iloc[-1]
        print(f"Realized Vol       : {rv:.3f}")
    print("═" * 80 + "\n")
