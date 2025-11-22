# src/pyfundlib/cli.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import click

from .automation.runner import AutomationRunner
from .backtester.engine import Backtester
from .ml.predictor import MLPredictor
from .ml.tracking import tracker
from .utils.logger import add_file_handler
from .utils.plotter import Plotter


# Global CLI options
class Config:
    def __init__(self):
        self.verbose = False
        self.output_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-o", "--output", type=click.Path(), default="results", help="Output directory")
@click.pass_context
def cli(ctx, verbose: bool, output: str):
    """PyFundLib ⚡ - The Ultimate Algo Trading Framework

    Open-source. Production-ready. Built for alpha.
    """
    ctx.obj = Config()
    ctx.obj.verbose = verbose
    ctx.obj.output_dir = Path(output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx.obj.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    if verbose:
        from .utils.logger import logger

        logger.setLevel(logging.DEBUG)

    add_file_handler(str(ctx.obj.output_dir), "cli.log")
    click.echo(click.style("PyFundLib CLI Started", fg="cyan", bold=True))
    click.echo(f"Output → {ctx.obj.output_dir}\n")

    if ctx.invoked_subcommand is None:
        click.echo(cli.get_help(ctx))


@cli.command()
@click.argument("ticker", type=str)
@click.option("--strategy", "-s", type=str, default="rsi_mean_reversion", help="Strategy name")
@click.option("--years", "-y", type=int, default=5, help="Years of data")
@click.option("--initial-capital", type=float, default=100_000, help="Starting capital")
@click.option("--plot/--no-plot", default=True, help="Show equity curve")
@pass_config
def backtest(config, ticker: str, strategy: str, years: int, initial_capital: float, plot: bool):
    """Run a full backtest with beautiful output"""
    click.echo(click.style(f"Backtesting {ticker.upper()} → {strategy}", fg="green", bold=True))

    bt = Backtester(
        ticker=ticker.upper(),
        strategy_name=strategy,
        years=years,
        initial_capital=initial_capital,
    )

    results = bt.run()

    # Save results
    report_path = (
        config.output_dir / f"backtest_{ticker}_{strategy}_{datetime.now().strftime('%H%M')}.html"
    )
    results.generate_report(report_path)
    click.echo(click.style(f"Report → {report_path}", fg="blue"))

    if plot:
        Plotter.equity_curve(
            results.equity_curve,
            title=f"{ticker} - {strategy} - CAGR: {results.cagr:.1%} | MaxDD: {results.max_drawdown:.1%}",
            save_path=config.output_dir / f"equity_{ticker}.png",
        )

    click.echo(
        click.style(
            f"Backtest complete! CAGR: {results.cagr:+.1%} | Sharpe: {results.sharpe:.2f}",
            bold=True,
        )
    )


@cli.command()
@click.option("--ticker", multiple=True, help="Train specific tickers (default: all)")
@click.option("--trials", type=int, default=150, help="Optuna trials per model")
def train(ticker, trials: int):
    """Train or retrain ML models"""
    click.echo(click.style("ML Model Training Started", fg="magenta", bold=True))

    predictor = MLPredictor()
    if ticker:
        for t in ticker:
            click.echo(f"Training model for {t.upper()}...")
            # Your per-ticker training logic
    else:
        predictor.train_all(trials=trials)

    click.echo(click.style("Training completed!", fg="green", bold=True))


@cli.command()
def live():
    """Start live/paper trading"""
    click.echo(click.style("LIVE TRADING MODE ACTIVATED", fg="red", bold=True))
    click.confirm("Are you sure you want to start live trading?", abort=True)

    runner = AutomationRunner(mode="live")
    runner.start()


@cli.command()
def paper():
    """Start paper trading"""
    click.echo(click.style("Paper Trading Started", fg="yellow", bold=True))
    runner = AutomationRunner(mode="paper")
    runner.start()


@cli.command()
def models():
    """List all registered models"""
    df = tracker.list_models()
    if df.empty:
        click.echo("No models found.")
    else:
        click.echo("\nRegistered Models:")
        click.echo(df.to_string(index=False))


@cli.command()
@click.argument("model_name")
@click.option("--version", type=int)
def promote(model_name: str, version: int):
    """Promote a model to Production"""
    tracker.promote_model(model_name, version)
    click.echo(click.style(f"{model_name} v{version} → Production", fg="green", bold=True))


@cli.command()
def dashboard():
    """Launch Streamlit dashboard"""
    import subprocess
    import sys

    subprocess.call([sys.executable, "-m", "streamlit", "run", "examples/05_dashboard.py"])


# Easter egg
@cli.command(hidden=True)
def alpha():
    click.echo(click.style("You have found the alpha.", fg="bright_magenta", bold=True))
    click.echo("Now go make money. 💰🚀")


if __name__ == "__main__":
    cli()
