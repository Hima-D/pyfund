# src/pyfundlib/automation/runner.py
from __future__ import annotations

import sys
from datetime import datetime

from ..utils.logger import get_logger
from .jobs.generate_signals import generate_signals_job
from .jobs.rebalance_portfolio import rebalance_job
from .jobs.retrain_ml import retrain_job
from .scheduler import scheduler

logger = get_logger(__name__)


class AutomationRunner:
    """
    The central nervous system of your autonomous trading fund.
    One command → fully automated alpha generation, signals, and execution.
    """

    def __init__(self, mode: str = "paper"):
        self.mode = mode.lower()
        if self.mode not in ["paper", "live"]:
            raise ValueError("mode must be 'paper' or 'live'")

        logger.info(f"AutomationRunner initialized in {self.mode.upper()} mode")

    def start(self) -> None:
        """Start the full autonomous trading loop"""
        logger.info("PYFUNDLIB AUTONOMOUS TRADING SYSTEM STARTING")
        logger.info(f"Mode: {self.mode.upper()} | Time: {datetime.now()}")

        # === Register all jobs ===
        try:
            # Weekly ML retraining (Sunday 2 AM ET)
            scheduler.add_daily_job(
                func=retrain_job,
                hour=2,
                minute=0,
                day_of_week="sun",
                id="retrain_ml_weekly",
                name="Weekly ML Retraining",
            )

            # Signal generation — every 15 mins during US market hours
            scheduler.add_job(
                func=generate_signals_job,
                trigger="cron",
                minute="*/15",
                hour="9-16",
                day_of_week="mon-fri",
                timezone="US/Eastern",
                id="generate_signals_intraday",
                name="Intraday Signal Generation",
            )

            # Portfolio rebalance — 3:55 PM ET (right before close)
            scheduler.add_job(
                func=rebalance_job,
                trigger="cron",
                hour=15,
                minute=55,
                day_of_week="mon-fri",
                timezone="US/Eastern",
                id="rebalance_daily",
                name="Daily Portfolio Rebalance",
            )

            logger.info("All jobs scheduled successfully")
            logger.info("Your fund is now fully autonomous. Go live your life.")

            # Start the scheduler (non-blocking, graceful shutdown)
            scheduler.start()

        except Exception as e:
            logger.critical(f"Failed to start automation: {e}", exc_info=True)
            sys.exit(1)


# Global runner (CLI friendly)
runner = AutomationRunner(mode="paper")  # default safe


def main():
    """Entry point for CLI: pyfundlib automate"""
    import argparse

    parser = argparse.ArgumentParser(description="Start pyfundlib autonomous trading")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="paper = dry-run, live = real money (be careful!)",
    )
    parser.add_argument(
        "--confirm-live", action="store_true", help="Required flag to start in live mode"
    )

    args = parser.parse_args()

    if args.mode == "live" and not args.confirm_live:
        print("LIVE MODE REQUIRES --confirm-live FLAG")
        print("This will trade real money. Add --confirm-live to proceed.")
        sys.exit(1)

    global runner
    runner = AutomationRunner(mode=args.mode)
    runner.start()


if __name__ == "__main__":
    main()
