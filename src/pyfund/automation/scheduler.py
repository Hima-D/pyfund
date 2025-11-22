# src/pyfundlib/automation/scheduler.py
from __future__ import annotations

import signal
import sys
from collections.abc import Callable

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.executors.pool import ProcessPoolExecutor, ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Scheduler:
    """
    Production-ready automation scheduler for pyfundlib.
    Never blocks. Never dies. Always on time.
    """

    def __init__(
        self,
        jobstore: str = "memory",  # "memory" or "sqlite"
        db_url: str = "sqlite:///jobs.db",
        max_workers: int = 10,
        coalesce: bool = True,
        max_instances: int = 3,
    ):
        jobstores = {}
        executors = {
            "default": ThreadPoolExecutor(max_workers),
            "processpool": ProcessPoolExecutor(5),
        }
        job_defaults = {
            "coalesce": coalesce,
            "max_instances": max_instances,
            "misfire_grace_time": 60,
        }

        if jobstore == "sqlite":
            jobstores["default"] = SQLAlchemyJobStore(url=db_url)

        self.sched = BackgroundScheduler(
            jobstores=jobstores or None,
            executors=executors,
            job_defaults=job_defaults,
        )

        # Event listeners
        self.sched.add_listener(self._job_error_listener, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
        self.sched.add_listener(self._job_success_listener, EVENT_JOB_EXECUTED)

        self._setup_signal_handlers()

        logger.info(f"Scheduler initialized | Jobstore: {jobstore} | Workers: {max_workers}")

    def add_job(
        self,
        func: Callable,
        trigger: str = "cron",
        id: str | None = None,
        name: str | None = None,
        **trigger_args,
    ) -> None:
        """Add a job with smart defaults"""
        if trigger == "cron":
            trigger_obj = CronTrigger(**trigger_args)
        else:
            raise ValueError("Only cron trigger supported for now")

        job = self.sched.add_job(
            func,
            trigger=trigger_obj,
            id=id or func.__name__,
            name=name or func.__name__.replace("_", " ").title(),
            replace_existing=True,
        )
        logger.info(f"Job added: {job.name} | Trigger: {trigger_args}")

    def add_daily_job(
        self,
        func: Callable,
        hour: int,
        minute: int = 0,
        timezone: str = "US/Eastern",
        **kwargs,
    ) -> None:
        """Convenience for daily jobs"""
        self.add_job(
            func,
            trigger="cron",
            hour=hour,
            minute=minute,
            timezone=timezone,
            **kwargs,
        )

    def start(self) -> None:
        """Start the scheduler (non-blocking)"""
        if self.sched.running:
            logger.warning("Scheduler already running")
            return

        logger.info("Starting automation scheduler...")
        self.sched.start()

        # Default jobs (uncomment to enable)
        # from .jobs.generate_signals import generate_signals_job
        # from .jobs.rebalance_portfolio import rebalance_job
        # from .jobs.retrain_ml import retrain_ml_job

        # self.add_daily_job(generate_signals_job, hour=4, minute=30)
        # self.add_daily_job(rebalance_job, hour=15, minute=55)
        # self.add_daily_job(retrain_ml_job, hour=2, minute=0, day_of_week="sun")

        try:
            logger.info("Scheduler running — press Ctrl+C to stop")
            while True:
                pass  # Keep main thread alive
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down scheduler...")
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown"""
        if self.sched.running:
            self.sched.shutdown(wait=True)
        logger.info("Scheduler stopped gracefully")

    def _setup_signal_handlers(self):
        """Handle SIGTERM/SIGINT gracefully (Docker/K8s friendly)"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum} — shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _job_error_listener(self, event):
        logger.error(f"Job {event.job_id} failed/missed: {event.exception}")

    def _job_success_listener(self, event):
        logger.debug(f"Job {event.job_id} executed successfully")


# Global scheduler instance
scheduler = Scheduler(jobstore="sqlite")  # Persists across restarts
