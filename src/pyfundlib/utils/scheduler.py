# src/pyfundlib/utils/scheduler.py
from __future__ import annotations

from typing import Any, Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)

class Scheduler:
    """
    Institutional-grade job scheduler for pyfundlib.
    Supports persistent jobs via SQLite and background execution.
    """

    def __init__(self, db_url: str = "sqlite:///pyfund_jobs.sqlite"):
        jobstores = {
            'default': SQLAlchemyJobStore(url=db_url)
        }
        self.scheduler = BackgroundScheduler(jobstores=jobstores)
        logger.info(f"Scheduler initialized | Jobstore: {db_url}")

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

    def add_job(
        self, 
        func: Callable, 
        trigger: str = "interval", 
        job_id: Optional[str] = None,
        **trigger_args: Any
    ):
        """
        Add a job to the scheduler.
        Trigger can be 'interval', 'cron', or 'date'.
        """
        job = self.scheduler.add_job(
            func, 
            trigger=trigger, 
            id=job_id, 
            replace_existing=True, 
            **trigger_args
        )
        logger.info(f"Job added: {job.id} | Trigger: {trigger} | Args: {trigger_args}")
        return job

    def remove_job(self, job_id: str):
        self.scheduler.remove_job(job_id)
        logger.info(f"Job removed: {job_id}")

    def list_jobs(self):
        return self.scheduler.get_jobs()
