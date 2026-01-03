# src/pyfund/utils/__init__.py

"""
Utilities for PyFundLib:
- Statistical tests and validation
- Logging, caching, plotting, monitoring, scheduling, etc.
"""

from .statistical_tests import (
    StatisticalValidator,           # class
    validator,                      # global instance
    print_report,                   # top-level function
)

from .statistical_validation import (
    print_validation,
    validate_strategy
)

# Other utility modules can be imported here as needed:
# from .logger import get_logger
# from .cache import cache_data
# from .plotter import plot_series
# from .scheduler import run_scheduler
# from .monitor import monitor_metrics
