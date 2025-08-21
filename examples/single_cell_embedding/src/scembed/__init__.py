"""Single-cell embedding comparison utilities for slurm_sweep examples."""

from importlib.metadata import version

from .aggregation import scIBAggregator
from .evaluation import IntegrationEvaluator
from .factory import get_method_instance

__all__ = ["IntegrationEvaluator", "scIBAggregator", "get_method_instance"]

__version__ = version("scembed")
