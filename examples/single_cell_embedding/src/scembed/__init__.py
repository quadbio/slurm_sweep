"""Single-cell embedding comparison utilities for slurm_sweep examples."""

from importlib.metadata import version

from .aggregation import scIBAggregator
from .evaluation import IntegrationEvaluator

__all__ = ["IntegrationEvaluator", "scIBAggregator"]

__version__ = version("scembed")
