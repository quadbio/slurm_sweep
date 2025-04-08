from importlib.metadata import version

from ._logging import logger
from .sweep_class import SweepManager

__all__ = ["SweepManager", "logger"]

__version__ = version("slurm_sweep")
