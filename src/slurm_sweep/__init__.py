from importlib.metadata import version

from .sweep_class import SweepManager

__all__ = ["SweepManager"]

__version__ = version("slurm_sweep")
