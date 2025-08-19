"""Dependency checking for scembed."""

import importlib
import types

from packaging.version import parse

from . import version


class Checker:
    """
    Checks availability and version of a Python module dependency.

    Adapted from the scGLUE package: https://github.com/gao-lab/GLUE

    Parameters
    ----------
    name
        Name of the dependency
    package_name
        Name of the package to check version for (if different from module name)
    vmin
        Minimal required version
    install_hint
        Install hint message to be printed if dependency is unavailable
    """

    def __init__(
        self, name: str, package_name: str | None = None, vmin: str | None = None, install_hint: str | None = None
    ) -> None:
        self.name = name
        self.package_name = package_name or name
        self.vmin = parse(vmin) if vmin else vmin
        vreq = f" (>={self.vmin})" if self.vmin else ""
        self.vreq_hint = f"This function relies on {self.name}{vreq}."
        self.install_hint = install_hint

    def check(self) -> None:
        """Check if the dependency is available and meets the version requirement."""
        try:
            importlib.import_module(self.name)
        except ModuleNotFoundError as e:
            raise RuntimeError(" ".join(filter(None, [self.vreq_hint, self.install_hint]))) from e

        if self.vmin:
            try:
                v = parse(version(self.package_name))
                if v < self.vmin:
                    raise RuntimeError(
                        " ".join(
                            [
                                self.vreq_hint,
                                f"Detected version is {v}.",
                                "Please install a newer version.",
                                self.install_hint or "",
                            ]
                        )
                    )
            except (ImportError, ValueError, TypeError):
                # If version checking fails, just warn but don't fail
                pass


INSTALL_HINTS = types.SimpleNamespace(
    # GPU-accelerated neighbor search
    faiss_gpu="To speed up k-NN search on GPU, install faiss-gpu: pip install faiss-gpu",
    # RAPIDS ecosystem for GPU acceleration
    rapids_singlecell="To use GPU-accelerated single-cell analysis, install rapids-singlecell: "
    "pip install rapids-singlecell",
    # Integration methods - CPU
    pyliger="To use LIGER integration, install pyliger: pip install pyliger",
    scanorama="To use Scanorama integration, install scanorama: pip install scanorama",
    # Integration methods - GPU
    harmony_pytorch="To use GPU-accelerated Harmony, install harmony-pytorch: pip install harmony-pytorch",
    scvi_tools="To use scVI-tools methods (scVI, scANVI), install scvi-tools: pip install scvi-tools",
    scarches="To use scPoli integration, install scarches: pip install scarches",
    # Data analysis and visualization
    lightning="To use PyTorch Lightning loggers, install lightning: pip install lightning",
)

CHECKERS = {
    # GPU-accelerated neighbor search
    "faiss-gpu": Checker("faiss", package_name="faiss-gpu", vmin="1.7.0", install_hint=INSTALL_HINTS.faiss_gpu),
    # RAPIDS ecosystem
    "rapids-singlecell": Checker(
        "rapids_singlecell", package_name="rapids-singlecell", vmin=None, install_hint=INSTALL_HINTS.rapids_singlecell
    ),
    # Integration methods - CPU
    "pyliger": Checker("pyliger", vmin="0.2.0", install_hint=INSTALL_HINTS.pyliger),
    "scanorama": Checker("scanorama", vmin="1.7.0", install_hint=INSTALL_HINTS.scanorama),
    # Integration methods - GPU
    "harmony-pytorch": Checker(
        "harmony", package_name="harmony-pytorch", vmin=None, install_hint=INSTALL_HINTS.harmony_pytorch
    ),
    "scvi-tools": Checker("scvi", package_name="scvi-tools", vmin="1.0.0", install_hint=INSTALL_HINTS.scvi_tools),
    "scarches": Checker("scarches", vmin="0.6.0", install_hint=INSTALL_HINTS.scarches),
    # Data analysis and visualization
    "lightning": Checker("lightning", package_name="lightning", vmin=None, install_hint=INSTALL_HINTS.lightning),
}


def check_deps(*args) -> None:
    """
    Check whether certain dependencies are installed.

    Parameters
    ----------
    args
        A list of dependencies to check
    """
    for item in args:
        if item not in CHECKERS:
            raise RuntimeError(f"Dependency '{item}' is not registered in CHECKERS.")
        CHECKERS[item].check()
