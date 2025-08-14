"""Base class for integration methods."""

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import anndata as ad


class BaseIntegrationMethod(ABC):
    """Abstract base class for single-cell integration methods."""

    def __init__(self, adata: ad.AnnData, output_dir: str | Path | None = None, **kwargs):
        """
        Initialize the integration method.

        Parameters
        ----------
        adata
            Annotated data object to validate and store.
        output_dir
            Directory for saving outputs. If None, creates a temporary directory.
        **kwargs
            Method-specific parameters.
        """
        self.name = self.__class__.__name__.replace("Method", "")
        self.params = kwargs
        self.is_fitted = False
        self.embedding_key = f"X_{self.name.lower()}"

        # Common data keys - can be overridden if needed
        self.batch_key = "batch"
        self.cell_type_key = "cell_type"
        self.hvg_key = "highly_variable"
        self.counts_layer = "counts"

        # Validate and store the data
        self.validate_adata(adata)
        self.adata = adata

        # Setup output directories
        self._temp_dir = None  # Store TemporaryDirectory object to prevent premature deletion
        if output_dir is None:
            self._temp_dir = TemporaryDirectory()
            output_dir = Path(self._temp_dir.name)
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # Create sub-directories
        self.output_dir = output_dir
        self.models_dir = output_dir / "models"
        self.logs_dir = output_dir / "logs"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialized {self.name} method, saving outputs to '{self.output_dir}'.")

    def validate_adata(self, adata: ad.AnnData) -> None:
        """
        Validate the AnnData object has required keys and structure.

        Parameters
        ----------
        adata
            Annotated data object to validate.

        Raises
        ------
        ValueError
            If required keys are missing or data is malformed.
        """
        # Check required observation keys
        if self.batch_key not in adata.obs.columns:
            raise ValueError(f"Batch key '{self.batch_key}' not found in adata.obs")

        # Check if counts layer exists (if not using X directly)
        if self.counts_layer != "X" and self.counts_layer not in adata.layers:
            print(f"Warning: Counts layer '{self.counts_layer}' not found in adata.layers")

        # Check for highly variable genes (most methods will need this)
        if self.hvg_key not in adata.var.columns:
            print(f"Warning: HVG key '{self.hvg_key}' not found in adata.var")

        print(f"Data validation passed for {self.name} method.")

    @abstractmethod
    def fit(self) -> None:
        """Fit the integration method to the data.

        Uses self.adata which was validated during initialization.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self) -> ad.AnnData:
        """Transform the data and add embedding to obsm.

        Uses self.adata which was validated during initialization.

        Returns
        -------
        ad.AnnData
            Data with integration embedding added to .obsm[self.embedding_key].
        """
        raise NotImplementedError

    def fit_transform(self) -> ad.AnnData:
        """
        Fit the method and transform the data.

        Returns
        -------
        ad.AnnData
            Data with integration embedding.
        """
        self.fit()
        return self.transform()

    def save_model(self, path: Path) -> Path | None:
        """
        Save the trained model (for deep learning methods).

        Parameters
        ----------
        path
            Directory to save the model.

        Returns
        -------
        Optional[Path]
            Path to saved model file, None if method doesn't support saving.
        """
        # Default implementation - subclasses can override
        _ = path  # Silence unused parameter warning
        return None

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the fitted model.

        Returns
        -------
        Dict[str, Any]
            Dictionary with model information.
        """
        return {
            "method": self.name,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "embedding_key": self.embedding_key,
        }

    def __repr__(self) -> str:
        """String representation of the method."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        status = "fitted" if self.is_fitted else "not fitted"
        data_info = f"{self.adata.n_obs} cells × {self.adata.n_vars} genes"
        return f"{self.__class__.__name__}({params_str}) [{status}, {data_info}]"
