"""Base class for integration methods."""

import gzip
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from slurm_sweep._logging import logger


class BaseIntegrationMethod(ABC):
    """Abstract base class for single-cell integration methods."""

    def __init__(
        self,
        adata: ad.AnnData,
        output_dir: str | Path | None = None,
        validate_spatial: bool = False,
        batch_key: str = "batch",
        cell_type_key: str = "cell_type",
        hvg_key: str = "highly_variable",
        counts_layer: str = "counts",
        spatial_key: str = "spatial",
        **kwargs,
    ):
        """
        Initialize the integration method.

        Parameters
        ----------
        adata
            Annotated data object to validate and store.
        output_dir
            Directory for saving outputs. If None, creates a temporary directory.
        validate_spatial
            Whether to validate spatial data requirements.
        batch_key
            Key in adata.obs for batch information.
        cell_type_key
            Key in adata.obs for cell type information.
        hvg_key
            Key in adata.var for highly variable genes.
        counts_layer
            Key in adata.layers for count data.
        spatial_key
            Key in adata.obsm for spatial coordinates.
        **kwargs
            Method-specific parameters.
        """
        self.name = self.__class__.__name__.replace("Method", "")
        self.params = kwargs
        self.is_fitted = False
        self.embedding_key = f"X_{self.name.lower()}"

        # Data keys - configurable for different datasets
        self.batch_key = batch_key
        self.cell_type_key = cell_type_key
        self.hvg_key = hvg_key
        self.counts_layer = counts_layer
        self.spatial_key = spatial_key

        # Validate and store the data
        self.validate_adata(adata)
        if validate_spatial:
            self.validate_spatial_adata(adata)
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
        self.embedding_dir = output_dir / "embeddings"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized %s method, saving outputs to '%s'.", self.name, self.output_dir)

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
            logger.warning("Counts layer '%s' not found in adata.layers", self.counts_layer)
        else:
            count_data = adata.layers[self.counts_layer] if self.counts_layer != "X" else adata.X
            is_integer = np.all((count_data.data if issparse(count_data) else count_data) % 1 == 0)

            if not is_integer:
                logger.warning("Counts layer '%s' contains non-integer values", self.counts_layer)

        # Check for highly variable genes (most methods will need this)
        if self.hvg_key not in adata.var.columns:
            logger.warning("HVG key '%s' not found in adata.var. Initializing as all genes HVG.", self.hvg_key)
            adata.var[self.hvg_key] = True

        logger.info("Data validation passed for %s method.", self.name)

    def validate_spatial_adata(self, adata: ad.AnnData) -> None:
        """
        Validate spatial-specific data requirements.

        Parameters
        ----------
        adata
            Annotated data object to validate.

        Raises
        ------
        ValueError
            If required spatial keys are missing or data is malformed.
        """
        # Check for spatial coordinates using the configured spatial_key
        if self.spatial_key not in adata.obsm:
            raise ValueError(f"Spatial coordinates not found. Expected '{self.spatial_key}' in adata.obsm")

        # Check spatial coordinates format
        spatial_coords = adata.obsm[self.spatial_key]
        if spatial_coords.shape[1] != 2:
            raise ValueError("Spatial coordinates must have 2 dimensions (x, y)")

        # Check for precomputed spatial neighbors (methods will compute these if missing)
        spatial_keys = [
            key
            for key in adata.obsm.keys() | adata.obsp.keys()
            if any(prefix in key.lower() for prefix in ["spatial", "index_neighbor", "distance_neighbor"])
        ]

        if not spatial_keys:
            logger.warning("No precomputed spatial neighbors found. Spatial methods will compute these during setup.")

        logger.info("Spatial data validation passed for %s method.", self.name)

    @abstractmethod
    def fit(self) -> None:
        """Fit the integration method to the data.

        Uses self.adata which was validated during initialization.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self) -> None:
        """Transform the data and add embedding to obsm.

        Uses self.adata which was validated during initialization.
        Modifies self.adata in place by adding embedding to .obsm[self.embedding_key].
        """
        raise NotImplementedError

    def fit_transform(self) -> None:
        """
        Fit the method and transform the data.

        Modifies self.adata in place.
        """
        self.fit()
        self.transform()

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

    def save_embedding(
        self,
        format_type: Literal["parquet", "pickle", "h5"] = "parquet",
        filename: str | None = None,
        compression: bool = True,
    ) -> Path:
        """
        Save embedding to file with preserved cell names as index.

        Parameters
        ----------
        format_type
            Format to save embedding in. Options: 'parquet', 'pickle', or 'h5'.
        filename
            Custom filename (without extension). If None, uses method name.
        compression
            Whether to use compression (gzip for all formats).

        Returns
        -------
        Path
            Path to the saved embedding file.

        Raises
        ------
        ValueError
            If method is not fitted or embedding key not found in adata.obsm.
        """
        if not self.is_fitted:
            raise ValueError("Method must be fitted before saving embedding")
        if self.embedding_key not in self.adata.obsm:
            raise ValueError(f"Embedding key '{self.embedding_key}' not found in adata.obsm")

        filename = filename or f"embedding_{self.name.lower()}"

        # Create DataFrame (shared for parquet/pickle)
        emb_df = pd.DataFrame(
            data=self.adata.obsm[self.embedding_key],
            index=self.adata.obs_names,
            columns=[f"dim_{i}" for i in range(self.adata.obsm[self.embedding_key].shape[1])],
        )

        if format_type == "parquet":
            file_path = self.embedding_dir / f"{filename}.parquet"
            emb_df.to_parquet(file_path, compression="gzip" if compression else None)

        elif format_type == "pickle":
            file_path = self.embedding_dir / f"{filename}.pkl.gz"
            with gzip.open(file_path, "wb") as f:
                pickle.dump(emb_df, f)

        elif format_type == "h5":
            file_path = self.embedding_dir / f"{filename}.h5"
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset(
                    "embedding", data=self.adata.obsm[self.embedding_key], compression="gzip" if compression else None
                )
                hf.create_dataset("cell_names", data=[n.encode() for n in self.adata.obs_names])
                hf.create_dataset(
                    "dim_names", data=[f"dim_{i}".encode() for i in range(self.adata.obsm[self.embedding_key].shape[1])]
                )
        else:
            raise ValueError(f"Unsupported format_type: {format_type}. Choose from 'parquet', 'pickle', 'h5'")

        logger.info("Saved %s embedding to '%s'", self.name, file_path)
        return file_path

    def _filter_none_params(self, params: dict) -> dict:
        """Filter out None values to allow library defaults."""
        return {k: v for k, v in params.items() if v is not None}

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
