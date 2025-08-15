"""GPU-based integration methods."""

import os
import tempfile
from pathlib import Path

from .base import BaseIntegrationMethod


class HarmonyMethod(BaseIntegrationMethod):
    """Harmony integration method (GPU-accelerated)."""

    def __init__(self, adata, theta: float = 2.0, **kwargs):
        """
        Initialize Harmony method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        theta
            Diversity clustering penalty parameter.
        """
        super().__init__(adata, theta=theta, **kwargs)
        self.theta = theta

    def fit(self):
        """Fit Harmony - no explicit fitting needed."""
        if "X_pca" not in self.adata.obsm:
            raise ValueError("PCA embedding 'X_pca' not found in adata.obsm. Run PCA first.")
        self.is_fitted = True

    def transform(self):
        """Apply Harmony integration."""
        try:
            from harmony import harmonize
        except ImportError as exc:
            raise ImportError("harmony-pytorch is required for Harmony integration") from exc

        # Use precomputed PCA embedding from data preprocessing
        harmony_embedding = harmonize(
            self.adata.obsm["X_pca"], self.adata.obs, batch_key=self.batch_key, theta=self.theta
        )

        # Add embedding to data
        self.adata.obsm[self.embedding_key] = harmony_embedding


class scVIMethod(BaseIntegrationMethod):
    """scVI integration method."""

    def __init__(
        self, adata, n_latent: int = 30, n_layers: int = 2, max_epochs: int = 100, accelerator: str = "auto", **kwargs
    ):
        """
        Initialize scVI method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        n_latent
            Dimensionality of latent space.
        n_layers
            Number of hidden layers.
        max_epochs
            Maximum epochs for scVI training.
        accelerator
            Accelerator type for training. Options: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto".
        """
        super().__init__(
            adata, n_latent=n_latent, n_layers=n_layers, max_epochs=max_epochs, accelerator=accelerator, **kwargs
        )
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.model = None

    def fit(self):
        """Fit scVI model."""
        try:
            import scvi
        except ImportError as exc:
            raise ImportError("scvi-tools is required for scVI integration") from exc

        # Subset to highly variable genes
        adata_hvg = self.adata[:, self.adata.var[self.hvg_key]].copy()

        # Setup scVI with counts layer
        scvi.model.SCVI.setup_anndata(adata_hvg, layer=self.counts_layer, batch_key=self.batch_key)

        # Create and train model
        self.model = scvi.model.SCVI(adata_hvg, n_latent=self.n_latent, n_layers=self.n_layers, gene_likelihood="nb")
        self.model.train(max_epochs=self.max_epochs, early_stopping=True, accelerator=self.accelerator)
        self.is_fitted = True

    def transform(self):
        """Get scVI latent representation."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before transform")

        # Get latent representation
        latent = self.model.get_latent_representation()
        self.adata.obsm[self.embedding_key] = latent

    def save_model(self, path: Path) -> Path | None:
        """Save scVI model."""
        if self.model is None:
            return None

        model_dir = path / "scvi_model"
        self.model.save(str(model_dir), overwrite=True)
        return model_dir


class scANVIMethod(BaseIntegrationMethod):
    """scANVI integration method."""

    def __init__(
        self,
        adata,
        n_latent: int = 30,
        n_layers: int = 2,
        max_epochs: int = 100,
        max_epochs_scanvi: int = 50,
        accelerator: str = "auto",
        **kwargs,
    ):
        """
        Initialize scANVI method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        n_latent
            Dimensionality of latent space.
        n_layers
            Number of hidden layers.
        max_epochs
            Maximum epochs for scVI pretraining.
        max_epochs_scanvi
            Maximum epochs for scANVI training.
        accelerator
            Accelerator type for training. Options: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto".
        """
        super().__init__(
            adata,
            n_latent=n_latent,
            n_layers=n_layers,
            max_epochs=max_epochs,
            max_epochs_scanvi=max_epochs_scanvi,
            accelerator=accelerator,
            **kwargs,
        )
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.max_epochs_scanvi = max_epochs_scanvi
        self.accelerator = accelerator
        self.scvi_model = None
        self.model = None

    def fit(self):
        """Fit scANVI model."""
        try:
            import scvi
        except ImportError as exc:
            raise ImportError("scvi-tools is required for scANVI integration") from exc

        # Subset to highly variable genes
        adata_hvg = self.adata[:, self.adata.var[self.hvg_key]].copy()

        # Setup scVI first
        scvi.model.SCVI.setup_anndata(adata_hvg, layer=self.counts_layer, batch_key=self.batch_key)

        # Train scVI model first
        self.scvi_model = scvi.model.SCVI(
            adata_hvg, n_latent=self.n_latent, n_layers=self.n_layers, gene_likelihood="nb"
        )
        self.scvi_model.train(max_epochs=self.max_epochs, early_stopping=True, accelerator=self.accelerator)

        # Create scANVI from scVI
        self.model = scvi.model.SCANVI.from_scvi_model(
            self.scvi_model,
            adata=adata_hvg,
            labels_key=self.cell_type_key,
            unlabeled_category="Unknown",
        )
        self.model.train(max_epochs=self.max_epochs_scanvi, accelerator=self.accelerator)
        self.is_fitted = True

    def transform(self):
        """Get scANVI latent representation."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before transform")

        # Get latent representation
        latent = self.model.get_latent_representation()
        self.adata.obsm[self.embedding_key] = latent

    def save_model(self, path: Path) -> Path | None:
        """Save scANVI model."""
        if self.model is None:
            return None

        model_dir = path / "scanvi_model"
        self.model.save(str(model_dir), overwrite=True)
        return model_dir


class scPoliMethod(BaseIntegrationMethod):
    """scPoli integration method."""

    def __init__(self, adata, embedding_dims: int = 5, n_epochs: int = 50, pretraining_epochs: int = 40, **kwargs):
        """
        Initialize scPoli method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        embedding_dims
            Dimensionality of condition embeddings.
        n_epochs
            Total number of training epochs.
        pretraining_epochs
            Number of pretraining epochs.
        """
        super().__init__(
            adata, embedding_dims=embedding_dims, n_epochs=n_epochs, pretraining_epochs=pretraining_epochs, **kwargs
        )
        self.embedding_dims = embedding_dims
        self.n_epochs = n_epochs
        self.pretraining_epochs = pretraining_epochs
        self.model = None

    def fit(self):
        """Fit scPoli model."""
        try:
            from scarches.models.scpoli import scPoli
        except ImportError as exc:
            raise ImportError("scarches is required for scPoli integration") from exc

        # Subset to highly variable genes
        adata_hvg = self.adata[:, self.adata.var[self.hvg_key]].copy()

        # Early stopping configuration
        early_stopping_kwargs = {
            "early_stopping_metric": "val_prototype_loss",
            "mode": "min",
            "threshold": 0,
            "patience": 20,
            "reduce_lr": True,
            "lr_patience": 13,
            "lr_factor": 0.1,
        }

        # Create and train scPoli model
        # Note: scPoli expects raw counts in .X, so we need to copy from layer
        adata_hvg.X = adata_hvg.layers[self.counts_layer].copy()

        self.model = scPoli(
            adata=adata_hvg,
            condition_keys=self.batch_key,
            cell_type_keys=self.cell_type_key,
            embedding_dims=self.embedding_dims,
            recon_loss="nb",
        )

        self.model.train(
            n_epochs=self.n_epochs,
            pretraining_epochs=self.pretraining_epochs,
            early_stopping_kwargs=early_stopping_kwargs,
            eta=5,
        )
        self.is_fitted = True

    def transform(self):
        """Get scPoli latent representation."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before transform")

        # Get latent representation
        latent = self.model.get_latent(self.adata, mean=True)
        self.adata.obsm[self.embedding_key] = latent

    def save_model(self, path: Path) -> Path | None:
        """Save scPoli model."""
        if self.model is None:
            return None

        # scPoli uses different saving mechanism
        model_dir = path / "scpoli_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model using scPoli's save method
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.save(tmp_dir, overwrite=True)
            # Copy to our desired location
            os.system(f"cp -r {tmp_dir}/* {model_dir}/")

        return model_dir


class ResolVIMethod(BaseIntegrationMethod):
    """ResolVI integration method for spatial transcriptomics."""

    def __init__(
        self,
        adata,
        n_latent: int = 10,
        n_layers: int = 2,
        n_hidden: int = 32,
        max_epochs: int = 50,
        semisupervised: bool = False,
        accelerator: str = "auto",
        **kwargs,
    ):
        """
        Initialize ResolVI method.

        Parameters
        ----------
        adata
            Annotated data object to integrate. Must contain spatial coordinates.
        n_latent
            Dimensionality of latent space.
        n_layers
            Number of hidden layers.
        n_hidden
            Number of nodes per hidden layer.
        max_epochs
            Maximum epochs for ResolVI training.
        semisupervised
            Whether to use semi-supervised mode with cell type labels.
        accelerator
            Accelerator type for training. Options: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto".
        """
        super().__init__(
            adata,
            validate_spatial=True,  # Enable spatial validation
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            max_epochs=max_epochs,
            semisupervised=semisupervised,
            accelerator=accelerator,
            **kwargs,
        )

        # Store ResolVI-specific parameters
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.max_epochs = max_epochs
        self.semisupervised = semisupervised
        self.accelerator = accelerator
        self.model = None

    def fit(self):
        """Fit ResolVI model."""
        try:
            import scvi
        except ImportError as exc:
            raise ImportError("scvi-tools is required for ResolVI") from exc

        # Setup ResolVI data registration
        # ResolVI setup_anndata will automatically compute spatial neighbors if missing
        scvi.external.RESOLVI.setup_anndata(
            self.adata,
            layer=self.counts_layer,
            batch_key=self.batch_key,
            labels_key=self.cell_type_key if self.semisupervised else None,
        )

        # Create ResolVI model
        self.model = scvi.external.RESOLVI(
            self.adata,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            semisupervised=self.semisupervised,
        )

        # Train the model
        self.model.train(max_epochs=self.max_epochs, accelerator=self.accelerator)
        self.is_fitted = True

    def transform(self):
        """Get ResolVI latent representation."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before transform")

        # Get latent representation
        latent = self.model.get_latent_representation()
        self.adata.obsm[self.embedding_key] = latent

    def save_model(self, path: Path) -> Path | None:
        """Save ResolVI model."""
        if self.model is None:
            return None

        model_dir = path / "resolvi_model"
        self.model.save(str(model_dir), overwrite=True)
        return model_dir
