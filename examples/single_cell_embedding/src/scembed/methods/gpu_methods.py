"""GPU-based integration methods."""

import os
import tempfile
from pathlib import Path

import wandb

from slurm_sweep._logging import logger

from .base import BaseIntegrationMethod


def _get_wandb_logger(run_id: str | None = None, project: str = "scvi-training"):
    """
    Create a wandb logger for scVI training if wandb is available and initialized.

    Parameters
    ----------
    run_id
        Existing wandb run ID to use. If None and wandb is initialized, uses current run.
    project
        Project name for new wandb runs.

    Returns
    -------
    pytorch_lightning.loggers.WandbLogger or None
        WandbLogger instance if wandb is available and initialized, None otherwise.
    """
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        logger.debug("pytorch_lightning/lightning not available, skipping wandb logging")
        return None

    # Check if wandb is initialized
    if wandb.run is None and run_id is None:
        logger.debug("No active wandb run and no run_id provided, skipping wandb logging")
        return None

    try:
        if run_id is not None:
            # Use existing run
            wandb_logger = WandbLogger(id=run_id, resume="allow")
        elif wandb.run is not None:
            # Use current active run
            wandb_logger = WandbLogger(experiment=wandb.run)
        else:
            # Create new run (fallback)
            wandb_logger = WandbLogger(project=project)

        logger.info("Created wandb logger for scVI training")
        return wandb_logger
    except (ValueError, RuntimeError, OSError) as e:
        logger.warning("Failed to create wandb logger: %s", e)
        return None


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

        # Add wandb logging if available
        wandb_logger = _get_wandb_logger()
        trainer_kwargs = {}
        if wandb_logger is not None:
            trainer_kwargs["logger"] = wandb_logger

        self.model.train(
            max_epochs=self.max_epochs, early_stopping=True, accelerator=self.accelerator, **trainer_kwargs
        )
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

        # Step 1: Train scVI model using existing scVIMethod
        logger.info("Training scVI model for scANVI pretraining")
        scvi_method = scVIMethod(
            self.adata,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            batch_key=self.batch_key,
            cell_type_key=self.cell_type_key,
            hvg_key=self.hvg_key,
            counts_layer=self.counts_layer,
        )
        scvi_method.fit()

        # Store the scVI model
        self.scvi_model = scvi_method.model
        if self.scvi_model is None:
            raise ValueError("scVI model training failed")

        # Step 2: Create scANVI from scVI
        logger.info("Creating scANVI from pretrained scVI model")
        self.model = scvi.model.SCANVI.from_scvi_model(
            self.scvi_model,
            labels_key=self.cell_type_key,
            unlabeled_category="Unknown",
        )

        # Step 3: Train scANVI with wandb logging
        logger.info("Training scANVI model")
        wandb_logger = _get_wandb_logger()
        trainer_kwargs = {}
        if wandb_logger is not None:
            trainer_kwargs["logger"] = wandb_logger
        self.model.train(max_epochs=self.max_epochs_scanvi, accelerator=self.accelerator, **trainer_kwargs)
        self.is_fitted = True

    def transform(self):
        """Get scANVI latent representation."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before transform")

        # Get latent representation
        latent = self.model.get_latent_representation()
        self.adata.obsm[self.embedding_key] = latent

    def save_model(self, path: Path) -> Path | None:
        """Save scANVI model and pretrained scVI model."""
        if self.model is None:
            return None

        # Save main scANVI model
        model_dir = path / "scanvi_model"
        self.model.save(str(model_dir), overwrite=True)

        # Save pretrained scVI model if available
        if self.scvi_model is not None:
            scvi_dir = path / "scvi_pretrained_model"
            self.scvi_model.save(str(scvi_dir), overwrite=True)
            logger.info("Saved pretrained scVI model to %s", scvi_dir)

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

        # Subset to highly variable genes (same as used during training)
        adata_hvg = self.adata[:, self.adata.var[self.hvg_key]].copy()

        # Get latent representation
        latent = self.model.get_latent(adata_hvg, mean=True)
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
            prepare_data_kwargs={"spatial_rep": self.spatial_key},
        )

        # Create ResolVI model
        self.model = scvi.external.RESOLVI(
            self.adata,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            semisupervised=self.semisupervised,
        )

        # Train the model with wandb logging
        wandb_logger = _get_wandb_logger()
        trainer_kwargs = {}
        if wandb_logger is not None:
            trainer_kwargs["logger"] = wandb_logger
        self.model.train(max_epochs=self.max_epochs, accelerator=self.accelerator, **trainer_kwargs)
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


class scVIVAMethod(BaseIntegrationMethod):
    """scVIVA integration method for spatial transcriptomics with neighborhood modeling."""

    def __init__(
        self,
        adata,
        embedding_method: str = "scvi",
        k_nn: int = 20,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        max_epochs: int = 400,
        embedding_n_latent: int = 30,
        embedding_n_layers: int = 2,
        embedding_max_epochs: int = 100,
        embedding_max_epochs_scanvi: int = 50,
        accelerator: str = "auto",
        **kwargs,
    ):
        """
        Initialize scVIVA method.

        Parameters
        ----------
        adata
            Annotated data object to integrate. Must contain spatial coordinates and cell type labels.
        embedding_method
            Method to compute expression embeddings. Options: "scvi", "scanvi".
        k_nn
            Number of nearest neighbors for spatial graph construction.
        n_latent
            Dimensionality of scVIVA latent space.
        n_hidden
            Number of nodes per hidden layer for scVIVA.
        n_layers
            Number of hidden layers for scVIVA.
        dropout_rate
            Dropout rate for scVIVA neural networks.
        max_epochs
            Maximum epochs for scVIVA training.
        embedding_n_latent
            Dimensionality of expression embedding latent space.
        embedding_n_layers
            Number of hidden layers for expression embedding method.
        embedding_max_epochs
            Maximum epochs for expression embedding training.
        embedding_max_epochs_scanvi
            Maximum epochs for scANVI training (only used when embedding_method="scanvi").
        accelerator
            Accelerator type for training. Options: "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto".
        """
        super().__init__(
            adata,
            validate_spatial=True,  # Enable spatial validation
            embedding_method=embedding_method,
            k_nn=k_nn,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            max_epochs=max_epochs,
            embedding_n_latent=embedding_n_latent,
            embedding_n_layers=embedding_n_layers,
            embedding_max_epochs=embedding_max_epochs,
            embedding_max_epochs_scanvi=embedding_max_epochs_scanvi,
            accelerator=accelerator,
            **kwargs,
        )

        # Store scVIVA-specific parameters
        self.embedding_method = embedding_method
        self.k_nn = k_nn
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs
        self.accelerator = accelerator

        # Store expression embedding parameters
        self.embedding_n_latent = embedding_n_latent
        self.embedding_n_layers = embedding_n_layers
        self.embedding_max_epochs = embedding_max_epochs
        self.embedding_max_epochs_scanvi = embedding_max_epochs_scanvi

        # Initialize models
        self.embedding_model = None
        self.model = None

    def fit(self):
        """Fit scVIVA model with expression embedding computation."""
        try:
            import scvi
        except ImportError as exc:
            raise ImportError("scvi-tools is required for scVIVA") from exc

        # Step 1: Compute expression embedding using existing methods
        logger.info("Computing %s expression embedding for scVIVA", self.embedding_method)

        if self.embedding_method == "scvi":
            self.embedding_model = scVIMethod(
                self.adata,
                n_latent=self.embedding_n_latent,
                n_layers=self.embedding_n_layers,
                max_epochs=self.embedding_max_epochs,
                accelerator=self.accelerator,
                batch_key=self.batch_key,
                cell_type_key=self.cell_type_key,
                hvg_key=self.hvg_key,
                counts_layer=self.counts_layer,
            )
            self.embedding_model.fit_transform()
            expression_embedding_key = "X_scvi"

        elif self.embedding_method == "scanvi":
            self.embedding_model = scANVIMethod(
                self.adata,
                n_latent=self.embedding_n_latent,
                n_layers=self.embedding_n_layers,
                max_epochs=self.embedding_max_epochs,
                max_epochs_scanvi=self.embedding_max_epochs_scanvi,
                accelerator=self.accelerator,
                batch_key=self.batch_key,
                cell_type_key=self.cell_type_key,
                hvg_key=self.hvg_key,
                counts_layer=self.counts_layer,
            )
            self.embedding_model.fit_transform()
            expression_embedding_key = "X_scanvi"
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")

        # Step 2: Run scVIVA preprocessing to compute spatial neighborhoods
        logger.info("Running scVIVA preprocessing with k_nn=%d", self.k_nn)

        # Run preprocessing to compute neighborhoods
        # Using SCVIVA class method for preprocessing
        scvi.external.SCVIVA.preprocessing_anndata(
            adata=self.adata,
            k_nn=self.k_nn,
            sample_key=self.batch_key,
            labels_key=self.cell_type_key,
            cell_coordinates_key=self.spatial_key,
            expression_embedding_key=expression_embedding_key,
        )

        # Step 3: Setup scVIVA data registration
        scvi.external.SCVIVA.setup_anndata(
            self.adata,
            layer=self.counts_layer,
            batch_key=self.batch_key,
            sample_key=self.batch_key,
            labels_key=self.cell_type_key,
            cell_coordinates_key=self.spatial_key,
            expression_embedding_key=expression_embedding_key,
        )

        # Step 4: Create and train scVIVA model
        logger.info("Training scVIVA model")
        self.model = scvi.external.SCVIVA(
            self.adata,
            n_latent=self.n_latent,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
        )

        # Train the model with wandb logging
        wandb_logger = _get_wandb_logger()
        trainer_kwargs = {}
        if wandb_logger is not None:
            trainer_kwargs["logger"] = wandb_logger
        self.model.train(
            max_epochs=self.max_epochs,
            early_stopping=True,
            accelerator=self.accelerator,
            **trainer_kwargs,
        )
        self.is_fitted = True

    def transform(self):
        """Get scVIVA latent representation."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before transform")

        # Get latent representation
        latent = self.model.get_latent_representation()
        self.adata.obsm[self.embedding_key] = latent

    def save_model(self, path: Path) -> Path | None:
        """Save scVIVA model and embedding model."""
        if self.model is None:
            return None

        # Save main scVIVA model
        model_dir = path / "scviva_model"
        self.model.save(str(model_dir), overwrite=True)

        # Save embedding model if available
        if self.embedding_model is not None:
            embedding_dir = self.embedding_model.save_model(path)
            if embedding_dir:
                logger.info("Saved embedding model to %s", embedding_dir)

        return model_dir
