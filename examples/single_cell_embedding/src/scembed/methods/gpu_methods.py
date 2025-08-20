"""GPU-based integration methods."""

import os
import tempfile
from pathlib import Path

from scembed.check import check_deps
from scembed.utils import _get_wandb_logger
from slurm_sweep._logging import logger

from .base import BaseIntegrationMethod


class HarmonyMethod(BaseIntegrationMethod):
    """Harmony integration method (GPU-accelerated)."""

    def __init__(self, adata, theta: float | None = None, pca_key: str = "X_pca", **kwargs):
        """
        Initialize Harmony method.

        For most input parameters, if None is passed, the default value from the Harmony library will be used.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        theta
            Diversity clustering penalty parameter.
        pca_key
            Key for PCA embedding in adata.obsm.
        """
        super().__init__(adata, theta=theta, pca_key=pca_key, **kwargs)
        self.theta = theta
        self.pca_key = pca_key

    def fit(self):
        """Fit Harmony - no explicit fitting needed."""
        if self.pca_key not in self.adata.obsm:
            raise ValueError(f"PCA embedding '{self.pca_key}' not found in adata.obsm. Run PCA first.")
        self.is_fitted = True

    def transform(self):
        """Apply Harmony integration."""
        check_deps("harmony-pytorch")
        from harmony import harmonize

        # Prepare harmony parameters, filtering out None values
        harmony_params = self._filter_none_params(
            {
                "theta": self.theta,
            }
        )

        # Use precomputed PCA embedding from data preprocessing
        harmony_embedding = harmonize(
            self.adata.obsm[self.pca_key], self.adata.obs, batch_key=self.batch_key, **harmony_params
        )

        # Add embedding to data
        self.adata.obsm[self.embedding_key] = harmony_embedding


class scVIMethod(BaseIntegrationMethod):
    """scVI integration method."""

    def __init__(
        self,
        adata,
        n_latent: int | None = None,
        n_layers: int | None = None,
        n_hidden: int | None = None,
        gene_likelihood: str | None = None,
        max_epochs: int | None = None,
        early_stopping: bool | None = None,
        accelerator: str | None = None,
        **kwargs,
    ):
        """
        Initialize scVI method.

        For most input parameters, if None is passed, the default value from the scVI library will be used.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        n_latent
            Dimensionality of latent space.
        n_layers
            Number of hidden layers.
        n_hidden
            Number of nodes per hidden layer.
        gene_likelihood
            Gene likelihood distribution.
        max_epochs
            Maximum epochs for scVI training.
        early_stopping
            Whether to use early stopping during training.
        accelerator
            Accelerator type for training.
        """
        super().__init__(
            adata,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            gene_likelihood=gene_likelihood,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            accelerator=accelerator,
            **kwargs,
        )
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gene_likelihood = gene_likelihood
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.accelerator = accelerator
        self.model = None

    def fit(self):
        """Fit scVI model."""
        check_deps("scvi-tools")
        import scvi

        # Subset to highly variable genes
        adata_hvg = self.adata[:, self.adata.var[self.hvg_key]].copy()

        # Setup scVI with counts layer
        scvi.model.SCVI.setup_anndata(adata_hvg, layer=self.counts_layer, batch_key=self.batch_key)

        # Prepare scVI model parameters, filtering out None values
        scvi_params = self._filter_none_params(
            {
                "n_latent": self.n_latent,
                "n_layers": self.n_layers,
                "n_hidden": self.n_hidden,
                "gene_likelihood": self.gene_likelihood,
            }
        )

        # Create and train model
        self.model = scvi.model.SCVI(adata_hvg, **scvi_params)
        logger.info("Set up scVI model: %s", self.model)

        # Add wandb logging if available
        wandb_logger = _get_wandb_logger()

        # Prepare training parameters, filtering out None values
        train_params = self._filter_none_params(
            {
                "max_epochs": self.max_epochs,
                "early_stopping": self.early_stopping,
                "accelerator": self.accelerator,
            }
        )
        if wandb_logger is not None:
            train_params["logger"] = wandb_logger

        self.model.train(**train_params)
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
        scvi_params: dict | None = None,
        max_epochs: int | None = None,
        early_stopping: bool | None = None,
        unlabeled_category: str = "unknown",
        accelerator: str | None = None,
        **kwargs,
    ):
        """
        Initialize scANVI method.

        For most input parameters, if None is passed, the default value from the scANVI library will be used.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        scvi_params
            Parameters for scVI model (n_latent, n_layers, etc.).
        max_epochs
            Maximum epochs for scANVI training.
        early_stopping
            Whether to use early stopping during training.
        unlabeled_category
            Category name for unlabeled cells.
        accelerator
            Accelerator type for training.
        """
        super().__init__(
            adata,
            scvi_params=scvi_params,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            unlabeled_category=unlabeled_category,
            accelerator=accelerator,
            **kwargs,
        )
        self.scvi_params = scvi_params or {}
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.unlabeled_category = unlabeled_category
        self.accelerator = accelerator
        self.scvi_model = None
        self.model = None

    def fit(self):
        """Fit scANVI model."""
        check_deps("scvi-tools")
        import scvi

        # Step 1: Train scVI model using existing scVIMethod
        logger.info("Training scVI model for scANVI pretraining")
        scvi_method = scVIMethod(
            self.adata,
            batch_key=self.batch_key,
            cell_type_key=self.cell_type_key,
            hvg_key=self.hvg_key,
            counts_layer=self.counts_layer,
            **self.scvi_params,
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
            unlabeled_category=self.unlabeled_category,
        )
        logger.info("Set up scANVI model: %s", self.model)

        # Step 3: Train scANVI with unified parameter handling
        logger.info("Training scANVI model")
        wandb_logger = _get_wandb_logger()

        train_params = self._filter_none_params(
            {
                "max_epochs": self.max_epochs,
                "early_stopping": self.early_stopping,
                "accelerator": self.accelerator,
            }
        )
        if wandb_logger is not None:
            train_params["logger"] = wandb_logger

        self.model.train(**train_params)
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

    def __init__(
        self,
        adata,
        embedding_dims: int | list[int] | None = None,
        latent_dim: int | None = None,
        hidden_layer_sizes: list[int] | None = None,
        dr_rate: float | None = None,
        use_mmd: bool | None = None,
        mmd_on: str | None = None,
        beta: float | None = None,
        use_bn: bool | None = None,
        use_ln: bool | None = None,
        embedding_max_norm: float | None = None,
        n_epochs: int | None = None,
        pretraining_epochs: int | None = None,
        recon_loss: str | None = None,
        eta: float | None = None,
        unknown_ct_names: list[str] | None = None,
        **kwargs,
    ):
        """
        Initialize scPoli method.

        For most input parameters, if None is passed, the default value from the scPoli library will be used.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        embedding_dims
            Dimensionality of condition embeddings.
        latent_dim
            Bottleneck layer (z) size.
        hidden_layer_sizes
            List of hidden layer sizes for encoder network.
        dr_rate
            Dropout rate applied to all layers.
        use_mmd
            Whether to use MMD loss on latent dimension.
        mmd_on
            Layer for MMD loss calculation ('z' or 'y').
        beta
            Scaling factor for MMD loss.
        use_bn
            Whether to apply batch normalization.
        use_ln
            Whether to apply layer normalization.
        embedding_max_norm
            Max norm allowed for conditional embeddings.
        n_epochs
            Total number of training epochs.
        pretraining_epochs
            Number of pretraining epochs.
        recon_loss
            Reconstruction loss type.
        eta
            Eta parameter for training.
        unknown_ct_names
            List of unknown cell type names.
        """
        super().__init__(
            adata,
            embedding_dims=embedding_dims,
            latent_dim=latent_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            dr_rate=dr_rate,
            use_mmd=use_mmd,
            mmd_on=mmd_on,
            beta=beta,
            use_bn=use_bn,
            use_ln=use_ln,
            embedding_max_norm=embedding_max_norm,
            n_epochs=n_epochs,
            pretraining_epochs=pretraining_epochs,
            recon_loss=recon_loss,
            eta=eta,
            unknown_ct_names=unknown_ct_names,
            **kwargs,
        )
        self.embedding_dims = embedding_dims
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dr_rate = dr_rate
        self.use_mmd = use_mmd
        self.mmd_on = mmd_on
        self.beta = beta
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.embedding_max_norm = embedding_max_norm
        self.n_epochs = n_epochs
        self.pretraining_epochs = pretraining_epochs
        self.recon_loss = recon_loss
        self.eta = eta
        self.unknown_ct_names = unknown_ct_names
        self.model = None

    def fit(self):
        """Fit scPoli model."""
        check_deps("scarches")
        from scarches.models.scpoli import scPoli

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

        # Prepare scPoli model parameters, filtering out None values
        scpoli_params = self._filter_none_params(
            {
                "condition_keys": self.batch_key,
                "cell_type_keys": self.cell_type_key,
                "embedding_dims": self.embedding_dims,
                "latent_dim": self.latent_dim,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "dr_rate": self.dr_rate,
                "use_mmd": self.use_mmd,
                "mmd_on": self.mmd_on,
                "beta": self.beta,
                "use_bn": self.use_bn,
                "use_ln": self.use_ln,
                "embedding_max_norm": self.embedding_max_norm,
                "recon_loss": self.recon_loss,
                "unknown_ct_names": self.unknown_ct_names,
            }
        )

        self.model = scPoli(
            adata=adata_hvg,
            **scpoli_params,
        )

        # Prepare training parameters, filtering out None values
        train_params = self._filter_none_params(
            {
                "n_epochs": self.n_epochs,
                "pretraining_epochs": self.pretraining_epochs,
                "eta": self.eta,
            }
        )

        self.model.train(
            early_stopping_kwargs=early_stopping_kwargs,
            **train_params,
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
        n_hidden: int | None = None,
        n_hidden_encoder: int | None = None,
        n_latent: int | None = None,
        n_layers: int | None = None,
        dropout_rate: float | None = None,
        dispersion: str | None = None,
        gene_likelihood: str | None = None,
        background_ratio: float | None = None,
        median_distance: float | None = None,
        semisupervised: bool = False,
        mixture_k: int | None = None,
        downsample_counts: bool | None = None,
        max_epochs: int | None = None,
        early_stopping: bool | None = None,
        lr: float | None = None,
        lr_extra: float | None = None,
        weight_decay: float | None = None,
        eps: float | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        batch_size: int | None = None,
        accelerator: str | None = None,
        unlabeled_category: str = "unknown",
        **kwargs,
    ):
        """
        Initialize ResolVI method.

        For most input parameters, if None is passed, the default value from the ResolVI library will be used.

        Parameters
        ----------
        adata
            Annotated data object to integrate. Must contain spatial coordinates.
        n_hidden
            Number of nodes per hidden layer.
        n_hidden_encoder
            Number of nodes per hidden layer in encoder.
        n_latent
            Dimensionality of latent space.
        n_layers
            Number of hidden layers.
        dropout_rate
            Dropout rate for neural networks.
        dispersion
            Dispersion parameter ('gene', 'gene-batch').
        gene_likelihood
            Gene likelihood ('nb', 'poisson').
        background_ratio
            Background ratio parameter.
        median_distance
            Median distance parameter.
        semisupervised
            Whether to use semi-supervised mode with cell type labels.
        mixture_k
            Mixture parameter K.
        downsample_counts
            Whether to downsample counts.
        max_epochs
            Maximum epochs for ResolVI training.
        early_stopping
            Whether to use early stopping during training.
        lr
            Learning rate for optimization.
        lr_extra
            Learning rate for extra parameters.
        weight_decay
            Weight decay regularization.
        eps
            Optimizer eps parameter.
        n_steps_kl_warmup
            Number of steps for KL warmup.
        n_epochs_kl_warmup
            Number of epochs for KL warmup.
        batch_size
            Batch size for training.
        accelerator
            Accelerator type for training.
        unlabeled_category
            Unlabeled category name.
        """
        super().__init__(
            adata,
            validate_spatial=True,  # Enable spatial validation
            n_hidden=n_hidden,
            n_hidden_encoder=n_hidden_encoder,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            background_ratio=background_ratio,
            median_distance=median_distance,
            semisupervised=semisupervised,
            mixture_k=mixture_k,
            downsample_counts=downsample_counts,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            lr=lr,
            lr_extra=lr_extra,
            weight_decay=weight_decay,
            eps=eps,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            batch_size=batch_size,
            accelerator=accelerator,
            unlabeled_category=unlabeled_category,
            **kwargs,
        )

        # Store ResolVI-specific parameters
        self.n_hidden = n_hidden
        self.n_hidden_encoder = n_hidden_encoder
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood
        self.background_ratio = background_ratio
        self.median_distance = median_distance
        self.semisupervised = semisupervised
        self.mixture_k = mixture_k
        self.downsample_counts = downsample_counts
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.lr = lr
        self.lr_extra = lr_extra
        self.weight_decay = weight_decay
        self.eps = eps
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.unlabeled_category = unlabeled_category
        self.model = None

    def fit(self):
        """Fit ResolVI model.

        Note: ResolVI will compute spatial neighbors internally in `_prepare_data`, which is called from `setup_anndata`.
        To compute neighbors, it will attempt to use rapids_singlecell and fall back to scanpy if necessary. It simply computes
        `n_neighbors` + 5 (by default, 15) neighbors in the given spatial representation, treating each batch separately.
        """
        check_deps("scvi-tools")
        import scvi

        # Setup ResolVI data registration
        # ResolVI setup_anndata will automatically compute spatial neighbors if missing

        scvi.external.RESOLVI.setup_anndata(
            self.adata,
            layer=self.counts_layer,
            batch_key=self.batch_key,
            labels_key=self.cell_type_key if self.semisupervised else None,
            prepare_data_kwargs={"spatial_rep": self.spatial_key},
            unlabeled_category=self.unlabeled_category,
        )

        # Prepare ResolVI model parameters, filtering out None values
        model_params = self._filter_none_params(
            {
                "n_hidden": self.n_hidden,
                "n_hidden_encoder": self.n_hidden_encoder,
                "n_latent": self.n_latent,
                "n_layers": self.n_layers,
                "dropout_rate": self.dropout_rate,
                "dispersion": self.dispersion,
                "gene_likelihood": self.gene_likelihood,
                "background_ratio": self.background_ratio,
                "median_distance": self.median_distance,
                "semisupervised": self.semisupervised,
                "mixture_k": self.mixture_k,
                "downsample_counts": self.downsample_counts,
            }
        )

        # Create ResolVI model
        self.model = scvi.external.RESOLVI(
            self.adata,
            **model_params,
        )
        logger.info("Set up ResolVI model: %s", self.model)

        # Prepare training parameters, filtering out None values
        train_params = self._filter_none_params(
            {
                "max_epochs": self.max_epochs,
                "early_stopping": self.early_stopping,
                "lr": self.lr,
                "lr_extra": self.lr_extra,
                "weight_decay": self.weight_decay,
                "eps": self.eps,
                "n_steps_kl_warmup": self.n_steps_kl_warmup,
                "n_epochs_kl_warmup": self.n_epochs_kl_warmup,
                "batch_size": self.batch_size,
                "accelerator": self.accelerator,
            }
        )

        # Add wandb logger if available
        wandb_logger = _get_wandb_logger()
        if wandb_logger is not None:
            train_params["logger"] = wandb_logger

        self.model.train(**train_params)
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
        scvi_params: dict | None = None,
        scanvi_params: dict | None = None,
        k_nn: int | None = None,
        n_latent: int | None = None,
        n_hidden: int | None = None,
        n_layers: int | None = None,
        dropout_rate: float | None = None,
        max_epochs: int | None = None,
        early_stopping: bool | None = None,
        lr: float | None = None,
        accelerator: str | None = None,
        batch_size: int | None = None,
        **kwargs,
    ):
        """
        Initialize scVIVA method.

        For most input parameters, if None is passed, the default value from the scVIVA library will be used.

        Parameters
        ----------
        adata
            Annotated data object to integrate. Must contain spatial coordinates and cell type labels.
        embedding_method
            Method to compute expression embeddings. Options: "scvi", "scanvi". Default is "scvi".
        scvi_params
            Parameters to pass to the scVI embedding method. If None, uses method defaults.
        scanvi_params
            Parameters to pass to the scANVI embedding method. If None, uses method defaults.
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
        early_stopping
            Whether to use early stopping during training.
        lr
            Learning rate for scVIVA training.
        accelerator
            Accelerator type for training.
        batch_size
            Batch size for training.
        """
        super().__init__(
            adata,
            validate_spatial=True,  # Enable spatial validation
            embedding_method=embedding_method,
            scvi_params=scvi_params,
            scanvi_params=scanvi_params,
            k_nn=k_nn,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            lr=lr,
            accelerator=accelerator,
            batch_size=batch_size,
            **kwargs,
        )

        # Store scVIVA-specific parameters
        self.embedding_method = embedding_method
        self.scvi_params = scvi_params or {}
        self.scanvi_params = scanvi_params or {}
        self.k_nn = k_nn
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.lr = lr
        self.accelerator = accelerator
        self.batch_size = batch_size

        # Initialize models
        self.embedding_model = None
        self.model = None

    def fit(self):
        """Fit scVIVA model with expression embedding computation."""
        check_deps("scvi-tools")
        import scvi

        # Step 1: Compute expression embedding using existing methods
        logger.info("Computing %s expression embedding for scVIVA", self.embedding_method)

        # Validate embedding method choice
        if self.embedding_method not in ["scvi", "scanvi"]:
            raise ValueError(f"embedding_method must be 'scvi' or 'scanvi', got: {self.embedding_method}")

        # Prepare embedding parameters and create embedding model
        if self.embedding_method == "scvi":
            embedding_params = self.scvi_params.copy()
            embedding_params.update(
                {
                    "batch_key": self.batch_key,
                    "cell_type_key": self.cell_type_key,
                    "hvg_key": self.hvg_key,
                    "counts_layer": self.counts_layer,
                }
            )
            self.embedding_model = scVIMethod(self.adata, **embedding_params)
            self.embedding_model.fit_transform()
            expression_embedding_key = "X_scvi"
            # Transfer embedding to main adata since methods now work on copies
            self.adata.obsm[expression_embedding_key] = self.embedding_model.adata.obsm[expression_embedding_key]

        else:  # embedding_method == "scanvi"
            embedding_params = self.scanvi_params.copy()
            embedding_params.update(
                {
                    "batch_key": self.batch_key,
                    "cell_type_key": self.cell_type_key,
                    "hvg_key": self.hvg_key,
                    "counts_layer": self.counts_layer,
                }
            )
            self.embedding_model = scANVIMethod(self.adata, **embedding_params)
            self.embedding_model.fit_transform()
            expression_embedding_key = "X_scanvi"
            # Transfer embedding to main adata since methods now work on copies
            self.adata.obsm[expression_embedding_key] = self.embedding_model.adata.obsm[expression_embedding_key]

        # Step 2: Run scVIVA preprocessing to compute spatial neighborhoods
        logger.info("Running scVIVA preprocessing")

        # Prepare preprocessing parameters, filtering out None values for k_nn only
        preprocessing_params = {
            "adata": self.adata,
            "sample_key": self.batch_key,
            "labels_key": self.cell_type_key,
            "cell_coordinates_key": self.spatial_key,
            "expression_embedding_key": expression_embedding_key,
        }

        # Add k_nn only if not None (let scVIVA use its default otherwise)
        if self.k_nn is not None:
            preprocessing_params["k_nn"] = self.k_nn

        # Run preprocessing to compute neighborhoods
        scvi.external.SCVIVA.preprocessing_anndata(**preprocessing_params)

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

        # Prepare scVIVA model parameters, filtering out None values
        model_params = self._filter_none_params(
            {
                "n_latent": self.n_latent,
                "n_hidden": self.n_hidden,
                "n_layers": self.n_layers,
                "dropout_rate": self.dropout_rate,
            }
        )

        self.model = scvi.external.SCVIVA(self.adata, **model_params)
        logger.info("Set up scVIVA model: %s", self.model)

        # Prepare training parameters, filtering out None values
        train_params = self._filter_none_params(
            {
                "max_epochs": self.max_epochs,
                "early_stopping": self.early_stopping,
                "accelerator": self.accelerator,
                "batch_size": self.batch_size,
            }
        )

        # Handle learning rate via plan_kwargs if specified
        if self.lr is not None:
            train_params["plan_kwargs"] = {"lr": self.lr}

        # Add wandb logger if available
        wandb_logger = _get_wandb_logger()
        if wandb_logger is not None:
            train_params["logger"] = wandb_logger

        self.model.train(**train_params)
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
