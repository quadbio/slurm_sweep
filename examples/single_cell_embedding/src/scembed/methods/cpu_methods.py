"""CPU-based integration methods."""

import numpy as np

from slurm_sweep.check import check_deps

from .base import BaseIntegrationMethod


class PrecomputedEmbeddingMethod(BaseIntegrationMethod):
    """Method for using pre-computed embeddings (e.g., PCA, UMAP, etc.)."""

    def __init__(self, adata, embedding_key: str = "X_pca", **kwargs):
        """
        Initialize precomputed embedding method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        embedding_key
            Key in .obsm containing the precomputed embedding to use.
        """
        super().__init__(adata, embedding_key=embedding_key, **kwargs)
        # Override the default name and embedding key for precomputed embeddings
        self.name = embedding_key.replace("X_", "").upper()
        self.embedding_key = embedding_key
        self.source_embedding_key = embedding_key

    def fit(self):
        """No fitting needed for precomputed embeddings."""
        if self.source_embedding_key not in self.adata.obsm:
            raise ValueError(f"Embedding '{self.source_embedding_key}' not found in adata.obsm")
        self.is_fitted = True

    def transform(self):
        """Use the precomputed embedding."""
        if self.source_embedding_key not in self.adata.obsm:
            raise ValueError(f"Embedding '{self.source_embedding_key}' not found in adata.obsm")

        # Copy the embedding to the standard method key if different
        if self.source_embedding_key != self.embedding_key:
            self.adata.obsm[self.embedding_key] = self.adata.obsm[self.source_embedding_key].copy()


class LIGERMethod(BaseIntegrationMethod):
    """LIGER integration method."""

    def __init__(self, adata, k: int = 30, lambda_reg: float = 5.0, **kwargs):
        """
        Initialize LIGER method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        k
            Number of factors for matrix factorization.
        lambda_reg
            Regularization parameter.
        """
        super().__init__(adata, k=k, lambda_reg=lambda_reg, **kwargs)
        self.k = k
        self.lambda_reg = lambda_reg

    def fit(self):
        """Fit LIGER - no explicit fitting needed."""
        self.is_fitted = True

    def transform(self):
        """Apply LIGER integration."""
        check_deps("pyliger")
        import pyliger

        # Setup raw counts for LIGER (data already validated)
        self.adata.X = self.adata.layers[self.counts_layer].copy()

        # Use HVG subset (we know it exists from validation)
        adata_hvg = self.adata[:, self.adata.var[self.hvg_key]].copy()

        # Split by batch for LIGER
        batch_cats = adata_hvg.obs[self.batch_key].cat.categories
        adata_list = [adata_hvg[adata_hvg.obs[self.batch_key] == b].copy() for b in batch_cats]

        for i, ad in enumerate(adata_list):
            ad.uns["sample_name"] = batch_cats[i]
            ad.uns["var_gene_idx"] = np.arange(adata_hvg.n_vars)

        # Run LIGER
        liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
        liger_data.var_genes = adata_hvg.var_names
        pyliger.normalize(liger_data)
        pyliger.scale_not_center(liger_data)
        pyliger.optimize_ALS(liger_data, k=self.k, value_lambda=self.lambda_reg)
        pyliger.quantile_norm(liger_data)

        # Combine results
        embedding = np.zeros((self.adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
        for i, b in enumerate(batch_cats):
            batch_mask = self.adata.obs[self.batch_key] == b
            embedding[batch_mask] = liger_data.adata_list[i].obsm["H_norm"]

        self.adata.obsm[self.embedding_key] = embedding


class ScanoramaMethod(BaseIntegrationMethod):
    """Scanorama integration method."""

    def __init__(self, adata, sigma: float = 15.0, alpha: float = 0.1, **kwargs):
        """
        Initialize Scanorama method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        sigma
            Neighborhood factor.
        alpha
            Alignment score minimum cutoff.
        """
        super().__init__(adata, sigma=sigma, alpha=alpha, **kwargs)
        self.sigma = sigma
        self.alpha = alpha

    def fit(self):
        """Fit Scanorama - no explicit fitting needed."""
        self.is_fitted = True

    def transform(self):
        """Apply Scanorama integration."""
        check_deps("scanorama")
        import scanorama

        # Use HVG subset (we know it exists from validation)
        adata_hvg = self.adata[:, self.adata.var[self.hvg_key]].copy()

        # Split by batch
        batch_cats = adata_hvg.obs[self.batch_key].cat.categories
        adata_list = [adata_hvg[adata_hvg.obs[self.batch_key] == b].copy() for b in batch_cats]

        # Run Scanorama
        scanorama.integrate_scanpy(adata_list, dimred=50, sigma=self.sigma, alpha=self.alpha)

        # Combine results
        embedding = np.zeros((self.adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
        for i, b in enumerate(batch_cats):
            batch_mask = self.adata.obs[self.batch_key] == b
            embedding[batch_mask] = adata_list[i].obsm["X_scanorama"]

        self.adata.obsm[self.embedding_key] = embedding
