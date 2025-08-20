"""CPU-based integration methods."""

import numpy as np

from scembed.check import check_deps

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

    def __init__(
        self,
        adata,
        # optimize_ALS parameters
        k: int = 20,  # Required parameter with common default
        value_lambda: float | None = None,
        thresh: float | None = None,
        max_iters: int | None = None,
        nrep: int | None = None,
        rand_seed: int | None = None,
        print_obj: bool | None = None,
        # quantile_norm parameters
        quantiles: int | None = None,
        ref_dataset: str | None = None,
        min_cells: int | None = None,
        dims_use: list | None = None,
        do_center: bool | None = None,
        max_sample: int | None = None,
        num_trees: int | None = None,
        refine_knn: bool | None = None,
        knn_k: int | None = None,
        use_ann: bool | None = None,
        **kwargs,
    ):
        """
        Initialize LIGER method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        k
            Number of factors for matrix factorization.
        value_lambda
            Regularization parameter.
        thresh
            Convergence threshold.
        max_iters
            Maximum number of iterations.
        nrep
            Number of restarts.
        rand_seed
            Random seed for reproducibility.
        print_obj
            Whether to print objective values.
        quantiles
            Number of quantiles for quantile normalization.
        ref_dataset
            Reference dataset name for normalization.
        min_cells
            Minimum cells per cluster for quantile normalization.
        dims_use
            Factors to use for quantile normalization.
        do_center
            Whether to center data when scaling factors.
        max_sample
            Maximum cells for quantile normalization per cluster.
        num_trees
            Number of trees for approximate nearest neighbor search.
        refine_knn
            Whether to refine cluster assignments using KNN.
        knn_k
            Number of nearest neighbors for KNN graph.
        use_ann
            Whether to use approximate nearest neighbors.
        """
        super().__init__(
            adata,
            k=k,
            value_lambda=value_lambda,
            thresh=thresh,
            max_iters=max_iters,
            nrep=nrep,
            rand_seed=rand_seed,
            print_obj=print_obj,
            quantiles=quantiles,
            ref_dataset=ref_dataset,
            min_cells=min_cells,
            dims_use=dims_use,
            do_center=do_center,
            max_sample=max_sample,
            num_trees=num_trees,
            refine_knn=refine_knn,
            knn_k=knn_k,
            use_ann=use_ann,
            **kwargs,
        )
        # optimize_ALS parameters
        self.k = k
        self.value_lambda = value_lambda
        self.thresh = thresh
        self.max_iters = max_iters
        self.nrep = nrep
        self.rand_seed = rand_seed
        self.print_obj = print_obj
        # quantile_norm parameters
        self.quantiles = quantiles
        self.ref_dataset = ref_dataset
        self.min_cells = min_cells
        self.dims_use = dims_use
        self.do_center = do_center
        self.max_sample = max_sample
        self.num_trees = num_trees
        self.refine_knn = refine_knn
        self.knn_k = knn_k
        self.use_ann = use_ann

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

        # Run LIGER preprocessing
        liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
        liger_data.var_genes = adata_hvg.var_names
        pyliger.normalize(liger_data)
        pyliger.scale_not_center(liger_data)

        # Filter None parameters for optimize_ALS to use library defaults
        optimize_params = self._filter_none_params(
            {
                "k": self.k,
                "value_lambda": self.value_lambda,
                "thresh": self.thresh,
                "max_iters": self.max_iters,
                "nrep": self.nrep,
                "rand_seed": self.rand_seed,
                "print_obj": self.print_obj,
            }
        )

        pyliger.optimize_ALS(liger_data, **optimize_params)

        # Filter None parameters for quantile_norm to use library defaults
        quantile_params = self._filter_none_params(
            {
                "quantiles": self.quantiles,
                "ref_dataset": self.ref_dataset,
                "min_cells": self.min_cells,
                "dims_use": self.dims_use,
                "do_center": self.do_center,
                "max_sample": self.max_sample,
                "num_trees": self.num_trees,
                "refine_knn": self.refine_knn,
                "knn_k": self.knn_k,
                "use_ann": self.use_ann,
                "rand_seed": self.rand_seed,  # Can be shared with optimize_ALS
            }
        )

        pyliger.quantile_norm(liger_data, **quantile_params)

        # Combine results
        embedding = np.zeros((self.adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
        for i, b in enumerate(batch_cats):
            batch_mask = self.adata.obs[self.batch_key] == b
            embedding[batch_mask] = liger_data.adata_list[i].obsm["H_norm"]

        self.adata.obsm[self.embedding_key] = embedding


class ScanoramaMethod(BaseIntegrationMethod):
    """Scanorama integration method."""

    def __init__(
        self,
        adata,
        sigma: float | None = None,
        alpha: float | None = None,
        knn: int | None = None,
        approx: bool | None = None,
        batch_size: int | None = None,
        verbose: bool | int | None = None,
        dimred: int | None = None,
        hvg: int | None = None,
        return_dense: bool | None = None,
        union: bool | None = None,
        seed: int | None = None,
        sketch: bool | None = None,
        sketch_method: str | None = None,
        sketch_max: int | None = None,
        **kwargs,
    ):
        """
        Initialize Scanorama method.

        Parameters
        ----------
        adata
            Annotated data object to integrate.
        sigma
            Correction smoothing parameter on Gaussian kernel.
        alpha
            Alignment score minimum cutoff.
        knn
            Number of nearest neighbors to use for matching.
        approx
            Use approximate nearest neighbors, greatly speeds up matching runtime.
        batch_size
            The batch size used in the alignment vector computation.
        verbose
            When True or not equal to 0, prints logging output.
        dimred
            Dimensionality of integrated embedding.
        hvg
            Use this number of top highly variable genes based on dispersion.
        return_dense
            Return numpy.ndarray matrices instead of scipy.sparse.csr_matrix.
        union
            Whether to compute the union or intersection of genes.
        seed
            Random seed to use.
        sketch
            Apply sketching-based acceleration by first downsampling the datasets.
        sketch_method
            Apply the given sketching method to the data. Only used if sketch=True.
        sketch_max
            If a dataset has more cells than sketch_max, downsample using sketch_method.
        """
        super().__init__(adata, **kwargs)
        self.sigma = sigma
        self.alpha = alpha
        self.knn = knn
        self.approx = approx
        self.batch_size = batch_size
        self.verbose = verbose
        self.dimred = dimred
        self.hvg_param = hvg  # Renamed to avoid conflict with self.hvg_key
        self.return_dense = return_dense
        self.union = union
        self.seed = seed
        self.sketch = sketch
        self.sketch_method = sketch_method
        self.sketch_max = sketch_max

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

        # Prepare parameters, filtering None values to use library defaults
        scanorama_params = self._filter_none_params(
            {
                "sigma": self.sigma,
                "alpha": self.alpha,
                "knn": self.knn,
                "approx": self.approx,
                "batch_size": self.batch_size,
                "verbose": self.verbose,
                "dimred": self.dimred,
                "hvg": self.hvg_param,
                "return_dense": self.return_dense,
                "union": self.union,
                "seed": self.seed,
                "sketch": self.sketch,
                "sketch_method": self.sketch_method,
                "sketch_max": self.sketch_max,
            }
        )

        # Run Scanorama
        scanorama.integrate_scanpy(adata_list, **scanorama_params)

        # Combine results
        embedding = np.zeros((self.adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
        for i, b in enumerate(batch_cats):
            batch_mask = self.adata.obs[self.batch_key] == b
            embedding[batch_mask] = adata_list[i].obsm["X_scanorama"]

        self.adata.obsm[self.embedding_key] = embedding
