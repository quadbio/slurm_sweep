"""Integration evaluation utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

from scembed.check import check_deps
from scembed.utils import faiss_brute_force_nn, subsample_adata
from slurm_sweep._logging import logger


class IntegrationEvaluator:
    """Evaluator for single-cell integration methods."""

    def __init__(
        self,
        adata: ad.AnnData,
        embedding_key: str,
        batch_key: str = "batch",
        cell_type_key: str = "cell_type",
        ignore_cell_types: list[str] | None = None,
        output_dir: str | Path | None = None,
        baseline_embedding_key: str = "X_pca_unintegrated",
    ):
        """
        Initialize integration evaluator.

        Parameters
        ----------
        adata
            Annotated data object with integration embedding.
        embedding_key
            Key in .obsm containing the integration embedding.
        batch_key
            Key in .obs for batch information.
        cell_type_key
            Key in .obs for cell type labels.
        output_dir
            Directory for saving evaluation outputs. If None, creates temporary directory.
        baseline_embedding_key
            Key in .obsm containing the unintegrated baseline embedding. If this embedding
            doesn't exist, it will be computed automatically.
        ignore_cell_types
            List of cell types to ignore during evaluation.
        """
        self.embedding_key = embedding_key
        self.batch_key = batch_key
        self.cell_type_key = cell_type_key
        self.baseline_embedding_key = baseline_embedding_key

        # Always make a copy to avoid modifying the original
        if ignore_cell_types is None:
            self.adata = adata.copy()
        else:
            if isinstance(ignore_cell_types, str):
                ignore_cell_types = [ignore_cell_types]

            mask = adata.obs[self.cell_type_key].isin(ignore_cell_types)
            self.adata = adata[~mask].copy()
            logger.info("Ignoring cell types: %s, filtered out %d cells", ignore_cell_types, mask.sum())

        # Setup output directories
        self._temp_dir = None
        if output_dir is None:
            self._temp_dir = TemporaryDirectory()
            output_dir = Path(self._temp_dir.name)
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Validate required embedding exists
        if self.embedding_key not in adata.obsm:
            raise ValueError(f"Integration embedding '{self.embedding_key}' not found in adata.obsm")

        # Ensure unintegrated baseline exists
        self._ensure_unintegrated_baseline()

        # Storage for results
        self.scib_metrics: pd.DataFrame | None = None

        logger.info("Initialized evaluator for '%s', saving to '%s", embedding_key, self.output_dir)

    def _ensure_unintegrated_baseline(self) -> None:
        """Ensure unintegrated PCA baseline exists for scIB evaluation."""
        if self.baseline_embedding_key not in self.adata.obsm:
            logger.info("Computing unintegrated PCA baseline as '%s'...", self.baseline_embedding_key)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.tl.pca(self.adata, n_comps=50, key_added=self.baseline_embedding_key)

    def evaluate_scib(
        self,
        min_max_scale: bool = False,
        use_faiss: bool = False,
        subsample_to: int | None = None,
        subsample_strategy: Literal["naive", "proportional"] = "naive",
        subsample_key: str | None = None,
        subset_to: tuple[str, str | int | list[str | int]] | None = None,
    ) -> None:
        """
        Evaluate integration using scIB metrics.

        Parameters
        ----------
        min_max_scale
            Whether to apply min-max scaling to results.
        use_faiss
            Whether to use FAISS GPU-accelerated nearest neighbor search.
        subsample_to
            If provided, subsample to this many cells before evaluation.
        subsample_strategy
            Strategy for subsampling when subsample_to is provided.
        subsample_key
            Key for proportional subsampling. If None, uses batch_key for proportional strategy.
        subset_to
            Tuple of a key in .obs and a list of categories to subset to.
        """
        logger.info("Computing scIB metrics...")

        # Apply subsampling if requested
        if subsample_to is not None and subsample_to < self.adata.n_obs:
            if subsample_key is None:
                subsample_key = self.batch_key
            adata_work = subsample_adata(
                self.adata, n_obs=subsample_to, strategy=subsample_strategy, proportional_key=subsample_key
            )
        else:
            adata_work = self.adata

        # Subset if requrested
        if subset_to is not None:
            key, values = subset_to
            if not isinstance(values, (list | tuple)):
                values = [values]
            mask = adata_work.obs[key].isin(values)
            adata_work = adata_work[mask].copy()

        # Filter cells without cell type annotations
        before_filter = adata_work.shape[0]
        cell_mask = adata_work.obs[self.cell_type_key].isna()
        adata_filtered = adata_work[~cell_mask]
        after_filter = adata_filtered.shape[0]

        logger.info("Filtered %d cells without %s annotations", before_filter - after_filter, self.cell_type_key)
        logger.info("Evaluating on %s cells", f"{after_filter:,}")

        # Set up neighbor computer
        neighbor_computer = None
        if use_faiss:
            try:
                check_deps("faiss-gpu")
                neighbor_computer = faiss_brute_force_nn
                logger.info("Using FAISS GPU-accelerated neighbor search")
            except RuntimeError:
                logger.info("FAISS not available, falling back to default neighbor search")

        # Set up benchmarker
        bm = Benchmarker(
            adata_filtered,
            batch_key=self.batch_key,
            label_key=self.cell_type_key,
            embedding_obsm_keys=[self.embedding_key],
            pre_integrated_embedding_obsm_key=self.baseline_embedding_key,
            bio_conservation_metrics=BioConservation(isolated_labels=False),
            batch_correction_metrics=BatchCorrection(),
            n_jobs=-1,
        )

        # Run benchmark
        if neighbor_computer is not None:
            bm.prepare(neighbor_computer=neighbor_computer)
        else:
            bm.prepare()
        bm.benchmark()

        # Store evaluation results
        self.scib_metrics = bm.get_results(min_max_scale=min_max_scale)
        logger.info("scIB metrics evaluation completed.")

    def compute_and_show_embeddings(
        self,
        key_added: str = "X_umap",
        use_rapids: bool = False,
        additional_colors: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Compute and visualize UMAP embedding.

        Parameters
        ----------
        key_added
            Key in .obsm for storing UMAP embedding.
        use_rapids
            Whether to use rapids_singlecell for acceleration.
        additional_colors
            Additional keys in .obs for coloring the UMAP plot. By default, we color in cell type and batch information.
        kwargs
            Additional keyword arguments for scanpy.pp.embedding
        """
        logger.info("Computing UMAP embedding...")

        if use_rapids:
            try:
                check_deps("rapids-singlecell")
                import rapids_singlecell as rsc

                # Compute UMAP with RAPIDS
                rsc.pp.neighbors(self.adata, use_rep=self.embedding_key, n_neighbors=15)
                rsc.tl.umap(self.adata, key_added=key_added)
            except RuntimeError:
                logger.info("RAPIDS not available, falling back to scanpy")
                use_rapids = False

        if not use_rapids:
            # Compute UMAP with scanpy
            sc.pp.neighbors(self.adata, use_rep=self.embedding_key, n_neighbors=15)
            sc.tl.umap(self.adata, key_added=key_added)

        # Plot embeddings
        if additional_colors is None:
            additional_colors = []
        if isinstance(additional_colors, str):
            additional_colors = [additional_colors]

        colors = [self.cell_type_key, self.batch_key] + additional_colors

        sc.pl.embedding(self.adata, basis=key_added, color=colors, show=False, **kwargs)
        plt.savefig(self.figures_dir / "umap_evaluation.png", bbox_inches="tight")
        plt.close()

        logger.info("UMAP embeddings plotted and saved to %s", self.figures_dir)

    def get_summary_metrics(self) -> dict[Any, Any]:
        """
        Get summary metrics from scIB evaluation.

        Returns
        -------
        dict
            Dictionary with key metrics for logging.
        """
        if self.scib_metrics is None:
            raise ValueError("Run evaluate_scib() first")

        method_results = self.scib_metrics.loc[self.embedding_key]
        return method_results.to_dict()

    def __repr__(self) -> str:
        """String representation of the evaluator."""
        data_info = f"{self.adata.n_obs:,} cells × {self.adata.n_vars:,} genes"
        embedding_status = "✓" if self.embedding_key in self.adata.obsm else "✗"
        evaluation_status = "evaluated" if self.scib_metrics is not None else "not evaluated"

        return (
            f"IntegrationEvaluator("
            f"embedding='{self.embedding_key}' {embedding_status}, "
            f"batch_key='{self.batch_key}', "
            f"cell_type_key='{self.cell_type_key}', "
            f"{evaluation_status}, "
            f"{data_info})"
        )
