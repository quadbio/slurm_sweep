"""Integration evaluation utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from slurm_sweep._logging import logger


class IntegrationEvaluator:
    """Evaluator for single-cell integration methods."""

    def __init__(
        self,
        adata: ad.AnnData,
        embedding_key: str,
        batch_key: str = "batch",
        cell_type_key: str = "cell_type",
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
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.batch_key = batch_key
        self.cell_type_key = cell_type_key
        self.baseline_embedding_key = baseline_embedding_key

        # Setup output directories
        self._temp_dir = None
        if output_dir is None:
            self._temp_dir = TemporaryDirectory()
            output_dir = Path(self._temp_dir.name)
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.results_dir = output_dir / "results"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

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

    def evaluate_scib(self, min_max_scale: bool = False) -> None:
        """
        Evaluate integration using scIB metrics.

        Parameters
        ----------
        min_max_scale
            Whether to apply min-max scaling to results.
        """
        try:
            from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
        except ImportError as exc:
            raise ImportError("scib-metrics is required for evaluation") from exc

        logger.info("Computing scIB metrics...")

        # Filter cells without cell type annotations
        before_filter = self.adata.shape[0]
        cell_mask = self.adata.obs[self.cell_type_key].isna()
        adata_filtered = self.adata[~cell_mask]
        after_filter = adata_filtered.shape[0]

        logger.info("Filtered %d cells without %s annotations", before_filter - after_filter, self.cell_type_key)
        logger.info("Evaluating on %s cells", f"{after_filter:,}")

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
        bm.prepare()
        bm.benchmark()

        # Store evaluation results
        self.scib_metrics = bm.get_results(min_max_scale=min_max_scale)
        logger.info("scIB metrics evaluation completed.")

    def compute_and_show_embeddings(self, key_added: str = "X_umap", use_rapids: bool = False) -> None:
        """
        Compute and visualize UMAP embedding.

        Parameters
        ----------
        key_added
            Key in .obsm for storing UMAP embedding.
        use_rapids
            Whether to use rapids_singlecell for acceleration.
        """
        logger.info("Computing UMAP embedding...")

        if use_rapids:
            try:
                import rapids_singlecell as rsc

                # Compute UMAP with RAPIDS
                rsc.pp.neighbors(self.adata, use_rep=self.embedding_key, n_neighbors=15)
                rsc.tl.umap(self.adata, key_added=key_added)
            except ImportError:
                logger.info("RAPIDS not available, falling back to scanpy")
                use_rapids = False

        if not use_rapids:
            # Compute UMAP with scanpy
            sc.pp.neighbors(self.adata, use_rep=self.embedding_key, n_neighbors=15)
            sc.tl.umap(self.adata, key_added=key_added)

        # Plot embeddings
        colors = [self.cell_type_key, self.batch_key]
        with plt.rc_context({"figure.figsize": (8, 6)}):
            sc.pl.embedding(self.adata, basis=key_added, color=colors, show=False, wspace=0.7)
            plt.savefig(self.figures_dir / "umap_evaluation.png", bbox_inches="tight")
            plt.close()

        logger.info("UMAP embeddings plotted and saved to %s", self.figures_dir)

    def get_summary_metrics(self) -> dict[str, Any]:
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
