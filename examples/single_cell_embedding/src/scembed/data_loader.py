"""Data loading utilities for single-cell integration examples."""

from pathlib import Path

import scanpy as sc


def load_lung_atlas(cache_dir: str | Path | None = None, subset_hvg: bool = False):
    """
    Load the lung atlas dataset for integration benchmarking.

    This dataset is used in the scIB metrics tutorial and contains
    multiple batches suitable for integration method comparison.

    Parameters
    ----------
    cache_dir
        Directory to cache the downloaded data. If None, uses scanpy's default.
    subset_hvg
        Whether to subset to highly variable genes.

    Returns
    -------
    adata
        Annotated data object with:
        - Raw counts in .X
        - Normalized counts in .layers["counts"]
        - Batch information in .obs["batch"]
        - Cell type annotations in .obs["cell_type"]
    """
    print("Loading lung atlas dataset...")

    # Set cache directory if provided
    cache_dir = Path(cache_dir) if cache_dir else Path("data")
    filename = cache_dir / "lung_atlas.h5ad"

    # Download the data
    adata = sc.read(filename, backup_url="https://figshare.com/ndownloader/files/24539942", cache=True)

    print(f"Loaded dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"Batches: {adata.obs['batch'].nunique()}")
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # Basic preprocessing for method comparison
    print("Performing basic preprocessing...")

    # Compute highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")

    # Compute PCA for methods that need it
    # Normalize and log-transform the main data (raw counts are preserved in layers)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)

    # Store unintegrated baseline
    adata.obsm["X_pca_unintegrated"] = adata.obsm["X_pca"].copy()

    print("Dataset preprocessing completed.")
    print(f"Highly variable genes: {adata.var['highly_variable'].sum():,}")

    # Subset to highly variable genes if requested
    if subset_hvg:
        print("Subsetting to highly variable genes...")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"Subsetted dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    return adata
