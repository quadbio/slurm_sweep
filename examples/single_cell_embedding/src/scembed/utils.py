import gzip
import pickle
from pathlib import Path
from typing import Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import wandb
from scib_metrics.nearest_neighbors import NeighborsResults

from scembed.check import check_deps
from slurm_sweep._logging import logger


def faiss_brute_force_nn(X: np.ndarray, k: int):
    """GPU brute force nearest neighbor search using faiss."""
    check_deps("faiss-gpu")
    import faiss

    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(X.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(X)
    distances, indices = gpu_index.search(X, k)
    del index
    del gpu_index
    # distances are squared
    return NeighborsResults(indices=indices, distances=np.sqrt(distances))


def subsample_adata(
    adata: ad.AnnData,
    n_obs: int,
    strategy: Literal["naive", "proportional"] = "naive",
    proportional_key: str = "batch",
    random_state: int = 42,
) -> ad.AnnData:
    """
    Subsample AnnData object using different strategies.

    Parameters
    ----------
    adata
        AnnData object to subsample.
    n_obs
        Target number of observations after subsampling.
    strategy
        Subsampling strategy:
        - "naive": Random sampling
        - "proportional": Maintain proportions of categories in proportional_key
    proportional_key
        Key in .obs for proportional sampling (e.g., "batch", "cell_type").
    random_state
        Random seed for reproducibility.

    Returns
    -------
    ad.AnnData
        Subsampled AnnData object (or original if no subsampling needed).
    """
    if n_obs >= adata.n_obs:
        logger.info("Requested subsample size (%d) >= current size (%d), returning original data", n_obs, adata.n_obs)
        return adata

    np.random.seed(random_state)

    if strategy == "naive":
        indices = np.random.choice(adata.n_obs, size=n_obs, replace=False)
    elif strategy == "proportional":
        # Maintain proportions of categories
        category_counts = adata.obs[proportional_key].value_counts()
        category_proportions = category_counts / category_counts.sum()

        indices = []
        for category, proportion in category_proportions.items():
            category_mask = adata.obs[proportional_key] == category
            category_indices = np.where(category_mask)[0]
            n_category_samples = int(np.round(n_obs * proportion))

            if n_category_samples > 0 and len(category_indices) > 0:
                n_to_sample = min(n_category_samples, len(category_indices))
                sampled_indices = np.random.choice(category_indices, size=n_to_sample, replace=False)
                indices.extend(sampled_indices)

        indices = np.array(indices)
        # Ensure we don't exceed the requested number
        if len(indices) > n_obs:
            indices = np.random.choice(indices, size=n_obs, replace=False)
    else:
        raise ValueError(f"Unknown subsampling strategy: {strategy}")

    logger.info("Subsampled from %d to %d cells using %s strategy", adata.n_obs, len(indices), strategy)

    return adata[indices].copy()


def _download_artifact_by_run_id(
    run_id: str,
    entity: str,
    project: str,
    artifact_name: str,
    download_dir: Path,
) -> Path | None:
    """
    Download a specific artifact from a wandb run.

    Parameters
    ----------
    run_id
        The run ID of the specific run.
    entity
        The wandb entity (user or team name).
    project
        The wandb project name.
    artifact_name
        The name of the artifact to download.
    download_dir
        The directory where the artifact should be downloaded.

    Returns
    -------
    Path | None
        The local directory where the artifact was downloaded, or None if failed.
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Find the artifact
    artifacts = [artifact for artifact in run.logged_artifacts() if artifact_name in artifact.name]
    if not artifacts:
        logger.debug("No artifact with name '%s' found for run %s", artifact_name, run_id)
        return None

    # Download the artifact
    artifact = artifacts[0]
    artifact_dir = artifact.download(root=download_dir)
    logger.debug("Downloaded %s from run %s to: %s", artifact_name, run_id, artifact_dir)
    return Path(artifact_dir)


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
        check_deps("lightning")
        from lightning.pytorch.loggers import WandbLogger
    except RuntimeError:
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


def load_embedding(file_path: Path | str) -> pd.DataFrame:
    """Load a saved embedding file back into a DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)

    elif file_path.name.endswith(".pkl.gz"):
        with gzip.open(file_path, "rb") as f:
            return pickle.load(f)

    elif file_path.suffix == ".h5":
        with h5py.File(file_path, "r") as hf:
            return pd.DataFrame(
                data=hf["embedding"][:],
                index=[n.decode() for n in hf["cell_names"][:]],
                columns=[n.decode() for n in hf["dim_names"][:]],
            )

    else:
        raise ValueError(f"Unsupported file format: {file_path.name}. Supported: .parquet, .pkl.gz, .h5")
