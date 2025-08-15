# Comparing embeddings for single-cell RNA-sequencing data

Single-cell RNA-sequencing (scRNA-seq) measures gene expression in individual cells and generates large datasets. Typically, these datasets consist of several samples, each corresponding to a combination of covariates (e.g. patient, time point, disease status, technology, etc.). Analyzing these vast datasets (often containing millions of cells for thousands of genes) is facilitated by data integration approaches, which learn lower-dimensional representations that remove the effects of certain unwanted covariates (such as experimental batch, the chip the data was run on, etc).

## Overview
Here, we use `slurm_sweep` to efficiently parallelize and track different data integration approaches on a small test dataset, and we compare their performance in terms of [scIB metrics](https://scib-metrics.readthedocs.io/en/stable/). For each data integration method, we compute a shared latent space, quantify integration performance in terms of batch correction and bio conservation, visualize the latent space with UMAP, store the model and embedding coordinates, and store all relevant data on wandb, so that we can retrieve it after the sweep.

The setup here is that we have one shared config file for all CPU-based methods and one for all GPU-based methods - in reality, you can adjust granularity as you need, for example, you could have one config file per method to define exactly the resources that this method needs to run.

### Methods Implemented
- **GPU-based methods**: scVI, scANVI, scPoli, ResolVI, scVIVA
- **CPU-based methods**: Harmony, LIGER, Scanorama, PCA (baseline)

### Dataset
- **Lung Atlas**: Multi-batch single-cell dataset from the scIB benchmarking suite
- **Automatic download**: Uses scanpy's built-in caching system

### Evaluation
- **scIB metrics**: Standardized benchmarking for integration quality
- **UMAP visualization**: Visual assessment of integration
- **Artifact tracking**: Models and embeddings stored in wandb

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n scembed python=3.12
conda activate scembed

# Install the package with desired method groups
cd examples/single_cell_embedding

# Basic installation (core functionality only)
pip install -e .

# For CPU methods only
pip install -e ".[cpu]"

# For GPU methods only
pip install -e ".[gpu]"

# To use GPU acceleration in evaluation
pip install -e ".[fast_metrics]"

# Install all optional dependencies
pip install -e ".[all]"
```

**Note**: If you encounter C++ compilation errors (e.g., with `louvain` or `annoy`), install those packages via conda first:
```bash
mamba install louvain python-annoy
```

### 2. Dependency Groups

The package uses optional dependency groups to minimize installation overhead:

- **Base**: Core functionality (scanpy, scib-metrics, wandb)
- **`[cpu]`**: CPU-based methods (e.g. Harmony, LIGER, Scanorama)
- **`[gpu]`**: GPU-based methods (e.g. scVI, scANVI, scPoli)
- **`[fast_metrics]`**: Accelerated evaluation with `faiss` and `RAPIDS`
- **`[all]`**: All optional dependencies


## Outputs

### Per Method
- **Integration embedding**: Stored in wandb as table
- **scIB metrics**: Comprehensive benchmarking scores
- **UMAP plots**: Visualization by cell type and batch
- **Model weights**: For deep learning methods

### Summary Metrics
- **scib_total_score**: Overall integration quality
- **scib_bio_conservation**: Preservation of biological signal
- **scib_batch_correction**: Removal of batch effects

## References

- [scIB metrics documentation](https://scib-metrics.readthedocs.io/)
- [scVI-tools](https://docs.scvi-tools.org/)
- [scanpy](https://scanpy.readthedocs.io/)
- [slurm_sweep](https://github.com/quadbio/slurm_sweep)
