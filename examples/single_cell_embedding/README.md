# Comparing embeddings for single-cell RNA-sequencing data

Single-cell RNA-sequencing (scRNA-seq) measures gene expression in individual cells and generates large datasets. Typically, these datasets consist of several samples, each corresponding to a combination of covariates (e.g. patient, time point, disease status, technology, etc.). Analyzing these vast datasets (often containing millions of cells for thousands of genes) is facilitated by data integration approaches, which learn lower-dimensional representations that remove the effects of certain unwanted covariates (such as experimental batch, the chip the data was run on, etc).

## Overview
Here, we use `slurm_sweep` to efficiently parallelize and track different data integration approaches on a small test dataset, and we compare their performance in terms of [scIB metrics](https://scib-metrics.readthedocs.io/en/stable/). For each data integration method, we compute a shared latent space, quantify integration performance in terms of batch correction and bio conservation, visualize the latent space with UMAP, store the model and embedding coordinates, and store all relevant data on wandb, so that we can retrieve it after the sweep.

### Methods Implemented
- **GPU-based methods**: scVI, scANVI, scPoli
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
pip install -e .[cpu]

# For GPU methods only
pip install -e .[gpu]

# For all methods
pip install -e .[all]

# For development
pip install -e .[all,dev]
```

**Note**: If you encounter C++ compilation errors (e.g., with `louvain` or `annoy`), install those packages via conda first:
```bash
mamba install -y louvain annoy -c conda-forge
```

### 2. Dependency Groups

The package uses optional dependency groups to minimize installation overhead:

- **Base**: Core functionality (scanpy, scib-metrics, wandb)
- **`[cpu]`**: CPU-based methods (Harmony, LIGER, Scanorama)
- **`[gpu]`**: GPU-based methods (scVI, scANVI, scPoli)
- **`[all]`**: All integration methods
- **`[dev]`**: Development tools (pytest, black, ruff)

### 3. Test Locally

```bash
# Run a simple test to verify everything works
python test_example.py
```

### 4. Run with slurm_sweep

```bash
# For GPU-based methods
slurm_sweep config_gpu.yaml

# For CPU-based methods
slurm_sweep config_cpu.yaml
```

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

## Extending the Example

### Adding New Methods
1. Create method class in `methods/cpu_methods.py` or `methods/gpu_methods.py`
2. Inherit from `BaseIntegrationMethod`
3. Implement `fit()` and `transform()` methods
4. Add to configuration file

### Custom Datasets
1. Modify `data_loader.py` to load your dataset
2. Ensure required columns: `batch`, `cell_type`
3. Update preprocessing as needed

### Additional Metrics
1. Extend `evaluation.py` with custom metrics
2. Add to summary metrics dictionary
3. Log to wandb for tracking

## References

- [scIB metrics documentation](https://scib-metrics.readthedocs.io/)
- [scVI-tools](https://docs.scvi-tools.org/)
- [scanpy](https://scanpy.readthedocs.io/)
- [slurm_sweep](https://github.com/quadbio/slurm_sweep)
