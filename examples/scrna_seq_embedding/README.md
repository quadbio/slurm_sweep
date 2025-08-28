# Comparing embeddings for single-cell RNA-sequencing data

Single-cell RNA-sequencing (scRNA-seq) measures gene expression in individual cells and generates large datasets. Typically, these datasets consist of several samples, each corresponding to a combination of covariates (e.g. patient, time point, disease status, technology, etc.). Analyzing these vast datasets (often containing millions of cells for thousands of genes) is facilitated by data integration approaches, which learn lower-dimensional representations that remove the effects of certain unwanted covariates (such as experimental batch, the chip the data was run on, etc).

## Overview
Here, we use `slurm_sweep` to efficiently parallelize and track different data integration approaches on a small test dataset, and we compare their performance in terms of [scIB metrics](https://scib-metrics.readthedocs.io/en/stable/). For each data integration method, we compute a shared latent space, quantify integration performance in terms of batch correction and bio conservation, visualize the latent space with UMAP, store the model and embedding coordinates, and store all relevant data on wandb, so that we can retrieve it after the sweep.

The setup here is that we have one shared config file for all CPU-based methods and one for all GPU-based methods - in reality, you can adjust granularity as you need, for example, you could have one config file per method to define exactly the resources that this method needs to run.

To facilitate running different methods, we use the `scembed` package, which you need to install separately via `pip install scembed`.

## References

- [scIB metrics documentation](https://scib-metrics.readthedocs.io/)
- [scVI-tools](https://docs.scvi-tools.org/)
- [scanpy](https://scanpy.readthedocs.io/)
- [slurm_sweep](https://github.com/quadbio/slurm_sweep)
- [scembed](https://github.com/quadbio/scembed)
