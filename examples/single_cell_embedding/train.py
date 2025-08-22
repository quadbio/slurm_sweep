"""Main training script for single-cell integration method comparison."""

import scanpy as sc
import wandb
from scembed import get_method_instance
from scembed.evaluation import IntegrationEvaluator

from slurm_sweep._logging import logger

sc.set_figure_params(frameon=False, fontsize=12)

# Define some constants for this dataset
CT_KEY = "cell_type"
BATCH_KEY = "batch"
UNLABELED_CATEGORY = "unknown"
PCA_KEY = "X_pca"


def main():
    """Main training function."""
    # Initialize wandb
    wandb.init()

    # Extract method and parameters from config
    config = wandb.config.config
    method_name = config["method"]

    # Extract method-specific parameters (everything except 'method')
    method_params = {k: v for k, v in config.items() if k != "method"}
    method_params.update(
        {"batch_key": BATCH_KEY, "cell_type_key": CT_KEY, "pca_key": PCA_KEY, "unlabeled_category": UNLABELED_CATEGORY}
    )

    logger.info("Method: %s", method_name)
    logger.info("Method params: %s", method_params)

    # Load and preprocess data
    logger.info("Loading lung atlas dataset...")
    adata = sc.read("data/lung_atlas.h5ad", backup_url="https://figshare.com/ndownloader/files/24539942")
    logger.info("Loaded: %s cells × %s genes", f"{adata.n_obs:,}", f"{adata.n_vars:,}")

    # Simple preprocessing
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
    sc.tl.pca(adata, n_comps=30, mask_var="highly_variable")

    # Create method instance
    logger.info("Initializing %s method...", method_name)
    method = get_method_instance(adata, method_name, method_params)

    # Fit and transform
    logger.info("Fitting method...")
    method.fit()

    logger.info("Transforming data...")
    method.transform()

    # Save model if applicable
    model_path = method.save_model(method.models_dir)
    if model_path is not None:
        logger.info("Model saved to %s", model_path)
        wandb.log_model(str(model_path), name="trained_model")

    # Save embedding coordinates
    logger.info("Saving embedding coordinates...")
    emb_path = method.save_embedding(format_type="parquet")

    # Log embedding as wandb artifact
    logger.info("Logging embedding to wandb...")
    try:
        artifact = wandb.Artifact("embedding", type="dataset")
        artifact.add_file(str(emb_path))
        wandb.log_artifact(artifact)
        logger.info("Successfully logged embedding artifact to wandb")
    except Exception as e:  # noqa
        logger.error("Failed to log embedding artifact to wandb: %s", e)

    # Evaluate integration
    logger.info("Evaluating integration...")
    evaluator = IntegrationEvaluator(
        adata=method.adata,
        embedding_key=method.embedding_key,
        batch_key=method.batch_key,
        cell_type_key=method.cell_type_key,
        output_dir=method.output_dir,
        baseline_embedding_key=method.pca_key,  # Use existing PCA from preprocessing
    )

    # Run scIB evaluation
    evaluator.evaluate_scib()
    wandb.summary["scib"] = evaluator.scib_metrics.loc[method.embedding_key].to_dict()

    # Generate UMAP plots
    evaluator.compute_and_show_embeddings(wspace=0.7)
    wandb.log({"umap_evaluation": wandb.Image(str(evaluator.figures_dir / "umap_evaluation.png"))})

    logger.info("✓ Run completed successfully")


if __name__ == "__main__":
    main()
