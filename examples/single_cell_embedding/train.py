"""Main training script for single-cell integration method comparison."""

import scanpy as sc
import wandb
from scembed.evaluation import IntegrationEvaluator
from scembed.methods import (
    BaseIntegrationMethod,
    HarmonyMethod,
    LIGERMethod,
    PrecomputedEmbeddingMethod,
    ScanoramaMethod,
    scANVIMethod,
    scPoliMethod,
    scVIMethod,
)


def get_method_instance(adata, method_name: str, method_params: dict) -> BaseIntegrationMethod:
    """
    Create an instance of the specified integration method.

    Parameters
    ----------
    adata
        Annotated data object.
    method_name
        Name of the integration method.
    method_params
        Parameters for the method.

    Returns
    -------
    BaseIntegrationMethod
        Instance of the integration method.
    """
    method_map = {
        "pca": lambda adata, **kwargs: PrecomputedEmbeddingMethod(adata, embedding_key="X_pca", **kwargs),
        "harmony": HarmonyMethod,
        "liger": LIGERMethod,
        "scanorama": ScanoramaMethod,
        "scvi": scVIMethod,
        "scanvi": scANVIMethod,
        "scpoli": scPoliMethod,
    }

    method_class = method_map.get(method_name.lower())
    if method_class is None:
        raise ValueError(f"Unknown method: {method_name}")

    return method_class(adata, **method_params)


def main():
    """Main training function."""
    # Initialize wandb
    wandb.init()

    # Extract method and parameters from config - simple and direct
    method_name = wandb.config.method
    method_params = getattr(wandb.config, method_name.lower(), {}).get("parameters", {})

    print(f"Method: {method_name}")
    print(f"Method params: {method_params}")

    # Load and preprocess data
    print("Loading lung atlas dataset...")
    adata = sc.read("data/lung_atlas.h5ad", backup_url="https://figshare.com/ndownloader/files/24539942")
    print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Simple preprocessing
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
    sc.tl.pca(adata, n_comps=30, mask_var="highly_variable")

    # Create method instance
    print(f"Initializing {method_name} method...")
    method = get_method_instance(adata, method_name, method_params)

    # Fit and transform
    print("Fitting method...")
    method.fit()

    print("Transforming data...")
    method.transform()

    # Save model if applicable
    model_path = method.save_model(method.models_dir)
    if model_path is not None:
        print(f"Model saved to {model_path}")
        wandb.log_model(str(model_path), name="trained_model")

    # Evaluate integration
    print("Evaluating integration...")
    evaluator = IntegrationEvaluator(
        adata=method.adata,
        embedding_key=method.embedding_key,
        batch_key=method.batch_key,
        cell_type_key=method.cell_type_key,
        output_dir=method.output_dir,
    )

    # Run scIB evaluation
    evaluator.evaluate_scib()
    wandb.summary["scib"] = evaluator.scib_metrics.loc[method.embedding_key].to_dict()

    # Generate UMAP plots
    evaluator.compute_and_show_embeddings()
    wandb.log({"umap_evaluation": wandb.Image(str(evaluator.figures_dir / "umap_evaluation.png"))})

    print("✓ Run completed successfully")


if __name__ == "__main__":
    main()
