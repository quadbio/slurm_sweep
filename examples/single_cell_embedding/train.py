"""Main training script for single-cell integration method comparison."""

import sys
from pathlib import Path

import wandb

# Add the src directory to the path for local imports
sys.path.append(str(Path(__file__).parent / "src"))

from scembed.data_loader import load_lung_atlas
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


def extract_method_params(config: dict, method_name: str) -> dict:
    """Extract method-specific parameters from wandb config."""
    # Look for method-specific parameter sections
    param_key = f"{method_name.lower()}_params"
    if param_key in config:
        return config[param_key]

    # Fallback to extracting parameters by method name prefix
    method_params = {}
    prefix = f"{method_name.lower()}_"

    for key, value in config.items():
        if key.startswith(prefix):
            param_name = key[len(prefix) :]
            method_params[param_name] = value

    return method_params


def main():
    """Main training function."""
    # Initialize wandb
    wandb.init()

    # Extract method and parameters from config
    method_name = wandb.config.get("method")
    if method_name is None:
        raise ValueError("Method name must be specified in config")

    method_params = extract_method_params(dict(wandb.config), method_name)
    print(f"Method: {method_name}")
    print(f"Method params: {method_params}")

    # Load data
    print("Loading data...")
    adata = load_lung_atlas()

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
