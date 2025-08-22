"""Factory for creating integration method instances."""

from slurm_sweep._logging import logger


def get_method_instance(adata, method_name: str, method_params: dict):
    """
    Create an instance of the specified integration method.

    Parameters
    ----------
    adata
        Annotated data object.
    method_name
        Name of the integration method. Methods starting with "precomputed"
        (e.g., "precomputed_pca", "precomputed_umap") will be mapped to
        PrecomputedEmbeddingMethod.
    method_params
        Parameters for the method.

    Returns
    -------
    BaseIntegrationMethod
        Instance of the integration method.

    Examples
    --------
    >>> from scembed.factory import get_method_instance
    >>> method = get_method_instance(adata, "harmony", {"theta": 2.0})
    >>> method = get_method_instance(adata, "scvi", {"n_latent": 10})
    >>> method = get_method_instance(adata, "precomputed", {"embedding_key": "X_pca"})
    >>> method = get_method_instance(adata, "precomputed_pca", {"embedding_key": "X_pca"})
    >>> method = get_method_instance(adata, "precomputed_umap", {"embedding_key": "X_umap"})
    """
    from scembed.methods import (
        HarmonyMethod,
        LIGERMethod,
        PrecomputedEmbeddingMethod,
        ResolVIMethod,
        ScanoramaMethod,
        scANVIMethod,
        scPoliMethod,
        scVIMethod,
        scVIVAMethod,
    )

    # Available integration methods
    method_map = {
        "harmony": HarmonyMethod,
        "liger": LIGERMethod,
        "precomputed": PrecomputedEmbeddingMethod,
        "resolvi": ResolVIMethod,
        "scanvi": scANVIMethod,
        "scanorama": ScanoramaMethod,
        "scpoli": scPoliMethod,
        "scvi": scVIMethod,
        "scviva": scVIVAMethod,
    }

    method_name_lower = method_name.lower()

    # Handle precomputed variants (precomputed, precomputed_pca, precomputed_umap, etc.)
    if method_name_lower.startswith("precomputed"):
        method_class = PrecomputedEmbeddingMethod
    else:
        method_class = method_map.get(method_name_lower)

    if method_class is None:
        available_methods = ", ".join(sorted(method_map.keys()))
        raise ValueError(
            f"Unknown method: {method_name}. Available methods: {available_methods} "
            f"or any method starting with 'precomputed' (e.g., 'precomputed_pca')"
        )

    logger.info("Creating %s method instance", method_name)
    return method_class(adata, **method_params)
