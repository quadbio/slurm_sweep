"""WandB sweep results aggregation and visualization for scIB benchmarking."""

from dataclasses import fields

import numpy as np
import pandas as pd
import scanpy as sc
import wandb
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from scib_metrics.benchmark._core import metric_name_cleaner
from tqdm import tqdm

from slurm_sweep._logging import logger


class scIBAggregator:
    """Aggregator for WandB sweep results with scIB metrics visualization.

    Retrieves runs from a WandB project, organizes them by integration method,
    and provides visualization capabilities using scIB metrics formatting.

    Parameters
    ----------
    entity
        WandB entity name.
    project
        WandB project name.
    """

    def __init__(self, entity: str, project: str):
        self.entity = entity
        self.project = project

        # Raw data storage
        self.raw_df: pd.DataFrame | None = None

        # Processed data organized by method
        self.method_data: dict[str, dict[str, pd.DataFrame | Benchmarker]] = {}

        # Data quality tracking
        self.n_runs_fetched = 0
        self.n_runs_filtered = 0
        self.missing_config_runs: list[str] = []
        self.missing_metrics_runs: list[str] = []

        # Available metrics in the data
        self.available_scib_metrics: set[str] = set()

        # Mapping of metric display names to their types
        self.metric_to_type: dict[str, str] = self._build_metric_type_mapping()

        logger.info("Initialized scIBAggregator for %s/%s", entity, project)

    def fetch_runs(self) -> None:
        """Fetch runs from WandB and process into internal storage."""
        logger.info("Fetching runs from %s/%s...", self.entity, self.project)

        # Reset state for fresh fetch
        self.raw_df = None
        self.method_data = {}
        self.n_runs_fetched = 0
        self.n_runs_filtered = 0
        self.missing_config_runs = []
        self.missing_metrics_runs = []
        self.available_scib_metrics = set()

        # Initialize WandB API
        api = wandb.Api()

        # Fetch all runs for the project
        runs = api.runs(f"{self.entity}/{self.project}")

        # Extract parameter configurations and final results
        data = []
        for run in tqdm(runs, desc="Processing runs"):
            run_data = {
                "run_id": run.id,
                "name": run.name,
                **run.config,
                **dict(run.summary),
            }
            data.append(run_data)

        # Convert to DataFrame
        self.raw_df = pd.DataFrame(data)
        self.n_runs_fetched = len(self.raw_df)

        logger.info("Fetched %d runs", self.n_runs_fetched)

        # Process the data
        self._process_runs()

        logger.info(
            "Processing complete: %d methods, %d runs filtered out", len(self.method_data), self.n_runs_filtered
        )

    def _process_runs(self) -> None:
        """Process raw runs into method-organized data structures."""
        if self.raw_df is None:
            raise ValueError("No runs fetched. Call fetch_runs() first.")

        # Filter runs with missing configs
        valid_runs = []
        for idx, row in self.raw_df.iterrows():
            config = row.get("config")
            if config is None or not isinstance(config, dict) or "method" not in config:
                self.missing_config_runs.append(row["run_id"])
                continue
            valid_runs.append(idx)

        if self.missing_config_runs:
            logger.warning(
                "Filtered out %d runs with missing/invalid configs: %s",
                len(self.missing_config_runs),
                self.missing_config_runs[:5],
            )

        valid_df = self.raw_df.loc[valid_runs].copy()

        # Extract scIB metrics and validate
        scib_metrics_data = []
        config_data = []
        other_logs_data = []

        # Define aggregate metrics to exclude from individual metrics
        aggregate_metrics = {"Batch correction", "Bio conservation", "Total"}

        for _, row in valid_df.iterrows():
            config = row["config"]
            method = config["method"]

            # Extract scIB metrics from the 'scib' column
            scib_data = row.get("scib", {})

            # Handle various types of scib data (dict, SummarySubDict, etc.)
            if scib_data is None:
                self.missing_metrics_runs.append(row["run_id"])
                continue

            # Convert to regular dict if it's a wandb SummarySubDict or similar
            if hasattr(scib_data, "keys") and hasattr(scib_data, "items"):
                scib_data = dict(scib_data)

            # Now check if it's empty or not a dict
            if not isinstance(scib_data, dict) or len(scib_data) == 0:
                self.missing_metrics_runs.append(row["run_id"])
                continue

            # Separate actual scIB metrics from aggregates
            actual_scib_metrics = {k: v for k, v in scib_data.items() if k not in aggregate_metrics}

            # Filter config to only relevant parameters using original config dict
            filtered_config = self._filter_config_for_method(config, method)
            config_entry = {
                "run_id": row["run_id"],
                **filtered_config,
            }
            config_data.append(config_entry)

            # Build other logs entry (everything else from the run)
            other_logs_entry = {
                "run_id": row["run_id"],
                "method": method,
                **{k: v for k, v in row.items() if k not in ["config", "scib", "run_id"]},
            }
            other_logs_data.append(other_logs_entry)

            # Build scIB metrics entry (only actual metrics, no aggregates)
            metrics_entry = {"run_id": row["run_id"], "method": method, **actual_scib_metrics}
            scib_metrics_data.append(metrics_entry)

            # Track available metrics (only actual metrics)
            self.available_scib_metrics.update(actual_scib_metrics.keys())

        if self.missing_metrics_runs:
            logger.warning(
                "Filtered out %d runs with missing scIB metrics: %s",
                len(self.missing_metrics_runs),
                self.missing_metrics_runs[:5],
            )

        # Update filter count
        self.n_runs_filtered = len(self.missing_config_runs) + len(self.missing_metrics_runs)

        # Create dataframes
        if not config_data:
            logger.warning("No valid runs found with both config and scIB metrics")
            return

        full_config_df = pd.DataFrame(config_data).set_index("run_id")
        full_metrics_df = pd.DataFrame(scib_metrics_data).set_index("run_id")
        full_other_logs_df = pd.DataFrame(other_logs_data).set_index("run_id")

        # Organize by method and create Benchmarker objects
        methods = full_config_df["method"].unique()
        for method in methods:
            method_mask = full_config_df["method"] == method
            method_runs = full_config_df.index[method_mask]

            method_config_df = full_config_df.loc[method_runs]
            method_metrics_df = full_metrics_df.loc[method_runs]
            method_other_logs_df = full_other_logs_df.loc[method_runs]

            # Create Benchmarker object for this method's scIB metrics
            try:
                benchmarker = self._create_benchmarker_for_method(method, method_metrics_df)
            except (ValueError, KeyError, ImportError) as e:
                logger.warning("Failed to create Benchmarker for method %s: %s", method, e)
                # Fallback to storing as DataFrame if Benchmarker creation fails
                benchmarker = method_metrics_df

            self.method_data[method] = {
                "configs": method_config_df,
                "scib_benchmarker": benchmarker,
                "other_logs": method_other_logs_df,
            }

        logger.info("Available scIB metrics: %s", sorted(self.available_scib_metrics))

    def _filter_config_for_method(self, config: dict, method: str) -> dict:
        """Filter config to only relevant parameters for the given method.

        Parameters
        ----------
        config
            Full configuration dictionary from WandB run
        method
            Method name to filter parameters for

        Returns
        -------
        dict
            Filtered config containing only parameters used by this method
        """
        # Start with method name
        filtered_config = {"method": method}

        # Find all non-null parameters in the config (excluding 'method')
        for key, value in config.items():
            if key != "method" and value is not None:
                filtered_config[key] = value

        return filtered_config

    def _build_metric_type_mapping(self) -> dict[str, str]:
        """Build mapping from metric names to metric types using scIB dataclasses."""
        metric_to_type = {}

        for name, display_name in metric_name_cleaner.items():
            if any(name.startswith(field.name) for field in fields(BioConservation)):
                metric_to_type[display_name] = "Bio conservation"
            if any(name.startswith(field.name) for field in fields(BatchCorrection)):
                metric_to_type[display_name] = "Batch correction"

        return metric_to_type

    def _create_benchmarker_for_method(self, scib_df: pd.DataFrame) -> Benchmarker:
        """Create a Benchmarker object for a specific method's results.

        Parameters
        ----------
        method
            Method name
        scib_df
            DataFrame with scIB metrics for this method (runs as rows, metrics as columns)

        Returns
        -------
        Benchmarker
            Benchmarker object with results loaded
        """
        # Transpose so metrics are rows and runs are columns (as expected by scIB)
        scib_plot = scib_df.drop(columns=["method"]).T.copy()

        # Add metric type information
        scib_plot["Metric Type"] = scib_plot.index.map(self.metric_to_type)

        # Filter to only metrics that have type mappings
        valid_metrics = [metric for metric in scib_plot.index if metric in self.metric_to_type]
        scib_plot = scib_plot.loc[valid_metrics]

        # Reorder rows based on the order in self.metric_to_type
        ordered_metrics = [metric for metric in self.metric_to_type.keys() if metric in valid_metrics]
        scib_plot = scib_plot.loc[ordered_metrics]

        # Create minimal dummy AnnData for Benchmarker initialization
        n_obs = 10
        n_vars = 5
        dummy_adata = sc.AnnData(np.random.random((n_obs, n_vars)))
        dummy_adata.obs["batch"] = ["batch1", "batch2"] * (n_obs // 2)
        dummy_adata.obs["celltype"] = ["type1", "type2"] * (n_obs // 2)

        # Create Benchmarker instance
        biocons = BioConservation(isolated_labels=False)  # isolated labels is computationally expensive
        bm = Benchmarker(
            adata=dummy_adata,
            batch_key="batch",
            label_key="celltype",
            embedding_obsm_keys=list(scib_plot.columns[:-1]),  # Exclude 'Metric Type' column
            bio_conservation_metrics=biocons,
            batch_correction_metrics=BatchCorrection(),
            n_jobs=-1,
        )

        # Inject the results directly (skip actual benchmarking)
        bm._results = scib_plot
        bm._benchmarked = True  # Mark as benchmarked to avoid re-running

        return bm

    def get_best_runs_summary(self, sort_by: str = "Total") -> pd.DataFrame:
        """
        Get the best run per method based on specified metric.

        Parameters
        ----------
        sort_by
            Metric to sort by for selecting best run per method.
            Default is "Total" (overall scIB score).

        Returns
        -------
        pd.DataFrame
            DataFrame with best run per method, indexed by method name.
        """
        if not self.method_data:
            raise ValueError("No data available. Call fetch_runs() first.")

        best_runs = []
        for method, data in self.method_data.items():
            benchmarker = data["scib_benchmarker"]
            configs_df = data["configs"]

            # Get metrics DataFrame from Benchmarker
            try:
                results_df = benchmarker.get_results()
                # Remove the "Metric Type" row if present
                if "Metric Type" in results_df.index:
                    metrics_df = results_df.drop(index="Metric Type")
                else:
                    metrics_df = results_df
            except (AttributeError, KeyError, ValueError):
                # Fallback if benchmarker is actually a DataFrame (for failed Benchmarker creation)
                if isinstance(benchmarker, pd.DataFrame):
                    metrics_df = benchmarker.drop(columns=["method"])
                else:
                    logger.warning("Could not extract metrics for method '%s', skipping", method)
                    continue

            if sort_by not in metrics_df.columns:
                logger.warning("Metric '%s' not found for method '%s', skipping", sort_by, method)
                continue

            # Find best run
            best_idx = metrics_df[sort_by].idxmax()

            # Combine config and metrics for best run
            best_run = {
                "method": method,
                "run_id": best_idx,
                **configs_df.loc[best_idx].to_dict(),
                **metrics_df.loc[best_idx].to_dict(),
            }
            best_runs.append(best_run)

        if not best_runs:
            logger.warning("No valid runs found with metric '%s'", sort_by)
            return pd.DataFrame()

        return pd.DataFrame(best_runs).set_index("method")

    def get_method_runs(self, method: str) -> dict[str, pd.DataFrame | Benchmarker]:
        """
        Get all data for a specific method.

        Parameters
        ----------
        method
            Method name to retrieve data for.

        Returns
        -------
        dict
            Dictionary with 'configs' (DataFrame), 'scib_benchmarker' (Benchmarker or DataFrame),
            and 'other_logs' (DataFrame).
        """
        if method not in self.method_data:
            raise ValueError(f"Method '{method}' not found. Available methods: {list(self.method_data.keys())}")

        return self.method_data[method]

    @property
    def available_methods(self) -> list[str]:
        """List of available methods in the data."""
        return list(self.method_data.keys())

    def __repr__(self) -> str:
        """String representation with data summary."""
        if self.raw_df is None:
            return f"scIBAggregator({self.entity}/{self.project}, no data fetched)"

        n_methods = len(self.method_data)
        n_valid_runs = self.n_runs_fetched - self.n_runs_filtered
        n_metrics = len(self.available_scib_metrics)

        status_parts = []
        if self.missing_config_runs:
            status_parts.append(f"{len(self.missing_config_runs)} missing configs")
        if self.missing_metrics_runs:
            status_parts.append(f"{len(self.missing_metrics_runs)} missing metrics")

        status = f" ({', '.join(status_parts)})" if status_parts else ""

        return (
            f"scIBAggregator("
            f"{self.entity}/{self.project}, "
            f"{n_valid_runs}/{self.n_runs_fetched} runs, "
            f"{n_methods} methods, "
            f"{n_metrics} metrics"
            f"{status})"
        )
