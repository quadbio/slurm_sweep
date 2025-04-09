from typing import Any

import yaml

from slurm_sweep.constants import SlurmDefaults


class ConfigValidator:
    """A class to validate and process the configuration file for slurm-sweep."""

    def __init__(self, config_path: str) -> None:
        """
        Initialize the ConfigValidator with the path to the configuration file.

        Parameters
        ----------
        config_path
            Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config: dict[str, Any] = {}

    def __repr__(self) -> str:
        """Return a string representation of the ConfigValidator instance."""
        return f"ConfigValidator(config_path={self.config_path!r}, config={self.config!r})"

    def load_config(self) -> None:
        """
        Load the configuration file and validate its structure.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        yaml.YAMLError
            If the configuration file is not a valid YAML file.
        """
        try:
            with open(self.config_path, encoding="utf-8") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}") from exc

    def validate(self) -> None:
        """
        Validate the configuration file.

        Raises
        ------
        ValueError
            If the mandatory `wandb` block is missing or invalid.
        """
        # Validate the `wandb` block
        if "wandb" not in self.config:
            raise ValueError("The `wandb` block is mandatory and must be defined in the configuration file.")

        # Validate the `wandb` block structure
        wandb_config = self.config["wandb"]
        if not isinstance(wandb_config, dict):
            raise ValueError("The `wandb` block must be a dictionary.")
        if "program" not in wandb_config or "method" not in wandb_config or "parameters" not in wandb_config:
            raise ValueError(
                "The `wandb` block must include `program`, `method`, and `parameters` keys. "
                "Refer to https://docs.wandb.ai/guides/sweeps/define-sweep-configuration/ for details."
            )

        # Add default values for optional blocks
        self.config.setdefault("general", {})
        self.config.setdefault(
            "slurm",
            {
                "time": SlurmDefaults.TIME,
                "mem_per_cpu": SlurmDefaults.MEM_PER_CPU,
                "job_name": SlurmDefaults.JOB_NAME,
                "output": SlurmDefaults.OUTPUT,
                "error": SlurmDefaults.ERROR,
            },
        )

    def get_config(self) -> dict[str, Any]:
        """
        Get the validated and processed configuration.

        Returns
        -------
        dict
            The validated configuration dictionary.
        """
        return self.config
