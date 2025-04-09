import typer

from slurm_sweep._logging import logger
from slurm_sweep.sweep_class import SweepManager
from slurm_sweep.utils import ConfigValidator

app = typer.Typer(help="Run hyperparameter sweeps on SLURM clusters using wandb.")


@app.command()
def main(config: str = typer.Option(..., "-c", "--config", help="Path to the configuration YAML file.")):
    """Main entry point for the CLI."""
    # Validate and load the configuration
    validator = ConfigValidator(config_path=config)
    validator.load_config()
    validator.validate()
    config_data = validator.get_config()

    # Extract general, slurm, and wandb configurations
    general_config = config_data["general"]
    slurm_config = config_data["slurm"]
    wandb_config = config_data["wandb"]

    # Initialize the SweepManager
    sm = SweepManager()

    # Register the sweep
    logger.info("Registering the sweep...")
    sm.register_sweep(
        sweep_config=wandb_config,
        project_name=general_config.get("project_name"),
        entity=general_config.get("organization"),
    )

    # Submit the jobs
    logger.info("Submitting jobs to SLURM...")
    sm.submit_jobs(
        slurm_parameters=slurm_config,
        mamba_env=general_config.get("mamba_env"),
        modules=general_config.get("modules"),
    )


if __name__ == "__main__":
    app()
