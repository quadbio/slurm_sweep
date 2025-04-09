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
        time=slurm_config["time"],
        mem_per_cpu=slurm_config["mem_per_cpu"],
        job_name=slurm_config["job_name"],
        output=slurm_config["output"],
        array=range(0, 2),  # Default array range; can be extended
        mamba_env=general_config.get("mamba_env"),
        modules=slurm_config.get("modules"),  # Optional modules
    )


if __name__ == "__main__":
    app()
