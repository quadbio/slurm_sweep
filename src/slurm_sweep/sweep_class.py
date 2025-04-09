import wandb
from simple_slurm import Slurm

from slurm_sweep._logging import logger


class SweepManager:
    """A class to manage wandb sweeps and submit SLURM jobs."""

    def __init__(self) -> None:
        """Initialize the SweepManager with general configuration."""
        self.project_name: str | None = None
        self.entity: str | None = None
        self.sweep_id: str | None = None

    def __repr__(self) -> str:
        """
        Return a string representation of the SweepManager instance.

        Returns
        -------
        str
            A string showing the current state of the SweepManager instance.
        """
        return f"SweepManager(project_name={self.project_name!r}, entity={self.entity!r}, sweep_id={self.sweep_id!r})"

    def register_sweep(
        self, sweep_config: dict, project_name: str = "uncategorized", entity: str | None = None
    ) -> None:
        """
        Register a sweep with wandb using the provided sweep configuration.

        Parameters
        ----------
        project_name
            The name of the wandb project.
        entity
            The name of the wandb entity or team.
        sweep_config
            The configuration dictionary for the wandb sweep.
        """
        wandb.login()

        self.project_name = project_name
        self.entity = entity

        self.sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=self.project_name,
            entity=self.entity,
        )
        logger.info("Sweep registered with ID '%s'", self.sweep_id)

    def submit_jobs(
        self,
        slurm_parameters: dict | None = None,
        mamba_env: str | None = None,
        job_file: str = "submit.sh",
        convert: bool = False,
        shell: str = "/bin/bash",
        modules: str | None = None,
    ) -> None:
        """
        Submit a SLURM job array to run the wandb agent for the sweep.

        Parameters
        ----------
        slurm_parameters
            A dictionary of SLURM parameters to pass to the `Slurm` class.
        mamba_env
            Mamba environment to activate.
        job_file
            The slurm submission script will be written here.
        convert
            Whether to escape bash variables (see the ``sbatch`` method in ``simple_slurm``).
        shell
            The shell to use for the SLURM job.
        modules
            A space-separated string of modules to load (e.g., "stack eth_proxy").
        """
        if not self.sweep_id:
            raise ValueError("Sweep ID is not set. Please register the sweep first.")

        # Initialize SLURM with user-provided parameters
        slurm_parameters = slurm_parameters or {}
        slurm = Slurm(**slurm_parameters)

        # Load required modules if specified
        if modules:
            slurm.add_cmd(f"module load {modules}")

        # Activate mamba/conda environment
        if mamba_env:
            slurm.add_cmd("source $HOME/.bashrc")  # Source user-specific bashrc
            slurm.add_cmd(f"mamba activate {mamba_env}")

        # Add the wandb agent command
        command = f'wandb agent "{self.sweep_id}"'
        slurm.add_cmd(command)

        logger.info("Submitting SLURM job with the following configuration:\n%s", slurm)
        slurm.sbatch(
            shell=shell,
            job_file=job_file,
            convert=convert,
        )
