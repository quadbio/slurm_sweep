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

    def register_sweep(self, sweep_config: dict, project_name: str | None = None, entity: str | None = None) -> None:
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
        time: str = "1:00:00",
        mem_per_cpu: str = "1G",
        job_name: str = "sweep",
        output: str = f"slurm_logs/wandb_grid_{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out",
        array: tuple | range = range(0, 2),
        mamba_env: str | None = None,
        job_file: str = "submit.sh",
        convert: bool = True,
        shell: str = "/bin/bash",
        modules: str | None = None,
    ) -> None:
        """
        Submit a SLURM job array to run the wandb agent for the sweep.

        Parameters
        ----------
        time
            The maximum runtime for the SLURM job (e.g., "00:30:00").
        mem_per_cpu
            The memory allocated per CPU (e.g., "8G").
        job_name
            The name of the SLURM job.
        output
            The file path for the SLURM job's standard output.
        array
            Job array specification.
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

        # Escape bash variables for the SLURM script
        command = f'wandb agent "{self.entity}/{self.project_name}/{self.sweep_id}"'

        slurm = Slurm(
            time=time,
            mem_per_cpu=mem_per_cpu,
            job_name=job_name,
            output=output,
            array=array,
        )

        # Load required modules if specified
        if modules:
            slurm.add_cmd(f"module load {modules}")

        # Activate mamba/conda environment
        if mamba_env:
            slurm.add_cmd("source $HOME/.bashrc")  # Source user-specific bashrc
            slurm.add_cmd(f"mamba activate {mamba_env}")

        # Add the wandb agent command
        slurm.add_cmd(command)

        logger.info("Submitting SLURM job with the following configuration: %s", slurm)
        slurm.sbatch(shell=shell, job_file=job_file, convert=convert)
