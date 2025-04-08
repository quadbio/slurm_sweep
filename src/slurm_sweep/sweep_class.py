import wandb
from simple_slurm import Slurm


class SweepManager:
    """A class to manage wandb sweeps and submit SLURM jobs."""

    def __init__(self, project_name: str, organization: str) -> None:
        """
        Initialize the SweepManager with general configuration.

        Parameters
        ----------
        project_name
            The name of the wandb project.
        organization
            The name of the wandb organization or team.
        """
        self.project_name = project_name
        self.organization = organization
        self.sweep_id: str | None = None

    def register_sweep(self, sweep_config: dict) -> None:
        """
        Register a sweep with wandb using the provided sweep configuration.

        Parameters
        ----------
        sweep_config
            The configuration dictionary for the wandb sweep.
        """
        wandb.login()

        self.sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=self.project_name,
            entity=self.organization,
        )
        print(f"Sweep registered with ID: {self.sweep_id}")

    def submit_jobs(
        self,
        time: str,
        mem_per_cpu: str,
        job_name: str,
        output: str,
        error: str,
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
        error
            The file path for the SLURM job's standard error.
        """
        if not self.sweep_id:
            raise ValueError("Sweep ID is not set. Please register the sweep first.")

        command = f'wandb agent "{self.organization}/{self.project_name}/{self.sweep_id}"'

        slurm = Slurm(
            time=time,
            mem_per_cpu=mem_per_cpu,
            job_name=job_name,
            output=output,
            error=error,
        )
        slurm.sbatch(command)
        print("SLURM job array submitted.")
