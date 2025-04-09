class SlurmDefaults:
    """Default SLURM configuration constants."""

    TIME = "1:00:00"
    MEM_PER_CPU = "4G"
    JOB_NAME = "sweep"
    OUTPUT = "slurm-%j.out"
    ERROR = "slurm-%j.err"
