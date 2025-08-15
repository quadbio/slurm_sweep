#!/bin/bash

#SBATCH --array               0-7
#SBATCH --job-name            scembed_cpu
#SBATCH --mem-per-cpu         32G
#SBATCH --output              slurm_logs/cpu_%A_%a.out
#SBATCH --time                01:00:00

module load stack eth_proxy
source $HOME/.bashrc
mamba activate slurm_sweep
wandb agent "spatial_vi/scembed_cpu_comparison/9j7niwyf"
