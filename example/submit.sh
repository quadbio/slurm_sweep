#!/bin/bash

#SBATCH --array               0-1
#SBATCH --job-name            sweep
#SBATCH --mem-per-cpu         1G
#SBATCH --output              slurm_logs/wandb_grid_%A_%a.out
#SBATCH --time                1:00:00

module load stack eth_proxy
source $HOME/.bashrc
mamba activate rapids_jax
wandb agent "spatial_vi/sweep_test/gy3rndlc"
