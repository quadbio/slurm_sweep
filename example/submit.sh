#!/bin/bash

#SBATCH --array               0-2
#SBATCH --job-name            sweep_test_2
#SBATCH --mem-per-cpu         8G
#SBATCH --output              slurm_logs/wandb_grid_%A_%a.out
#SBATCH --time                00:30:00

module load stack eth_proxy
source $HOME/.bashrc
mamba activate py312
wandb agent "spatial_vi/sweep_test_2/cp0qvbng"
