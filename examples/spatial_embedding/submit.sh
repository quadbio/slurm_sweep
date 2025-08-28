#!/bin/bash

#SBATCH --array               0-3
#SBATCH --gpus                rtx_4090:1
#SBATCH --job-name            scembed_resolvi
#SBATCH --mem-per-cpu         32G
#SBATCH --output              slurm_logs/resolvi_%A_%a.out
#SBATCH --time                01:00:00

module load stack eth_proxy
source $HOME/.bashrc
mamba activate scembed
wandb agent --count 1 "spatial_vi/scembed_test_spatial_rc1/o8jzlhkp"
