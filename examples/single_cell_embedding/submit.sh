#!/bin/bash

#SBATCH --array               0-15
#SBATCH --gpus                rtx_4090:1
#SBATCH --job-name            scembed_gpu
#SBATCH --mem-per-cpu         16G
#SBATCH --output              slurm_logs/gpu_%A_%a.out
#SBATCH --time                01:00:00

module load stack eth_proxy
source $HOME/.bashrc
mamba activate scembed
wandb agent "spatial_vi/scembed_test/wt48api1"
