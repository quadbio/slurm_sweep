#!/bin/bash

#SBATCH --job-name            scembed_pca
#SBATCH --mem-per-cpu         32G
#SBATCH --output              slurm_logs/pca_%A_%a.out
#SBATCH --time                01:00:00

module load stack eth_proxy
source $HOME/.bashrc
mamba activate slurm_sweep
wandb agent "spatial_vi/scembed_test_spatial_2/oreacgyh"
