#!/bin/bash

#SBATCH --partition           test
#SBATCH --time                01:00:00

module load test_module
source $HOME/.bashrc
mamba activate test_env
wandb agent "test_entity/test_project/test-sweep-id"
