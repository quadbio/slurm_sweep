# Copilot Instructions for slurm_sweep

## Project Overview

**slurm_sweep** is a lightweight CLI tool for running hyperparameter sweeps on
SLURM clusters using Weights & Biases (W&B). It bridges W&B sweeps with SLURM
job arrays via [simple_slurm][], enabling efficient parallel hyperparameter
optimization.

### Domain Context
- **SLURM**: HPC workload manager for job scheduling on clusters. Uses job
  arrays for parallel execution.
- **W&B Sweeps**: Hyperparameter optimization tool that tracks experiments,
  supports grid/random/bayesian search.
- **Workflow**: 1) Define sweep in `config.yaml`, 2) Write training script
  (`train.py`) that logs to W&B, 3) Submit SLURM job array where each job
  runs a W&B agent.

### Key Dependencies
- **Core**: wandb, simple-slurm, typer (CLI), numpy, session-info2

## Architecture

### Core Components
1. **`src/slurm_sweep/cli.py`**: CLI commands via Typer
   - `configure-sweep`: Register W&B sweep, generate `submit.sh` script
   - `validate-config`: Validate YAML config file
2. **`src/slurm_sweep/sweep_class.py`**: `SweepManager` class
   - `register_sweep()`: Create W&B sweep via API
   - `write_script()`: Generate SLURM submission script
3. **`src/slurm_sweep/utils.py`**: `ConfigValidator` for YAML validation
4. **`examples/`**: Example configs and training scripts

## Project-Specific Patterns

### Config File Structure (YAML)
Three sections: `general` (project/entity/env), `slurm` (job params), `wandb`
(sweep config). See `examples/` for templates.

### CLI Usage
```bash
slurm-sweep validate-config config.yaml
slurm-sweep configure-sweep config.yaml
sbatch submit.sh
```

## Common Gotchas

1. **W&B login**: Must be logged in (`wandb login`) before running `configure-sweep`
2. **SLURM array size**: Match array range to number of parallel jobs needed (e.g., `array: "0-9"` = 10 jobs)
3. **Agent count**: Set `count: 1` in general config to avoid multiple agents per job (recommended for SLURM)
4. **Module loading**: Modules are space-separated in config (e.g., `modules: "stack eth_proxy"`)
5. **Environment activation**: Mamba env must exist on compute nodes, not just login node
6. **Training script**: Must call `wandb.init()` and log metrics via `wandb.log()`

## Related Resources

- **Examples**: `examples/` folder with various use cases
- **W&B Sweeps docs**: https://docs.wandb.ai/guides/sweeps/
- **simple_slurm**: https://github.com/amq92/simple_slurm

[simple_slurm]: https://github.com/amq92/simple_slurm
