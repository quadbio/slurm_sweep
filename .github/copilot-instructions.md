# Copilot Instructions for slurm_sweep

## Important Notes
- Avoid drafting summary documents or endless markdown files. Just summarize in chat what you did, why, and any open questions.
- When running terminal commands, activate the appropriate environment first (`mamba activate slurm_sweep`).
- Rather than making assumptions, ask for clarification when uncertain.
- **GitHub workflows**: Use GitHub CLI (`gh`) when possible. For GitHub MCP server tools, ensure Docker Desktop is running first (`open -a "Docker Desktop"`).

## Project Overview

**slurm_sweep** is a lightweight CLI tool for running hyperparameter sweeps on SLURM clusters using Weights & Biases (W&B). It bridges W&B sweeps with SLURM job arrays via [simple_slurm][], enabling efficient parallel hyperparameter optimization. Users provide a `config.yaml` and `train.py` script, then submit via `slurm-sweep configure-sweep config.yaml && sbatch submit.sh`.

### Domain Context (Brief)
- **SLURM**: HPC workload manager for job scheduling on clusters. Uses job arrays for parallel execution.
- **W&B Sweeps**: Hyperparameter optimization tool that tracks experiments, supports grid/random/bayesian search.
- **Workflow**: 1) Define sweep in `config.yaml`, 2) Write training script (`train.py`) that logs to W&B, 3) Submit SLURM job array where each job runs a W&B agent.

### Key Dependencies
- **Core**: wandb, simple-slurm, typer (CLI), numpy, session-info2
- **Optional**: scembed (for examples)

## Architecture & Code Organization

### Module Structure (simple Python package)
- Type annotations use modern syntax: `str | None` instead of `Optional[str]`
- Supports Python 3.11, 3.12, 3.13 (see `pyproject.toml`)
- CLI entry point via Typer: `slurm-sweep` command

### Core Components
1. **`src/slurm_sweep/cli.py`**: CLI commands via Typer
   - `configure-sweep`: Register W&B sweep, generate `submit.sh` script
   - `validate-config`: Validate YAML config file
2. **`src/slurm_sweep/sweep_class.py`**: `SweepManager` class
   - `register_sweep()`: Create W&B sweep via API
   - `write_script()`: Generate SLURM submission script
3. **`src/slurm_sweep/utils.py`**: `ConfigValidator` for YAML validation
4. **`examples/`**: Example configs and training scripts for various use cases

## Development Workflow

### Environment Management (Hatch-based)
```bash
# Testing - NEVER use pytest directly
hatch test                    # run pytest in isolated environment

# Environment inspection
hatch env show                # list environments
```

### Testing Strategy
- Tests live in `tests/`, use pytest
- Run via `hatch test` to ensure proper environment isolation
- Config validation tests check YAML structure

### Code Quality Tools
- **Ruff**: Linting and formatting (120 char line length)
- **Biome**: JSON/JSONC formatting with trailing commas
- **Pre-commit**: Auto-runs ruff, biome. Install with `pre-commit install`
- Use `git pull --rebase` if pre-commit.ci commits to your branch

## Key Configuration Files

### `pyproject.toml`
- **Build**: `hatchling` with `hatch-vcs` for git-based versioning
- **Dependencies**: Minimal (wandb, simple-slurm, typer)
- **Ruff**: 120 char line length, NumPy docstring convention
- **CLI**: Entry point `slurm-sweep` → `slurm_sweep.main:app`

### Version Management
- Version from git tags via `hatch-vcs`
- Release: Create GitHub release with tag `vX.X.X`
- Follows **Semantic Versioning**

## Project-Specific Patterns

### Config File Structure (YAML)
Three sections: `general` (project/entity/env), `slurm` (job params), `wandb` (sweep config). See `examples/` for templates.

### CLI Usage
```bash
# 1. Validate config
slurm-sweep validate-config config.yaml

# 2. Configure sweep (creates submit.sh)
slurm-sweep configure-sweep config.yaml

# 3. Submit to SLURM
sbatch submit.sh
```

### Generated Submission Script
- Loads specified modules (if any)
- Activates mamba environment (if specified)
- Runs `wandb agent` with sweep ID
- Uses SLURM array indices for parallelization

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
- **CONTRIBUTING**: `CONTRIBUTING.md` for development guidelines

[simple_slurm]: https://github.com/amq92/simple_slurm
