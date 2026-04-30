# AGENTS.md — slurm_sweep

`slurm_sweep` is a small CLI tool that bridges Weights & Biases sweeps with SLURM
job arrays via [simple_slurm][]. It registers a W&B sweep from a YAML config and
emits a `submit.sh` script that runs `wandb agent` inside a SLURM array.

Key dependencies: `wandb`, `simple-slurm`, `typer`, `session-info2`.
Python: 3.11–3.14.

## Trust Order

When sources disagree:
1. PR description and changed code
2. This file (`AGENTS.md`)
3. `REVIEW_GUIDE.md`
4. Tests under `tests/`
5. Example configs and training scripts under `examples/`

This file owns invariants and the where-to-find table. Review-specific workflow
lives in `REVIEW_GUIDE.md`.

## Review Guidelines

For GitHub PR reviews, use `REVIEW_GUIDE.md` as the canonical review workflow and
source of review-specific risk areas, testing checks, and documentation-impact
checks. This file only owns the project invariants and source-of-truth map.

## Where To Find What

| Topic | Source of truth |
|-------|-----------------|
| CLI commands (`configure-sweep`, `validate-config`) | `src/slurm_sweep/cli.py` |
| Console-script entry shim (`slurm-sweep` → `app`) | `src/slurm_sweep/main.py` |
| Sweep registration + `submit.sh` generation | `src/slurm_sweep/sweep_class.py` (`SweepManager`) |
| YAML config schema + validation | `src/slurm_sweep/utils.py` (`ConfigValidator`) |
| Logger setup (`logger`) | `src/slurm_sweep/_logging.py` |
| Public API | `src/slurm_sweep/__init__.py` — exports `SweepManager`, `logger` only |
| Example configs and training scripts | `examples/` (`basic_config.yaml`, `simple/`, `scrna_seq_embedding/`, `spatial_embedding/`) |
| Test fixtures | `tests/conftest.py` |
| Contributor guide | `docs/contributing.md` |
| PR review workflow & risk areas | `REVIEW_GUIDE.md` |

## Critical Invariants

### Config schema (enforced by `ConfigValidator`)
- Three top-level blocks: `general`, `slurm`, `wandb`. Unknown blocks trigger a
  warning and are ignored.
- `wandb` is mandatory and must contain `program`, `method`, and `parameters`
  (per the W&B sweep spec).
- `general` is mandatory and must contain `entity` and `project_name`.
- `slurm` is optional; defaults to `{}` if missing.

### Generated `submit.sh` shape
- `simple_slurm.Slurm(**slurm_parameters)` builds the `#SBATCH` header.
- Optional `module load <modules>` (space-separated string in config).
- Optional mamba activation: `source $HOME/.bashrc && mamba activate <env>`. The
  env must exist on **compute** nodes, not just the login node.
- Final command: `wandb agent [--count N] "<entity>/<project>/<sweep_id>"`.
- `count: 1` is the recommended SLURM setting — one agent per array task.

### CLI / entry point
- `pyproject.toml` declares `scripts.slurm-sweep = "slurm_sweep.main:app"`.
  `main.py` is a one-line re-export of `cli.app`; do not move logic into it.
- `wandb.login()` is called inside `SweepManager.register_sweep()`, so the user
  must already be logged in when running `configure-sweep`.

### Public API
- Only `SweepManager` and `logger` are exported from the top level. Everything
  else (`ConfigValidator`, internal helpers) is reachable only via submodule
  imports. Keep this surface minimal — additions require justification.

### Training scripts (user-supplied)
- Must call `wandb.init()` and read hyperparameters from `wandb.config`.
- Metrics must be logged via `wandb.log()` for the sweep to optimize anything.
- `examples/simple/train.py` is the canonical minimal template.

## Development Commands

Python 3.11 and 3.14 are the matrix endpoints (see `[tool.hatch.envs.hatch-test.matrix]`).

```bash
uv sync                             # install dev deps via dependency-groups
uv run --group test pytest          # quick local test run
uvx hatch test                      # run tests on the highest matrix Python
uvx hatch test --all                # full matrix (3.11 stable, 3.14 stable, 3.14 pre)
pre-commit run --all-files          # lint + format (ruff, biome, pyproject-fmt)
```

CLI smoke-tests (no SLURM/W&B needed for `validate-config`):

```bash
slurm-sweep validate-config examples/basic_config.yaml
slurm-sweep configure-sweep examples/simple/config.yaml   # requires `wandb login`
```

[simple_slurm]: https://github.com/amq92/simple_slurm
