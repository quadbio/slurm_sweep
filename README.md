# slurm_sweep: hyperparameter sweeps with W&B on SLURM clusters

[![Tests][badge-tests]][tests]

[badge-tests]: https://github.com/quadbio/slurm_sweep/actions/workflows/test.yaml/badge.svg


`slurm_sweep` is the missing (small) piece to efficiently run hyperparameter sweeps on SLURM clusters by combining the power of weights and biases ([W&B][]) and [simple_slurm][]. It allows you to efficiently parallelize sweeps with job arrays, while tracking experiments and results on W&B. All you need is:

- [W&B][] account.
- `config.yaml` file that defines your sweep.
- `train.py` script, that specifies the actual training and evaluation.

## Getting started

Create an account on [W&B][] and take a look at our examples in the [examples][] folder. These contain both `config.yaml` and `train.py` scripts.

### The config file
You need config file in `yaml` format. This file should have three sections:
- `general`: you need to define at least the `project_name` and the `entity` for the sweep on wandB.
- `slurm`: any valid slurm option. Depends on your cluster, see the `simple_slurm` docs.
- `wandb`: [standard W&B config](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration/) for a hyperparameter sweep.

### The training script
This needs to be a python script that defines the training and evaluation logic. It should call `wandb.init()` and retrieve parameters from `wandb.config`. It can log values using `wandb.log`. See the [W&B docs](https://docs.wandb.ai/guides/sweeps/).

### Submission
Once you're ready, you can test your config file using `slurm-sweep validate_config config.yaml`. If this passes, create a submission script using `slurm-sweep configure-sweep config.yaml`, and submit with `sbatch submit.sh`.

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install slurm_sweep:

<!--
1) Install the latest release of `slurm_sweep` from [PyPI][]:

```bash
pip install slurm_sweep
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/quadbio/slurm_sweep.git@main
```

## Release notes

See the [changelog][].

## Contact
If you found a bug, please use the [issue tracker][].


[examples]: https://github.com/quadbio/slurm_sweep/tree/main/examples
[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/quadbio/slurm_sweep/issues
[tests]: https://github.com/quadbio/slurm_sweep/actions/workflows/test.yaml
[changelog]: https://github.com/quadbio/slurm_sweep/blob/main/CHANGELOG.md
[pypi]: https://pypi.org/project/slurm_sweep
[simple_slurm]: https://github.com/amq92/simple_slurm
[W&B]:  https://wandb.ai/site/
