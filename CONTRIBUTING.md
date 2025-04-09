# Contributing Guide

This project builds upon the [Scanpy developer guide][scanpy developer guide], which provides extensive documentation for developers. This document summarizes the most important information to help you get started contributing to this project.

If you're unfamiliar with Git or making pull requests on GitHub, please refer to the [Scanpy developer guide][scanpy developer guide].

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html

---

## Installing Development Dependencies

In addition to the packages required to _use_ this package, you'll need additional Python packages to [run tests](#writing-tests).

### Using Hatch

The easiest way to manage dependencies is with [Hatch environments][hatch environments]. You can run the following commands:

```bash
hatch test  # Defined in the [tool.hatch.envs.hatch-test] section of pyproject.toml
```

### Using Pip

If you prefer managing environments manually, you can use `pip`:

```bash
cd slurm_sweep
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

[hatch environments]: https://hatch.pypa.io/latest/tutorials/environment/basic-usage/

---

## Code Style

This project uses [pre-commit][] to enforce consistent code styles. On every commit, pre-commit checks will either automatically fix issues or raise an error message.

### Setting Up Pre-Commit Locally

To enable pre-commit locally, run the following command in the root of the repository:

```bash
pre-commit install
```

Pre-commit will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub. If you didn't run `pre-commit` before pushing changes to GitHub, it will automatically commit fixes to your pull request or show an error message.

If `pre-commit.ci` adds a commit to a branch you're working on locally, use the following command to integrate the changes:

```bash
git pull --rebase
```

While `pre-commit.ci` is useful, we strongly encourage installing and running pre-commit locally to understand its usage.

### Editor Integration

Most editors support _autoformat on save_. Consider enabling this option for [Ruff][ruff-editors] and [Biome][biome-editors].

[pre-commit]: https://pre-commit.com/
[pre-commit.ci]: https://pre-commit.ci/
[ruff-editors]: https://docs.astral.sh/ruff/integrations/
[biome-editors]: https://biomejs.dev/guides/integrate-in-editor/

---

## Writing Tests

This project uses [pytest][] for automated testing. Please write tests for every function added to the package.

### Running Tests Locally

#### Using Hatch

```bash
hatch test  # Test with the highest supported Python version
# or
hatch test --all  # Test with all supported Python versions
```

#### Using Pip

```bash
source .venv/bin/activate
pytest
```

### Continuous Integration

Continuous integration automatically runs tests on all pull requests and tests against the minimum and maximum supported Python versions.

Additionally, a CI job tests against pre-releases of all dependencies (if available). This helps detect incompatibilities with new package versions early, giving you time to fix issues or reach out to the developers of the dependency before the package is widely released.

[pytest]: https://docs.pytest.org/

---

## Publishing a Release

### Updating the Version Number

This package uses `hatch-vcs` to infer version numbers. Follow [Semantic Versioning][semver]:

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version for incompatible API changes,
> 2. MINOR version for backwards-compatible functionality additions, and
> 3. PATCH version for backwards-compatible bug fixes.

Once updated, commit and push your changes. Navigate to the "Releases" page on GitHub, specify `vX.X.X` as the tag name, and create a release. For more details, see [managing GitHub releases][managing GitHub releases].

This will automatically create a Git tag and trigger a GitHub workflow to publish the release on [PyPI][].

[semver]: https://semver.org/
[managing GitHub releases]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
[pypi]: https://pypi.org/
