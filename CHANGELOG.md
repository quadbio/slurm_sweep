# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [v0.3.0]

### Removed

- Support for Python 3.11; Python 3.12 or newer is now required [#34]
- Unused `numpy` runtime dependency [#25]

### Changed

- Update the cookiecutter-scverse template to v0.8.0 [#25], [#34]

## [v0.2.1]

### Removed

- Support for Python 3.10 [#20]

### Changed

- Update the cookiecutter-scverse template to v0.6.0 [#19]

## [v0.2.0]

### Added

- PyPI badge and installation instructions [#6]
- Added two examples for data integration: one for scRNA-seq data [#13], [#14], [#15], [#16] and one for spatial data (seqFISH) [#17]
- Added a count parameter [#9]

## [v0.1.3]

### Changed

- Attempt again to publish to PyPI

## [v0.1.2]

### Changed

- Attempt to publish to PyPI

## [v0.1.1]

### Added

- Set up pre-commit.ci and codecov [#4]
- Release on PyPI

## [v0.1.0]

Basic version of this package to create a `submit.sh` script and initialize a hyperparameter sweep on W&B.
