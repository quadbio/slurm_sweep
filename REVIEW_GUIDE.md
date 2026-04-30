# slurm_sweep Review Guide

This file is the canonical, agent-neutral source of truth for automated PR review in this repo.
It is written for **agents performing PR reviews on GitHub** — use the imperative voice and be concrete.

**Scope: review only.** Your job is to produce review comments and suggestions on the PR. Do **not** push commits, modify files, or apply fixes yourself. Any changes are the author's call. Flag issues, ask questions, and suggest concrete diffs in comments when helpful — but leave the decision and the edits to the user.

Use `AGENTS.md` for architecture, invariants, and commands. Use this guide for review workflow, risk areas, testing checks, documentation-impact checks, and test lookup.

## Review-First Workflow

1. Read the PR body first when it is present.
2. Check CI status (`gh pr checks <num>`, `gh run view <run-id> --log-failed`) and investigate any test or lint failures before commenting. Do not run the test suite locally — rely on CI.
3. Identify changed modules and map them to matching tests (see [Changed-Path Test Lookup](#changed-path-test-lookup)).
4. Check whether the change touches a repo invariant from `AGENTS.md`.
5. Prioritize behavioral regressions in the generated `submit.sh`, config-schema breakage, and CLI-surface changes over style feedback.
6. Verify that docs (human- and agent-facing) did not become stale — see [Documentation Impact](#documentation-impact).

## High-Risk Areas

- **Config schema (`utils.py` / `ConfigValidator`)**:
  changes here silently break every user's YAML. New required fields, renamed keys, or removed defaults must be called out in the PR body and exercised by tests.
- **Generated `submit.sh` (`sweep_class.py` / `SweepManager.write_script`)**:
  module loading, mamba activation, and the final `wandb agent` line are the contract with users' SLURM clusters. Watch for changes to `--count`, shell quoting, and the default `convert=False`.
- **CLI (`cli.py`)**:
  Typer command and option signatures are user-facing. Renames or removed defaults are breaking changes.
- **Entry-point shim (`main.py`)**:
  must remain a one-line re-export of `cli.app` (the console script `slurm-sweep` resolves through it).
- **Public API surface (`__init__.py`)**:
  only `SweepManager` and `logger` are exported. Re-exports of internals (e.g. `ConfigValidator`) commit the project to an API; flag unnecessary additions.
- **W&B SDK drift**:
  `wandb.login()`, `wandb.sweep(...)` signatures and return types vary across versions. New W&B calls should be tested against at least the pinned floor.
- **`count` semantics**:
  `count: 1` per agent is the recommended SLURM pattern. Changes to its propagation into the agent command affect every multi-task array.
- **Examples (`examples/`)**:
  examples are documentation. Any config-schema, CLI, or `submit.sh` change must be reflected in the relevant example.

## Changed-Path Test Lookup

| Touched file | Tests to look at |
|--------------|------------------|
| `src/slurm_sweep/cli.py`, `src/slurm_sweep/main.py` | `tests/test_cli.py` |
| `src/slurm_sweep/sweep_class.py` | `tests/test_sweep_manager.py` |
| `src/slurm_sweep/utils.py` | `tests/test_config_validation.py` |
| `src/slurm_sweep/_logging.py`, `__init__.py`, cross-cutting | `tests/test_basic.py`, `tests/conftest.py` |

## Testing

Apply these checks whenever the PR touches code or tests. **Read CI output rather than running tests locally.**

**New code.** Confirm that new behavior is covered by tests.
- Reuse fixtures from `tests/conftest.py` rather than creating parallel ones.
- Prefer `pytest.mark.parametrize` over many near-identical tests.
- Favor few meaningful tests over many redundant ones; flag low-value tests that only duplicate existing coverage.

**Failing tests.** If CI is red, do not wave it through.
- Inspect which tests fail and why (`gh pr checks <num>`, `gh run view <run-id> --log-failed`).
- Distinguish critical regressions (config-schema, `submit.sh` shape, CLI surface) from trivial or flaky failures.
- Surface critical failures back to the author and ask them to fix before merge.

**Modified tests.** Scrutinize *how* existing tests were changed.
- PRs that only relax thresholds, remove assertions, delete cases, or loosen `parametrize` matrices are a red flag — tests-working-around-tests defeats the purpose.
- Require an explicit justification in the PR body for any weakened assertion; do not accept silently.

## Documentation Impact

A single behavioral or API change often touches docs in multiple places. Check both audiences and ask the author to update what is stale. Point to the **owning file** for each topic rather than duplicating content in your review.

**Human-facing docs.**
- Public API or CLI changes → `README.md` and `docs/contributing.md` if relevant.
- Config-schema changes → `examples/basic_config.yaml` (the canonical commented template) and `examples/README.md`.
- `submit.sh` shape changes → check that `examples/*/submit.sh` references in the example READMEs still match.

**Agent-facing docs (repo root and `.github/`).**
- Invariants or development commands changed → `AGENTS.md` (Critical Invariants, Development Commands).
- Review workflow, risk areas, or testing conventions changed → `REVIEW_GUIDE.md` (this file).
- Repo structure or top-level pointers moved → `AGENTS.md` "Where To Find What" table, `CLAUDE.md`, `.github/copilot-instructions.md`.

If behavior changes but the relevant docs do not, call it out explicitly in the review and request the update.

## Review Checklist

- Does the change preserve the invariants in `AGENTS.md`?
- Does CI pass, and were any failures investigated? (See [Testing](#testing).)
- Is test coverage adequate and non-redundant, and are modified tests not simply weakened? (See [Testing](#testing).)
- Does it alter the generated `submit.sh`, config schema, or CLI surface in a way that needs explicit mention in the PR body?
- Is the public API surface still minimal (`SweepManager` and `logger` only at the top level)?
- Are all affected human- and agent-facing docs updated, including examples? (See [Documentation Impact](#documentation-impact).)
- Is the PR scope tight — no unrelated changes bundled in?
