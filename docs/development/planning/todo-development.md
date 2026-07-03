---
title: Development Roadmap
---

# Development Roadmap

Planned improvements to the DynVision toolbox itself (code, workflow, and
tooling — as distinct from the documentation tasks tracked in
[Documentation TODOs](todo-docs.md)).

Each item links to the detailed design note in `development/planning/` where one
exists. Status markers: ⬜ planned · 🚧 in progress · ✅ done (kept briefly for
context, then folded into release notes).

---

## Compatibility & Packaging

- ⬜ **Python 3.12 / 3.13 support** — currently pinned to `>=3.11,<3.14` but only
  tested on 3.11. The only blocker is FFCV: `torch>=2.4.0` /
  `torchvision>=0.19.0` are required for 3.12, and FFCV's C++ extension must be
  verified to build against them. All other dependencies are already
  3.12-compatible. See
  [Python 3.12 compatibility notes](python-3.12-compatibility.md).
- ⬜ **CI matrix across Python versions** — extend the test workflow to run on
  3.11, 3.12, and 3.13 once FFCV builds cleanly, so regressions surface early.
- ⬜ **Custom-model template files** — the README refers to model templates that
  do not yet exist in the repo. Ship a minimal, copy-pasteable template (a
  `BaseModel` subclass plus a config stub) under `examples/`.

## Configuration & Parameters

- 🚧 **Config-driven transform system** — move transform selection out of
  hardcoded logic in `dynvision/data/transforms.py` into the config layer so
  presets are inspectable and overridable from YAML. Fixes the fragile
  substring-matching lookup (e.g. `ffcv_test_imagenette` failing) and makes the
  torch/FFCV composition rules consistent. See
  [Transform configuration roadmap](transforms.md).
- ⬜ **Parameter-provenance logging** — replace the generic `(override)` /
  `(adjusted)` markers with explicit provenance (which precedence tier —
  `default` / `config` / `cli` / `override` — set each value, and whether it was
  later mutated at runtime or derived by a validator). See the *Provenance
  Tagging Plan* in [Logging modernization](logging-rework-plan.md).
- ⬜ **Document mode-detection precedence** — the interaction between
  `log_level="DEBUG"` and `epochs <= 5` (and other mode triggers) needs a single
  authoritative decision tree in code and docs so mode activation is
  predictable.

## Workflow & Reproducibility

- ✅ **Parallel experiment processing** — `process_test_data` split into a
  per-output processing stage plus aggregation, enabling parallelism and
  reduced disk pressure for large (30 GB+) response files. See
  [Parallel experiment processing](parallel_experiment_processing.md).
- ⬜ **Config snapshot isolation** — editing config files while a cluster
  workflow is running can leak the updated config into jobs Snakemake submits
  later, breaking within-run reproducibility. Freeze the resolved config at
  workflow start and point running jobs at the snapshot. See
  [Snakecharm config stability](snakecharm-config-stability-issue.md).
- ⬜ **Standardise experiment-config wildcards** — mark which wildcards are
  required vs optional in experiment paths so users don't produce incomplete
  targets.

## Logging & Diagnostics

- 🚧 **Centralised, structured logging** — route every entrypoint through
  `dynvision.utils.configure_logging`, let each Pydantic params class own its
  `summary_sections`, and trim duplicate/chatty output between params modules
  and runtime scripts. See [Logging modernization](logging-rework-plan.md).

## Performance & Visualization

- 🚧 **Memory-efficient response plotting** — `plot_responses.py` OOMs on
  ~450 MB / ~7.5 M-row `test_data.csv`. Load only required columns, downcast,
  aggregate before rendering, and stop Seaborn from recomputing error bars per
  trace. Shared helpers move to `dynvision/utils/visualization_utils.py`. See
  [Visualization refactor plan](visualizations.md).
- ⬜ **FFCV guidance** — document when FFCV helps vs. adds overhead, and the
  fallback behaviour when it is unavailable, alongside the existing install
  troubleshooting.
- ⬜ **Mixed-precision guidance** — GPU requirements, numerical-stability caveats,
  and representative speedups for `precision: "bf16-mixed"` / `"16-mixed"`.

## Testing

- ⬜ **Broaden the test suite** — extend coverage of temporal-dynamics
  correctness, recurrence integration, data-loading pipelines, parameter
  validation, and model initialization / coordination-network building.

---

## Naming Cleanup

- ⬜ **Single canonical project name** — reconcile the mixed use of
  `rhythmic_visual_attention` (Makefile), `Modeling_Dynamical_Vision`
  (`project_paths.py` `project_name`), and `DynVision` (`toolbox_name`) so
  commands and paths are consistent, and drop the user-specific default
  `working_dir`.

---

!!! note "Where completed work lives"
    Finished items are summarised in release notes rather than accumulating
    here. The detailed design notes in `development/planning/` are retained as a
    historical record even after their work lands.
