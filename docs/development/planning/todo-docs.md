---
title: Documentation TODOs
---

# Documentation TODOs

Open documentation tasks: missing pages, doc/code mismatches, and content gaps.
Toolbox/code work is tracked separately in the
[Development Roadmap](todo-development.md).

> Last swept 2026-07-03. Resolved items are dropped rather than archived here;
> the historical record lives in git history and release notes.

---

## Missing Pages & Sections

- ✅ **Custom experiments** — [Custom Experiments guide](../../user-guide/custom-experiments.md) (added 2026-07-03).
- ⬜ **Visualization gallery** — `user-guide/visualization.md` is thin; add
  screenshot examples of each plot type.
- ⬜ **Monitoring callbacks** — what metrics are logged, where, and how to add
  custom ones (no reference page for `callbacks.py`).
- ⬜ **StorageBuffer API** — mentioned in `model-base.md` but has no dedicated
  reference page (methods, usage, memory considerations).
- ⬜ **Parameter override examples** — CLI / YAML / Snakemake override patterns
  in one place.
- ⬜ **Transforms reference** — complete list of available transforms and their
  parameters (only mentioned in `transform-configuration.md`).
- ⬜ **Complete parameter reference** — exhaustive list of all parameters by
  component.
- ⬜ **Migration guide** — for users upgrading between versions.

### Code modules without a reference page

- **High priority** (user-facing): `recurrence`, `temporal`, `dynamics_solver`,
  `integration_strategy`, `transforms`, `callbacks`, `datamodule`.
- **Medium**: `dataloader`, `datasets`, `ffcv_*`, `noise`, `retina`,
  `supralinearity`, `bias`.
- **Low** (internal): `*_utils`, `*_params`, `project_paths`, `snake*`,
  `mode_registry`.

---

## Doc / Code Consistency

- ⬜ **Parameter system integration** (`user-guide/parameter-handling.md`) — the
  Pydantic parameter system is documented, but its interaction with the
  `@alias_kwargs` decorator, the Snakemake config system, and Lightning's
  hyperparameter tracking is not. Clarify precedence and show the systems
  working together. *(cross-refs the provenance work in the
  [Development Roadmap](todo-development.md))*
- ⬜ **Mode-detection decision tree** (`user-guide/parameter-handling.md`) —
  document what happens when multiple mode triggers apply and the override
  precedence.
- ⬜ **Model initialization sequence** (`reference/model-base.md`) — add the
  missing `_verify_architecture()` step to match `base/temporal.py`.
- ⬜ **Experiment-config wildcards** (`user-guide/workflows.md`) — mark
  required vs optional wildcards so users don't build incomplete paths.

---

## Content Quality

- ⬜ **Code-example style guide** — examples vary in whether they show imports,
  use full class paths, or include type hints. Establish and apply one style.
- ⬜ **Type hints in examples** — the codebase uses type hints extensively;
  examples mostly don't.
- ⬜ **Code-of-conduct surfacing** — currently in `not_in_nav`; decide whether
  it should appear in the nav or footer.

---

## Notes for Contributors

When updating documentation:

- Verify every class name, method signature, and file path against the actual
  code before writing.
- Include working, tested code examples.
- Use terminology consistent with existing docs.
- Link related sections; add diagrams for complex concepts.

When changing code:

- Update the affected docs in the same change.
- Add docstrings in the existing NumPy style (mkdocstrings renders them).
- Update the [Claude Code Guide](../guides/claude-guide.md) if the architecture
  shifts significantly.
