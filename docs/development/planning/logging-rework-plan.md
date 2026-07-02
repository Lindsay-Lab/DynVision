# DynVision Logging Modernization Notes

_This living document tracks the ongoing effort to centralize and structure logging across DynVision. Update it as changes land._

## Task Statement
- Reduce redundant or overly chatty logging between params modules and runtime scripts.
- Provide consistent, structured summaries owned by Pydantic parameter classes.
- Establish reusable logging utilities and document the patterns for future development.

## Current Strategy
1. **Map existing logging surfaces** — audit params, runtime scripts, workflow rules, and model utilities to understand current verbosity and duplication.
2. **Centralize configuration** — rely on `dynvision.utils.configure_logging` in `BaseParams` so every entrypoint respects CLI `log_level`, standard formatters, and optional log files.
3. **Structured summaries** — let each params class define `summary_sections` (with `SummaryItem`) and surface them through `BaseParams.log_summary` / `CompositeParams.log_overview`.
4. **Runtime integration** — refactor scripts (init/train/test) to call the param summaries instead of reformatting dictionaries; use `log_section` for any runtime-only context.
5. **Helper ecosystem** — expand `logging_utils` with formatters that keep info concise (tables, diffs); encourage DEBUG for large payloads.
6. **Documentation pass** — once behavior stabilizes, fold this plan into the official dev guides alongside examples of expected log output.

### Provenance Tagging Plan
Goal: replace the generic `(override)` / `(adjusted)` markers with explicit provenance metadata that explains *which precedence layer* produced the value and whether it was later mutated.

#### Dimensions to Track
1. **Precedence Source** – which tier in the resolution stack provided the value last:
	- `default` – implicit value (none provided by config/CLI/override). Note: Pydantic models often rely on upstream defaults (config files) rather than defining Python defaults directly; if a field remains unset, downstream consumers (model/dataloader classes) may fall back to their own defaults.
	- `config:<path>` – loaded from a config file; `<path>` can encode mode sections (e.g., `config:init.model`).
	- `cli` – provided via CLI/Snakemake arguments after alias resolution.
	- `override` – supplied through `override_kwargs` / programmatic injection (highest static priority).

2. **Mutation Type** – whether the value was changed after instantiation:
	- `runtime` – adjusted via `update_field` (dataset inference, validation correction).
	- `derived` – computed from other fields during validators (e.g., implicit defaults that are functions of other params).

We record both dimensions so a log entry can say `(config:init.model; runtime)` if a config value was later tweaked by inference.

#### Implementation Sketch
1. **Source tracking:**
	- While merging parameters in `BaseParams.get_params_from_cli_and_config`, maintain `_value_provenance: Dict[str, Provenance]` where `Provenance` captures `source` and optional `scope` (config section or alias).
	- When `update_field` mutates a value, append the mutation marker (`runtime`). Validators that compute fields can flag `derived`.
	- For composites, flatten keys (`model.n_classes`) and merge child provenance maps when constructing the parent.

2. **Log rendering:**
	- Extend `SummaryItem`/`build_section` to accept a `provenance_formatter` callback that turns the provenance record into a compact suffix string.
	- Format as `(source[; scope][; mutation])`, omitting segments that are redundant. Examples: `(default)`, `(config:init.model)`, `(cli; runtime)`.

3. **Legend (for docs/log appendix):**
	- `default` – Field used its class default.
	- `config:<section>` – Value came from configuration files; `<section>` points to the nested key (e.g., `train.trainer`).
	- `cli` – Provided on the command line or via Snakemake wildcards.
	- `override` – Injected programmatically (e.g., from `override_kwargs`).
	- `runtime` – Modified after instantiation (dataset inference, validators calling `update_field`).
	- `derived` – Computed during validation rather than specified directly.

	Multiple tags combine with `;`. If a config section ultimately came from CLI (e.g., Snakemake templating), we show the highest-precedence layer actually applied (`cli`).

4. **Presentation tweaks:**
	- Skip entries when `include_defaults=False` **and** provenance is `default` with no mutations. Clarify in docs that `default` here means “no explicit value supplied” (the instantiated class may still apply its own internal default at construction time).
	- Always show mutation tags (`runtime`, `derived`) even if the final value equals the original to make adjustments visible.
	- Surface the legend in developer docs and optionally provide a `--show-provenance-legend` flag or DEBUG-level print during CLI runs for onboarding.

## Progress Log
- **[Done]** Created `dynvision/utils/logging_utils.py` with `configure_logging`, `log_section`, `format_value`, `SummaryItem`, and `build_section` helpers.
- **[Done]** Updated `BaseParams` to track dynamic overrides, expose `summary_sections`, and provide `log_summary`.
- **[Done]** Extended `CompositeParams.log_overview` to cascade component summaries.
- **[Done]** Added structured summaries to `DataParams`, `TestingParams`, and `ModelParams` via `summary_sections`.
- **[Done]** Refactored `runtime/init_model.py` to use the new logging helpers and delegate configuration summaries to `InitParams`.
- **[Done]** Introduced provenance tracking across `BaseParams`/`CompositeParams`, so summaries label values with their source (`config`, `cli`, `override`, `runtime`) instead of the generic `(override)/(adjusted)` markers.
- **[Done]** Applied the same integration to `runtime/train_model.py` and `runtime/test_model.py`, delegating their run summaries to `TrainingParams.log_training_overview` and `TestingParams.log_testing_overview`.
- **[Done]** Wired dataset creation logging through `DataParams.log_dataset_creation` so init/train/test emit `(default)` markers sourced from `get_dataset` signatures.
- **[Done]** Swept remaining params classes (trainer/optimizer controls) so summaries now cover gradient clipping, validation limits, strategy kwargs, and early stopping provenance.
- **[Done]** Extracted dataset/dataloader wiring into `dynvision/data/datamodule.py`, giving training (`DataModule`), initialization (`SimpleDataModule`), and testing (`TestingDataModule`) a shared `DataInterface` and the same logging diff helpers.
- **[Done]** Migrated `runtime/test_model.py` onto `TestingDataModule`, so sampler instantiation, preview logging, and dataloader provenance flow through `DataParams` instead of bespoke helpers.
- **[Done]** Demoted preview-phase dataset/dataloader logs in `DataInterface` to DEBUG while keeping diff tracking for active contexts, reducing INFO noise during init/train/test.
- **[Done]** Auto-null `prefetch_factor` when `num_workers=0`, including values injected through `dataloader_kwargs`, so Lightning no longer emits multiprocessing warnings for intentional single-thread loaders.
- **[Done]** Updated `_build_callable_entries` so dataset/dataloader logs only emit default markers the first time—subsequent stages now show deltas/new/removed entries instead of restating unchanged defaults.
- **[Done]** Gated preview batch summaries in `runtime/train_model.py` and `runtime/test_model.py` so they stay at DEBUG by default but automatically elevate to INFO when `verbose` is set, giving opt-in noise when diagnosing data issues.
- **[Done]** Replaced the ad-hoc resume banners in `runtime/train_model.py` with a structured `checkpoint_selection` section that documents whether the run is resuming from a Lightning checkpoint or initializing from the saved state dict (including the effective weight source).
- **[Done]** Upgraded `TrainingParams._validate_required_paths` to emit a `training_dataset_paths` DEBUG section instead of a raw `print`, recording which dataset link/FFCV trio was validated without polluting stdout.
- **[Planned]** Document the logging pattern in `docs/development/guides` once the refactor settles.

### Latest Observations (Train Run 2025-11-13)
- Provenance labels now render as `(config:data)` / `(cli:model)` etc.; INFO stream is readable but still lengthy. Evaluate collapsing repeated path/value rows that appear in both the params overview and subsequent `creating_*` sections.
- Preview logs default to DEBUG but flip to INFO automatically when `verbose` is requested, so operators can surface the preview diffs without changing code; consider whether a dedicated CLI switch is still needed.
- `creating_standarddataloader` now only prints actual diffs, but we still duplicate a few context headers; consider collapsing nested sections or adding a compact summary banner atop each run.
- Lightning still emits the `prefetch_factor` warning when `num_workers=0`; decide whether to auto-null that field or downgrade the warning to DEBUG when the configuration intentionally forces single-thread loading.
- `torch.load(... weights_only=False)` future warning surfaces during state dict loads; track follow-up to adopt the safer default once PyTorch updates land.
- Coordination dtype warning (`list index out of range`) remains; log is informative but might warrant a one-time INFO followed by DEBUG repeats if it persists each run.
- The new `checkpoint_selection` section keeps resume/fresh metadata in a single place, but we still need before/after samples to ensure downstream tools parse it correctly.
- `training_dataset_paths` now captures the validated dataset link/train/val trio at DEBUG level; validate that this provides enough breadcrumbing when path resolution fails outside of verbose mode.
- `ModelParams.log_model_creation` and `ModelParams.log_configuration` currently emit overlapping summaries; consolidate them so a single helper owns the structured model overview (ideally reusing `log_section`).


## Open Questions / Follow-Ups
- Should `configure_logging` support structured JSON outputs for cluster monitoring, or stick with plain text for now?
- How to best expose debug-level deep dives (full config dumps) without cluttering INFO logs—dedicated CLI flag?
- Need to confirm Snakemake entrypoints call `setup_logging` exactly once to avoid handler duplication.

## Next Actions
1. Extend provenance tagging through derived/preprocessor-driven adjustments (mark as `derived`) and ensure runtime scripts surface the legend when verbose output is requested.
2. Consolidate `ModelParams.log_model_creation` / `log_configuration` into a single helper that emits sectioned tables via `log_section`, then update runtime entrypoints to call only that surface. Capture before/after log snippets to validate INFO-volume improvements.
3. Audit remaining ad-hoc logging statements across runtime/scripts; for each, decide whether to remove, demote to DEBUG, or migrate into the owning Params helper so the structure stays consistent with the formalized logging style.
4. Restructure `creating_dataset`/`creating_standarddataloader` outputs to highlight deltas instead of repeating full config dumps; keep preview dataset logs at DEBUG unless `verbose` is set.
5. ✅ (2025-11-17) Guard dataloader kwargs so `prefetch_factor` disappears whenever `num_workers=0`, with an INFO log explaining the adjustment.
6. Provide guidance for adopting `torch.load(..., weights_only=True)` once dependent codepaths are ready; track in follow-up issue.
7. Verify CLI/Snakemake workflows respect log levels after noise demotion; capture before/after log excerpts for documentation and the forthcoming developer guide update.
8. Update `docs/development/guides/claude-guide.md` (or a new `data-processing.md` companion) with the new DataModule responsibilities so DynVision-specific logging expectations stay in the toolbox docs while the generic AI style guide remains framework-agnostic.
9. Explore beautifying high-level summaries (tables, headers, minimal visual markers) to improve scanability while staying within plain-text logging constraints.
	- Prototype a `log_section_table` helper (inspired by the weight-check tables inside `dynvision/base/monitoring.py`) that auto-aligns columns and keeps headers to 1–2 lines while still emitting plain text.
	- Apply the helper to the heaviest INFO blocks first (`training_run`, `creating_trainer`, `checkpoint_selection`) so operators can skim the columns, keeping richer provenance data in the suffix markers.
	- Gate the textual table rendering behind INFO, but emit a one-line summary and keep the detailed tables at DEBUG when `verbose` is off; document the convention alongside examples so Snakemake logs remain predictable.

10. Add regression notes/tests (even simple scripts) to ensure `SimpleDataModule`/`TestingDataModule` continue logging provenance correctly when new params are introduced and that the preview→active diff remains obvious despite the DEBUG demotion.
11. Review `dynvision/base/monitoring.py` and port the existing banners/memory diagnostics onto `log_section` helpers (e.g., `training_start`, `system_resources`) so model-internal instrumentation matches the Params-driven style.

### Monitoring Alignment Gameplan
- **Snapshot current behavior** — grab a short training log to capture the existing `_log_system_info`, `_log_memory_usage`, and batch preview outputs so we can verify the new structure preserves content.
- **Refactor batching hooks** — have `_log_system_info`, `_log_memory_usage`, `_validate_batch_data` (first batch), and `_log_training_summary` emit `log_section` blocks with the same keys the params summaries use (`model_name`, `n_classes`, memory stats, batch shapes).
- **Keep warnings loud** — retain emoji/⚠️ warnings for NaNs, invalid labels, and high loss, but make the “happy path” info flow through `log_section` so INFO streams stay tabular.
- **Rollout plan** — update MonitoringMixin first, then rerun `runtime/train_model.py` on a dry run to capture before/after logs and link them back here before touching downstream docs.
