# Temporal units: one principle for ms and steps

Status: implemented. User-facing documentation:
[Temporal units](../../user-guide/temporal-units.md). Changes: `CHANGELOG.md`.

## Problem

DynVision mixes two time units without a rule for which quantity uses which.
Delays and `tau` are specified in milliseconds and converted with `dt`; the
simulation window, the idle period and every stimulus protocol are specified in
steps and never see `dt`. Changing `dt` therefore changes the physical
experiment, not only its resolution: with `tsteps=20, stim=5`, a model at
`dt=0.5` trains on a 10 ms window and sees a 2.5 ms stimulus, while the same
configuration at `dt=2` trains on 40 ms and sees 10 ms. Downstream, results
carry no time base, so plots convert every model with one configured `dt`.

## Decisions

1. **Option B.** Physical quantities are specified in ms at every interface
   (config files, CLI, file names). The engine (model internals, loaders, loss
   masks) runs on integer steps. Conversion happens at one strict boundary.
2. **Strict conversion.** `utils/timing.to_timesteps` raises on an off-grid
   value instead of rounding. This reverses the rounding fallback of
   `942795a` (`test_rounds_rather_than_truncating`). A rounded off-grid value is
   the silent error this design exists to prevent.
3. **Pydantic integration.** Conversion is part of parameter validation, not a
   parallel mechanism:
   - Quantities a component can convert alone (it holds `dt`) are converted in
     that component's existing `mode="after"` validator. `ModelParams` handles
     delays, `t_simulation` and `t_idle`.
   - Quantities that need `dt` from another component (loader durations on
     `DataParams`) are converted by `coordinate_component_timing(config)` in
     `utils/timing.py`. It is called from an after-validator on
     `TrainingParams`, `TestingParams` and `InitParams`, mirroring
     `coordinate_component_dtypes()`.
   - Loader durations become declared `DataParams` fields, so they are
     validated and summarized instead of travelling as `__pydantic_extra__`.
   - Derived step values are written with `update_field(...,
     mutation_tag="derived")`, so provenance tells a user-set step value from a
     derived one.
4. **Naming: a `t` prefix means milliseconds.** This extends the existing
   convention (`tff`, `trc`, `tfb`, `tsk`). New quantities get new names, and
   no existing name changes meaning:

   | Quantity | Config field (ms) | File-name alias | Legacy step field / alias |
   |---|---|---|---|
   | simulation window | `t_simulation` | `tsim` | `n_timesteps` / `tsteps` |
   | data sequence | `t_sequence` | `tseq` | `data_timesteps` / `dsteps` |
   | idle period | `t_idle` | `tidle` | `idle_timesteps` / `idle` |
   | stimulus | `t_stimulus` | `tstim` | `stimulus_duration` / `stim` |
   | pre-stimulus blank | `t_intro` | `tintro` | `intro_duration` / `intro` |
   | inter-stimulus interval | `t_interval` | `tinterval` (alias `tisi`) | `interval_duration` / `interval` |

   `loss_reaction_time` / `lossrt` is already in ms and keeps its name.
   Experiment configs keep each key at the position of the legacy key it
   replaces, so a translated legacy identifier equals the new one character
   for character (decision 11 relies on this).
5. **Legacy keys are translated on read, never rewritten.** A legacy step key
   stays valid and keeps its meaning (steps). The params layer converts it to
   the canonical ms field with `ms = steps · dt`, which is always exact, and
   logs a deprecation warning. Supplying a legacy key and its ms counterpart
   together raises. Existing paths, checkpoints and result files are never
   renamed. Processing derives ms columns from legacy step columns using each
   result's own `dt`, so old and new results aggregate on one ms column.
6. **Anchored path readers.** `extract_param_from_string` and its siblings
   match a key only at a token boundary. Today `stim` matches inside
   `tstim=10`, and `pattern` matches inside `shufflepattern=…`.
7. **Time convention.** The engine already integrates step `k` (0-based, after
   idle) over `[k·dt, (k+1)·dt)` and records the state at its end. So
   `time_ms = (k + 1)·dt`, with `t = 0` at the end of the idle period. This is
   a correction of processing and plotting, which place the state at `k·dt`; the
   engine does not change.
8. **Buffer offsets are not delays.** The `+1` in `layer_connections.py:113`
   and in the `dyrcnn.py` skip indices is an intended buffer-index offset: the
   source layer has already stored the current step. It stays, with the step
   count from `to_timesteps`. Recurrence reads its own buffer before the current
   step is stored, so its minimum delay is one step (`get_hidden_state` maps 0
   to 1). Validation therefore requires `t_recurrence ≥ dt` instead of the
   current `≥ 1` (ms compared with a step count).
9. **No behaviour change for on-grid configurations.** Every existing
   configuration with on-grid delays (all trained models) produces identical
   delay indices and identical forward outputs. A golden test pins this.
10. **Mechanical guard.** A test scans `dynvision/` (by AST) for `*`, `/`, `//`
    and `%` with `dt` as an operand outside `utils/timing.py`. Two uses are
    exempt by design, with reasons: the integration step in
    `dynamics_solver.py`, and the `dt/tau` diagnostics on `ModelParams`. All
    other current sites start on an allowlist; each item removes its entries,
    and the last item asserts the allowlist is empty.
11. **Existing results are reachable under new names by symlinks.** A helper
    (`scripts/link_legacy_temporal_paths.py`) mirrors every legacy-key path
    under `models/` and `reports/` to its ms-key name. See *Legacy paths and
    Snakemake* below.

## Legacy paths and Snakemake

Verified against Snakemake 9.1.11 (source and a toy workflow):

- **Metadata is keyed by the literal path** (`persistence._record_path`
  base64-encodes the path string; no `realpath`). A new name therefore has no
  recorded params, code, input or software stack, and those rerun triggers
  cannot fire. Only mtime decides.
- **Outputs use the link's own mtime; inputs use `max(link, target)`**
  (`Mtime.local(follow_symlinks=…)` in
  `snakemake_interface_storage_plugins/io.py`). Links made with a plain
  `ln -s` get the creation time, so an input link created after its output
  link reads as "updated input" and triggers a rerun. Observed: rule `b` reran
  on the new name while the old name was up to date.
- **Fix: give every link its target's mtime** with
  `os.utime(link, ns=…, follow_symlinks=False)`. The new-name graph then has
  exactly the old graph's mtime relationships, in any creation order.
  Observed: "Nothing to be done" on the new name.
- **Checksums only cover old paths.** For inputs under 1 MB, Snakemake skips
  a rerun when the recorded content checksum still matches, even if the input
  is newer than the output. New names have no recorded checksums. An old tree
  that is up to date only through checksums therefore shows reruns under the
  new names. The helper reports such links.
- **Links must be file-level, inside real new directories.** Before a job
  runs, Snakemake removes its outputs with `os.remove`. On a file symlink this
  removes the link and leaves the old file intact (observed: same inode, same
  content). Through a directory symlink, the same call would delete the real
  old file.

The helper does the following:

- **What it links.** Every file under a legacy-key model directory
  (`models/`), and every file under a test directory (`reports/…/{test_id}/`).
  Stage-1 `test_data.csv` cannot be regenerated without re-running the GPU
  test, because `process_single_test` deletes `test_responses.pt`.
- **Stage-2 aggregates.** `reports/…/test_data.csv` is linked only with
  `--include-aggregates`, for mirrors that hold no per-test files. Otherwise
  it is regenerated in the new format.
- **Hashed test identifiers.** Compressed mode names test directories
  `md5("{loader}_{args}")[:8].lstrip(":hash=")`. The `lstrip` removes a
  character set, so hashes that start with `a` lose those characters. This is
  kept, because every existing directory name depends on it. The helper
  recovers the legacy spec from a frozen copy of the pre-migration experiment
  config, translates it, and re-hashes it with the workflow's own function.
- **`dt` for the translation** comes from the identifier in the same path; a
  `dt` in the test spec overrides the model's.
- **Unmappable paths.** A legacy step key next to `dt=*` (a dt sweep at a
  fixed step count) has no single ms name. Such paths are reported and skipped,
  as are unknown hashes and paths with an incomplete marker in
  `.snakemake/incomplete/`.
- **Default mode.** The helper runs as a dry run. `--apply` creates the links,
  writes a manifest, and mirrors mtimes; `--undo` removes exactly the links in
  the manifest.

## Inventory before this change (line numbers at `942795a`)

Legend: **ms** = defined in milliseconds, **steps** = defined in steps,
**ratio** = enters only as `dt/tau`.

### 1. Model definition and preparation

| Quantity | Unit | Conversion today | Site |
|---|---|---|---|
| `tau` | ratio | `dt / tau` in Euler / RK4 | `model_components/dynamics_solver.py:85-121` |
| `t_feedforward` | ms | `int(t/dt)` truncates | `base/temporal.py:137`, `model_components/recurrence.py:1091` |
| `t_recurrence` | ms | `int(t/dt)` truncates | `model_components/recurrence.py:827` |
| delay buffer length | ms | `int(history_length/dt)` | `model_components/recurrence.py:826` |
| `t_skip`, `t_feedback` | ms | `int(round(t/dt))` (+1 offset for skip), copied into three classes | `models/dyrcnn.py:294-297, 478-481, 693-696` |
| skip connections | ms | `int(t/dt)` (+1 offset) | `model_components/layer_connections.py:113` |
| `idle_timesteps` | steps | none | `base/temporal.py`, alias `idle` |
| `t_recurrence >= 1` | mixed | ms compared with a step count | `params/model_params.py:622` |

Already strict on `dev`: `ModelParams.delay_*` properties, the reaction mask
(`base/temporal.py`), and the multiple check in both `ModelParams` and
`TemporalBase`.

Observed: `int(0.6/0.2) == 2`, so a 0.6 ms delay at `dt=0.2` runs as 0.4 ms.

### 2. Training

| Quantity | Unit | Conversion today | Site |
|---|---|---|---|
| simulation window `n_timesteps` (`tsteps`) | steps | none; part of the model identifier | `params/model_params.py` |
| `data_presentation_pattern` | relative | stretched over `n_timesteps` with floor indexing; uneven chunks accepted silently (`1011` over 10 steps gives chunks of 3, 2, 5) | `base/temporal.py` `_expand_pattern` |
| `truncated_bptt_timesteps` | steps | computational, not physical; stays in steps | `base/temporal.py` |
| `data_timesteps` vs `n_timesteps` | steps | reconciled by count | `params/training_params.py:408-423` |

### 3. Testing

| Quantity | Unit | Conversion today | Site |
|---|---|---|---|
| `dsteps`, `stim`, `intro`, `interval` | steps | none; loaders never see `dt`; `stim`/`intro`/`interval` are undeclared extras | `data/dataloader.py` Stimulus* loaders, `params/data_params.py:466` |
| `idle` in experiment `data_args` | steps | routed to `model.idle_timesteps` | `configs/config_experiments.yaml` |
| path readers | — | `re.search(f"{key}=…")` without a boundary | `utils/string_utils.py:63-108`, `processing/process_wandb_data.py:27,96` |

### 4. Test-result processing

`test_data.csv` records only `times_index`. A `dt` column appears only when
`dt` is the swept category (parsed from the path). The resolved config
(`test_outputs.csv.config.yaml`) holds `model.dt` and the idle steps, but
processing does not read them.

### 5. Visualization

Every site below takes one `--dt` from the workflow config (default `2`):

- `time_ms = times_index · dt + idle · dt`: `visualization/plot_responses.py`, `plot_dynamics.py`, `plot_performance.py`, `plot_training.py`, `plot_unrolling.py`
- parameter labels and metrics: `utils/visualization_utils.py:58`, `plot_responses.py:140`, `plot_dynamics.py` (`peak_time`, tick labels), `plot_performance.py:1279`
- workflow: `workflow/snake_visualizations.smk` (5 rules), and downstream project workflows that pass `--dt` the same way
- per-trace max normalization in ridge plots (`plot_responses.py` `_normalize_trace`) makes the first-pulse height depend on later stimuli

## What this does not achieve

Trained models will not be dt-invariant at `dt=2, tau=5` even with every
quantity in ms:

- **Euler error.** `dt/tau = 0.4`. A toy leaky unit with delayed recurrence
  deviates by 2–5% in peak ratio between `dt=2` and the continuum limit.
- **Separate training runs.** Each `dt` is a separate optimization problem.

dt-invariance holds in the continuum limit (`dt ≪ tau`). That can be tested on
fixed weights, not across trained models.

## Implementation

The work landed in these steps, each with its own tests:

| Step | Content |
|---|---|
| T0 | `utils/timing.py`: strict ms/step conversion, the temporal-quantity registry, and the AST guard against `dt` arithmetic (`tests/utils/test_dt_arithmetic_guard.py`, allowlist empty) |
| M1 | Every model delay goes through `to_timesteps`; sub-step delays are rejected; a golden forward test (`tests/models/golden/dt_golden.pt`) pins on-grid behaviour |
| M2 | `t_simulation` and `t_idle` on `ModelParams`; legacy `tsteps` / `idle` translated on read |
| R1 | Presentation-pattern chunks must divide the simulation window exactly |
| S1 | Path readers match a key only as a whole key |
| S2 | Declared ms loader fields on `DataParams`, converted by `coordinate_component_timing` |
| S3 | Experiment and workflow configs in ms keys; the pre-migration copy is frozen in `configs/legacy/config_experiments_steps.yaml` |
| L1 | `scripts/link_legacy_temporal_paths.py` links legacy-named results to their ms names |
| P1 | `processing/time_base.py`: results carry `dt`, `idle_steps`, `time_ms` and ms columns for legacy step parameters |
| V1 | Plots read time from the data; ridge plots use one scale per model |
| C1 | `processing/sample_selection.py` warns when compared test runs rest on different samples |
| C2 | `scripts/prune_test_results.py` removes test results whose stored samples differ from their experiment |
| C3 | Storage caps can no longer truncate a test run silently |

## Open follow-ups

- In `dynamic` mode the noise loader draws a new frame every step, so the
  noise frame rate is `1/dt`. A frame interval in ms would make it
  dt-invariant.
- The `tsim` category values 28 and 52 ms (14 and 26 steps at `dt=2`) do not
  divide into 4-chunk patterns. Since R1, a joint sweep with pattern `1011`
  raises an error instead of using uneven chunks.
- `dt`-invariance of trained models can be tested only on fixed weights in the
  continuum limit (see above).
