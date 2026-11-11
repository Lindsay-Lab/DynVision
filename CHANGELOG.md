# Changelog

All notable changes to DynVision will be documented in this file.

## [Unreleased]

### Changed
- **Temporal quantities are specified in milliseconds (breaking for new
  configs, legacy keys still accepted)**: the simulation window, idle period,
  data sequence and stimulus protocol were counted in steps and never saw `dt`,
  so changing `dt` changed the experiment, not its resolution (`tsteps=20,
  stim=5` was a 10 ms window with a 2.5 ms stimulus at `dt=0.5` and 40 ms with
  10 ms at `dt=2`). They are now `t_simulation`/`tsim`, `t_idle`/`tidle`,
  `t_sequence`/`tseq`, `t_stimulus`/`tstim`, `t_intro`/`tintro` and
  `t_interval`/`tinterval` (alias `tisi`), declared on `ModelParams` and
  `DataParams` and converted to steps during validation
  (`utils/timing.coordinate_component_timing`). Legacy step keys keep their
  meaning, are translated with `ms = steps * dt`, and log a deprecation
  warning. Experiment and workflow configs use the ms keys; the dt sweep
  category is `[0.5, 1, 2]`; the previous experiment config is frozen in
  `configs/legacy/config_experiments_steps.yaml`. Guide:
  `docs/user-guide/temporal-units.md`.
- **Strict ms -> step conversion**: `utils/timing.to_timesteps` raises on an
  off-grid value instead of rounding (reverses the rounding fallback of
  `942795a`). All model delay conversions go through it; on-grid models keep
  identical delay indices and reaction masks, and their forward output matches
  within `rtol=1e-6, atol=1e-7` (`tests/models/test_dt_golden.py`) -- below the
  4e-6 that a one-step delay shift moves every recorded case. A difference that
  is not bit-identical is reported as a warning, since a build that evaluates
  the arithmetic in another order rounds differently.
  `t_recurrence` must be at least `dt` (was compared with the step count 1).
- **Presentation patterns stretch over whole chunks**: a pattern whose length
  does not divide the window raises instead of producing uneven chunks
  (`1011` over 10 steps gave chunks of 3, 2, 5).
- **Results carry their time base**: processed test data record `dt`,
  `idle_steps` and `time_ms = (times_index + 1) * dt` (`t = 0` at the end of
  the idle period; the engine records each step's state at its end), plus ms
  columns for swept legacy step parameters. `processing/time_base.ensure_time_base`
  adds them to results written before.
- **Plots read time from the data**: the `--dt` / `--idle-timesteps` arguments
  and workflow fallbacks (default `dt=2`) are gone. Ridge plots with
  `normalize='max'` scale once per model across parameter values (per-trace
  normalization made the first-pulse height depend on later stimuli); the
  stimulus indicator is drawn per distinct protocol; dynamics Panel A uses a
  parameter value every model was run with.
- **Loader defaults**: `num_workers` is 8 (was 4) for training and testing;
  the test loader prefetches one batch per worker (`prefetch_factor: 1`, was
  `null`).
- **Delay validation now rejects off-grid durations (breaking)**: `t_feedforward`,
  `t_recurrence`, `t_feedback`, `t_skip` and `loss_reaction_time` are measured in
  milliseconds but integrated on a fixed `dt` grid, so each must be a whole
  number of `dt` steps. A configuration that violated this was previously
  accepted and silently converted, with the conversion sites disagreeing:
  `int(value / dt)` truncated for the feedforward and recurrent delays while
  `math.ceil` rounded up for the reaction window. `dt=3` with `t_recurrence=4`
  and `loss_reaction_time=4` therefore ran a 3 ms recurrence and a 6 ms
  reaction window. Validation now raises `DynVisionValidationError` (params
  layer) or `ValueError` (model construction), naming the parameter and both
  neighbouring valid values.
- **`dt` must be positive**: previously unchecked, and a non-positive `dt` made
  the delay comparisons undefined.
- **Composite alias maps are derived from their components**: `TrainingParams`
  and `InitParams` listed only a subset of `ModelParams`' aliases, so the model
  timing aliases (`trc`, `tff`, `tfb`, `tsk`, `lossrt`) and others were absent
  from them while present in `TestingParams`. A bare key such as `--trc 4` was
  therefore attributed to the model component unscoped, stored as an inert extra
  attribute (`extra="allow"`), and only reached the model because
  `get_model_kwargs` merges `__pydantic_extra__` and the model classes re-open
  the alias in their own `@alias_kwargs` decorator. That route bypassed
  parameter validation entirely. `CompositeParams.get_component_aliases` now
  scopes each component's own aliases, so adding an alias to a component routes
  it through every composite that contains it.

### Added
- **`scripts/link_legacy_temporal_paths.py`**: exposes models and test
  results stored under legacy step-key paths under their ms-key names, as
  file-level symlinks carrying their target's mtime, so Snakemake plans no
  reruns for existing results (dry run by default, `--apply`, `--undo`).
- **Anchored path readers**: `extract_param_from_string` and its siblings match
  a key only at a token boundary (`stim` no longer matches inside `tstim=10`).
- **Sample-selection warning** (`processing/sample_selection.py`): aggregating
  test runs or loading them for a plot logs a warning when the runs evaluated
  different samples or recorded layer responses for different samples.
- **`scripts/prune_test_results.py`**: reports the test results whose stored
  samples differ from the rest of their experiment (per layer and in total)
  and, with `--apply`, deletes them for a rerun, so all runs of an experiment
  can rest on the same, largest sample set. `scripts/link_legacy_temporal_paths.py`
  now imports `dynvision` from its own checkout, so it runs without
  `PYTHONPATH` or an install.
- **`dynvision/cluster/interactive.sh`**: queues an interactive job
  (`srun --pty`) or a detached command (`--sbatch`) inside the container,
  overlays and conda env that `executor_wrapper.sh` gives every rule, and
  prints the environment it landed in. Placeholders as in the other cluster
  scripts.
- **Ridge response plots**: the `layers` dimension can take any of the three roles
  (`column`, `subplot`, `hue`), not only the subplot.
- **Guard test** `tests/utils/test_dt_arithmetic_guard.py`: fails on `dt`
  arithmetic outside `utils/timing.py` (two exempt sites: the integration step
  and the `dt/tau` diagnostics).
- **`ModelParams.recurrence_target` and `DataParams.data_loader` are declared
  fields.** Both were previously reachable only as aliases
  (`rctarget`, `data_loader`) whose target was not a declared field, so the value
  was kept by `extra="allow"` and reached its consumer only via the
  `__pydantic_extra__` merge in `ModelParams.get_model_kwargs` or by direct
  attribute access in `DataModule.setup_dataloader`. Neither was validated, and
  neither appeared in the summary or in provenance. `recurrence_target` is
  constrained to `input` / `middle` / `output` (case-insensitively); the
  `rctarget` alias is unchanged.
- **`tests/params/test_component_alias_routing.py`**,
  **`tests/params/test_timing_multiples.py`** and
  **`tests/params/test_recurrence_target_and_data_loader.py`**:
  completeness and precedence of component alias routing across all three
  composites, the dt-multiple constraint on both validation layers, and the two
  newly declared fields including their forwarding into model kwargs.
- **`tests/base/test_recurrence_target_middle.py`**: the `'middle'` target guard.

### Fixed
- **Snakefiles import only the workflow helpers**: `key_value_pattern` moved
  from `utils/string_utils.py` to `workflow/snake_utils.py` (re-exported), so
  `snake_utils.smk` no longer imports `dynvision.utils` and its dependencies.
- **The legacy-path link helper aborted on a misplaced model directory**:
  an `rsync` that copies a directory into a directory leaves
  `models/<class>/<idA>/<idB>/...`, and such a copy can hold a relative symlink
  that no longer resolves at its new depth. The helper planned a link to it and
  `os.stat` raised `FileNotFoundError` from `mirror_mtime`, stopping `--apply`
  halfway through its manifest. Nested model/test directories are now reported
  as `misplaced` and unresolvable sources as `broken`, both skipped; a link
  that would dangle is removed instead of keeping a run alive.
- **A capped response buffer stopped the test run early, silently**: an
  unlimited records buffer (`-1`) is never full, but `len(records) >= -1` read
  as full, so `early_test_stop` fired as soon as the response cap was reached
  and the run's metrics then described a prefix of the test set. Early stopping
  now requires a finite cap on both buffers.
- **Storage defaults store every sample**: `store_test_responses` and
  `store_test_records` default to `-1` (unlimited). A positive cap is now named
  in a warning, because `store_test_records` truncates the *evaluated* set --
  its records carry the predictions -- and `store_test_responses` truncates the
  stored activations to a prefix of the samples. `store_test_responses: -1`
  with `response_resolution: unit` is rejected as impossible to fit.
- **`RecurrentConnectedConv2d` crashed on `recurrence_target='middle'` without
  `mid_channels`**: `_calculate_recurrence_output_dims` subscripts
  `kernel_size` / `stride` / `padding`, which are only normalised to `(y, x)`
  tuples when `mid_channels` is set; otherwise they stay scalars and the
  subscript raised `TypeError: 'int' object is not subscriptable`. `DyRCNNx2`
  and `DyRCNNx4` build every recurrent layer without `mid_channels`, so
  `rctarget=middle` -- a value in `config_experiments.yaml`'s `rctarget`
  category -- crashed on them. The invalid combination is now rejected at layer
  construction with a message naming `mid_channels`.
- **`get_data_loader_class` on an unset name**: it raised `TypeError: argument of
  type 'NoneType' is not iterable` from a substring test. Now that `data_loader`
  is a field whose unset value is `None`, it raises a `ValueError` naming the
  parameter and listing the registry.
- **`ModelParams.get_timing_summary()` raised `AttributeError`**: it returned
  `effective_tau_steps`, `total_processing_time` and `total_processing_steps`,
  none of which existed, and `is_stable` raised `TypeError` whenever `dt` or
  `tau` was unset. `effective_tau_steps` (`tau / dt`) is now a real property, the
  two ambiguous `total_processing_*` keys are gone, and every field is
  `None`-safe. This is documented public API in the parameter-handling guide.
- **Model-level re-check of effective durations**: `TemporalBase` validates the
  values it actually integrates, which catches a duration left unset in the
  configuration and supplied by a model class default. `t_feedback` and
  `t_skip` are checked only when feedback or skip is enabled, since they are
  inert otherwise.

## [0.1.0] - 2026-06-23

### Added
- **ActivityLoss**: Signed EI-balance mode for excitation/inhibition regularization (renamed from EnergyLoss)
- **Layer-resolution response storage**: Reduces test output size by storing per-layer rather than full tensors
- **Jang et al. (2021) data loader**: Human/CNN noise benchmark dataset for psychometric comparisons
- **Visualization enhancements**: Plot reference models, weight caption metrics, parameter counting, benchmark fetching
- **Seed series 9000+**: Extended seed configuration for robustness experiments
- **Cluster config updates**: New storage paths and SLURM account support
- **Comprehensive documentation**: Expanded user guide, API reference, and developer docs
- **Test suite**: 170 tests covering base, params, data, workflow, cluster, and losses modules

### Changed
- **scikit-learn**: Relaxed version constraint from `~=1.1.0` to `>=1.2.0,<2`
- **Makefile**: Fixed project naming (`rva` → `dvn`, `rhythmic_visual_attention` → `dynvision`)
- **project_paths.py**: Renamed `project_name` to `DynVision_Working` for clarity
- **Documentation**: Fixed 6+ broken references, missing images, and stale class names across reference docs
- **README**: Updated badges, installation instructions, citation with preprint DOI
- **Development status**: Updated classifier from "Planning" to "Beta"

### Removed
- 7 deprecated/backup files (`*_old.py`, `*_backup.py`) — no code references remained

### Fixed
- **ActivityLoss `__del__` guard**: Prevent `AttributeError` when destructor fires before init completes

### Known Issues
- Python 3.12 not yet supported (blocked on FFCV compatibility testing)
- 7 pre-existing test failures (3 ActivityLoss API mismatch, 4 params `data_timesteps` validation)
- Makefile contains user-specific paths (not tracked in git)

### Upcoming (Post-0.1)
- Manuscript-specific code separation to manuscript repository
- Python 3.12 support after FFCV validation
- Logging standardization across params module
- Pre-commit hooks and CI/CD pipeline
- Quick-start Jupyter notebook
- Zenodo DOI archival (see `.zenodo.json`; DOI assigned on first release)

---

## [0.0.1] - Initial development

Pre-release development phase. Core architecture established:
- RCNN models with biologically-inspired dynamics (DyRCNNx2/4/8, AlexNet, ResNet, CorNetRT, CordsNet)
- Continuous-time ODE solvers (Euler, RK4)
- 6 recurrence types (self, full, depthpointwise, pointdepthwise, local, localdepthwise)
- Snakemake workflow system for reproducible experiments
- Pydantic-based parameter validation
- FFCV data loading
- PyTorch Lightning training integration
