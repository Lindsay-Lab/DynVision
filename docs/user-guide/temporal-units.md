# Temporal Units: Milliseconds and Steps

DynVision integrates its models on a fixed grid of `dt` milliseconds. Every
quantity with a biological meaning is specified in **milliseconds**; the engine
runs on **integer steps**; the conversion happens once, exactly, during
parameter validation. This guide shows how to specify time, what happens to
existing configurations and results, and how results and plots carry their
time base.

## The rule

- A **`t` prefix means milliseconds**: `tsim`, `tstim`, `tff`, `trc`, ...
- Everything else that counts time counts **steps**, and is either a legacy key
  (below) or computational (`truncated_bptt_timesteps`).
- A millisecond value must be a **whole number of `dt` steps**. An off-grid
  value raises instead of being rounded, because a rounded window or delay runs
  a different experiment than the one configured:

  ```text
  t_simulation=40 ms is not an integer multiple of dt=3 ms (40/3 = 13.3333 timesteps).
  ... Use 39 ms (13 step(s)) or 42 ms (14 step(s)).
  ```

- `tau` is exempt: it enters the dynamics only as `dt / tau`.

Changing `dt` therefore changes only the resolution of an experiment, not its
duration: `tsim=40, tstim=10` is a 40 ms window with a 10 ms stimulus at every
`dt` that divides both.

## Quantities

| Quantity | Config field (ms) | Key (ms) | Engine field (steps) | Legacy key (steps) |
|---|---|---|---|---|
| simulation window | `t_simulation` | `tsim` | `n_timesteps` | `tsteps` |
| idle period | `t_idle` | `tidle` | `idle_timesteps` | `idle` |
| data sequence | `t_sequence` | `tseq` | `data_timesteps` | `dsteps` |
| stimulus | `t_stimulus` | `tstim` | `stimulus_duration` | `stim` |
| pre-stimulus blank | `t_intro` | `tintro` | `intro_duration` | `intro` |
| inter-stimulus interval | `t_interval` | `tinterval` (`tisi`) | `interval_duration` | `interval` |
| delays | `t_feedforward`, `t_recurrence`, `t_feedback`, `t_skip` | `tff`, `trc`, `tfb`, `tsk` | buffer indices | — |
| loss reaction window | `loss_reaction_time` | `lossrt` | reaction mask | — |

The model owns `dt` and converts its own fields (`t_simulation`, `t_idle`,
delays). The loader durations live on `DataParams` and are converted with the
model's `dt` when the composite parameters are validated. Derived step values
are tagged `derived` in the parameter provenance.

The recurrent delay must be at least one step (`t_recurrence >= dt`): the
recurrence reads its own buffer before the current step is stored.

## Specifying time

In a model identifier, a workflow config or on the command line, use the ms
keys:

```yaml
# config_workflow.yaml
model_args:
  tsim: 40   # 20 steps at dt=2, 80 steps at dt=0.5
  dt: 2
  trc: 6

# config_experiments.yaml
interval:
  parameter: tinterval
  data_loader: StimulusInterval
  data_args:
    tseq: 80
    tintro: 2
    tstim: 10
    tinterval: [2, 4, 6, 8, 12, 16, 20]
    tidle: 40
```

```bash
python dynvision/runtime/test_model.py ... --dt 2 --tseq 40 --tstim 10
```

Experiment configs keep every ms key at the position of the step key it
replaced, so identifiers built from them line up with translated legacy
identifiers character for character.

## Legacy step keys

A legacy key (`tsteps`, `dsteps`, `stim`, `intro`, `interval`, `idle`) stays
valid and keeps its meaning: it counts steps. It is translated on read with
`ms = steps * dt`, which is always exact, and logs a deprecation warning once
per process. Giving both forms is accepted when they agree at the model's `dt`
and rejected when they disagree.

A legacy step value names a different duration at every `dt`: `tsteps=20` was
a 10 ms window at `dt=0.5` and a 40 ms window at `dt=2`. The experiment configs
before the switch are frozen in `dynvision/configs/legacy/config_experiments_steps.yaml`.

## Existing models and results

Existing directories keep their legacy names; nothing is renamed. To make them
reachable under the ms names, run the link helper from the project directory
that holds `models/` and `reports/`:

```bash
python scripts/link_legacy_temporal_paths.py .            # dry run: what would be linked
python scripts/link_legacy_temporal_paths.py . --apply    # create the links
python scripts/link_legacy_temporal_paths.py . --undo     # remove exactly those links
```

It creates a file-level symlink at the new path for every file in a
legacy-named model or test directory, and gives each link its target's mtime,
so Snakemake sees the old graph's time relations under the new names. It skips
and reports what has no single ms name: a legacy step key next to `dt=*` (a dt
sweep at a fixed step count), unknown hashes, and incomplete outputs.

Then check with a dry run before running anything:

```bash
snakemake -n <targets under the new names>
```

Two things can still plan reruns, under old and new names alike:

- **Script inputs.** `train_model` and `init_model` list their scripts as
  ordinary inputs. If git rewrote a script after a model was trained, its mtime
  is newer and Snakemake plans retraining. `--allowed-rules test_model
  process_test_data` (see `test_model_variations`) tests existing models
  without training them.
- **Checksums.** For inputs under 1 MB, Snakemake can skip a rerun because the
  recorded checksum still matches; that record exists only for the old paths.
  The helper lists such links.

The new names carry no Snakemake metadata, so parameter and code changes do not
trigger reruns there; only mtimes decide.

## Results carry their time base

Processed test results (`test_data.csv`) record per row:

- `dt` and `idle_steps` of the run (from its resolved config),
- `time_ms = (times_index + 1) * dt`: the engine integrates step `k` over
  `[k*dt, (k+1)*dt)` and records the state at its end; `t = 0` is the end of
  the idle period,
- ms columns for swept legacy step parameters (`interval` -> `tinterval`, ...),
  converted with each row's own `dt`.

Results written before this change hold only `times_index`.
`dynvision.processing.time_base.ensure_time_base(df, path)` adds the time base,
reading `dt` from the table or from `dt=` in its path; it never assumes a
default `dt`. Old and new results therefore aggregate on one ms axis.

## Plots

- Every plot reads `time_ms` from the data; there is no `--dt` or
  `--idle-timesteps` argument. Load results with
  `dynvision.utils.visualization_utils.load_result_data(paths)`.
- Ridge plots with `normalize='max'` scale each model (hue) once across all
  parameter values, so the first pulse keeps its height in every ridge.
- The stimulus indicator is drawn per distinct stimulus protocol: models with
  the same stimulus in ms share one gray line, models with different protocols
  (e.g. legacy step protocols at different `dt`) get one line each in their
  color.
- Panel A of the dynamics figures shows the middle parameter value that every
  model was run with.

## Comparing runs: sample selections

A test run evaluates a set of samples and stores layer responses for a subset
(`store_test_responses` batches; the other rows hold `NaN`). Averages over runs
with different selections mix different images. When results are aggregated
(`aggregate_experiment_data`) or loaded for a plot (`load_result_data`), the
selections of all runs are compared, and one warning lists the distinct ones:

```text
Compared test runs rest on different samples (plotting): 2 selections among 63 runs. ...
  - 1280 samples, responses recorded for 384: 47 run(s), e.g. interval=1, dt=0.5, ...
  - 2560 samples, responses recorded for IT 512, V1 32, V2 512, V4 512: 16 run(s), e.g. ...
```

Rerun the deviating tests with the same data group and `store_test_responses`
to remove the mismatch.

## What this does not achieve

Specifying time in ms makes experiments comparable across `dt`; it does not
make trained models `dt`-invariant. At `dt=2, tau=5` (`dt/tau = 0.4`) the
Euler error alone changes peak ratios by a few percent against the continuum
limit, and each `dt` is a separate training run. `dt`-invariance holds for
fixed weights as `dt << tau`.
