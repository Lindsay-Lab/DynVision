# Designing Custom Experiments

## Overview

In DynVision an **experiment** is a named recipe for *how a stimulus is
presented to a model over time*, and a **category** is a named *model attribute
to vary*. Together they let you sweep a set of models against a temporal probe
and collect comparable results — all driven from configuration, without editing
any Python.

This guide shows how to:

1. Define a new experiment in `config_experiments.yaml`.
2. Optionally define a category (model-attribute sweep).
3. Run the experiment through the `*_model_variations` Snakemake rules.
4. Understand where results land.

!!! goal "Goal"
    Add a custom stimulus-presentation experiment and run it across a sweep of
    model variants, producing test results and comparison plots.

## Prerequisites

- A working Snakemake environment, run from `dynvision/workflow/` as described
  in the project README.
- At least one trained model available (or willing to let the workflow train
  it on demand). See [Training](training.md).
- Familiarity with [Workflow Management](workflows.md) and the
  [Temporal Data Presentation](temporal-data-presentation.md) concepts, since an
  experiment is essentially a configured data loader.

## Anatomy of an Experiment

Experiments live under the `experiment_config` key in
`dynvision/configs/config_experiments.yaml`. Each experiment is a named block:

```yaml
experiment_config:

  contrast:                     # <- experiment name
    parameter: contrast         # the swept stimulus variable (the x-axis)
    data_loader: StimulusContrast   # which data loader presents the stimulus
    data_args:                  # arguments passed to that data loader
      dsteps: 30                # total presentation timesteps
      intro: 1                  # blank lead-in steps before stimulus onset
      stim: 20                  # steps the stimulus is shown
      contrast: [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]   # swept values
      idle: 20                  # trailing blank steps after the stimulus
```

| Field | Required | Meaning |
|-------|----------|---------|
| `parameter` | ✅ | The stimulus variable that is swept — becomes the x-axis of response plots. Its values are usually the list given under the same key in `data_args`. |
| `data_loader` | ✅ | The data-loader class that realises the presentation (e.g. `StimulusDuration`, `StimulusContrast`, `StimulusInterval`, `StimulusNoise`). See [Data Processing](data-processing.md). |
| `data_args` | ✅ | Keyword arguments forwarded to the data loader. Providing a **list** for a key turns it into a swept dimension. |
| `status` | ⬜ | Optional override of which model checkpoint(s) to test, e.g. `[trained-epoch=99, trained-epoch=199]` to probe intermediate checkpoints. Defaults to `config.status` (typically `trained`). |

!!! tip "Timestep budget"
    For most presentation loaders the timeline is `intro + stim + idle` and
    should match `dsteps`. Keep them consistent so the stimulus window falls
    where you expect within the recorded response.

### A minimal custom experiment

Suppose you want a "flash" experiment: a very brief stimulus followed by a long
blank, to study how long the recurrent dynamics ring after stimulus offset.

```yaml
experiment_config:

  # ... existing experiments ...

  flash:
    parameter: stim
    data_loader: StimulusDuration
    data_args:
      dsteps: 80
      intro: 5
      stim: [1, 2, 4]      # three brief flash durations to compare
      idle: 74             # long tail to watch the response decay
```

That is the entire definition — no code changes required.

## Categories: Sweeping Model Attributes

An experiment varies the *stimulus*. To also vary the *model*, use a
**category** — a named list of values for one model argument, defined under
`experiment_config.categories`:

```yaml
experiment_config:
  categories:
    rctype: ['depthpointwise', 'self', 'full', 'local', 'none']  # recurrence type
    tsteps: [4, 8, 14, 20, 26]
    tau: [3, 5, 9]
    feedback: ['false', 'add', 'mul']
    # ...
```

When you pass `--config category=rctype`, the workflow builds the Cartesian
product of your default `model_args` with each value of that category, so one
command trains/tests/plots the whole family of model variants.

## Running the Experiment

Three rules consume experiments and categories. Each takes
`--config category=<name>` (which model attribute to vary) and, for testing and
plotting, `--config experiment=<name>` (which stimulus probe to apply).

=== "Train the variants"

    ```bash
    snakemake train_model_variations --config \
      category=rctype \
      model_name=DyRCNNx4 \
      data_name=cifar100
    ```

    Trains every `rctype` variant of `DyRCNNx4` on `cifar100`.

=== "Test with the experiment"

    ```bash
    snakemake test_model_variations --config \
      category=rctype \
      experiment=flash \
      model_name=DyRCNNx4 \
      data_name=cifar100
    ```

    Runs the `flash` presentation on each trained variant. Missing or stale
    models are trained automatically first. To test **only** already-trained
    models, restrict the rules:

    ```bash
    snakemake test_model_variations --config category=rctype experiment=flash \
      --allowed-rules test_model process_test_data
    ```

=== "Plot the comparison"

    ```bash
    snakemake plot_model_variations --config \
      category=rctype \
      experiment=flash \
      plot=responses \
      model_name=DyRCNNx4 \
      data_name=cifar100
    ```

    Generates the requested plot (default `responses`) for each variant. Use
    `--allowed-rules plot_responses` to plot only existing results without
    re-running upstream steps.

!!! tip "Sweep multiple categories at once"
    `category` accepts a list: `--config category="[rctype, tau]"` iterates each
    category in turn. Likewise `experiment` and `plot` accept lists.

## Where Results Land

The `*_model_variations` rules expand into the standard DynVision output
hierarchy (see [Workflow Management](workflows.md#output-organization)):

- **Models** — `models/{model_name}/{model_name}{model_args}_{seed}/{data_name}/{status}.pt`
- **Test data** — `reports/{experiment}/{model_name}{model_args}_{seed}/{data_name}:{data_group}_{status}/test_data.csv`
- **Figures** — `reports/figures/{experiment}/{model_name}{model_args}_{seed}/{data_name}:{data_group}_{status}/{plot}.png`

Because the experiment name is part of the path, results from different
experiments never collide, and a category sweep produces one sibling directory
per model variant.

## Checklist

- [ ] Added a named block under `experiment_config` with `parameter`,
      `data_loader`, and `data_args`.
- [ ] Confirmed `intro + stim + idle` matches `dsteps`.
- [ ] (Optional) Added or reused a `categories` entry for the model sweep.
- [ ] Verified the chosen `data_loader` exists in `dynvision/data/dataloader.py`.
- [ ] Ran `test_model_variations` (or the `--allowed-rules` variant) and found
      `test_data.csv` under `reports/{experiment}/…`.

## Related

- [Workflow Management](workflows.md) — Snakemake targets, wildcards, and output
  layout.
- [Temporal Data Presentation](temporal-data-presentation.md) — how the
  presentation loaders build stimulus timelines.
- [Data Processing](data-processing.md) — the available data loaders.
- [Model Testing](model-testing.md) — the single-model testing path that these
  rules build on.
