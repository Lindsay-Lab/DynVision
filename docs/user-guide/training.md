# Training Models and Generating Checkpoints

## Overview

DynVision trains models through a two-rule Snakemake pipeline: a model is first
initialized into a state dict, then trained into a final state dict while
PyTorch Lightning checkpoints are written periodically. This guide explains how
to launch a training run, where the resulting files are written, and how to
recover usable weights from the Lightning checkpoints.

The training entry point is `dynvision/runtime/train_model.py`, orchestrated by
the `train_model` rule in `dynvision/workflow/snake_runtime.smk`.

## Prerequisites

- An interim training dataset prepared under
  `project_paths.data.interim/{data_name}/train_all` (with the matching
  `train_all.ready` marker). See [Data Processing](data-processing.md).
- If `use_ffcv` is enabled, the FFCV `train.beton` and `val.beton` files under
  `project_paths.data.processed/{data_name}/train_all/`.
- A configured Snakemake environment, run from the `dynvision/workflow/`
  directory as described in the project README.
- Training defaults defined in `dynvision/configs/config_defaults.yaml`
  (optimizer, scheduler, epochs, batch size, loss). See the
  [Configuration Reference](../reference/configuration.md).

## Workflow Summary

| Stage | Snakemake rule | Key script | Output |
|-------|----------------|------------|--------|
| Initialization | `init_model` | `runtime/init_model.py` | `…/{data_name}/init.pt` |
| Training | `train_model` | `runtime/train_model.py` | `…/{data_name}/trained.pt` |
| Checkpoint export | `best_checkpoint_to_statedict` | `utils/checkpoint_to_statedict.py` | `…/{data_name}/trained-best.pt` |

Models are stored hierarchically:

```
models/
└── {model_name}/
    └── {model_name}{model_args}_{seed}/
        └── {data_name}/
            ├── init.pt                 # initialized state dict
            ├── trained.pt              # final trained state dict
            ├── trained.config.yaml     # resolved configuration export
            ├── trained-best-{epoch}-{metric}.ckpt   # top-k Lightning checkpoints
            └── trained-last-{epoch}-{metric}.ckpt   # final Lightning checkpoint
```

## Step 1: Initialize the Model

The `init_model` rule creates a model from the configuration and saves its
initial state dict. It infers input shape from the training dataset.

```bash
snakemake init_model \
  --config model_name=DyRCNNx4 data_name=cifar100 seed=0
```

You normally do not need to run this rule explicitly — Snakemake builds it
automatically as a dependency of `train_model`. Run it on its own when you want
to inspect or reuse the initialized weights.

## Step 2: Train the Model

The `train_model` rule consumes `init.pt` and produces `trained.pt`. Because the
output path embeds `{model_name}`, `{model_args}`, `{seed}`, and `{data_name}`,
you select a configuration by setting those values in `--config`:

```bash
snakemake train_model \
  --config model_name=DyRCNNx4 data_name=cifar100 seed=0 \
  model_args="{rctype:full}"
```

During training the rule:

- Loads the initialized weights from
  `models/{model_name}/{model_name}{model_args}_{seed}/{data_name}/init.pt`.
- Trains with the optimizer, scheduler, loss, and epoch count defined in
  `config_defaults.yaml` (overridable through `--config`).
- Writes periodic and best Lightning checkpoints (`.ckpt`) into the same
  `{data_name}` directory using the `monitor_checkpoint` metric.
- Saves the final state dict as `trained.pt` and exports the resolved
  configuration alongside it as `trained.config.yaml`.

If a Lightning checkpoint already exists for the run, training resumes from it
(restoring model, optimizer, and epoch). If no checkpoint is found, training
starts from `init.pt`.

### Requesting a Trained Model by Path

Because Snakemake resolves rules by output file, you can request a trained model
directly by its target path instead of by rule name:

```bash
snakemake models/DyRCNNx4/DyRCNNx4:rctype=full_0/cifar100/trained.pt
```

Snakemake runs `init_model` first if `init.pt` is missing, then `train_model`.

## Step 3: Override Training Parameters

Any default in `config_defaults.yaml` can be overridden on the command line.
Model-specific arguments go through `model_args`; run-level training arguments
are passed directly:

```bash
snakemake train_model \
  --config model_name=DyRCNNx4 data_name=cifar100 seed=0 \
  model_args="{rctype:full+tsteps:20}" \
  epochs=200 \
  batch_size=192 \
  learning_rate=0.0008 \
  use_ffcv=True
```

Learning-rate parameter groups apply different factors to recurrent and feedback
weights (`lr_parameter_groups` in `config_defaults.yaml`), so the recurrent and
feedback connections train with a smaller effective learning rate than the
feedforward weights.

## Step 4: Generate a Clean State Dict from Checkpoints

The final `trained.pt` is written when training completes. To extract the
best-performing weights from the Lightning `.ckpt` files — for example after an
interrupted run, or to select the top checkpoint by validation metric — use the
`best_checkpoint_to_statedict` rule:

```bash
snakemake best_checkpoint_to_statedict \
  --config model_name=DyRCNNx4 data_name=cifar100 seed=0 \
  model_args="{rctype:full}"
```

This produces `trained-best.pt` in the run directory, a plain state dict ready
for evaluation with the [`test_model` workflow](model-testing.md).

## Step 5: Parameter Sweeps and Multiple Seeds

Snakemake expands list-valued arguments into separate runs, so you can train a
whole comparison set with a single command:

```bash
snakemake train_model \
  --config model_name=DyRCNNx4 data_name=cifar100 \
  seed="[0, 1, 2]" \
  model_args="{rctype:[full, self, depthpointwise]}"
```

Each combination is written to its own hierarchical output directory and is
skipped if its `trained.pt` already exists. See
[Workflow Management](workflows.md) for more on parameter sweeps.

## Common Issues and Solutions

- **Missing input dataset**: If Snakemake reports a missing `train_all.ready`
  marker, prepare the dataset first (see [Data Processing](data-processing.md)).
- **FFCV files not found**: When `use_ffcv=True`, the `train.beton` and
  `val.beton` files must exist; generate them or set `use_ffcv=False` to fall
  back to the standard PyTorch data pipeline.
- **GPU out-of-memory**: Lower `batch_size`, increase `accumulate_grad_batches`
  to keep the effective batch size, or enable a lower-precision setting via
  `precision`.
- **Resuming unintentionally**: Training resumes automatically when a `.ckpt`
  exists in the run directory. Remove or rename the checkpoints to force a fresh
  start from `init.pt`.

## Related Resources

- [Model Testing](model-testing.md) — evaluate the `trained.pt` you produce here.
- [Workflow Management](workflows.md) — Snakemake rules, wildcards, and sweeps.
- [Custom Models](custom-models.md) — define new architectures to train.
- [Configuration Reference](../reference/configuration.md) — full list of
  training parameters and their defaults.
