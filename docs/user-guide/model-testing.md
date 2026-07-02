# Model Testing and Test Data Processing

## Overview
DynVision separates model evaluation into two coordinated steps:

1. Running the `test_model` workflow rule to produce raw classifier predictions and layer responses.
2. Aggregating those artifacts with `process_test_data.py` to build analysis-ready tables.

This guide explains how to configure each phase, what files are created, and how to customize metrics for downstream visualization.

## Prerequisites
- A trained model checkpoint generated with the `train_model` rule.
- Test data prepared under `project_paths.data.interim`.
- A populated experiment entry in `dynvision/configs/config_experiments.yaml` (defines loaders, parameters, and data arguments).
- Snakemake environment configured as described in the project README.

## Workflow Summary
| Stage | Snakemake rule | Key script | Outputs |
|-------|----------------|------------|---------|
| Evaluation | `test_model` (from `workflow/snake_runtime.smk`) | `SCRIPTS/runtime/test_model.py` | `test_outputs.csv`, `test_responses.pt` per experiment slice |
| Processing | `process_test_data` (from `workflow/snake_visualizations.smk`) | `SCRIPTS/visualization/process_test_data.py` | Consolidated `test_data.csv` with metrics |

## Step 1: Configure the Experiment
Each experiment in `config_experiments.yaml` specifies:

- `parameter`: the primary sweep variable injected into filenames.
- `data_loader`: class responsible for assembling stimuli.
- `data_args`: mapping of loader arguments; values may be scalars or lists to expand across combinations.
- Optional `status` entries (e.g., `trained-epoch=99`) to select intermediate checkpoints.

Example excerpt:
```yaml
duration:
  status: trained
  parameter: stim
  data_loader: StimulusDuration
  data_args:
    dsteps: 30
    intro: 1
    stim: [1, 3, 5, 10, 20]
    idle: 20
```
Snakemake expands `data_args` and `status` to enumerate concrete runs. Category lists in `experiment_config.categories` (e.g., `rctype`, `trc`) provide wildcard values for comparisons.

### Extending Experiment Types
To introduce a novel stimulus protocol:

1. Implement a DataLoader subclass in `dynvision/data/dataloader.py` (use `StandardDataLoader` or the temporal loaders as templates). Provide aliases via `@alias_kwargs` so configuration keys (for example `stim`, `intro`) map cleanly onto constructor arguments.
2. Register the class name in the `DATALOADER_CLASSES` dictionary so `get_data_loader` can resolve it during workflow execution.
3. Reference the new loader in `config_experiments.yaml` by setting `data_loader` and supplying the required `data_args`. Snakemake will automatically expand the experiment combinations and pass them into the `test_model` rule.

## Step 2: Run the `test_model` Rule
From `dynvision/workflow/` execute:
```bash
snakemake test_model \
  --config experiment=duration model_name=DyRCNNx4 data_name=cifar100 seed=0
```
The rule:

- Loads the trained weights from `project_paths.models/{model_name}`.
- Mounts the test dataset symlink specified by `data_loader` and `data_group`.
- Calls `SCRIPTS/runtime/test_model.py` with batch size, normalization stats, and any `model_args`/`data_args` supplied via configuration.
- Emits two artifacts under `project_paths.reports/{data_loader}/<formatted-run-id>/`:
  - `test_outputs.csv`: per-sample classifier predictions, labels, confidences, and metadata.
  - `test_responses.pt`: serialized tensor dictionary with layer responses (including `classifier` logits when requested).

## Step 3: Inspect Intermediate Results
Before aggregation, verify the evaluation pass:

- `test_outputs.csv` columns include `sample_index`, `times_index`, `label_index`, `guess_index`, and other task-specific fields produced by the runtime script.
- `test_responses.pt` should contain layer tensors keyed by module name. Missing tensors usually indicate disabled logging in the model configuration.

## Step 4: Process Test Data
The visualization workflow calls `process_test_data.py` via the `process_test_data` rule:
```bash
snakemake process_test_data \
  --config experiment=duration model_name=DyRCNNx4 data_name=cifar100 seed=0 category=rctype
```
Key parameters injected by the rule:

- `--responses` / `--test_outputs`: glob-expanded lists of matching `.pt` and `.csv` files.
- `--parameter`: experiment-level sweep key (e.g., `stim`).
- `--category`: comparison axis taken from `experiment_config['categories']`.
- `--additional_parameters`: optional metadata to extract from directory names (default `epoch`).
- `--measures`: metrics to compute (layer statistics, confidence scores, top-k accuracy, classifier unit activations).
- `--sample_resolution`: choose `sample` (per image) or `class` (aggregated by `first_label_index`).
- `--remove_input_responses`: remove `.pt` responses after successful processing to save space.

### Script Responsibilities
Inside `process_test_data.py`:

1. Validates metadata consistency between `.pt` and `.csv` filenames using `extract_param_from_string`.
2. Loads the CSV with `load_df` and augments it via `process_test_performance`, adding `first_label_index` and `accuracy` indicators.
3. Optionally computes classifier-derived metrics (confidence, top-k accuracy, top-N unit activations) when `test_responses.pt` contains a `classifier` tensor.
4. Streams layer responses through `process_layer_responses_incremental` to assemble large-scale statistics without exhausting memory.
5. Attaches requested metadata columns (primary parameter, category, additional parameters) for downstream plotting.
6. Writes a unified `test_data.csv` per batch; Snakemake concatenates batches into the final report path under `project_paths.reports/{experiment}/`.

### Custom Invocation
You can call the script directly, for example:
```bash
python dynvision/processing/process_test_data.py \
  --responses path/to/test_responses.pt \
  --test_outputs path/to/test_outputs.csv \
  --output reports/duration/run_01/test_data.csv \
  --parameter stim \
  --category rctype \
  --measures response_avg response_std accuracy_top3 \
  --additional_parameters epoch tau \
  --sample_resolution sample \
  --fail_on_missing_inputs False
```
Use `--fail_on_missing_inputs False` to skip missing file pairs without aborting the run—helpful when partial evaluations finish.

## Step 5: Utilize the Processed Dataset
Downstream visualization rules (e.g., `plot_performance`, `plot_responses`) consume the `test_data.csv` files produced above. Each CSV contains:

- Metadata columns (experiment parameter, category, additional parameters).
- Temporal indices (`times_index`) and presentation identifiers (`first_label_index`).
- Layer statistics (`response_avg`, `response_std`, etc.).
- Performance measures (`accuracy`, `accuracy_topK`, confidence metrics).
- Optional classifier activation columns (`classifier_topN`, `_id`).

### Metadata and Index Columns
- **Index columns**: `sample_index` tracks individual images when `sample` resolution is selected; `times_index` marks timestep positions; `first_label_index` records the earliest valid class per sample (used for grouping and presentation-level aggregation).
- **Parameter column**: named after the experiment’s `parameter` entry (e.g., `stim`, `contrast`) and repeats the sweep value extracted from the file path.
- **Category column**: matches the Snakemake wildcard specified in the workflow invocation (e.g., `rctype`, `trc`), enabling comparisons across architectural variants.
- **Additional parameters**: any names supplied via `--additional_parameters` are extracted from response directories and inserted verbatim, allowing downstream filters like `epoch == 99` or `tau == "5"`.

### Available Measure Columns
`process_test_data.py` and `process_single_test.py` organize measures into four categories:

- **Layer metrics** (`response_avg`, `response_std`, `spatial_variance`, `feature_variance`): computed per layer and timestep from response tensors. When `--sample_resolution sample` is used, they emit columns such as `{layer_name}_response_avg`. Under `class` resolution the same metrics aggregate over presentations (`first_label_index`).
- **Confidence metrics** (`guess_confidence`, `label_confidence`, `first_label_confidence`): derived from classifier logits. Values reflect softmax probabilities and remain at the same resolution as the CSV input.
- **Top-k accuracy metrics** (`accuracy_top3`, `accuracy_top5`, etc.): Boolean indicators per timestep showing whether the ground-truth label appears in the model’s top-k predictions. With class resolution, they are averaged and accompanied by standard deviation columns.
- **Classifier activations** (`classifier_topN`, `classifier_topN_id`): capture the activation magnitude and corresponding unit index for the most active classifier channels, useful for feature analysis.

### Column Naming Conventions
- Scalar columns retain their measure name (e.g., `accuracy`, `label_confidence`).
- Layer metrics follow `{layer}_{measure}` for sample resolution; class resolution introduces `_avg` and `_std` suffixes when values vary within a presentation.
- Additional parameters requested via `--additional_parameters` appear as plain columns (e.g., `epoch`, `tau`).
- When resolution is `class`, `sample_index` is removed and summary columns are keyed by `first_label_index` and `times_index`.

These tables can also be loaded manually into analysis notebooks:
```python
import pandas as pd

test_data = pd.read_csv("logs/reports/duration/duration_DyRCNNx4:.../test_data.csv")
filtered = test_data.query("first_label_index == 5 and times_index <= 10")
```

## Common Issues and Solutions
- **Mismatched metadata**: The processor raises `ValueError` if filename parameters disagree between `.pt` and `.csv`. Confirm Snakemake wildcards produce aligned paths.
- **Missing classifier tensor**: Confidence metrics require `responses["classifier"]`. Enable classifier logging in the model or remove those measures.
- **Memory warnings**: Lower `--batch_size` or adjust `--memory_limit_gb` when processing very large response sets.
- **Empty outputs**: Ensure `test_outputs.csv` contains rows for every sample; rerun `test_model` if the evaluation terminated early.

## Related Resources
- [Training](training.md) for training models and generating checkpoints.
- `docs/development/guides/ai-style-guide.md` for workflow conventions.
- `docs/reference/workflow-overview.md` (if available) for a schematic of Snakemake rules.
- Source scripts in `dynvision/runtime/test_model.py` and `dynvision/processing/process_test_data.py` for implementation details.
