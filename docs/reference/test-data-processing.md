# Test Data Processing and Aggregation

This reference describes the two-stage pipeline for processing and aggregating test data in DynVision experiments.

## Overview

The test data processing system operates in two stages:

1. **Stage 1: Single Test Processing** (`process_single_test.py`) - Converts individual test outputs into structured CSV format
2. **Stage 2: Experiment Aggregation** (`aggregate_experiment_data.py`) - Combines multiple test results with metadata extraction

### Purpose

- Transform raw test outputs (PyTorch tensors, model outputs) into analysis-ready tabular data
- Extract metadata from configuration files and file paths
- Aggregate results across experimental conditions (parameters, categories, status values)
- Support flexible data resolution (sample-level or class-level aggregation)

### Key Features

- **Automatic metadata extraction** from config files and file paths
- **Path-based parameter extraction** for wildcards (category, status, epoch)
- **Flexible status handling** supporting multiple model checkpoints
- **Epoch auto-extraction** from `trained-epoch=N` status values
- **Configurable aggregation** at sample or class level
- **Robust error handling** with optional failure tolerance

## File Structure

Test data follows this hierarchical structure:

```
reports/
└── {experiment}/
    └── {model_identifier}/
        └── {data_name}:{data_group}_{status}/
            └── {test_identifier}/
                ├── test_responses.pt          # Layer-wise neural responses
                ├── test_outputs.csv           # Model predictions and performance
                ├── test_outputs.csv.config.yaml  # Resolved parameters
                └── test_data.csv              # Processed unified data (Stage 1 output)
```

**Aggregated output:**
```
reports/
└── {experiment}/
    └── {model_name}:{args1}{category}=*{args2}_{seed}/
        └── {data_name}:{data_group}_{status}/
            └── test_data.csv              # Aggregated data (Stage 2 output)
```

## Stage 1: Single Test Processing

### Script

`dynvision/processing/process_single_test.py`

### Purpose

Processes individual test outputs by:

1. Loading neural responses from `.pt` files
2. Loading predictions and performance from `.csv` files
3. Computing layer-wise statistics (mean, std, variance)
4. Combining into unified tabular format

### Command-Line Interface

```bash
python process_single_test.py \
    --test_responses <path>/test_responses.pt \
    --test_outputs <path>/test_outputs.csv \
    --output <path>/test_data.csv \
    [--layer_measures response_avg response_std] \
    [--confidence_measures first_label_confidence] \
    [--accuracy_topk 1 5] \
    [--classifier_topk 1 5]
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--test_responses` | Path | Yes | Path to `test_responses.pt` file containing layer activations |
| `--test_outputs` | Path | Yes | Path to `test_outputs.csv` file with predictions |
| `--output` | Path | Yes | Output path for processed `test_data.csv` |
| `--layer_measures` | List[str] | No | Layer statistics to compute: `response_avg`, `response_std`, `spatial_variance`, `feature_variance` |
| `--confidence_measures` | List[str] | No | Confidence metrics to include: `guess_confidence`, `label_confidence`, `first_label_confidence` |
| `--accuracy_topk` | List[int] | No | Top-k accuracy values to compute (e.g., 1, 5) |
| `--classifier_topk` | List[int] | No | Top-k classifier confidence to extract |

### Output Format

The `test_data.csv` file contains:

| Column Group | Columns | Description |
|--------------|---------|-------------|
| Identifiers | `sample_idx`, `true_label`, `predicted_label` | Sample identification |
| Time | `times_index`, `times_ms` | Temporal indices and timestamps |
| Layer Responses | `{layer}_response_avg`, `{layer}_response_std`, ... | Per-layer statistics |
| Performance | `accuracy`, `top5_accuracy`, ... | Classification metrics |
| Confidence | `first_label_confidence`, ... | Model confidence scores |

## Stage 2: Experiment Aggregation

### Script

`dynvision/processing/aggregate_experiment_data.py`

### Purpose

Aggregates multiple processed test files by:

1. Loading test_data.csv files from multiple experimental conditions
2. Extracting metadata from configuration files and file paths
3. Adding metadata columns (parameter values, categories, status)
4. Concatenating into single dataset
5. Optionally aggregating to class level

### Command-Line Interface

```bash
python aggregate_experiment_data.py \
    --test_data <path1>/test_data.csv <path2>/test_data.csv ... \
    --test_configs <path1>/config.yaml <path2>/config.yaml ... \
    --output <output_path>/test_data.csv \
    --parameter <param_name> \
    --category <category_name> \
    [--additional_parameters seed status epoch] \
    [--sample_resolution sample|class] \
    [--fail_on_missing_inputs True|False]
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--test_data` | List[Path] | Yes | - | Paths to individual `test_data.csv` files |
| `--test_configs` | List[Path] | Yes | - | Paths to corresponding `.config.yaml` files |
| `--output` | Path | Yes | - | Output path for aggregated dataset |
| `--parameter` | str | Yes | - | Primary parameter key to extract from `data.*` namespace |
| `--category` | str | Yes | - | Category key to extract from model path wildcards |
| `--additional_parameters` | List[str] | No | `[]` | Additional parameters to extract (e.g., `seed`, `status`, `epoch`) |
| `--sample_resolution` | str | No | `'sample'` | Resolution: `'sample'` (keep all samples) or `'class'` (aggregate by class) |
| `--fail_on_missing_inputs` | bool | No | `True` | Whether to fail if input files are missing |

## Metadata Extraction

### FileMetadata Class

```python
@dataclass
class FileMetadata:
    """Container for extracted test metadata."""
    parameter_value: str              # Primary parameter value
    category_value: str               # Category value from path
    extra_values: Dict[str, Optional[str]]  # Additional parameters
    status_value: Optional[str]       # Status from path (e.g., "trained-epoch=150")
```

### Metadata Sources

#### 1. Configuration Files

Parameters extracted from `.config.yaml` files using namespaced keys:

- **Data namespace**: `data.{param_name}` (e.g., `data.dsteps`, `data.stim`)
- **Model namespace**: `model.{param_name}` (e.g., `model.tau`, `model.dt`)
- **Unscoped**: Direct key names as fallback

**Example config.yaml:**
```yaml
data.dsteps: 20
data.stim: "gratings"
model.tau: 8
model.dt: 2
seed: 42
```

#### 2. File Paths

Path-based wildcards extracted using pattern matching:

##### Category Extraction

Categories are extracted from the model identifier using `extract_param_from_string()`:

**Path structure:**
```
.../DyRCNNx8:rctype=none+tau=8_42/...
           └─ category=value ──┘
```

**Extraction pattern:** `{category}={value}` in model arguments string

##### Status Extraction

Status values are extracted from the dataset path component:

**Path structure:**
```
.../{data_name}:{data_group}_{status}/...
                            └─status─┘
```

**Example paths:**
```
.../imagenette:all_init/...           → status="init"
.../imagenette:all_trained/...        → status="trained"
.../imagenette:all_trained-best/...   → status="trained-best"
.../imagenette:all_trained-epoch=150/... → status="trained-epoch=150", epoch=150
```

##### Epoch Extraction

Epoch values are automatically extracted from status strings containing `epoch=N`:

**Pattern:** `epoch=(\d+)` within status string

When `status="trained-epoch=150"`:

- Status column: `"trained-epoch=150"`
- Epoch column: `150` (auto-added to metadata)

## Core Functions

### extract_status_from_path()

```python
def extract_status_from_path(file_path: Path) -> Optional[str]
```

Extracts status value from file path.

**Parameters:**

- `file_path` (Path): Path to config or test_data file

**Returns:**

- `str`: Status string (e.g., `"trained"`, `"trained-epoch=150"`)
- `None`: If status not found in path

**Example:**
```python
from pathlib import Path
from dynvision.processing.process_test_data import extract_status_from_path

path = Path("reports/exp1/model/imagenette:all_trained-epoch=150/test1/test_data.csv")
status = extract_status_from_path(path)
print(status)  # Output: "trained-epoch=150"
```

### parse_status_string()

```python
def parse_status_string(status: str) -> Tuple[str, Optional[int]]
```

Parses status string to extract base status and optional epoch.

**Parameters:**

- `status` (str): Status string from path

**Returns:**

- `Tuple[str, Optional[int]]`: (status, epoch) where epoch is `None` if not present

**Example:**
```python
from dynvision.processing.process_test_data import parse_status_string

status, epoch = parse_status_string("trained-epoch=150")
print(f"Status: {status}, Epoch: {epoch}")  # Output: Status: trained-epoch=150, Epoch: 150

status, epoch = parse_status_string("trained-best")
print(f"Status: {status}, Epoch: {epoch}")  # Output: Status: trained-best, Epoch: None
```

### extract_param_from_string()

```python
def extract_param_from_string(
    s: str,
    key: str,
    value_type: Optional[type] = None,
    assigner: str = "="
) -> Union[int, float, str, bool, None]
```

Extracts parameter value from string using key-value pattern.

**Parameters:**

- `s` (str): String containing parameter (e.g., model arguments)
- `key` (str): Parameter key to extract
- `value_type` (Optional[type]): Expected type (`int`, `float`, `str`, or `None` for auto-detect)
- `assigner` (str): Character separating key and value (default: `"="`)

**Returns:**

- Value of appropriate type, or `None` if not found

**Raises:**

- `ValueError`: If parameter not found or type mismatch

**Example:**
```python
from dynvision.utils import extract_param_from_string

model_args = "rctype=none+tau=8+dt=2.0"
rctype = extract_param_from_string(model_args, "rctype", str)
tau = extract_param_from_string(model_args, "tau", int)
dt = extract_param_from_string(model_args, "dt", float)

print(f"rctype={rctype}, tau={tau}, dt={dt}")
# Output: rctype=none, tau=8, dt=2.0
```

### aggregate_test_data()

```python
def aggregate_test_data(
    test_data_files: List[Path],
    config_files: List[Path],
    data_arg_key: str,
    category: str,
    extra_parameters: Sequence[str],
    resolution: str = "sample",
    fail_on_missing: bool = True,
    extract_status: bool = False,
) -> Tuple[pd.DataFrame, List[Path]]
```

Aggregates multiple test_data.csv files with metadata extraction.

**Parameters:**

- `test_data_files` (List[Path]): Paths to test_data.csv files
- `config_files` (List[Path]): Paths to corresponding config.yaml files
- `data_arg_key` (str): Primary parameter key to extract from data namespace
- `category` (str): Category key to extract from model path
- `extra_parameters` (Sequence[str]): Additional parameters to extract
- `resolution` (str): `'sample'` (keep all) or `'class'` (aggregate by class)
- `fail_on_missing` (bool): Whether to fail on missing input files
- `extract_status` (bool): Whether to extract status from file paths

**Returns:**

- `Tuple[pd.DataFrame, List[Path]]`: (aggregated_dataframe, list_of_successful_files)

**Example:**
```python
from pathlib import Path
from dynvision.processing.aggregate_experiment_data import aggregate_test_data

test_files = [
    Path("reports/exp1/model1/data_trained/test1/test_data.csv"),
    Path("reports/exp1/model2/data_trained/test2/test_data.csv"),
]
config_files = [
    Path("reports/exp1/model1/data_trained/test1/test_outputs.csv.config.yaml"),
    Path("reports/exp1/model2/data_trained/test2/test_outputs.csv.config.yaml"),
]

df, successful = aggregate_test_data(
    test_data_files=test_files,
    config_files=config_files,
    data_arg_key="dsteps",
    category="rctype",
    extra_parameters=["seed", "status", "epoch"],
    resolution="sample",
    extract_status=True,
)

print(f"Aggregated {len(successful)} files, {len(df)} total rows")
print(f"Columns: {df.columns.tolist()}")
```

## Status Values Reference

### Standard Status Values

| Status | Description | Epoch | Use Case |
|--------|-------------|-------|----------|
| `init` | Randomly initialized model | No | Baseline performance |
| `pretrained` | Loaded from pretrained weights | No | Transfer learning |
| `trained` | Fully trained model (final checkpoint) | No | Standard evaluation |
| `trained-best` | Best validation checkpoint | No | Best performance |
| `trained-last` | Last training checkpoint | No | Final state |
| `trained-epoch=N` | Specific training epoch N | Yes | Training dynamics analysis |

### Status Column Behavior

When `status` is included in `additional_parameters`:

1. **Automatic extraction** from file paths (no manual specification needed)
2. **Status column added** to output DataFrame with full status string
3. **Epoch auto-extraction** when status contains `epoch=N`
4. **Epoch column added** automatically when present in status

**Example DataFrame columns:**
```python
# With status="trained-epoch=150"
['sample_idx', 'true_label', 'times_index', 'dsteps', 'rctype',
 'seed', 'status', 'epoch', 'V1_response_avg', 'accuracy', ...]
#                    ↑        ↑
#     Auto-added: "trained-epoch=150"  and  150
```

## Data Resolution

### Sample Resolution (`sample`)

Keeps all individual samples with their full temporal dynamics.

**Output structure:**

- One row per sample per timestep
- All layer responses preserved
- All performance metrics at sample level

**Use case:** Detailed temporal analysis, single-sample trajectories

### Class Resolution (`class`)

Aggregates data by true label (class) across all samples.

**Aggregation method:**

- Layer responses: Mean across samples of same class
- Performance metrics: Mean across samples of same class
- Temporal dimension preserved (per-timestep aggregation)

**Output structure:**

- One row per class per timestep
- Reduced data size for class-level comparisons

**Use case:** Class-level performance comparison, reduced data analysis

## Snakemake Integration

### Rule: aggregate_experiment_data

**Input:**

- Expanded list of `test_data.csv` files across category values
- Corresponding `test_outputs.csv.config.yaml` files

**Parameters:**
```python
params:
    parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
    additional_parameters = ['seed', 'status', 'epoch'],
    sample_resolution = 'sample',  # or 'class'
    fail_on_missing_inputs = False,
```

**Output:**
```python
output:
    experiment_data = project_paths.reports \
        / '{experiment}' \
        / '{model_name}:{args1}{category}=*{args2}_{seed}' \
        / '{data_name}:{data_group}_{status}' \
        / 'test_data.csv'
```

**Note:** Wildcards in output path (`{status}`, `{category}`) are resolved by Snakemake expand functions, but values are also extracted from individual file paths for robustness.

## Common Workflows

### Workflow 1: Basic Aggregation

Aggregate test results across one parameter dimension:

```bash
python aggregate_experiment_data.py \
    --test_data results/exp1/model:rctype=*/data_trained/*/test_data.csv \
    --test_configs results/exp1/model:rctype=*/data_trained/*/config.yaml \
    --output results/exp1/aggregated.csv \
    --parameter dsteps \
    --category rctype \
    --additional_parameters seed
```

### Workflow 2: Multi-Status Aggregation

Aggregate across multiple model checkpoints (init, trained, best):

```bash
python aggregate_experiment_data.py \
    --test_data results/exp1/model/data_*/*/test_data.csv \
    --test_configs results/exp1/model/data_*/*/config.yaml \
    --output results/exp1/aggregated.csv \
    --parameter dsteps \
    --category rctype \
    --additional_parameters seed status epoch \
    --sample_resolution sample
```

Result includes `status` column with values: `"init"`, `"trained"`, `"trained-best"`, etc.

### Workflow 3: Training Dynamics

Aggregate across epoch checkpoints for training dynamics analysis:

```bash
# Assumes test data for trained-epoch=0, trained-epoch=50, ..., trained-epoch=300
python aggregate_experiment_data.py \
    --test_data results/exp1/model/data_trained-epoch=*/*/test_data.csv \
    --test_configs results/exp1/model/data_trained-epoch=*/*/config.yaml \
    --output results/exp1/training_dynamics.csv \
    --parameter dsteps \
    --category rctype \
    --additional_parameters seed status epoch
```

Result includes:

- `status`: Full status strings (`"trained-epoch=0"`, `"trained-epoch=50"`, ...)
- `epoch`: Extracted integers (`0`, `50`, ...)

## Performance Considerations

### Memory Usage

- **Sample resolution**: Memory scales with (n_samples × n_timesteps × n_conditions)
- **Class resolution**: Reduces memory by factor of n_samples / n_classes
- Large experiments (>1000 files) may require chunked processing

### Processing Time

- **File I/O**: Dominant factor for large experiments
- **Metadata extraction**: Minimal overhead (<1% of total time)
- **Concatenation**: O(n) in number of files

**Optimization tips:**

1. Use `fail_on_missing_inputs=False` for partial results during development
2. Process subsets of conditions first to validate pipeline
3. Use class resolution for exploratory analysis, sample resolution for final analysis

## Error Handling

### Common Errors

#### Missing Config Files

```
FileNotFoundError: Config file not found: .../config.yaml
```

**Solution:** Ensure config files are generated by TestingParams (automatic in Snakemake workflow)

#### Parameter Not Found

```
ValueError: Parameter 'data.dsteps' not found in config file
```

**Solution:** Check that parameter exists in config with correct namespace (`data.*`, `model.*`, or unscoped)

#### Status Extraction Failed

```
Warning: extract_status=True but no status found in path: .../test_data.csv
```

**Solution:** Verify path structure matches expected format: `{data}:{group}_{status}`

### Tolerance Modes

**Strict mode** (`fail_on_missing_inputs=True`, default):

- Fails immediately on missing files or parameters
- Use for production pipelines

**Tolerant mode** (`fail_on_missing_inputs=False`):

- Skips missing files with warnings
- Continues processing remaining files
- Use during development or for partial results

## Advanced Topics

### Custom Metadata Extraction

To extract custom metadata beyond standard parameters:

1. Add new keys to `additional_parameters` in Snakemake rule
2. Ensure keys exist in config files with appropriate namespace
3. For path-based extraction, extend `extract_status_from_path()` or add new extraction functions

### Multi-Dimensional Aggregation

For experiments varying multiple dimensions simultaneously:

```python
additional_parameters = ['seed', 'status', 'epoch', 'data_group', 'stimulus_type']
```

All parameters become columns in the output DataFrame, enabling multi-factor analysis.

### Integration with Visualization

Aggregated data is designed for direct use with DynVision visualization scripts:

```bash
# After aggregation
python plot_training.py \
    --test_data results/exp1/aggregated.csv \
    --accuracy_csv results/exp1/accuracy.csv \
    --loss_csv results/exp1/loss.csv \
    --output figures/exp1_overview.pdf \
    --category rctype \
    --parameter dsteps
```

## Related Documentation

- [Testing Guide](../user-guide/model-testing.md) - How to run tests and configure test parameters
- [Workflows Guide](../user-guide/workflows.md) - Complete workflow orchestration
- [Visualization Guide](../user-guide/visualization.md) - Plotting aggregated data
- [Configuration System](configuration.md) - Parameter management and config files

## Notes

### Path-Based vs Config-Based Parameters

**Path-based** (extracted from file paths):

- `category`: Model argument wildcard (e.g., `rctype=none`)
- `status`: Model checkpoint indicator (e.g., `trained-epoch=150`)
- `epoch`: Auto-extracted from status when present

**Config-based** (extracted from .config.yaml):

- All `data.*` parameters
- All `model.*` parameters
- Unscoped parameters (e.g., `seed`)

**Rationale:** Wildcards are structural elements of the file system hierarchy, while configuration parameters represent actual experiment settings.

### Automatic Epoch Handling

When `status` contains `epoch=N`:

- Epoch is automatically added to `extra_values`
- No need to manually add `'epoch'` to `additional_parameters` (though it's harmless to do so)
- Epoch column always present when status indicates specific epoch

This design ensures epoch information is never lost when analyzing training dynamics.

### Status String Preservation

The full status string (including epoch specification) is preserved in the `status` column to maintain complete checkpoint information. This allows filtering like:

```python
# Get only best checkpoints
best_data = df[df['status'] == 'trained-best']

# Get specific epoch
epoch_150 = df[df['status'] == 'trained-epoch=150']

# Get all epoch checkpoints (use epoch column)
all_epochs = df[df['epoch'].notna()]
```
