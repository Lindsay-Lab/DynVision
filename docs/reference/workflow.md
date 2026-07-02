# Workflow File Organization

This reference describes how DynVision organizes files for models, experiments, and results. Understanding this structure helps you locate files, interpret paths, and organize your own experiments.

## Overview

DynVision uses a hierarchical file organization system that:

- Organizes models by architecture, parameters, and training data
- Groups experimental results by experiment type
- Optionally uses compact test identifiers to avoid filesystem limitations for complex test protocols
- Maintains clear relationships between models and results

## Directory Structure

### Models

Models are stored in a hierarchical structure:

```
models/
  {model_name}/
    {model_name}{model_args}_{seed}/
      {data_name}/
        ├── init.pt                    # Initialized model
        ├── trained.pt                 # Trained model
        ├── trained-epoch=149.pt       # Intermediate checkpoints (optional)
        └── trained-best.pt            # Best checkpoint (optional)
```

**Example:**
```
models/DyRCNNx8/
  DyRCNNx8:tsteps=20+dt=2_0042/
    imagenette/
      ├── init.pt
      ├── trained.pt
      └── trained-epoch=149.pt
```

**Components:**

- `{model_name}`: Model architecture (e.g., `DyRCNNx8`, `CorNetRT`)
- `{model_args}`: Model parameters (e.g., `:tsteps=20+dt=2`)
- `{seed}`: Random seed (e.g., `0042`)
- `{data_name}`: Training dataset (e.g., `imagenette`, `cifar10`)

### Test Identifier Compression

For test protocols with very long parameter specifications, DynVision can optionally use hash-compressed test identifiers to avoid filesystem path length limitations. This is automatically handled by the workflow when needed.

**Uncompressed form (default):**
```
reports/uniformnoise/
  DyRCNNx8:tsteps=20+dt=2_0042/
    imagenette:all_trained/
      StimulusNoise:noisetype=uniform+noiselevel=0.2+stim=50/
        ├── test_outputs.csv
        ├── test_responses.pt
        └── test_outputs.csv.config.yaml
```

**Compressed form (when enabled):**
```
reports/uniformnoise/
  DyRCNNx8:tsteps=20+dt=2_0042/
    imagenette:all_trained/
      abc123ef/                    # Hash of test protocol
        ├── test_outputs.csv
        ├── test_responses.pt
        └── test_outputs.csv.config.yaml    # Contains full parameters
```

**How it works:**

1. Long test specifications are hashed to short identifiers (8 characters)
2. Configuration files (`.config.yaml`) preserve the full parameter specification
3. The `process_test_data.py` script reads parameters from config files, not paths
4. This is transparent to users - parameters are still accessible

### Reports (Test Results)

Test results are organized by experiment, model, and test protocol:

```
reports/
  {experiment}/
    {model_name}{model_args}_{seed}/
      {data_name}:{data_group}_{status}/
        {test_identifier}/
          ├── test_outputs.csv
          ├── test_responses.pt
          └── test_outputs.csv.config.yaml
```

**Example:**
```
reports/uniformnoise/
  DyRCNNx8:tsteps=20+dt=2_0042/
    imagenette:all_trained/
      StimulusNoise:noisetype=uniform+noiselevel=0.2/
        ├── test_outputs.csv
        ├── test_responses.pt
        └── test_outputs.csv.config.yaml
```

**Components:**

- `{experiment}`: Experiment type (e.g., `uniformnoise`, `response`)
- `{model_name}{model_args}_{seed}`: Full model identifier
- `{data_name}:{data_group}_{status}`: Training data and test split
- `{test_identifier}`: Test protocol (either `{data_loader}{data_args}` or hash if compressed)

### Processed Results

Aggregated results combine multiple test runs:

```
reports/
  {experiment}/
    {model_name}{model_args}_{seed}/
      {data_name}:{data_group}_{status}/
        └── test_data.csv
```

**Example:**
```
reports/uniformnoise/
  DyRCNNx8:tsteps=20+dt=2_0042/
    imagenette:all_trained/
      └── test_data.csv    # Combined results across noise levels
```

### Figures

Visualizations follow the same hierarchy as processed results:

```
figures/
  {experiment}/
    {model_name}{model_args}_{seed}/
      {data_name}:{data_group}_{status}/
        ├── responses.png
        └── performance.png
```

**Example:**
```
figures/uniformnoise/
  DyRCNNx8:tsteps=20+dt=2_0042/
    imagenette:all_trained/
      ├── responses.png
      └── performance.png
```

## Path Patterns

### Understanding the Naming Convention

DynVision uses a consistent syntax for paths:

**Model identifiers:**
```
{model_name}{model_args}_{seed}
```
- Arguments start with `:` and use `+` separators
- Example: `DyRCNNx8:tsteps=20+dt=2_0042`

**Training/test specifications:**
```
{data_name}:{data_group}_{status}
```
- Colon separates data name from group
- Underscore separates group from status
- Example: `imagenette:all_trained`

**Data loader specifications:**
```
{data_loader}{data_args}
```
- Arguments start with `:` and use `+` separators
- Example: `StimulusNoise:noisetype=uniform+noiselevel=0.2`

### Common Path Examples

**Model files:**
```
models/DyRCNNx8:tsteps=20+dt=2_0042/imagenette/trained.pt
models/CorNetRT:dt=2_0000/cifar10/init.pt
models/DyRCNNx4_0015/mnist/trained.pt
```

**Test outputs:**
```
reports/uniformnoise/DyRCNNx8:hash=a7f3c9d4/imagenette:all_trained/StimulusNoise:noisetype=uniform+noiselevel=0.2/test_outputs.csv

reports/response/CorNetRT:dt=2_0000/cifar10:one_init/StimulusDuration:dsteps=50+stim=25/test_outputs.csv
```

**Processed results:**
```
reports/uniformnoise/DyRCNNx8:tsteps=20+dt=2_0042/imagenette:all_trained/test_data.csv

reports/response/CorNetRT:dt=2_0000/cifar10:one_init/test_data.csv
```

**Figures:**
```
figures/uniformnoise/DyRCNNx8:tsteps=20+dt=2_0042/imagenette:all_trained/responses.png

figures/response/CorNetRT:dt=2_0000/cifar10:one_init/performance.png
```

## Experiments

### Experiment Types

Experiments group related tests and analyses. Common experiment types:

| Experiment | Description | Data Loader |
|------------|-------------|-------------|
| `response` | Temporal response characterization | `StimulusDuration` |
| `uniformnoise` | Uniform noise robustness | `StimulusNoise` (uniform) |
| `gaussianblurnoise` | Gaussian blur robustness | `StimulusNoise` (gaussianblur) |
| `contrast` | Contrast sensitivity | `StimulusContrast` |
| `adaptation` | Temporal adaptation effects | `StimulusInterval` |

### Experiment Organization

Each experiment contains:

1. **Raw test outputs** for individual model/test combinations
2. **Processed data** aggregating results across parameters
3. **Visualizations** summarizing findings

**Example experiment structure:**
```
uniformnoise/
  # Raw outputs
  DyRCNNx8:hash=a7f3c9d4/imagenette:all_trained/StimulusNoise:noiselevel=0.2/test_outputs.csv
  DyRCNNx8:hash=a7f3c9d4/imagenette:all_trained/StimulusNoise:noiselevel=0.4/test_outputs.csv
  ...

  # Processed results
  DyRCNNx8:tsteps=20+dt=2_0042/imagenette:all_trained/test_data.csv

  # Visualizations
  DyRCNNx8:tsteps=20+dt=2_0042/imagenette:all_trained/responses.png
```

## Working with Files

### Finding Model Files

To locate a trained model:

1. Navigate to `models/{model_name}/`
2. Look for directories matching your parameters
3. Check the training dataset subdirectory
4. Use `trained.pt` for trained models, `init.pt` for initialized

**Example:**
```bash
# Find all trained DyRCNNx8 models
find models/DyRCNNx8/ -name "trained.pt"

# Find models trained on imagenette
find models/ -path "*/imagenette/trained.pt"
```

### Finding Test Results

To locate test results:

1. Navigate to `reports/{experiment}/`
2. Find the model directory (may use hash identifier)
3. Look for the training/test specification
4. Check the data loader subdirectory

**Example:**
```bash
# Find all uniformnoise experiment results
find reports/uniformnoise/ -name "test_outputs.csv"

# Find results for specific model
find reports/uniformnoise/DyRCNNx8:*/ -name "test_outputs.csv"
```

### Resolving Test Parameters

To find the original test parameters when using compressed test identifiers:

```bash
# Check the config file in the test output directory
cat reports/uniformnoise/DyRCNNx8:tsteps=20_42/imagenette:all_trained/abc123ef/test_outputs.csv.config.yaml

# This contains all resolved parameters including data.* and model.* settings
```

### Navigating the Structure

```bash
# List all model architectures
ls models/

# List all experiments
ls reports/

# List models for an architecture
ls models/DyRCNNx8/

# List results for an experiment
ls reports/uniformnoise/

# Show full directory tree (limited depth)
tree -L 4 models/DyRCNNx8/
```

## File Formats

### Model Files (.pt)

PyTorch model state dictionaries containing:

- Model parameters (weights and biases)
- Configuration information
- Training metadata

**Load a model:**
```python
import torch

# Load state dictionary
state_dict = torch.load('models/DyRCNNx8:tsteps=20+dt=2_0042/imagenette/trained.pt')

# Extract components
model_state = state_dict['model_state']
config = state_dict['config']
```

### Test Outputs (.csv)

CSV files with test results:

- Model predictions
- Ground truth labels
- Performance metrics
- Per-sample information

**Columns typically include:**

- `sample_id`: Test sample identifier
- `true_label`: Ground truth class
- `predicted_label`: Model prediction
- `confidence`: Prediction confidence
- Additional experiment-specific columns

**Load test outputs:**
```python
import pandas as pd

df = pd.read_csv('reports/uniformnoise/.../test_outputs.csv')
```

### Test Responses (.pt)

PyTorch tensors with layer-wise neural responses:

- Activations from each network layer
- Temporal dynamics (for recurrent models)
- Used for detailed response analysis

**Load responses:**
```python
import torch

responses = torch.load('reports/uniformnoise/.../test_responses.pt')
```

### Configuration Files (.yaml)

YAML files documenting test configuration (automatically created for each test run):

- Data loader parameters (prefixed with `data.`)
- Model configuration (prefixed with `model.`)
- Test protocol details
- All resolved parameter values

**Location:** Alongside test outputs as `test_outputs.csv.config.yaml`

**Load configuration:**
```python
import yaml

with open('reports/uniformnoise/.../test_outputs.csv.config.yaml') as f:
    config = yaml.safe_load(f)

# Access parameters
noise_level = config['data.noiselevel']
model_tsteps = config['model.tsteps']
```

**Purpose:**

- Preserves full parameter specification when test identifiers are compressed
- Enables parameter extraction by `process_test_data.py` script
- Documents exact configuration used for reproducibility

## Tips and Best Practices

### Organizing Custom Experiments

When designing custom experiments:

1. **Choose descriptive names** that reflect the experiment's purpose
2. **Use consistent naming** across related experiments
3. **Document parameters** in configuration files
4. **Group related tests** under the same experiment name

### Managing Large Parameter Sweeps

For experiments with many parameter combinations:

1. **Test identifier compression** handles long test protocol specifications automatically
2. **Use full model identifiers** - they clearly show model configuration
3. **Check config files** when you need to verify exact parameter values
4. **Let Snakemake manage** file organization automatically
5. **Check processed results** (`test_data.csv`) for aggregated views across parameter sweeps

### Troubleshooting

**Can't find a file:**

- Verify the model identifier matches the trained model
- Check the experiment name
- Ensure the workflow has completed successfully
- Check Snakemake logs for errors

**Unclear which parameters were used:**

- Check `test_outputs.csv.config.yaml` files for full parameter specification
- Model identifiers in paths show model configuration directly
- Use `grep` or `find` to search for specific parameter values

**Need to clean up:**

- Remove intermediate files carefully
- Keep model files and final results (`test_data.csv`, figures)
- Use `snakemake --delete-all-output` to remove workflow outputs
- Back up important results before cleanup

## Related Resources

- [Workflow Management Guide](../user-guide/workflows.md) - Running experiments with Snakemake
- [Configuration Reference](configuration.md) - Configuring experiments
- [Developer Workflow Guide](../development/guides/workflow.md) - Technical implementation details
- [File Organization Planning](../development/planning/hash_model_args.md) - Design rationale
