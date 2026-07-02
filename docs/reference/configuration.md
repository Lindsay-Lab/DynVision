# Configuration Reference

DynVision uses a hierarchical configuration system based on YAML files to manage parameters across different components of the toolbox. This document provides a comprehensive reference for the configuration files and parameters.

## Configuration Organization

The configuration system is organized into several YAML files, each handling specific aspects of the toolbox:

```
dynvision/configs/
├── config_defaults.yaml       # Base configuration with sensible defaults
├── config_data.yaml           # Dataset-specific configurations
├── config_visualization.yaml  # General visualization settings
├── config_experiments.yaml    # Experiment-specific settings
├── config_workflow.yaml       # Workflow execution parameters
└── README.md                  # Configuration documentation
```

## Configuration Loading

The configuration files are loaded in the workflow file `snake_utils.smk` in the following order:

1. `config_defaults.yaml`
2. `config_data.yaml`
3. `config_visualization.yaml`
4. `config_experiments.yaml`
5. `config_workflow.yaml`

When DynVision runs, these files are loaded in sequence, so that in case of redundant or conflicting parameter definitions later definitions files taking precedence over earlier ones. The compiled configuration is then saved to `config_runtime.yaml` for reference and reproducibility.

The runtime scripts `init_model.py`, `train_model.py`, `test_model.py` get their required parameters as commandline arguments and accept both a path to a config file and explicit parameter values, e.g. `python init_model.py --config_path="../configs/config_runtime.yaml" --model_name AlexNet`, where all explicit parameter values overwrite the values in the config file. So, the scripts can be used flexibly within the workflow with the right parameter combinations (e.g. for parameter sweeps) and as stand-alone resources.

## Parameter Precedence and Defaults

DynVision uses a **sentinel-based parameter system** with a three-tier hierarchy for parameter resolution. This allows model classes to define their own defaults while still supporting framework-wide configuration.

### Three-Tier Hierarchy

Parameters are resolved in the following priority order:

1. **Explicit Values (Highest Priority)**
   Parameters explicitly set via CLI arguments or YAML config files always take precedence.
   ```bash
   python init_model.py --dt 1.0  # Overrides everything
   ```

2. **Framework Defaults (Medium Priority)**
   Parameters defined in `config_defaults.yaml` provide framework-wide defaults for common configurations.
   These defaults are used when no explicit value is provided but before falling back to model-specific defaults.
   ```yaml
   # config_defaults.yaml
   dt: 2  # Framework default used by most models
   ```

3. **Model Class Defaults (Lowest Priority)**
   Each model class defines its own defaults in the `__init__` method signature.
   These are used only when a parameter is not set at higher levels (i.e., when the parameter value is `None`).
   ```python
   class CorNetRT(DyRCNN):
       def __init__(
           self,
           dt: float = 2.0,  # Used if config.dt is None
           fixed_self_weight: float = 1.0,  # Model-specific default
           ...
       ):
   ```

### Sentinel-Based Parameter Passing

The Pydantic parameter classes use `None` as a **sentinel value** to distinguish between:

- **Explicitly set**: Value provided via CLI or YAML → passed to model
- **Not set**: Value is `None` → **not passed** to model, allowing model default to be used

This is implemented through automatic filtering in `get_model_kwargs()`, which removes all `None` values before passing parameters to the model constructor.

#### In YAML Files

```yaml
# Explicitly set framework default (passed to all models)
dt: 2

# Commented out = not loaded = None in Pydantic = not passed = model decides
# tau: 5

# Explicit null = None in Pydantic = not passed = model decides
recurrence_bias: ~

# Empty dict/list are still "set" (not None), so they ARE passed
optimizer_kwargs: {}  # Passed as empty dict to model
```

#### In Model Classes

```python
class CorNetRT(DyRCNN):
    def __init__(
        self,
        dt: float = 2.0,  # Used if config.dt is None
        fixed_self_weight: float = 1.0,  # Model-specific default
        recurrence_target: str = "middle",  # Model-specific default
        ...
    ):
        # If user doesn't set these in YAML, model defaults are used
        super().__init__(dt=dt, ...)
```

#### Parameter Flow Example

```
User YAML: (dt commented out, fixed_self_weight not mentioned)
    ↓
Pydantic loads: dt=None, fixed_self_weight=None
    ↓
get_model_kwargs() filters: {} (both None, so excluded)
    ↓
Model instantiation: CorNetRT()
    ↓
Model uses its defaults: dt=2.0, fixed_self_weight=1.0
```

```
User YAML: dt: 1.0
    ↓
Pydantic loads: dt=1.0, fixed_self_weight=None
    ↓
get_model_kwargs() filters: {"dt": 1.0} (fixed_self_weight excluded)
    ↓
Model instantiation: CorNetRT(dt=1.0)
    ↓
Model uses: dt=1.0 (explicit), fixed_self_weight=1.0 (default)
```

### Best Practice: When to Define Defaults

| Location | Purpose | Example |
|----------|---------|---------|
| **config_defaults.yaml** | Framework-wide defaults applicable to most models | `dt: 2`, `n_timesteps: 20` |
| **Model class `__init__`** | Model-specific defaults that may differ between architectures | `fixed_self_weight: 1.0` (CorNet-specific) |
| **Pydantic classes** | No defaults! Use `Optional[T] = None` for model parameters | `dt: Optional[float] = None` |

### Debugging Parameter Resolution

To see which parameters are being passed to your model:

```python
from dynvision.params import ModelParams

params = ModelParams.from_cli_and_config(config_path="config.yaml")
model_kwargs = params.get_model_kwargs(MyModel)
print(f"Parameters passed to model: {model_kwargs.keys()}")
# Any parameter not in this dict will use the model's default
```

## Core Configuration Files

The configuration system is divided into four main files, each responsible for a specific aspect of the toolbox:

### 1. config_defaults.yaml

Provides the foundational configuration layer with sensible defaults for all components. Contains:

- Model parameters (time steps, delays, neural dynamics)
- Temporal presentation parameters (patterns, reaction time masking)
- Basic training settings (batch size, epochs, optimizer)
- Response storage configuration
- Default loss function settings
- Base data parameters

**Key Temporal Parameters:**

- `n_timesteps`: Number of temporal processing steps in the model (commented out by default, allowing model-specific defaults)
- `data_presentation_pattern`: Controls stimulus/null presentation pattern (default: `[1]` for continuous stimulus)
- `loss_reaction_time`: Reaction time window for loss masking in milliseconds (default: `0`)
- `data_timesteps`: Number of timesteps for data loader temporal expansion (default: `1`)

See [Temporal Data Presentation Guide](../user-guide/temporal-data-presentation.md) for detailed usage.

These defaults can be overridden by other configuration files or command-line arguments.

### 2. config_data.yaml

Manages all dataset-specific configurations, including:

- Dataset resolutions and statistics (mean, standard deviation)
- Data loading settings (FFCV configuration)
- Dataset groupings and categories
- Mounted dataset specifications

This separation allows for easy addition of new datasets and modification of data processing parameters.

### 3. config_visualization.yaml

Defines general visualization parameters. This is still empty, and will be filled as different visualization are more formalized.

### 4. config_experiments.yaml

Defines experiment-specific configurations and parameter sweeps:

- Parameter categories for systematic exploration
- Experiment-specific data loading settings
- Stimulus and timing parameters
- Contrast and interval configurations

This separation allows for organized parameter exploration and experiment reproducibility.

### 5. config_workflow.yaml

Controls the current workflow execution parameters:

- Model selection and architecture settings
- Training hyperparameters for the current run
- Dataset and category selection
- Experiment selection and configuration

This file typically changes most frequently as it defines the specific experiment being run.

## Command-Line Overrides

Configuration values can be overridden via the command line using Snakemake's `--config` parameter:

```bash
snakemake --config model_name=DyRCNNx4 model_args="{rctype: full}"
```

This approach allows for flexible parameter exploration without modifying the configuration files.
The final configuration set is still accurately reported in the log files.

## Parameter Expansion

DynVision includes utilities for parameter expansion, which is particularly useful for hyperparameter sweeps. Any parameter that is included as wildcard in the init, train, or test rules can be set as a list of values to be expanded: `model_name`, `seed`, `data_name`, `status`, `data_loader`, `data_group`, as well as any arguments to the model or dataloader in `model_args` and `data_args`.
To make the expansion over `data_args` more compact for specific testing scenarios, you can also expand over `experiments`, as they are defined in `config_experiments.yaml`.


```yaml
model_args:

  - tsteps: [10, 20]
  - rctype: ["full", "self"]

experiment:

  - contrast
  - duration

data_name: imagenet
data_group:

  - imagenette
  - insects
```

would produce 4 model variants

```python
[":rctype=full+tsteps=10", ":rctype=full+tsteps=20", ":rctype=self+tsteps=10", ":rctype=self+tsteps=20"]
```

that are each tested on two subsets of imagenet, the easier imagenette selection and the more demanding differentiation of insects, both with varying contrasts and varying durations of image presentation.

## Project Paths Configuration

DynVision uses a flexible path management system that automatically adapts to different execution environments and provides a structured organization for project files. The system is implemented in `project_paths.py` and can be customized for your specific setup.

### Basic Configuration

The path system distinguishes between two main directories :

- **Working Directory**: Contains project-specific data, models, and outputs
- **Toolbox Directory**: Contains the DynVision source code and scripts

The system creates and manages the following directory structure:

```
working_dir/
├── data/
│   ├── raw/          # Original, immutable data
│   ├── external/     # Data from third-party sources
│   ├── interim/      # Intermediate data
│   └── processed/    # Final, canonical data
├── models/           # Trained models and checkpoints
├── notebooks/        # Jupyter notebooks
├── references/       # Data dictionaries, manuals, etc.
├── reports/
│   └── figures/      # Generated graphics and figures
└── logs/
    └── benchmarks/   # Performance benchmarks

toolbox_dir/
├── data/            # Data processing scripts
├── utils/           # Utility functions
├── models/          # Model implementations
├── losses/          # Loss function implementations
├── configs/         # Configuration files
├── workflow/        # Workflow definitions
└── visualization/   # Visualization tools
```

When you downloaded the DynVision repository, this `DynVision` folder represents the `working_dir` and the subfolder `dynvision` represents the `toolbox_dir`. You may split these, for example, to use the same toolbox codebase across multiple projects with separate data/output/log folders.

To configure the path system, you can modify these class attributes in `project_paths.py`:

```python
class project_paths_class:
    project_name = "your_project_name"
    toolbox_name = "DynVision"
    user_name = "your_username"
```

### Environment-Specific Setup

The system automatically detects cluster environments by checking the hostnames for common strings used naming hpc clusters including "hpc", "log-", "greene", "slurm", "compute", "node", "cluster" which you may extend to incorporate you specific cluster system.
When DynVision detects to be on a compute cluster, it can dynamically change path settings, for example, large data/log/results directories are moved to the scratch partition, and paths are automatically adjusted for cluster-specific locations

Example cluster path structure:
```
/home/username/project_name/     # Main project directory
/scratch/username/
├── data/                       # Raw data
└── project_name/
    ├── models/                 # Trained models
    ├── reports/                # Generated reports
    └── logs/                   # Large log files
```

### Accessing Paths

Once configured, you can access paths through the global `project_paths` instance:

```python
from dynvision.project_paths import project_paths

# Access data directories
raw_data_path = project_paths.data.raw
processed_data_path = project_paths.data.processed

# Access model and log directories
models_path = project_paths.models
logs_path = project_paths.logs

# Access script directories
model_scripts = project_paths.scripts.models
workflow_scripts = project_paths.scripts.workflow
```

To get the project path structure with changed base directories, you may also reinitialize the class

```python
from dynvision.project_paths import project_paths_class

project_paths = project_paths_class(
    working_dir="/path/to/your/project",
    toolbox_dir="/path/to/dynvision"
)
```

## Environment-Specific Configuration

DynVision adapts to different execution environments (defined in `config_defaults.yaml`):

**Local Environment**

```yaml
# Default values optimized for local development (such as quick debugging)
debug_batch_size: 3
debug_check_val_every_n_epoch: 1
debug_log_every_n_steps: 1
debug_accumulate_grad_batches: 1
debug_enable_progress_bar: True
```

**Cluster Environment**

```yaml
# Automatically adjusted for efficient cluster execution
batch_size: 256
check_val_every_n_epoch: 5
log_every_n_steps: 100
accumulate_grad_batches: 2
enable_progress_bar: False
```

## Best Practices

### 1. Configuration Management

- Maintain reasonable defaults in `config_defaults.yaml`
- Use specific configurations in specialized files
- Override only what's necessary

### 2. Experiment Reproducibility

- Use fixed random seeds
- Save runtime configurations
- Use version control for configuration files

### 3. Parameter Exploration

- Leverage parameter expansion for systematic exploration
- Use command-line overrides for quick tests
- Document parameter configurations in experiment logs

### 4. Path Management

- Use `project_paths` for consistent path handling
- Avoid hardcoded paths
- Use relative paths when possible

