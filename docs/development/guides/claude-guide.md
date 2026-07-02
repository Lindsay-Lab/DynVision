# Claude Code Guide for DynVision

This guide provides comprehensive context for Claude Code (claude.ai/code) when working with the DynVision codebase.

> **Important**: Before starting any work, consult the [AI Style Guide](ai-style-guide.md) which establishes fundamental principles for research software development including:
> - Research software best practices (scientific correctness, reproducibility, performance)
> - Approach and workflow (investigation → analysis → implementation)
> - Code organization, documentation, testing, and error handling standards
> - Communication and collaboration guidelines
>
> This guide (claude-guide.md) provides **DynVision-specific** context, while the AI Style Guide provides **general research software principles** that apply to all tasks.

## Project Overview

DynVision is a modular toolbox for constructing and evaluating recurrent convolutional neural networks (RCNNs) with biologically inspired dynamics. The framework combines PyTorch, PyTorch Lightning, and Snakemake to enable efficient experimentation with temporal visual processing models that bridge computational neuroscience and deep learning.

**Design Philosophy**: DynVision prioritizes biological plausibility while maintaining computational efficiency, focusing on continuous-time neural dynamics, heterogeneous temporal delays, and modular component composition.

**Key Scientific Goals**:

- Enable systematic exploration of recurrent dynamics in visual processing
- Bridge computational neuroscience and deep learning
- Maintain biological plausibility while ensuring computational efficiency
- Support reproducible experimentation through workflow automation
- Facilitate parameter sweeps and comparative analysis across model variants

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n dynvision python=3.11
conda activate dynvision

# Install PyTorch with CUDA support (recommended)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install in editable mode
pip install -e .
```

### Running Experiments
All Snakemake commands must be run from the workflow directory:
```bash
cd dynvision/workflow

# Run a basic experiment
snakemake --config experiment=duration model_name=AlexNet data_name=cifar100

# Run with parameter sweeps (expand with wildcards)
snakemake --config experiment=contrast model_name=DyRCNNx8 model_args="{rctype: [full, self, pointdepthwise]}"

# Run specific named rules (manuscript figures, temporal parameters, noise tests, etc.)
snakemake neural
snakemake noise
snakemake timeparams
```

### Code Quality
```bash
# Format code with black
make format

# Lint code
make lint
```

Note: Makefile targets reference old project name `rhythmic_visual_attention` instead of `dynvision` - this needs updating.

### Release & Publishing

DynVision uses [flit](https://flit.pypa.io/) as its build backend and GitHub Actions to
automatically publish to PyPI via OIDC trusted publishing.

**How to create a release:**

1. Update the version in `pyproject.toml` (e.g., `version = "0.1.0"`).
   For pre-releases, use `0.1.0rc1` — rc/beta tags are published to Test PyPI.

2. Update `CHANGELOG.md` with the new version section.

3. Commit and push, then create a version tag:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to 0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```

4. The `.github/workflows/publish.yml` action triggers on `v*` tags and will:
   - Build the package with `flit build`
   - Publish to **Test PyPI** if the tag contains `rc` or `beta`
   - Publish to **production PyPI** for all other `v*` tags

**Tag naming convention:**

| Tag            | Destination   | Example version    |
|----------------|---------------|--------------------|
| `v*rc*`        | Test PyPI     | `v0.1.0rc1`        |
| `v*beta*`      | Test PyPI     | `v0.1.0beta1`      |
| `v*` (other)   | PyPI          | `v0.1.0`           |

**Prerequisites (one-time setup):**

Add the Lindsay-Lab/DynVision repository as a trusted publisher on both:

- [PyPI](https://pypi.org/manage/project/dynvision/settings/publishing/)
- [Test PyPI](https://test.pypi.org/manage/project/dynvision/settings/publishing/)

### Data Management
Data is automatically downloaded and prepared on first use. Manual data preparation:
```bash
# From workflow directory
snakemake <project_paths.data.interim>/<dataset_name>/train_all.ready
```

## Working with DynVision: Key Principles

### Investigation-First Approach

Before making any changes:

1. **Understand the Scientific Context**:
      - What biological/computational principle is being implemented?
      - How does this relate to visual neuroscience or neural dynamics?
      - What are the mathematical foundations (ODEs, connectivity patterns, etc.)?

2. **Find Existing Patterns**:
      - Search for similar implementations in the codebase
      - Check if related functionality exists in model_components/
      - Review how other models implement similar features
      - Look for established parameter handling patterns

3. **Review Documentation**:
      - Check docstrings for scientific context and parameter meanings
      - Review config files to understand parameter ranges and defaults
      - Consult planning/todo-*.md for known issues and planned work
      - Reference software-patterns.md for architectural guidance

### Scientific Correctness is Paramount

- **Verify mathematical correctness**: Check implementations against equations in papers/docs
- **Validate dimensions**: Ensure tensor shapes match biological/theoretical expectations
- **Test limiting cases**: Verify behavior when parameters → 0, ∞, or special values
- **Check units**: Time constants in ms, delays in ms, dimensions in pixels, etc.
- **Numerical stability**: Watch for operations that can overflow/underflow

### Performance Considerations

DynVision runs on GPUs and HPC clusters, so:

- **Profile before optimizing**: Use PyTorch profiler to identify actual bottlenecks
- **Minimize CPU-GPU transfers**: Keep computations on device
- **Leverage PyTorch Lightning**: Use built-in optimizations (mixed precision, DDP)
- **Optimize recurrent loops**: These are often the bottleneck in temporal models
- **Memory efficiency**: Be mindful of storing activations across many timesteps

### Code Organization Principles

- **Separation of concerns**:
  - Scientific components (`model_components/`) are framework-agnostic when possible
  - Training infrastructure (`base/lightning.py`) wraps PyTorch Lightning
  - Workflow orchestration (`workflow/`) handles experimentation
  - Data handling (`data/`) is independent of models

- **Modularity**:
  - Components should be composable (e.g., different recurrence types, solvers, connections)
  - Avoid tight coupling between modules
  - Use dependency injection (pass components to constructors)

- **Configuration-driven**:
  - Most behavior should be controllable via YAML configs
  - Use Pydantic for validation and type safety
  - Support both long and short parameter names (aliases)

### Communication Guidelines

When proposing changes:

- **Explain the "why"**: Scientific motivation, not just technical implementation
- **Show trade-offs**: Performance vs clarity, biological plausibility vs computational cost
- **Provide alternatives**: Multiple valid approaches often exist
- **Estimate effort**: Simple refactor vs major architectural change
- **Consider users**: Code will be used by neuroscientists, not just ML engineers

### Common Workflows

See the sections below for detailed guidance on:

- [Adding a new model](#adding-a-new-model) (workflow section)
- [Modifying recurrent connections](#modifying-recurrent-connections) (architecture section)
- [Adding a new experiment](#adding-a-new-experiment) (workflow section)
- [Known issues](#known-issues-and-inconsistencies) (known issues section)

## Architecture Overview

### Multi-Inheritance Pattern (Method Resolution Order)

DynVision uses Python's multiple inheritance to compose functionality. The inheritance order is critical for proper MRO:

```
Model Classes (e.g., DyRCNNx4, AlexNet, ResNet)
    ↓
BaseModel (base/__init__.py)
    ↓ (inherits via MRO in this order:)
┌───────────────┬──────────────────┬────────────────────┬──────────────────┐
│               │                  │                    │                  │
TemporalBase    LightningBase    StorageBufferMixin   MonitoringMixin   DtypeDeviceCoordinatorMixin
(temporal.py)   (lightning.py)   (storage.py)         (monitoring.py)   (coordination.py)
```

**MRO Inheritance Order**:

1. `TemporalBase` - Provides core neural network methods (forward, _define_architecture)
2. `LightningBase` - Can call DynVision methods in training steps
3. `StorageBufferMixin` - Adds Lightning hooks for response storage
4. `MonitoringMixin` - Adds Lightning hooks for debugging/logging
5. `DtypeDeviceCoordinatorMixin` - Adds Lightning hooks for dtype/device coordination

### Key Base Classes

**TemporalBase** (`base/temporal.py`):

- Core neural dynamics: timesteps, temporal delays, data presentation patterns
- Manages delays for feedforward (t_feedforward), recurrent (t_recurrence), skip (t_skip), and feedback (t_feedback) connections
- Implements `_process_input_dimensions()`, `forward()`, abstract `_define_architecture()`
- Handles DataBuffer instances for delayed activations

**LightningBase** (`base/lightning.py`):

- PyTorch Lightning integration: training configuration, loss computation, optimizer setup
- Implements `model_step()`, `training_step()`, `validation_step()`, `configure_optimizers()`
- Supports multiple loss functions via `criterion_params` list
- Parameter grouping for different learning rates (regular, recurrence, feedback)

**DtypeDeviceCoordinator** (`base/coordination.py`):

- Auto-discovery network to coordinate dtype/device across modules with persistent state
- Builds coordination graph via `build_coordination_network()`
- Propagates dtype/device sync with `propagate_dtype_sync()`
- Only active in non-distributed setups (disabled when `WORLD_SIZE > 1`)

**StorageBuffer / StorageBufferMixin** (`base/storage.py`):

- `DataBuffer` class: Circular buffers for managing delayed activations across timesteps
- Response storage and retrieval via `get_responses()`, `get_dataframe()`
- Configurable CPU vs GPU storage for memory management

**Monitoring / MonitoringMixin** (`base/monitoring.py`):

- Activity recording during forward passes
- Parameter statistics logging: `log_param_stats()`
- Weight checking: `_check_weights()` for NaN/Inf detection

**Alternative Compositions**:

- `CoreModel`: TemporalBase + DtypeDeviceCoordinatorMixin only
- `MonitoredModel`: TemporalBase + MonitoringMixin + DtypeDeviceCoordinatorMixin (no Lightning)

### Core Components

**Model Components** (`dynvision/model_components/`):

- **recurrence.py**: Recurrent connection types
  - `SelfConnection`: Unit connects only to itself
  - `FullConnection`: Dense local connectivity (full conv)
  - `DepthPointwiseConnection`: Depthwise → pointwise
  - `PointDepthwiseConnection`: Pointwise → depthwise
  - `LocalLateralConnection`: 2D topographic organization
  - `LocalSeparableConnection`: Local + patchy long-range connections
  - `RecurrentConnectedConv2d`: Wrapper class managing all recurrence types

- **dynamics_solver.py**: ODE solvers for continuous-time dynamics
  - `EulerStep`: First-order Euler integration
  - `RungeKuttaStep`: 4th-order Runge-Kutta for higher accuracy
  - Both solve: τ dx/dt = -x + W(x)

- **integration_strategy.py**: How recurrence integrates with feedforward
  - Additive: x + recurrence(x)
  - Multiplicative: x * recurrence(x)

- **layer_connections.py**: Skip and feedback connections between layers
  - `Skip`: Bypass connections between non-adjacent layers
  - `Feedback`: Top-down modulation from higher to lower areas

- **retina.py**: Retinal preprocessing (Gaussian blur, center-surround)
- **supralinearity.py**: Power-law nonlinearity f(x) = k·sign(x)·|x|^n
- **topographic_recurrence.py**: Local/topographic connectivity patterns

**Models** (`dynvision/models/`):

- `dyrcnn.py`: DyRCNNx2, DyRCNNx4, DyRCNNx8 (2/4/8 layer RCNNs with biological features)
- `alexnet.py`: AlexNet variants with optional recurrence
- `resnet.py`: ResNet variants (18, 20, 44, 1202)
- `cornet_rt.py`: CorNet-RT recurrent timing model
- `cordsnet.py`: Cortico-cortical dynamics model

### Workflow System

**Snakemake Workflow** (`dynvision/workflow/`):

The workflow uses wildcard-based path patterns to enable parameter sweeps:
```
{model_name}{model_args}_{seed}_{data_name}_{status}.pt

Example: DyRCNNx8:tsteps=20+rctype=full+tau=5_0040_imagenette_trained.pt
```

**Workflow Files**:

- `Snakefile`: Main entry point, includes all sub-workflows, defines top-level targets
- `snake_utils.smk`: Shared utilities (config processing, path handling, argument parsing)
- `snake_data.smk`: Data download, preprocessing, dataset creation, FFCV conversion
- `snake_runtime.smk`: Model initialization (`init_model`), training (`train_model`), evaluation (`test_model`)
- `snake_experiments.smk`: Experiment-specific test configurations (duration, contrast, noise, etc.)
- `snake_visualizations.smk`: Result plotting and visualization
- `config_handler.py`: Processes YAML config files and applies wildcard substitutions

**Key Rules**:

- `init_model`: Initialize model architecture from config
- `train_model`: Train with PyTorch Lightning
- `test_model`: Evaluate on test scenarios (StimulusDuration, StimulusContrast, noise variants)
- `process_test_data`: Convert raw responses to pandas DataFrame
- `plot_*`: Various visualization rules

**Runtime Scripts** (`dynvision/runtime/`):

- `init_model.py`: Initialize model architecture, save state dict
- `train_model.py`: Training loop with PyTorch Lightning Trainer
- `test_model.py`: Evaluation on various test scenarios

### Configuration System

**Config Files** (`dynvision/configs/`):

- `config_defaults.yaml`: Default model/training/data parameters
- `config_runtime.yaml`: Training hyperparameters (epochs, batch size, optimizer settings)
- `config_data.yaml`: Dataset definitions, paths, preprocessing options
- `config_experiments.yaml`: Test scenario specifications (parameter to vary, data loaders, data_args)
- `config_workflow.yaml`: Default workflow parameters for `snakemake all`
- `config_visualization.yaml`: Plotting parameters, style settings
- `config_modes.yaml`: Named parameter presets (debug, large_dataset, distributed)

**Config Hierarchy**: defaults → specific configs → command-line overrides

**Parameter Handling System** (`dynvision/params/`):

A sophisticated Pydantic-based validation system with four layers:

1. **Configuration Layer**: YAML files with operational mode management
2. **Validation Layer**: Pydantic type checking and constraint enforcement
3. **Composition Layer**: Script-specific parameter combinations (ModelParams, TrainerParams, DataParams)
4. **Runtime Layer**: Model/trainer/dataloader instantiation

**Parameter Precedence** (lowest to highest):

1. YAML Configuration Files
2. Snakemake CLI (`snakemake --config param=value`)
3. Python Script CLI (arguments passed to scripts within rules)
4. Direct Override kwargs (programmatic)

**Key Parameter Classes**:

- `BaseParams`: Foundation with CLI parsing, config loading, alias resolution
- `ModelParams`: Neural architecture, biological parameters, optimizer config
- `TrainerParams`: PyTorch Lightning settings, system config
- `DataParams`: Dataset specification, data loading, preprocessing
- `TrainingParams`: Composite of Model + Trainer + Data params
- `InitParams`, `TestingParams`: Task-specific compositions

**Config Modes**: Auto-detected or explicit parameter overrides based on context

- `debug`: Triggered when log_level="DEBUG" or epochs ≤ 5
- `large_dataset`: Activated for ImageNet, COCO, OpenImages
- `distributed`: Must be explicitly enabled

### Data Pipeline

**Data Flow**:

### Shared DataModule + Logging Workflow

DynVision now routes every runtime entrypoint (init/train/test) through the shared helpers in `dynvision/data/datamodule.py`:

- `DataInterface` captures dataset/dataloader provenance (preview vs active) and pipes every log line through `DataParams.log_dataset_creation` / `BaseParams.log_dataloader_creation`. Always reuse this interface instead of ad-hoc logging to keep provenance tags consistent.
- `DataModule` (Lightning-ready) backs `runtime/train_model.py`. It expects a full `TrainingParams` object with FFCV/PyTorch paths and exposes `create_preview_loader()` before Lightning `setup()` runs. When editing training data flows, wire new arguments through `DataParams` helper methods so both preview and fit/val logs stay in sync.
- `SimpleDataModule` serves single-dataset workflows (currently `runtime/init_model.py`). Use it whenever you just need a preview batch for dimension inference; all kwargs should come from `DataParams.get_preview_dataloader_kwargs()` to avoid runtime mutation.
- `TestingDataModule` extends `SimpleDataModule` with sampler instantiation, batch-size guards, and additional debug logging for `runtime/test_model.py`. Any future testing scripts should import this class rather than rebuilding loaders manually.

When adjusting logging verbosity:

1. Prefer `log_section()` calls inside the appropriate Params or DataModule helper so INFO-level output stays structured.
2. Use the `context` argument (`preview`, `train`, `val`, `active`) to highlight diffs; preview-only noise should generally be demoted to DEBUG unless you are actively debugging dataset wiring.
3. If new parameters must be inferred at runtime, call `BaseParams.update_field(..., provenance="runtime")` so provenance tags reflect the adjustment across init/train/test logs.
1. Raw data downloaded to `data/raw/` (automatic on first use)
2. Preprocessed to `data/interim/` (train/val splits via symlinks, no data duplication)
3. FFCV `.beton` files created in `data/processed/` for fast loading
4. DataLoaders handle both FFCV and PyTorch loading modes

**Key Data Classes** (`dynvision/data/`):

- `ffcv_dataloader.py`: Fast FFCV-based loading with `OS_CACHE` and `QUASI_RANDOM` ordering
- `dataloader.py`: Standard PyTorch DataLoader wrapper
- `datasets.py`: Custom dataset classes
  - `StimulusDuration`: Present stimuli for varying durations
  - `StimulusContrast`: Present stimuli at different contrast levels
  - `StimulusInterval`: Repeated stimuli with varying intervals
  - Various noise variants (uniform, gaussian, poisson, phase-scrambled, etc.)
- `operations.py`: Data transformations, temporal dimension handling
- `transforms.py`: Augmentation pipelines

## Key Patterns and Conventions

### Parameter Aliases

Many parameters have shortened aliases (defined via `@alias_kwargs` decorator):

**Temporal Parameters**:

- `trc` → `t_recurrence` (recurrent delay in ms)
- `tff` → `t_feedforward` (feedforward delay in ms)
- `tsk` → `t_skip` (skip connection delay in ms)
- `tfb` → `t_feedback` (feedback delay in ms)
- `tsteps` → `n_timesteps`
- `dt` → integration time step (ms)
- `tau` → neural time constant (ms)

**Model Parameters**:

- `rctype` → `recurrence_type` (full, self, pointdepthwise, depthpointwise, local, localdepthwise)
- `rctarget` → `recurrence_target` (output, input, middle)
- `solver` → `dynamics_solver` (euler, rk4)
- `lossrt` → `loss_reaction_time` (ms after stimulus onset to apply loss)

**Training Parameters**:

- `lr` → `learning_rate`
- `ffonly` → `feedforward_only` (disable recurrence)
- `inadapt` → `input_adaption_weight`
- `supralin` → `supralinearity`

### Model Initialization Order

When creating a model, initialization follows this sequence:

1. `__init__()` sets attributes (dt, tau, delays, etc.)
2. `_process_input_dimensions()` determines batch/time dimensions
3. `_define_architecture()` constructs layers (**must be implemented by subclass**)
4. `_init_parameters()` initializes weights
5. `build_coordination_network()` discovers modules needing dtype/device sync (called by root node in distributed setup)
6. `set_residual_timesteps()` determines timesteps needed for full signal propagation

### Temporal Processing

- Models process inputs with shape: `(batch, timesteps, channels, height, width)`
- Delays stored in `DataBuffer` instances (circular buffers)
- Delays specified in milliseconds, converted to timesteps via `dt`
- Data presentation patterns control when inputs are shown:
  - `[1]` = all timesteps
  - `[0,1,1,1,0]` = specific temporal pattern (idle, stimulus, idle)

### Layer Operations Sequence

Layers can define a custom sequence of operations via `layer_operations` list:

**Standard Operations**:

- `layer`: Main computation (conv, linear)
- `tstep`: Dynamics evolution (EulerStep, RungeKuttaStep)
- `nonlin`: Nonlinearity (ReLU, supralinear, etc.)
- `pool`: Pooling
- `record`: Response storage
- `addskip`: Add skip connections
- `addfeedback`: Add feedback connections

Example:
```python
self.layer_operations = ["layer", "tstep", "nonlin", "record", "pool"]
```

### Experiment Wildcards

Snakemake uses wildcards in file paths to generate parameter sweeps:

**Wildcard Format**:
```
{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}

Components:

- model_name: DyRCNNx8, AlexNet, ResNet18
- model_args: :tsteps=20+rctype=full+tau=5
- seed: 0000, 0040 (for reproducibility)
- data_name: imagenette, cifar100, mnist
- status: init, trained, trained#minval (checkpoint selection)
- data_loader: StimulusDuration, StimulusContrast
- data_args: :stim=20+idle=10+dsteps=40
- data_group: all, snakes, mollusks (class subsets)
```

## Project Paths

Edit `dynvision/project_paths.py` to configure:

- `working_dir`: Root for data, models, reports (default: `/home/rgutzen/01_PROJECTS/rhythmic_visual_attention`)
- `toolbox_dir`: Codebase location (default: auto-detected)
- Automatically detects cluster environment (checks for SLURM) and redirects large data to scratch partition

**Important Directories**:

- `data/interim/{dataset}/`: Prepared datasets with symlinks
- `models/{model_name}/`: Trained model checkpoints (`.pt` files)
- `reports/{experiment}/{full_model_spec}/`: Test results and CSV files
  - `test_data.csv`: Processed outputs with labels, predictions, confidence
- `reports/figures/{experiment}/{full_model_spec}/`: Plots and visualizations
- `logs/`: Training logs, wandb logs, benchmarks
- `data/processed/`: FFCV `.beton` files for fast loading

## Biological Plausibility Features

### Continuous-Time Dynamics

Models use differential equations rather than discrete updates:
```
τ · dx/dt = -x + Φ[f(t, r_n, r_{n-1})]
```

Where:

- τ (tau): time constant controlling response speed (5-20ms typical)
- dt: integration time step (1-5ms typical)
- Φ: nonlinearity (ReLU, supralinear)
- f: combines feedforward, recurrent, and external inputs

### Temporal Delays

Different connection types have different propagation delays:

- **Feedforward** (t_feedforward): Typically 10ms (longer-range projections)
- **Recurrent** (t_recurrence): Typically 6ms (shorter-range lateral)
- **Skip** (t_skip): Variable, can match feedback
- **Feedback** (t_feedback): Typically > 30ms (top-down from higher areas)

These create **temporally heterogeneous responses** - signals arrive at different layers at different times, mimicking biological response latencies:

- V1: ~40-60ms after stimulus
- V2: ~50-70ms
- V4: ~60-80ms
- IT: ~80-120ms

### Biological Phenomena Captured

1. **Response Latency**: Different areas respond with characteristic delays
2. **Contrast-Dependent Timing**: Higher contrast → faster response onset, earlier peak
3. **Temporal Summation**: Subadditive integration over time (saturates, doesn't scale linearly)
4. **Adaptation**: Response decrease with sustained/repeated stimulation
5. **Short-Term Memory**: Recurrent connections maintain information persistence

## Common Workflows

### Adding a New Model

1. Create file in `dynvision/models/`:
```python
from dynvision.base import BaseModel

class MyModel(BaseModel):
    def _define_architecture(self):
        self.layer_names = ['V1', 'V2', 'classifier']
        self.V1 = nn.Conv2d(3, 64, 3)
        self.V2 = nn.Conv2d(64, 128, 3)
        self.classifier = nn.Linear(128, self.n_classes)
```

2. Add model to `dynvision/models/__init__.py`:
```python
from .my_model import MyModel
__all__ = [..., 'MyModel']
```

3. Configure in `config_workflow.yaml`:
```yaml
model_name: MyModel
model_args:
  tsteps: 20
  dt: 2
  tau: 5
```

4. Use with Snakemake:
```bash
snakemake --config model_name=MyModel experiment=contrast
```

### Adding a New Experiment

1. Define in `config_experiments.yaml`:
```yaml
experiment_config:
  my_experiment:
    status: trained
    parameter: my_param  # which parameter varies
    data_loader: MyDataLoader
    data_args:
      dsteps: 100
      my_param: [1, 2, 3, 4, 5]
```

2. Create data loader in `dynvision/data/datasets.py`:
```python
class MyDataLoader(TemporalDataset):
    def __init__(self, dataset, my_param=1, ...):
        # Implementation
```

3. Add visualization rules to `snake_visualizations.smk`

4. Create plotting function in `dynvision/visualization/`

### Modifying Recurrent Connections

1. Recurrence types defined in `recurrence.py` as classes:
   - `FullConnection`, `SelfConnection`, etc.
2. `RecurrentConnectedConv2d` wrapper manages all types
3. Integration strategies in `integration_strategy.py` (additive/multiplicative)
4. Delays managed by `DataBuffer` in `storage.py`

### Running on Clusters

DynVision includes cluster integration via Snakemake's cluster plugins with **automatic environment detection**:

```bash
# Basic cluster execution (using NYU Greene as example)
./cluster/snakecharm.sh -j100 --config experiment=contrast
# Cluster automatically detected via SLURM_JOB_ID
# Python wrapped with executor_wrapper.sh (singularity + conda)
```

**Automatic Detection:**

- Detects SLURM, PBS, LSF, SGE via environment variables
- No configuration needed for cluster vs. local execution
- See `is_cluster_execution()` in `workflow/snake_utils.smk`

**Manual cluster submission:**
```bash
# Custom cluster resources
snakemake --executor cluster-generic \
  --cluster-generic-submit-cmd "sbatch --cpus-per-task=4 --gres=gpu:1" \
  --jobs 50
```

See `cluster/` directory for cluster-specific configuration files and `cluster/executor_wrapper.sh` for environment setup.

## Testing and Validation

The repository does not currently have a formal test suite (no `tests/` directory or pytest configuration).

**When adding tests**:

- Create `tests/` directory
- Add pytest to dev dependencies in `pyproject.toml`
- Test critical paths: temporal dynamics, recurrence integration, data loading, parameter validation

## Known Issues and Inconsistencies

See [`docs/development/planning/todo-docs.md`](../planning/todo-docs.md) for a comprehensive list of documentation-implementation mismatches and areas needing improvement.

**Major Known Issues**:

1. Makefile targets reference old project name `rhythmic_visual_attention` instead of `dynvision`
2. `project_paths.py` has mixed naming (project_name vs toolbox_name)
3. Git status shows modified files in model components and coordination - review before committing
4. Parameter handling docs describe Pydantic system that exists in code but may need reconciliation with @alias_kwargs decorator system

## References and Documentation

- [Getting Started Guide](../../getting-started.md): First steps with DynVision
- [Design Philosophy](../../explanation/design-philosophy.md): Core design principles
- [Temporal Dynamics](../../explanation/temporal_dynamics.md): Understanding temporal properties
- [Biological Plausibility](../../explanation/biological-plausibility.md): Alignment with neural systems
- [Model Components Reference](../../reference/model-components.md): Core building blocks
- [Recurrence Types](../../reference/recurrence-types.md): Different recurrent implementations
- [Dynamics Solvers](../../reference/dynamics-solvers.md): ODE solvers
- [Configuration Reference](../../reference/configuration.md): Config file documentation
- [Workflows Guide](../../user-guide/workflows.md): Snakemake workflow management
- [Custom Models Guide](../../user-guide/custom-models.md): Creating custom architectures
- [Parameter Handling](../../user-guide/parameter-handling.md): Sophisticated parameter system
- [Software Patterns](software-patterns.md): Design patterns used in DynVision

## Quick Reference

**Most Common Commands**:
```bash
# Train and test a model
cd dynvision/workflow
snakemake --config experiment=duration model_name=DyRCNNx4 data_name=cifar100

# Parameter sweep
snakemake --config experiment=contrast model_args="{rctype: [full, self]}"

# Dry run to see what would execute
snakemake -n --config experiment=duration

# Force rerun specific rule
snakemake --forcerun test_model --config experiment=contrast
```

**Key Files to Edit**:

- Model architecture: `dynvision/models/<model_name>.py`
- Experiments: `dynvision/configs/config_experiments.yaml`
- Workflow: `dynvision/workflow/Snakefile` and `snake_*.smk`
- Paths: `dynvision/project_paths.py`
- Parameters: `dynvision/configs/config_defaults.yaml`

---

## How to Use This Guide

This guide provides **project-specific** context for DynVision. It should be used in conjunction with the [AI Style Guide](ai-style-guide.md):

**AI Style Guide** → General research software principles (how to approach any research software project)
**Claude Guide** → DynVision-specific details (architecture, conventions, workflows)

### Workflow for Claude Code

1. **Start with the AI Style Guide**: Understand general principles for research software development
2. **Read this guide**: Learn DynVision's specific architecture, patterns, and conventions
3. **Investigate before coding**: Search codebase, review docs, find existing patterns
4. **Apply principles**: Scientific correctness, performance awareness, maintainability
5. **Communicate clearly**: Explain scientific motivation, trade-offs, alternatives

### When Something is Unclear

If you encounter ambiguity or need clarification:

- Check if it's documented in [planning/todo-docs.md](../planning/todo-docs.md) as a known issue
- Search the codebase for similar implementations
- Review config files for parameter defaults and ranges
- Ask the user for scientific context or preferred approach
- Propose multiple alternatives with trade-offs explained

### Contributing

For guidelines on contributing to DynVision, see [Contributing Guide](../../contributing.md).

For understanding design patterns used in the project, see [Software Patterns](software-patterns.md).

For research software development expertise, see [Research Software Guide](research-software.md).

---

**Last Updated**: 2025-10-23

This guide evolves with the project. Suggestions for improvements are welcome!
