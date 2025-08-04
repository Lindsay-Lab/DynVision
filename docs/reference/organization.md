# DynVision Organization

This reference document explains the core organizational principles and structure of the DynVision toolbox.

## Core Principles

DynVision is organized around three fundamental principles:

1. **Modular Architecture**: Each component is designed as a self-contained module with clear responsibilities and interfaces.
2. **Biological Plausibility**: The structure reflects the hierarchical and recurrent nature of biological visual systems.
3. **Experimental Workflow**: The organization supports systematic experimentation from data preparation to result analysis.

## Component Organization

The toolbox is structured into specialized modules, each serving a distinct purpose:

```
dynvision/
├── base/                # Core base classes and coordination
├── model_components/    # Neural building blocks
├── models/              # Complete architectures
├── params/              # Parameter management system
├── data/                # Data management
├── losses/              # Training objectives
├── runtime/             # Execution handling
├── workflow/            # Experiment management
├── visualization/       # Analysis tools
├── utils/               # Shared utilities
├── configs/             # Configuration system
├── cluster/             # Distributed execution
└── project_paths.py     # Centralized path handling
```

## Module Structure and Purpose

### Base Module

The `base` module provides the fundamental infrastructure for the entire framework:

**Core Base Classes**:
   - `__init__.py`: Contains the BaseModel class that combines the other base classes
   - `dynvision.py`: Core forward functionality and temporal dynamics
   - `lightning.py`: PyTorch Lightning integration and training infrastructure
   - `coordination.py`: Device and dtype coordination across model components
   - `storage.py`: Efficient data buffering and memory management
   - `monitoring.py`: Training monitoring and performance tracking utilities

This module establishes the foundational architecture that all other components build upon.

### Model Components Module

The `model_components` module provides the fundamental building blocks for neural networks:

**Neural Components**:
   - `dynamics_solver.py`: Numerical integration methods for neural dynamics
   - `recurrence.py`: Recurrent connection implementations
   - `topographic_recurrence.py`: Spatially-constrained recurrent connections
   - `layer_connections.py`: Skip and feedback connectivity patterns
   - `supralinearity.py`: Nonlinear activation functions
   - `retina.py`: Retinal processing components

For details, see [Model Components Reference](model-components.md).

### Parameters Module

The `params` module implements a comprehensive parameter management system:

1. **Parameter Categories**:
   - `base_params.py`: Base parameter definitions and validation
   - `model_params.py`: Model-specific parameter configurations
   - `data_params.py`: Data processing and loading parameters
   - `training_params.py`: Training procedure parameters
   - `testing_params.py`: Evaluation and testing parameters
   - `trainer_params.py`: PyTorch Lightning trainer configurations
   - `init_params.py`: Model initialization parameters

2. **Organization**:
   - Centralized parameter validation
   - Type checking and constraint enforcement
   - Configuration inheritance and composition

### Models Module

The `models` module implements complete neural architectures:

1. **Core Implementations**:
   - Research models (DyRCNN)
   - Standard architectures (ResNet, AlexNet, CorNetRT, CordsNet)
   - Custom architectures

2. **Model Organization**:
   - Each model in a separate file
   - Consistent inheritance from base classes
   - Standardized configuration interface

### Data Module

The `data` module manages all data-related operations:

1. **Data Loading with PyTorch**:
   - `datasets.py`: Dataset implementations
   - `dataloader.py`: PyTorch data loaders

2. **Data Loading with FFCV**:
   - `ffcv_datasets.py`: Optimized dataset compression
   - `ffcv_dataloader.py`: Optimized loading pipelines
   - `ffcv_operations.py`: FFCV-specific operations

3. **Processing Pipeline**:
   - `transforms.py`: Data transformations
   - `operations.py`: Standard processing operations
   - `get_data.py`: Dataset acquisition and management

### Losses Module

The `losses` module implements training objectives:

1. **Loss Functions**:
   - `base_loss.py`: Abstract base classes
   - `cross_entropy_loss.py`: Classification losses
   - `energy_loss.py`: Biological energy constraint losses

2. **Organization**:
   - Modular implementation with consistent interfaces
   - Configurable parameters and composition support
   - Support for multi-objective optimization

### Runtime Module

The `runtime` module handles execution:

1. **Core Components**:
   - `init_model.py`: Model initialization procedures
   - `train_model.py`: Training execution routines
   - `test_model.py`: Evaluation and testing routines

2. **Integration**:
   - PyTorch Lightning integration
   - Experiment tracking and logging
   - Resource management and optimization

### Workflow Module

The workflow system orchestrates experiments through Snakemake:

1. **Core Workflows**:
   - `snake_data.smk`: Data preparation pipelines
   - `snake_runtime.smk`: Model execution workflows
   - `snake_visualizations.smk`: Analysis and visualization
   - `snake_experiments.smk`: Experiment orchestration
   - `Snakefile`: Main workflow coordination

2. **Management**:
   - `mode_manager.py`: config mode coordination
   - Modular rule definitions and dependency management
   - Resource allocation and distributed execution

For usage details, see the [Workflows Guide](../user-guide/workflows.md).

### Visualization Module

The `visualization` module provides analysis tools:

1. **Plot Types**:
   - `plot_classifier_responses.py`: Model response analysis
   - `plot_weight_distributions.py`: Weight distribution visualization
   - `plot_adaption.py`: Temporal adaptation analysis
   - `plot_experiment_outputs.py`: Experiment result comparison
   - `plot_confusion_matrix.py`: Classification performance analysis

2. **Components**:
   - `callbacks.py`: Runtime visualization callbacks
   - Specialized plotting functions with consistent interfaces
   - Analysis utilities for neural dynamics

### Utils Module

The `utils` module provides shared functionality:

1. **Utility Categories**:
   - `config_utils.py`: Configuration loading and validation
   - `data_utils.py`: Data manipulation operations
   - `model_utils.py`: Model construction and management
   - `torch_utils.py`: PyTorch helper functions
   - `type_utils.py`: Type checking and validation
   - `visualization_utils.py`: Plotting helper functions

2. **Organization**:
   - Function-specific files with clear interfaces
   - Consistent error handling and documentation
   - Shared type definitions and constants

### Configuration Module

The configuration system manages all settings:

1. **Config Files**:
   - `config_defaults.yaml`: Base default settings
   - `config_data.yaml`: Dataset and data processing settings
   - `config_experiments.yaml`: Experiment parameter definitions
   - `config_workflow.yaml`: Workflow execution settings
   - `config_visualization.yaml`: Visualization parameters
   - `config_modes.yaml`: Execution mode configurations
   - `config_runtime.yaml`: Runtime execution settings

2. **Organization**:
   - Hierarchical structure with inheritance
   - Override system for flexible configuration
   - Environment-specific adaptations

See the [Configuration Reference](configuration.md) for details.

### Cluster Module

The `cluster` module provides distributed execution capabilities:

1. **Execution Infrastructure**:
   - SLURM integration profiles
   - Distributed execution scripts
   - Job management utilities

2. **Development Tools**:
   - Remote development setup
   - Cluster-specific optimizations
   - Resource monitoring

## Extension Points

DynVision can be extended through several mechanisms:

1. **New Models**: Inherit from base classes in `base` and `model_components`
2. **Custom Components**: Add modules following the component interface patterns
3. **Parameter Sets**: Define new parameter configurations in the `params` module
4. **New Experiments**: Add specialized workflows in `workflow` and parameter sweeps in `configs`
5. **Additional Workflows**: Define new Snakemake rules and execution modes
6. **Visualization Tools**: Implement new analysis capabilities in `visualization`
7. **Loss Functions**: Add new training objectives in the `losses` module
8. **Utility Functions**: Contribute shared functionality to the `utils` module

For implementation details, refer to the [Custom Models Guide](../user-guide/custom-models.md).