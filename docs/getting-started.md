# Getting Started

This guide will walk you through the basic steps to get started, including installation, running a basic training-testing workflow, and working with a custom model.

## System Requirements

- **Python**: 3.11 or higher
- **CUDA-compatible GPU** (optional): For efficient model training
- **RAM**: 16GB minimum, 32GB recommended for larger experiments
- **Storage**: At least 50GB free space for datasets and experiment results
- **Operating System**: Linux (recommended), macOS, or Windows
- **Build Tools**: Git, CMake, and a C++ compiler

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/Lindsay-Lab/dynvision.git

# Create and activate mamba/conda environment
mamba create -n dynvision python=3.11
mamba activate dynvision

# Install PyTorch with CUDA support
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install DynVision in an editable mode
cd dynvision
pip install -e .
```

For detailed installation instructions, including OS-specific steps and troubleshooting, see the [Installation Guide](user-guide/installation.md).

## Project Setup

### Configure Project Paths

Edit `dynvision/project_paths.py` to set the proper paths to the toolbox directory (for the codebase) and optionally to a separate working directory (for data, outputs, logs). You may also set specific paths (e.g. for large data directories) that dynamically adapt when you are executing the toolbox on a compute cluster.

### Data Management

DynVision automatically handles dataset downloads and preparation. Standard datasets (CIFAR10, CIFAR100, MNIST) are automatically downloaded when required for the first time. All data segmentation and selection of subset is symlinking without data duplication, following the settings in `dynvision/configs/config_data.yaml`.

To trigger data preparation you can run:
```bash
# Download and prepare CIFAR10 dataset
snakemake <project_paths.data.interim>/cifar10/train_all/folder.link
```

For more information about data organization and custom datasets, see the [Data Processing Guide](user-guide/data-processing.md).

## Basic Usage

### 1. Simple Model Example

Here's a minimal example using DynVision model in a custom script:

```python
import torch
from dynvision.models import DyRCNNx4

# Create a variation of AlexNet with recurrency and activity evolution over time
model = DyRCNNx4(
    n_classes=10,                 # Number of output classes
    input_dims=(20, 3, 64, 64),   # (timesteps, channels, height, width)
    recurrence_type="full",       # Full recurrent connectivity
    dt=2,                         # time step (ms)
    tau=8                         # neural time constant (ms)
)

# Create a random input batch
batch = torch.randn(2, 20, 3, 64, 64)  # (batch, timesteps, channels, height, width)
outputs = model(batch)
print(f"Output shape: {outputs.shape}")  # [2, 20, 10]
```

### 2. Running an Experiment with Snakemake

DynVision uses [Snakemake](https://snakemake.readthedocs.io/) for workflow management.
Note that Snakemake commands must be called from within the workflow folder.
Here's how to run a basic experiment:

```bash
cd dynvision/workflow

# Train a model on CIFAR100 and evaluate contrast sensitivity
snakemake --config experiment=duration model_name=AlexNet model_args='{tsteps: 20, rctype: full}' data_name=cifar100 data_group=all
```

This command:
1. Initializes an AlexNet version with full recurrence
2. Downloads and prepares the CIFAR100 dataset if needed
3. Trains the model using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
4. Tests the model on input streams with the image shown for a varying number of timesteps within the 20 timestep simulation 
5. Stores the test results and classifier responses in a [pandas](https://pandas.pydata.org/docs/) DataFrame format in a csv file.

See the [Workflow Guide](user-guide/workflows.md) for more details.

### 3. Editing Configurations

Any workflow executed with snakemake reads the configuration from the yaml files `dynvision/configs/`. The files are structured into domains: `config_defaults.yaml`, `config_data.yaml`, `config_experiments.yaml`, and `config_visualization.yaml`. Additionally, there is `config_workflow.yaml` which may overwrite and specify parameters for a given workflow. Setting a single-valued parameter with a list, generally causes the workflow to scan over all list values, e.g. setting `model_name=['AlexNet', 'CorNetRT', 'DyRCNNx4']` and `recurrence_type=['full', 'self', 'pointdepthwise']` produces 9 models, 3 versions with different recurrence types of the 3 models. Superseeding also the workflow config file, config values can also be specified in command line calls with the `--config` flag as in the example in **2.** above.
The complete configuration settings for a specific workflow call are temporarily stored during runtime in `dynvision/configs/config_runtime.yaml` and also printed in the corresponding log files.

For more details see the [Configuration Guide](reference/configuration.md).

### 4. Analyzing Test Results

During training, validation, and testing, the unit responses of all layers can be recorded. To control for how many samples the responses during testing are recorded edit in`configs_defaults.yaml` the setting `store_test_responses: 100`.
The responses of all layers are stored in the dictionary model attribute `model.responses[layer_name]` as tensors.
Additionally, you can extract the activations of the classifying layer combined with the testing results in a handy dataframe.

```python
df = model.get_dataframe()
```

The test results are stored in a pandas DataFrame with the following columns:

- **`sample_index`**: The unique identifier for each sample in the dataset.
- **`times_index`**: Represents the time step of the input data.
- **`class_index`**: The unit index of the classifying layer corresponding to the possible label indices.
- **`response`**: The activations of the corresponding classifier unit output or prediction for the given input. 
- **`label_index`**: The index of the true label assigned to the sample.
- **`guess_index`**: The index of the predicted class with the highest score.
- **`image_index`**: A reference to the specific image of the label class in the dataset.
- **`label_set`**: The string representation of input over all timesteps (e.g. "-1-1-1-1202020202020" indicating null input (label_index=-1) for 4 timesteps and 6 timesteps of an image with label index 20).

For more details on analyzing test results, see the [Evaluation Guide](user-guide/evaluation.md).

## Understanding DynVision

### Key Modelling Concepts

1. **Explicit Time Dimension**: DynVision models work with sequences where time is an explicit dimension after the batch dimension: `(batch, timesteps, channels, height, width)`.

2. **Biological Time Unrolling**: Models simulate biological neural dynamics with configurable time steps and delays:
   - Integration time step (`dt`)
   - Neural time constant (`tau`)
   - Feedforward and recurrent delays (`t_feedforward`, `t_recurrent`)
   - Handling of residual timesteps (timesteps needed for the first input to reach the last model layer)

3. **Modular Components**: The framework provides a collection of interoperable components to assist model development, and features mechanism to flexibly rearrange their execution order, e.g.:
    ```python
    layer_operations=[
        "layer",        # apply (recurrent) convolutional layer
        "addskip",      # add skip connection activity
        "addfeedback",  # add feedback connection activity
        "tstep",        # apply dynamical systems ODE solver step
        "nonlin",       # apply nonlinearity
        "record",       # record activations in responses dict
        "delay",        # set and get delayed activations for next layer
        "pool",         # apply pooling
    ]
    ```

### Namespace Conventions

The state_dict of models, i.e. the values of all their parameters are stored in files with the following namespace:
```python
"{model_name}{model_args}_{seed}_{data_name}_{status}.pt"
```
The recorded unit activations during testing are stored in files with the following namespace:
```python
"{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt"
```
(the corresponding test outputs as dataframe have the same name but with ending in `test_outputs.csv`)

- `model_name` is the name of the model class defined in `dynvision/models/`
- `model_args` is the specification of all non-default model parameters beginning with a `:` and followed by key-value pairs separated by a `=`, and pairs separated by a `+`, e.g. `AlexNet:tsteps=20+rctype=full+tau=5.5`. `model_args` may also be an empty string.
- `seed` is a typically 4 digit number setting the random seed of the weight initialization and can also be used as model version number
- `data_name` is the lowercase name of the dataset used for training
- `status` is either `init` or `trained` before and after training
- `data_loader` is the name of the dataloader class defined in `dynvision/data/dataloader.py` (omitted the `DataLoader`), e.g. `StimulusDuration`
- `data_args`, just as the model_args are the non-default parameters to the dataloader class
- `data_group` is the subgroup of classification classes used for testing, and is either `all` or defined in `dynvision/configs/config_data.yaml

**Note:** to keep such file names short and the model_args and data_args without underscore, the class __init__ functions typically define alias keywords for longer parameter names such as `rctype ->  recurrence_type` and `tff -> t_feedforward`. See the [Configuration page](reference/configuration.md) for a full list of keyword aliases.

### Core Technologies

DynVision leverages several powerful tools:
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training management
- [FFCV](https://ffcv.io/) for optimized data loading
- [Snakemake](https://snakemake.readthedocs.io/) for workflow orchestration
- [Weight&Biases](https://wandb.ai) for logging and monitoring training parameters
- [Pandas](https://pandas.pydata.org/docs/) for results formatting and storage

## Next Steps

1. [**Explore Model Components**](reference/index.md): 
    - [Recurrence Types](reference/recurrence-types.md)
    - [Model Architectures](reference/model-components.md)

2. [**Try Tutorials**](tutorial/index.md):
   - [Basic Model Training](tutorial/basic-model-training.md)
   - [Creating Custom Models](tutorial/custom-model.md)

3. [**Understand the Framework**](explanation/index.md):
   - [Design Philosophy](explanation/design-philosophy.md)
   - [Temporal Dynamics](explanation/temporal_dynamics.md)
   - [Biological Plausibility](explanation/biological-plausibility.md)

4. [**Advanced Topics**](user-guide/index.md):
   - [Custom Data Processing](user-guide/data-processing.md)
   - [Model Evaluation](user-guide/evaluation.md)
   - [Result Visualization](user-guide/visualization.md)
   - [Workflow Management](user-guide/workflows.md)

## Getting Help

- File issues on GitHub
