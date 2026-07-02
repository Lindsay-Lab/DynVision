# Installation Guide

!!! goal "Goal"
    Get a working DynVision environment with all dependencies installed, so you
    can train and evaluate models.

This guide provides detailed instructions for installing DynVision and its dependencies on different operating systems.

## Prerequisites

Before installing DynVision, make sure you have the following:

- **Python 3.11 or higher**: DynVision requires Python 3.11+ for compatibility with all dependencies.
- **CUDA-compatible GPU** (recommended): While DynVision can run on CPU, a CUDA-compatible GPU is highly recommended for training models efficiently.
- **Git**: Required for cloning the repository.
- **Mamba/Conda**: Recommended for managing dependencies ([Miniforge](https://github.com/conda-forge/miniforge)).
- **Linux OS** (recommended): DynVision may also be compatible with other operation systems but is only tested and documented for Linux distributions.

## Basic Installation

### 1. Environment Setup

First, create and activate a conda or mamba environment. Mamba is a faster alternative to conda and is now the recommended environment handler. The mamba and conda commands are interchangeable.
See [https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html] for a guide how to install them.

**Using Mamba/Conda:**

```bash
# Create environment
mamba create -n dynvision python=3.11
mamba activate dynvision
```

### 2. Installing Dependencies

Most dependencies are automatically installed with the installation of DynVision, but some packages may require additional installation steps depending on the system.

**Install CUDA dependencies**
```bash
mamba install -c conda-forge cudatoolkit=11.3

# Install PyTorch with CUDA support (adjust CUDA version as needed)
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Install FFCV for optimized data loading**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y pkg-config libturbojpeg-dev

# Install FFCV
pip install ffcv
```

**Install Snakemake for workflow management**
```bash
# Install Snakemake
mamba install -c bioconda -c conda-forge snakemake

# Install Graphviz for workflow visualization
mamba install -c conda-forge graphviz
```

### 3. Install DynVision

**From PyPI (recommended for most users):**

```bash
pip install dynvision
```

**From source (for development):**

```bash
# Clone repository
git clone https://github.com/Lindsay-Lab/dynvision.git
cd DynVision

# Install package in development mode
pip install -e .
```

The `-e` flag installs the package in development mode, which means you can modify the code without needing to reinstall.

## Installing on a Cluster

For installing on an HPC cluster, you may need to use environment modules, docker, or singularity images. Please follow the instructions of the HPC documentation.

Note that if you want use of the feature that the snakemake workflow can submit jobs on its own to aid with parallelization and automatization, and your are using docker or singularity images, you might need another base environment that has snakemake installed and access to slurm commands (i.e. `srun` and `sbatch`).

For more details on cluster execution, see [user-guide/cluster-execution.md].

## Verifying Installation

After installation, you can verify that DynVision works correctly:

```bash
# Activate the environment if not already activated
conda activate dynvision

# Run a simple test
python -c "from dynvision.models import DyRCNNx4; model = DyRCNNx4(); print('DynVision successfully installed!')"
```

## Installing Optional Dependencies

### For Logging and Monitoring

Setup an account at [Weights & Biases](https://wandb.ai) and initialize a new project to track your training runs.

1. **Install Weights & Biases**:
    Install the `wandb` library using pip:

    ```bash
    pip install wandb  # For experiment tracking
    ```

2. **Login to Weights & Biases**:
    Authenticate your local environment with your Weights & Biases account:

    ```bash
    wandb login
    ```

    Follow the instructions to copy and paste your API key from the Weights & Biases website.

    Also make sure wandb is activated and syncing with the online dashboard:

    ```bash
    wandb enabled
    wandb online
    ```

3. **Setting a new project**:
    In `dynvision/project_paths.py`, make sure you have set the `project_name` to the name of your project.


For more details, refer to the [Weights & Biases Documentation](https://docs.wandb.ai/).


### For Development

```bash
pip install black isort flake8 mypy pytest  # Development tools
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues with PyTorch installation:

1. Visit the [official PyTorch installation page](https://pytorch.org/get-started/locally/)
2. Select your preferences (OS, package manager, compute platform)
3. Use the provided command to install PyTorch

### CUDA Compatibility

Make sure your CUDA drivers are compatible with the PyTorch version:

```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch version
# For CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Missing Python Modules

In case of package or dependency issues, it can help to instead install the package that cause the environment error with `mamba install pkg_name`, comment it out in the pyproject.toml, and run `pip install -e .` from the `DynVision` directory again.

## Next Steps

After installation, continue with the [Getting Started Guide](../getting-started.md) to learn how to use DynVision.
