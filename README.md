<p align="center">
  <img src="docs/assets/logo_white.svg" alt="DynVision Logo" width="400"/>
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![PyPI](https://img.shields.io/pypi/v/dynvision)](https://pypi.org/project/dynvision/)
[![CI](https://github.com/Lindsay-Lab/DynVision/actions/workflows/test.yml/badge.svg)](https://github.com/Lindsay-Lab/DynVision/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Preprint](https://badges.ws/badge/Preprint-10.1101%2F2025.08.11.669756%20-B31B1B)](https://doi.org/10.1101/2025.08.11.669756)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-brightgreen)](https://lindsay-lab.github.io/DynVision/)
<!-- Zenodo DOI badge: replace XXXXXXXX after first GitHub release triggers archival -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX) -->

DynVision is a modular toolbox for constructing and evaluating recurrent convolutional neural networks (RCNNs) with biologically inspired dynamics. It provides a flexible framework for exploring how recurrent connections and temporal dynamics shape visual processing in artificial neural networks and how these networks can be aligned with properties of biological visual systems.

<p align="center">
  <img src="docs/assets/overview.png" alt="DynVision Approach Overview" width="600"/>
</p>

## Key Features

- **Biologically Plausible Dynamics**: Implement neural dynamics governed by continuous differential equations with realistic time constants and delays
- **Diverse Recurrent Architectures**: Explore various recurrent connection types (self, full, depthwise, pointwise, local topographic)
- **Minimal Coding Requirements**: Customization of models, training hyperparameters, testing scenarios, data selection, parameter sweeps, visualizations can be achieved by editing by human-readable config files. For more elaborate extensions there are template files and guides.
- **Modular Components**: Easily combine and reconfigure biologically-inspired features:
  - Recurrent processing within and across areas
  - Skip and feedback connections
  - Retinal preprocessing
  - Supralinear activation
  - Adaptive input gain
- **Modular Operation Order**: Easily rearrange the execution order of layer operations like convolution, adding recurrence, applying delays, nonlinearity, pooling, recording activity, etc.
- **Efficient Workflow Management**: Leverages Snakemake for reproducible experiments and parameter sweeps
- **PyTorch Lightning Integration**: Standardized training with minimal boilerplate
- **Optimized Performance**: Fast data loading with FFCV, GPU acceleration, mixed precision
- **Pre-built Model Zoo**: Access pre-implemented architectures like AlexNet, CorNetRT, ResNet variants, CordsNet, and DyRCNNx4/8

<p align="center">
  <img src="docs/assets/rcnn_architecture.png" alt="DynVision Architecture" width="800"/>
</p>

## Installation

**Python requirement**: Python 3.11+ (3.12 support pending FFCV compatibility testing — see [Python 3.12 Compatibility](docs/development/planning/python-3.12-compatibility.md)).

```bash
# Install from PyPI (recommended for most users)
pip install dynvision
```

**From source (for development):**

```bash
# Clone repository
git clone https://github.com/Lindsay-Lab/dynvision.git
cd dynvision

# Create conda environment (recommended)
conda create -n dynvision python=3.11
conda activate dynvision

# Install PyTorch with CUDA (adjust as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install DynVision in editable mode
pip install -e .
```

For more detailed installation instructions, see the [Installation Guide](docs/user-guide/installation.md).

## Quick Start

```python
import torch
from dynvision.models import DyRCNNx4

# Create a 4-layer RCNN with recurrent connections
model = DyRCNNx4(
    n_classes=10,
    input_dims=(20, 3, 224, 224),  # (timesteps, channels, height, width)
    recurrence_type="full",        # Full recurrent connectivity
    dt=2,                          # Integration time step (ms)
    tau=5,                         # Neural time constant (ms)
    t_feedforward=0,               # feedforward delay (ms)
    t_recurrence=6,                # recurrence delay (ms)
)

# Forward pass with a batch of inputs
batch = torch.randn(1, 20, 3, 224, 224)  # (batch, timesteps, channels, height, width)
outputs = model(batch)
```

For a step-by-step tutorial, see the [Getting Started](docs/getting-started.md) guide.

## Example Experiments

DynVision includes pre-configured experiments to explore temporal response properties of different recurrent architectures:

```bash
# Train and run contrast response experiment with on multiple models
snakemake --config experiment=contrast model_name=['AlexNet', 'ResNet18', 'CorNetRT'] data_name=cifar100

# Evaluate stimulus duration effects with different recurrence types
snakemake -j4 --config experiment=duration model_name=DyRCNNx4 model_args="{rctype: [full, self, pointdepthwise]}"
```

<p align="center">
  <img src="docs/assets/performance_rctarget.png" alt="Noise robustness by recurrence target" width="600"/>
</p>

## Documentation

- [Getting Started](docs/getting-started.md): Beginner's tutorial
- [User Guide](docs/user-guide/index.md): How-to guides for common tasks
- [API Reference](docs/reference/index.md): Technical documentation
- [Explanation](docs/explanation/index.md): Understanding core concepts
- [Contributing](docs/contributing.md): How to contribute to the project
- [Developer Guide](docs/development/index.md): Resources for contributors and AI assistants

> **For Claude Code Users**: The comprehensive developer guide (formerly `CLAUDE.md` in project root) is now at [`docs/development/guides/claude-guide.md`](docs/development/guides/claude-guide.md).

## Citation

If you use DynVision in your research, please cite both the software and the preprint:

```bibtex
@article{Gutzen2025_2025.08.11.669756,
  title        = {Modeling Dynamical Vision with Biologically Plausible Recurrent Convolutional Networks},
  shorttitle   = {{{Modeling Dynamical Vision}}},
  author       = {Gutzen, Robin and Lindsay, Grace W.},
  year         = 2025,
  pages        = {2025.08.11.669756},
  publisher    = {bioRxiv},
  doi          = {10.1101/2025.08.11.669756},
}
```
<!-- @software{Gutzen_DynVision,
  author       = {Gutzen, Robin and Lindsay, Grace W.},
  title        = {DynVision: A Modeling Toolbox for Biologically Plausible
                   Recurrent Visual Networks},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXXX},
  url          = {https://github.com/Lindsay-Lab/DynVision},
  note         = {Replace XXXXXXXX with DOI from https://zenodo.org after first release}
} -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
