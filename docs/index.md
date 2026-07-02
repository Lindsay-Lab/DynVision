---
title:
hide:

  - toc
---

<p align="center">
  <img src="assets/logo.svg" alt="DynVision" class="only-light" style="max-width: 600px; width: 100%;">
</p>

DynVision is a modular toolbox for building and evaluating recurrent convolutional neural networks (RCNNs) with biologically plausible temporal dynamics.
The toolbox provides researchers with a framework to explore how details in the network architecture influence temporal dynamics and shape visual processing, while handling most of the overhead to achieve computational efficiently.
The toolbox provides a collection of biologically-inspired components including:

- realistic lateral recurrent connections
- flexible skip and feedback connections
- activity evolution governed by dynamical systems equations
- unrolling of biological time with heterogenous time delays for different connection types 

## Documentation Structure

Our documentation is organized into four main categories:

### Tutorials

Step-by-step guides for beginners to get started with DynVision.

- [Getting Started](getting-started.md): First steps with DynVision
- [Basic Model Training](tutorial/basic-model-training.md): Train your first model
- [Custom Model Creation](tutorial/custom-model.md): Build your own neural network architecture

### How-to Guides

Task-oriented guides for solving specific problems.

- [Installation](user-guide/installation.md): Detailed installation instructions
- [Custom Models](user-guide/custom-models.md): Define your own neural network architectures
- [Data Processing](user-guide/data-processing.md): Work with different datasets
- [Workflow Management](user-guide/workflows.md): Use Snakemake for experiments
- [Model Testing](user-guide/model-testing.md): Evaluate model performance

### Reference

Technical descriptions of DynVision's components.

- [Organization Overview](reference/organization.md): Structure of the toolbox
- [Model Components API](reference/model-components.md): Core building blocks
- [Recurrence Types](reference/recurrence-types.md): Different recurrent connection implementations
- [Dynamics Solvers](reference/dynamics-solvers.md): ODE solvers for neural dynamics
- [Configuration Reference](reference/configuration.md): Configuration file documentation

### Explanation

Conceptual understanding of DynVision's approach.

- [Biological Plausibility](explanation/biological-plausibility.md): Alignment with neural systems
- [Temporal Dynamics](explanation/temporal_dynamics.md): Understanding temporal properties
- [Design Philosophy](explanation/design-philosophy.md): Core design principles

## Citing DynVision

If you use DynVision in your research, please cite:

> Gutzen, R. & Lindsay, G. (2025). *Modeling Dynamical Vision with Biologically
> Plausible Recurrent Convolutional Networks.* bioRxiv.
> [doi:10.1101/2025.08.11.669756](https://doi.org/10.1101/2025.08.11.669756)

```bibtex
@article{gutzen2025modelingdynamical,
  title   = {Modeling Dynamical Vision with Biologically Plausible
             Recurrent Convolutional Networks},
  author  = {Gutzen, Robin and Lindsay, Grace},
  year    = {2025},
  journal = {bioRxiv},
  doi     = {10.1101/2025.08.11.669756},
  url     = {https://doi.org/10.1101/2025.08.11.669756}
}
```

## Contributing

DynVision is an open-source project, and we welcome contributions! See our [Contributing Guide](contributing.md) for information on how to get involved.

## Getting Support

If you have questions or run into issues:

1. Search the [GitHub Issues](https://github.com/Lindsay-Lab/dynvision/issues) to see if someone has encountered the same problem. Open a new issue if you can't find a solution.
2. Reach out via [Email](mailto:robin.gutzen@nyu.edu).

## License

DynVision is released under the MIT License. See the [LICENSE](https://github.com/Lindsay-Lab/dynvision/blob/main/LICENSE) file for more details.
