# Design Philosophy

This document explains the core design principles and architectural decisions that guided the development of DynVision. Understanding these principles will help you better utilize the toolbox and contribute to its development.

## Bridging Computational Neuroscience and Deep Learning

DynVision was designed to bridge two distinct but complementary approaches to understanding visual processing:

1. **Computational Neuroscience**: Focused on building biologically accurate models of neural systems, often prioritizing mechanistic explanations over task performance.

2. **Deep Learning**: Focused on building high-performance vision models, often prioritizing task performance over biological plausibility.

DynVision aims to create a framework where researchers can explore the middle ground between these approaches by:

- Starting with deep learning architectures (CNNs) that have demonstrated functional similarities to biological visual systems
- Adding biologically plausible temporal dynamics via recurrent connections
- Implementing continuous-time processing that better reflects neural computation
- Maintaining sufficient computational efficiency to train on meaningful visual tasks

## Core Design Principles

### 1. Modularity and Composability

DynVision is designed around the principle of modularity, where components can be mixed and matched to create different model architectures:

- **Layer-Operation Architecture**: Models are defined by a sequence of layers, and each layer contains a sequence of operations. This two-tier organization allows for flexible arrangement of computational components.

- **Interchangeable Components**: Recurrence types, dynamics solvers, connectivity patterns, and other components can be swapped without changing the overall model architecture.

- **Common Interfaces**: All components adhere to consistent interfaces, making it easy to combine them in novel ways or replace them with custom implementations.

This modularity enables researchers to conduct controlled experiments where specific components are varied while others are held constant, facilitating systematic exploration of the design space.

### 2. Biological Plausibility

DynVision prioritizes biological plausibility in several key aspects:

- **Continuous Dynamics**: Neural activity evolves according to differential equations rather than discrete updates, better reflecting the continuous nature of biological processing.

- **Temporal Delays**: Heterogeneous delays for different connection types (feedforward, recurrent, feedback) reflect the different conduction velocities and distances in neural circuits.

- **Topographic Organization**: Support for spatial organization of features that reflects the topographic arrangement of neural populations in the visual cortex.

- **Metabolic Constraints**: Activity losses and regularization terms that encourage metabolically efficient coding.

Importantly, biological plausibility is balanced with computational tractability, focusing on aspects that are likely to have functional significance rather than implementing every biological detail.

### 3. Experimental Reproducibility

Scientific reproducibility is a core design consideration:

- **Configuration-Driven**: Experiments are defined by configuration files rather than code changes, ensuring that all parameters are explicitly captured.

- **Workflow Management**: Integration with Snakemake for reproducible execution of entire experimental pipelines.

- **Deterministic Execution**: Support for fixed random seeds and controlled initialization to ensure result reproducibility.

- **Comprehensive Logging**: Detailed logging of parameters, model configurations, and training dynamics.

This emphasis on reproducibility aligns with best practices in computational science and facilitates collaboration between research groups.

### 4. Performance Optimizations

While prioritizing biological plausibility, DynVision incorporates numerous performance optimizations:

- **FFCV Integration**: Fast data loading using the FFCV library to minimize I/O bottlenecks.

- **Memory Efficiency**: Careful management of activation histories to balance biological fidelity with memory constraints.

- **Mixed Precision**: Support for float16 computation to accelerate training on modern GPUs.

- **Efficient Recurrent Operations**: Optimized implementations of recurrent connection patterns to minimize computational overhead.

These optimizations enable research on larger datasets and more complex models than would otherwise be feasible.

### 5. Adaptability for Research

DynVision is fundamentally a research tool, designed to be extended and modified:

- **Configurable Abstractions**: High-level abstractions with sensible defaults but extensive configurability.

- **Explicit Design Decisions**: Clear separation between scientific modeling choices and engineering implementations.

- **Instrumentation**: Built-in tools for analyzing and visualizing model behavior at multiple levels of detail.

- **Extension Points**: Well-defined interfaces for adding new components, models, and analysis methods.

The toolbox aims to lower the barrier to entry for researchers while still providing the flexibility needed for cutting-edge research.

## Architectural Decisions

Several key architectural decisions embody these design principles:

### Dynamics Formulation

DynVision uses a continuous-time dynamics formulation:

```
r_n(t) = r_n(t - dt) + (dt/τ) * (-r_n(t - dt) + Φ[f(t - dt, r_n, r_n-1)])
```

where:

- `r_n(t)` is the activity of layer n at time t
- `dt` is the time step size
- `τ` is the time constant
- `Φ` is a nonlinearity
- `f` combines feedforward, recurrent, and external inputs

This formulation allows for separate control of:

- Temporal precision (`dt`)
- Integration time constants (`τ`)
- Delays for different connection types

### Layer Organization

Models are organized into layers that correspond to regions in the visual cortex (e.g., V1, V2, V4, IT), with each layer containing:

1. **Feedforward connections** from previous layers
2. **Recurrent connections** within the layer
3. **Skip connections** that bypass certain layers
4. **Feedback connections** from higher layers

This organization facilitates comparison with neural recordings from corresponding brain regions.

### Signal Propagation Model

Signal propagation follows a 'biological unrolling of time' approach where:

- Signals propagate through the network with explicit delays
- The simulation time is extended beyond input presentation to allow signals to reach all layers
- Response timing emerges naturally from the network structure rather than being artificially synchronized

This approach produces temporally heterogeneous responses similar to those observed in biological visual systems.

## Relation to Other Frameworks

DynVision's design philosophy can be understood in relation to other frameworks:

- **vs. PyTorch/TensorFlow**: Adds biological temporal dynamics and recurrent processing on top of these general-purpose deep learning frameworks.

- **vs. BRIAN/NEST**: More focused on high-level visual processing and less on detailed neuron models, with greater emphasis on learning and task performance.

- **vs. CORnet/BrainScore**: Extends these brain-score optimized models with explicit temporal dynamics and recurrent processing.

- **vs. PsychRNN/TensorFlow RNN**: More focused on the specific constraints of visual cortex rather than general RNN architectures.

DynVision occupies a unique position, focusing on visual processing with biologically plausible dynamics at a level of abstraction that enables both computational efficiency and meaningful comparison with neural data.

## Practical Implications

The design philosophy of DynVision has several practical implications for users:

1. **Configuration over Code**: Most customizations should be made through configuration rather than code changes, leveraging the modular architecture.

2. **Biological Validation**: Models should be evaluated not just on task performance but also on the plausibility of their temporal dynamics.

3. **Computational Resources**: While optimized, biologically plausible models may require more computational resources than their non-recurrent counterparts.

4. **Iterative Exploration**: The toolbox is designed to support iterative exploration of model variants through its workflow management capabilities.

By understanding these design principles, researchers can more effectively leverage DynVision to explore the intersection of biological visual processing and deep learning.