# Biological Plausibility in DynVision

This document explores the concept of biological plausibility in DynVision and explains how the toolbox implements features inspired by the primate visual system.

## What is Biological Plausibility?

Biological plausibility refers to the degree to which an artificial neural network reflects properties and constraints of biological neural systems. While standard deep learning models like CNNs share some conceptual similarities with the visual cortex, they often lack many important features of biological neural processing, including:

1. **Recurrent connectivity**: Abundant feedback and lateral connections
2. **Temporal dynamics**: Time-dependent processing with continuous evolution
3. **Noise and stochasticity**: Variability and robustness to perturbations
4. **Metabolic constraints**: Energy efficiency considerations
5. **Topographic organization**: Spatial arrangement of neurons with similar properties
6. **Neural adaptation**: Changes in response properties over time

DynVision focuses on implementing several of these biologically plausible features, particularly recurrent connectivity and temporal dynamics, to create models that better reflect how the brain processes visual information.

## Biological Inspiration in the Visual System

The primate visual system has several key properties that inspire DynVision's architecture:

### Hierarchical Processing

Visual information in primates flows through a hierarchical pathway, often called the ventral visual stream:

1. **Retina**: Initial processing and encoding of visual input
2. **LGN** (Lateral Geniculate Nucleus): Relay and modulation of visual signals
3. **V1** (Primary Visual Cortex): Processing of basic visual features (edges, orientations)
4. **V2** (Secondary Visual Cortex): More complex features and contour integration
5. **V4**: Processing of shape and color information
6. **IT** (Inferotemporal Cortex): Object recognition and categorization

DynVision models like DyRCNNx4 implement this hierarchical structure with corresponding layers (V1, V2, V4, IT).

### Recurrent Connections

The visual cortex contains abundant recurrent connections:

1. **Lateral connections**: Within a single area (e.g., connections between neurons in V1)
2. **Feedback connections**: From higher areas to lower areas (e.g., V4 to V1)
3. **Skip connections**: Between non-adjacent areas (e.g., V1 to V4)

These connections are thought to enable:
- Contextual modulation of neural responses
- Perceptual grouping and figure-ground segregation
- Predictive processing and expectation
- Attentional modulation

DynVision implements various types of recurrent connections to capture these properties.

### Neural Dynamics

Biological neural processing unfolds over time, with:

1. **Continuous dynamics**: Neural activity evolves continuously rather than in discrete steps
2. **Differential propagation delays**: Different types of connections have different delays
3. **Integration and decay**: Neurons integrate inputs over time with characteristic time constants
4. **Temporal response properties**: Adaptation, facilitation, and suppression over time

DynVision models these dynamics using differential equations that govern how neural activity evolves over time.

## Biological Features in DynVision

DynVision implements several key biological features:

### 1. Recurrent Architectures

DynVision provides a range of recurrent connection types, each with different biological inspirations:

- **Self recurrence**: Models persistence of neural activity over time
- **Full recurrence**: Models dense local connectivity within cortical areas
- **Depthwise/Pointwise recurrence**: Models separable processing of spatial and feature information
- **Local recurrence**: Models topographic organization with cortical-like arrangements

These connections allow for context-dependent processing, where neural responses depend not only on the current input but also on the current state of the network.

### 2. Dynamical Systems Implementation

Rather than using discrete-time steps like traditional RNNs, DynVision implements continuous-time dynamics:

```
τ · dx/dt = -x + Φ[f(t, r_n, r_{n-1})]
```

Where:
- τ is the time constant
- x is the neural activity
- Φ is a nonlinearity
- f represents inputs from various sources

This approach:
- Better captures the temporal characteristics of neural responses
- Allows for different time constants in different areas
- Models the differential delays of various connection types

### 3. Biologically Motivated Delays

DynVision models the different delays associated with different connection types:

- **Feedforward delays** (t_feedforward): Time for signals to propagate from one area to the next
- **Recurrent delays** (t_recurrence): Time for signals to propagate within an area

These delays are typically set to biologically plausible values (e.g., t_feedforward = 10ms, t_recurrence = 6ms).

### 4. Supralinear Activation

The `SupraLinearity` module implements a power-law nonlinearity:

```
f(x) = k · sign(x) · |x|^n
```

This models the supralinear response properties observed in cortical neurons, where output increases more rapidly than linearly with input strength, which is important for:

- Allowing weak signals to be amplified by recurrent processing
- Enabling winner-take-all competition between neural populations
- Creating sensitivity to correlated inputs

### 5. Retinal Preprocessing

The `Retina` module models the representational bottleneck at the retina and LGN, which:

- Reduces dimensionality of visual input
- Enhances relevant features while suppressing noise
- Adapts to the statistics of the visual environment

## Experimental Evidence for Biological Plausibility

DynVision models can be evaluated on their ability to reproduce biological phenomena:

### 1. Response Timing

Neurons in different areas of the visual cortex respond with different latencies to visual stimuli. DynVision models this through:

- Hierarchical processing with feedforward delays
- Area-specific time constants
- Recurrent amplification of signals

### 2. Contrast-Dependent Dynamics

In the visual cortex, responses to high-contrast stimuli are both stronger and faster than responses to low-contrast stimuli. DynVision reproduces this through:

- The interaction between feedforward drive and recurrent processing
- Supralinear activation functions
- Neural dynamics with appropriate time constants

### 3. Adaptation Effects

Neural responses adapt over time when presented with sustained or repeated stimuli. DynVision models this through:

- Temporal dynamics with appropriate time constants
- Recurrent connections that can implement both facilitation and suppression
- Optional input adaptation mechanisms

### 4. Subadditive Temporal Summation

Neural responses to longer stimuli do not increase linearly with stimulus duration but instead saturate. DynVision models this through:

- The dynamical systems formulation with appropriate time constants
- Recurrent connections that provide both excitation and inhibition
- Neural responses that reach equilibrium states

## Research Applications

Researchers can use DynVision to test hypotheses about the role of recurrence in visual processing:

- How does recurrence contribute to object recognition?
- What is the functional significance of different recurrence types?
- How do feedback connections modulate feedforward processing?

## Limitations and Future Directions

While DynVision implements several biologically plausible features, many aspects of biological neural processing remain to be incorporated:

### Future Directions

Potential extensions to enhance biological plausibility:

1. **Separation of excitation and inhibition**: Implementing Dale's Law with distinct E and I populations
2. **Local field potentials**: Modeling population activity similar to EEG/MEG signals
3. **Prediction error coding**: Implementing predictive coding principles
4. **Diverse learning rules**: More biologically realistic learning rules
5. **Attention mechanisms**: Incorporating spatial and feature-based attention

## References

1. van Bergen & Kriegeskorte (2020). Going in circles is the way forward: The role of recurrence in visual inference.
2. Kietzmann et al. (2019). Recurrence is required to capture the representational dynamics of the human visual system.
3. Kar et al. (2019). Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior.
4. Groen et al. (2022). Temporal Dynamics of Neural Responses in Human Visual Cortex.
5. Rubin et al. (2015). The stabilized supralinear network: A unifying circuit motif underlying multi-input integration in sensory cortex.
