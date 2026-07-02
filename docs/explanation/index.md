---
title: Explanation
---

This section provides conceptual discussions and explanations of the ideas behind DynVision. Here you'll find in-depth coverage of the theoretical foundations, design principles, and biological inspirations that inform the toolbox.

These pages are **understanding-oriented**: read them to learn *why* DynVision
works the way it does. For practical steps see the
[How-to Guides](../user-guide/index.md); for factual lookups see the
[Reference](../reference/index.md).

## Core Concepts

- [**Biological Plausibility**](biological-plausibility.md): How DynVision implements biologically plausible features
- [**Temporal Dynamics**](temporal_dynamics.md): Understanding temporal processing in vision models
- [**Engineering vs. Biological Time**](engineering-vs-biological-time.md): The two unrolling conventions and delay‑conversion formulas
- [**Design Philosophy**](design-philosophy.md): The guiding principles behind DynVision's architecture

## Recurrent Processing

- [**Role of Recurrence**](role-of-recurrence.md): Why recurrent connections matter in visual processing
- [**Comparison to Neural Data**](comparison-to-neural-data.md): How model dynamics compare to ECoG recordings and human behavioural data
- [**Why Snakemake?**](why-snakemake.md): The reasoning behind DynVision's workflow orchestration

## Planned topics

The following conceptual pages are planned but not yet written. They are tracked
in the [Documentation TODOs](../development/planning/todo-docs.md) and will be
added on demand:

- Continuous vs. discrete time, time constants, propagation delays
- Visual-cortex organization & cortical connectivity
- Comparisons with standard CNNs, other RCNNs, and spiking networks
- Trade-offs in balancing biological fidelity and performance
