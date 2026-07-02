# Visualization and Analysis

This guide explains how to visualize and analyze results from DynVision experiments, helping you understand the temporal dynamics and response properties of your models.

## Overview

DynVision provides several visualization tools to analyze model behavior, particularly focusing on temporal dynamics and response properties that reflect biological neural systems. Key visualizations include:

1. **Response Time Courses**: Neural activity over time
2. **Weight Distributions**: Model parameter distributions
3. **Classifier Responses**: Classification behavior over time
4. **Adaptation Analysis**: Neural adaptation to repeated stimuli
5. **Contrast Response**: Response properties to varying contrast levels
6. **Duration Response**: Response properties to varying stimulus durations
7. **Interval Response**: Response properties to varying interstimulus intervals

## Visualization Workflow

DynVision's visualization system is integrated into the Snakemake workflow:

```
dynvision/workflow/
└── snake_visualizations.smk   # Contains visualization rules
```

Visualizations are typically generated automatically when running experiments:

```bash
# Run experiment and generate visualizations
snakemake experiment --config experiment=contrast
```

Results are saved in the reports directory:

```
reports/
└── figures/
    ├── contrast/            # Experiment-specific visualizations
    ├── weight_distributions/ # Weight analysis
    └── classifier_response/  # Classifier behavior
```

## Basic Visualization Commands

### Generate All Visualizations

To generate all visualizations for completed experiments:

```bash
snakemake plot_experiments_on_models
```

### Generate Specific Visualizations

To generate visualizations for a specific experiment:

```bash
# Generate contrast response visualizations
snakemake --config experiment=contrast plot_adaption
```

### Compare Models

To compare different models on the same experiment:

```bash
# Compare different recurrence types
snakemake --config experiment=contrast model_args="{rctype:[full,self,depthpointwise,pointdepthwise]}" plot_experiments_on_models
```

## Visualization Types and Interpretation

### 1. Response Time Courses

These visualizations show the evolution of neural activity over time in response to a stimulus.

**Key Features to Look For**:

- **Response Onset**: When neural activity begins to increase
- **Peak Time**: When activity reaches its maximum
- **Response Duration**: How long activity persists
- **Adaptation**: Whether activity decreases with sustained stimulation

**Interpretation Example**:

When analyzing response time courses, you can observe:

1. The V1 layer typically responds first, followed by V2, V4, and IT layers
2. Higher layers show progressively longer response latencies
3. Full recurrence often shows stronger adaptation than self recurrence

**Generation Command**:

```bash
snakemake --config experiment=response plot_adaption
```

### 2. Weight Distributions

These visualizations show the distribution of weights in different parts of the model.

**Key Features to Look For**:

- **Distribution Shape**: Whether weights follow a Gaussian-like distribution
- **Magnitude**: The range of weight values
- **Layer Differences**: How weight distributions differ across layers
- **Recurrence Weights**: Differences between feedforward and recurrent weights

The revised `plot_weight_distributions.py` script can aggregate checkpoints from
multiple seeds and compare groups by mapping dimensions to subplot rows,
columns, and violin hues. Use `--row`, `--column`, and `--hue` to choose between
`category` (e.g. recurrence type), `connection_type`, `status`, and `layer`;
provide `--category-key` when you need to extract grouping information from
checkpoint paths.

**Interpretation Example**:

When analyzing weight distributions, you can observe:

1. Feedforward weights often show broader distributions than recurrent weights
2. Higher layers may have slightly larger weight magnitudes than lower layers
3. Models may learn sparse connectivity patterns in recurrent connections

**Generation Command**:

```bash
snakemake plot_weight_distributions --config model_name=DyRCNNx4 model_args="{rctype:full}"
```

### 3. Contrast Response

These visualizations show how neural activity changes with stimulus contrast.

**Key Features to Look For**:

- **Response Magnitude**: How activity increases with contrast
- **Response Speed**: Whether high-contrast stimuli elicit faster responses
- **Layer Differences**: How contrast sensitivity differs across layers
- **Recurrence Effects**: How different recurrence types affect contrast sensitivity

**Interpretation Example**:

When analyzing contrast responses, you can observe:

1. Response magnitude typically increases with contrast in all layers
2. Higher layers (V4, IT) may show stronger contrast dependence than lower layers (V1, V2)
3. Full recurrence often shows stronger contrast response than self recurrence
4. High-contrast stimuli may elicit responses with shorter latencies

**Generation Command**:

```bash
snakemake --config experiment=contrast plot_adaption
```

### 4. Duration Response

These visualizations show how neural activity changes with stimulus duration.

**Key Features to Look For**:

- **Temporal Summation**: How activity accumulates with longer stimuli
- **Saturation**: Whether responses saturate with long stimuli (subadditive temporal summation)
- **Offset Response**: Whether there's a response when the stimulus ends
- **Layer Differences**: How temporal integration differs across layers

**Interpretation Example**:

When analyzing duration responses, you can observe:

1. Response magnitude may increase with stimulus duration but saturate (subadditive temporal summation)
2. Higher layers (V4, IT) may show stronger saturation than lower layers (V1, V2)
3. Full recurrence often shows stronger sustained activity than self recurrence
4. Some layers may show an offset response when the stimulus ends

**Generation Command**:

```bash
snakemake --config experiment=duration plot_adaption
```

### 5. Interval Response

These visualizations show how neural activity responds to repeated stimuli with varying intervals.

**Key Features to Look For**:

- **Repetition Suppression**: Whether the second response is weaker than the first
- **Recovery**: How the second response recovers with longer intervals
- **Layer Differences**: How adaptation differs across layers
- **Recurrence Effects**: How different recurrence types affect adaptation

**Interpretation Example**:

When analyzing interval responses, you can observe:

1. The response to the second stimulus is weaker than the first (repetition suppression)
2. Longer intervals lead to stronger recovery of the second response
3. Higher layers (V4, IT) show stronger adaptation than lower layers (V1, V2)
4. Full recurrence (blue) shows stronger adaptation effects than self recurrence (orange)

**Generation Command**:

```bash
snakemake --config experiment=interval plot_adaption
```

## Manual Visualization with Python

You can also create custom visualizations in Python:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dynvision.mode