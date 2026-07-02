# The Role of Recurrence in Visual Processing

!!! note "Understanding-oriented"
    This page discusses *why* recurrent connections matter for modeling the
    ventral visual stream. For the concrete connection types DynVision
    implements, see the
    [Recurrence Types reference](../reference/recurrence-types.md).

## Feedforward models are not enough

Standard convolutional networks are purely feedforward, yet the primate visual
system is densely recurrent. Lateral and feedback connections are thought to
support contextual modulation, figure–ground segregation, predictive
processing, and robust recognition under challenging conditions (occlusion,
noise, low contrast).

Empirically, recurrence has been shown to be necessary to capture the
representational *dynamics* of the human visual system (Kietzmann et al., 2019)
and to explain behavior on harder recognition tasks where feedforward models
fall short (Kar et al., 2019).

## What recurrence buys a model

DynVision lets researchers study several distinct functional contributions of
recurrence:

- **Temporal normalization** — recurrent inhibition can reproduce adaptation,
  sublinear temporal summation, and contrast-dependent response timing without
  explicit divisive-normalization operators.

- **Noise robustness** — recurrent dynamics can stabilize representations under
  input perturbations, approaching human-level robustness in some regimes.

- **Iterative inference** — recurrence allows the network to refine its estimate
  over biological time, rather than committing to a single feedforward pass.

A key empirical finding from DynVision is that these roles can *dissociate*:
different architectural placements of recurrence and different metabolic-cost
weightings push the network into qualitatively different dynamic regimes.

## Connections in DynVision

| Connection type | Biological analogue |
|-----------------|---------------------|
| Self / full lateral recurrence | Horizontal connections within a cortical area |
| Feedback | Top-down projections from higher to lower areas |
| Skip | Long-range projections between non-adjacent areas |

## Weight Distribution Insights

Analysis of learned weights reveals systematic patterns across recurrence types
that explain the two functional regimes:

<p align="center">
  <img src="../../assets/weights_rctype.png" alt="Weight distributions by recurrence type" width="700"/>
</p>

*Figure: Recurrent weight distributions for models trained with different
connection types. Self‑recurrence produces strongly negative (inhibitory) weights;
local recurrence produces a mix of negative values; full and sparser types remain
near zero. Feedforward weight distributions (not shown) are consistent across all
variants.*

<p align="center">
  <img src="../../assets/weights_activityloss_pattern=1.png" alt="Weight distributions by activity loss weight" width="700"/>
</p>

*Figure: Feedforward and recurrent weight distributions across activity‑loss
weighting. Recurrent weights remain consistently small and negative across all
regularisation strengths, while feedforward weights are unaffected.*

<p align="center">
  <img src="../../assets/weights_activityloss_pattern=1011.png" alt="Weight distributions with alternating presentation pattern" width="700"/>
</p>

*Figure: Same weight analysis but with the `1011` alternating presentation
pattern, confirming that the inhibitory‑weight regime is robust to the
temporal structure of input presentation.*

<p align="center">
  <img src="../../assets/weights_feedback.png" alt="Feedback weight distributions" width="700"/>
</p>

*Figure: Feedback weight distributions across different feedback‑connection
configurations. Multiplicative feedback shows marginally different
distributions compared to additive, but no systematic noise‑robustness
benefit across seeds.*

<p align="center">
  <img src="../../assets/weights_rctarget.png" alt="Weight distributions by recurrence target" width="700"/>
</p>

*Figure: Recurrent weight distributions for models trained with different
recurrence targets (input, middle, output). Middle‑target training produces
qualitatively different weight structures consistent with the noise‑robustness
regime.*

## Response Profiles

DynVision can capture detailed temporal response profiles that reveal how
different architectural choices shape neural dynamics:

<p align="center">
  <img src="../../assets/responses_tripytch_rctype_rctarget_seed.png" alt="Response profiles across recurrence configs and seeds" width="700"/>
</p>

*Figure: Per‑layer response trajectories across recurrence types, recurrence
targets, and random seeds. The three‑panel layout (type × target × seed)
exposes consistent response‑shaping effects of recurrence independent of
initialization variability.*

<p align="center">
  <img src="../../assets/responses_tripytch_pattern_skip_feedback.png" alt="Response profiles with skip and feedback connections" width="700"/>
</p>

*Figure: Response trajectories for models with skip and feedback connections
enabled. The additional connectivity pathways produce richer temporal dynamics
that propagate earlier and later than in purely feedforward networks.*

## Performance and Stability

<p align="center">
  <img src="../../assets/performance_feedback.png" alt="Performance with feedback connections" width="700"/>
</p>

*Figure: Accuracy across model variants with different feedback configurations.
Feedback connections provide no systematic noise‑robustness benefit across
seeds; multiplicative integration is only marginally preferable to additive.*

<p align="center">
  <img src="../../assets/performance_rctype_rctarget=middle.png" alt="Performance by recurrence type, middle-target" width="700"/>
</p>

*Figure: Accuracy comparison across recurrence types when recurrence targets the
middle layer computation. Full recurrence consistently enables the strongest
noise‑robustness effect.*

<p align="center">
  <img src="../../assets/stability_rctarget.png" alt="Training stability by recurrence target" width="700"/>
</p>

*Figure: Training stability analysis by recurrence target. The middle‑target
configuration shows the most stable convergence across seeds while producing
the strongest noise‑robustness gains.*

## See also

- Reference: [Recurrence Types](../reference/recurrence-types.md)
- Explanation: [Temporal Dynamics](temporal_dynamics.md)
- Explanation: [Biological Plausibility](biological-plausibility.md)
