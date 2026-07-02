# Benchmarking & Computational Performance

This reference documents the computational cost of different modeling choices
and provides the default training configuration from the published DynVision
manuscript. All values are sourced from the paper's Table 1 (training
configuration) and Table 2 (computational resource demands).

## Default Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Temporal Dynamics** | | |
| Time steps | 20 | Simulation timesteps |
| Resolution ($dt$) | 2 ms | Integration time step |
| Time constant ($\tau$) | 5 ms | Neural time constant |
| $\Delta_{FF}$ | 0 ms | Feedforward delay (engineering time) |
| $\Delta_{SK}$ | 0 ms | Skip‑connection delay (engineering time) |
| $\Delta_{RC}$ | 6 ms | Recurrent connection delay |
| Integration strategy | Additive | Activity integration |
| Dynamics solver | Euler | Numerical ODE solver |
| Recurrence target | Output | Recurrence applied to layer output |
| **Architecture** | | |
| Skip connections | Enabled | V1→V4, V2→IT |
| Feedback connections | Disabled | V1←V4, V2←IT |
| Normalization | None | No BatchNorm / LayerNorm |
| **Training** | | |
| Epochs | 200 | |
| Datasets | ImageNette | 224 px, 10 classes |
| Batch size | 192 | |
| Initial learning rate | 0.0008 | |
| LR schedule | CosineAnnealingLR | $T_{\max}=500$ |
| Recurrent LR factor | 0.2 | 5× smaller LR for recurrent |
| Feedback LR factor | 0.5 | 2× smaller LR for feedback |
| Optimizer | Adam | |
| Weight decay | 0.0005 | |
| **Weight Initialization** | | |
| Feedforward V1 | Kaiming | $\mu=0$, $\sigma=0.053$ |
| Feedforward V2 | Kaiming | $\mu=0$, $\sigma=0.050$ |
| Feedforward V4 | Kaiming | $\mu=0$, $\sigma=0.035$ |
| Feedforward IT | Kaiming | $\mu=0$, $\sigma=0.025$ |
| Recurrent weights | Truncated Normal | $\mu=0$, $\sigma=0.004$ |
| Skip & Feedback V1↔V4 | Truncated Normal | $\mu=0$, $\sigma=0.001$ |
| Skip & Feedback V2↔IT | Truncated Normal | $\mu=0$, $\sigma=0.001$ |
| Bias initialization | Zero | All biases set to 0 |
| **Loss** | | |
| Primary | Cross‑entropy | |
| Auxiliary | Activity Loss (×0.0002) | |
| Loss reaction time | 4 ms | |
| **Optimization** | | |
| Gradient accumulation | 4 batches | Effective batch size 768 |
| Precision | bf16‑mixed | |
| Gradient clipping | Norm, 1.0 | |
| **Hardware** | | |
| GPU | NVIDIA A100 | |
| CPU | 16 cores | |

## Computational Resource Demands

Values show the **change relative to the default** DyRCNNx8 configuration
(full recurrence, output target, 20 timesteps, resolution 2 ms, pattern `1`,
skip enabled, no feedback, loss reaction time 4 ms, activity loss weight
0.0002, idle timesteps 0) trained on ImageNette. GPU memory reported as the
90th percentile of `system.gpu.0.memoryAllocatedBytes`. All runs use FP32
precision, Lightning 1.9.5, NVIDIA A100‑SXM4‑80 GB, batch size 192, averaged
over 3 seeds. Gray rows denote the default value for that parameter category.

| Parameter | Value | Δ Time / Epoch (min) | Δ GPU Mem. (GB) | Δ # Params |
|-----------|-------|----------------------|-----------------|------------|
| *Default* | | 1.00 | 48 | 8,534,794 |
| **Recurrence Type** | | | | |
| | none | −0.2 | −6 | −3,398,410 |
| | self | −0.1 | +1 | −3,398,402 |
| | full | — | — | — |
| | depthpointwise | +0.2 | +6 | −3,016,254 |
| | pointdepthwise | +0.2 | +6 | −3,016,254 |
| | local | +0.6 | +11 | −3,398,338 |
| | localdepthwise | +0.9 | +23 | −3,387,384 |
| **Recurrence Target** | | | | |
| | input | 0 | −4 | −1,759,607 |
| | middle | 0 | +6 | 0 |
| | output | — | — | — |
| **Skip** | | | | |
| | True | — | — | — |
| | False | −0.2 | −4 | −93,345 |
| **Feedback** | | | | |
| | False | — | — | — |
| | Additive | +0.1 | +6 | +92,768 |
| | Multiplicative | +0.1 | +14 | +92,768 |
| **Timesteps** | | | | |
| | 8 | −0.6 | −25 | 0 |
| | 14 | −0.3 | −12 | 0 |
| | 20 | — | — | — |
| | 26 | +0.3 | +13 | 0 |
| **Presentation Pattern** | | | | |
| | 1 | — | — | — |
| | 1011 (shuffled) | 0 | +1 | 0 |
| | 1001 | 0 | +1 | 0 |
| | 1000 | 0 | +1 | 0 |
| **Loss Reaction Time** | | | | |
| | 0 | 0 | 0 | 0 |
| | 4 | — | — | — |
| | 8 | 0 | 0 | 0 |
| | 18 | 0 | +1 | 0 |
| | 28 | 0 | 0 | 0 |
| **Activity Loss Weight** | | | | |
| | 0 | −0.1 | −19 | 0 |
| | 0.0002 | — | — | — |
| | 0.02 | 0 | 0 | 0 |
| | 0.2 | 0 | 0 | 0 |
| | 1.0 | 0 | 0 | 0 |
| **Idle Timesteps** | | | | |
| | 0 | — | — | — |
| | 1 | +0.1 | +2 | 0 |
| | 5 | +0.2 | +4 | 0 |
| | 10 | +0.3 | +4 | 0 |
| | 20 | +0.5 | +4 | 0 |
| **Dataloader** | | | | |
| | FFCV — expanded in loader | −0.2 | −1 | 0 |
| | FFCV — expanded in model | −0.2 | −1 | 0 |
| | torch — expanded in loader | +0.4 | +3 | 0 |
| | torch — expanded in model | — | — | — |
| **Unrolling** | | | | |
| | engineering | — | — | — |
| | biological | +0.2 | +43 | 0 |

## Key Takeaways

- **52 % speedup over CORnet‑RT**: reimplementation runs at 8.86 s/epoch vs.
  original 13.51 s/epoch on A100 (averaged over first 80 epochs).

- **Engineering vs. biological time**: engineering time reduces epoch time by
  ~29 % and GPU memory from 2.39 GB to 2.13 GB in a representative DyRCNNx8
  training run. Biological unrolling nearly doubles GPU memory (~43 GB increase)
  because the extended delay buffer stores ~22 ms of hidden-state activations
  across all four layers.

- **Middle recurrence memory cost**: targeting recurrence at the *middle* stage
  increases GPU memory by ~6 GB relative to *output* despite identical channel
  dimensions — the autograd engine must retain both first-convolution output and
  recurrence output for the backward pass.

- **Activity loss memory**: disabling the activity loss frees ~19 GB GPU memory
  (no per-timestep activation norms to track).

- **FFCV dataloaders**: provide ~16 % speedup and ~4 % memory reduction for the
  default ImageNette setup, but benefits are modest because the 20-timestep
  forward pass is compute-bound. Expect larger gains for ImageNet-scale training
  or multi-GPU distributed setups.

## See Also

- [Engineering vs. Biological Time](../explanation/engineering-vs-biological-time.md) — delay conversion formulas
- [FFCV Dependency Guide](../development/dependencies/ffcv.md)
- [Model Testing (How‑to)](../user-guide/model-testing.md)
