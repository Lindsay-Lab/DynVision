# Optimizers and Schedulers Reference

Quick reference for available optimizers and learning rate schedulers in DynVision.

## Optimizers

DynVision supports all PyTorch optimizers via string identifiers. The optimizer is specified with the `optimizer` parameter.

### Commonly Used Optimizers

| Optimizer | String ID | Default Learning Rate | Best For | Key Parameters |
|-----------|-----------|----------------------|----------|----------------|
| **Adam** | `"Adam"` | 0.001 | General purpose, default choice | `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0` |
| **AdamW** | `"AdamW"` | 0.001 | Large models, better weight decay | `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01` |
| **SGD** | `"SGD"` | 0.01 | Fine-tuning, momentum-based training | `momentum=0.9`, `weight_decay=0`, `nesterov=False` |
| **RMSprop** | `"RMSprop"` | 0.01 | Recurrent networks, unstable gradients | `alpha=0.99`, `eps=1e-8`, `weight_decay=0`, `momentum=0` |
| **Adagrad** | `"Adagrad"` | 0.01 | Sparse gradients | `lr_decay=0`, `weight_decay=0`, `eps=1e-10` |

### Configuration Examples

**Basic configuration** (config YAML):
```yaml
optimizer: "Adam"
learning_rate: 0.001
optimizer_kwargs: {}
```

**With custom parameters**:
```yaml
optimizer: "AdamW"
learning_rate: 0.0005
optimizer_kwargs:
  betas: [0.9, 0.999]
  weight_decay: 0.01
  eps: 1e-8
```

**Command-line override**:
```bash
snakemake train_model --config \
  optimizer="SGD" \
  learning_rate=0.01 \
  optimizer_kwargs="{momentum:0.9,weight_decay:0.0001}"
```

## Learning Rate Schedulers

DynVision supports both PyTorch built-in schedulers and custom schedulers. The scheduler is specified with the `scheduler` parameter.

### DynVision Custom Schedulers

Located in `dynvision.losses.lr_scheduler`:

#### LinearWarmupCosineAnnealingLR

Combines linear warmup with cosine annealing for stable training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_epochs` | int | Required | Number of warmup epochs |
| `max_epochs` | int | Required | Total number of epochs |
| `warmup_start_lr` | float | 0.0 | Initial learning rate during warmup |
| `eta_min` | float | 0.0 | Minimum learning rate after annealing |

**Example**:
```yaml
scheduler: "LinearWarmupCosineAnnealingLR"
scheduler_kwargs:
  warmup_epochs: 10
  max_epochs: 100
  warmup_start_lr: 0.0
  eta_min: 1.0e-6
scheduler_configs:
  interval: "epoch"
  frequency: 1
```

### PyTorch Built-in Schedulers

All PyTorch schedulers are available via `torch.optim.lr_scheduler`:

#### CosineAnnealingLR (Default)

Cosine annealing without warmup.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `T_max` | int | 250 | Maximum number of iterations/epochs |
| `eta_min` | float | 0 | Minimum learning rate |

**Example**:
```yaml
scheduler: "CosineAnnealingLR"
scheduler_kwargs:
  T_max: 250
  eta_min: 0
```

#### StepLR

Decays learning rate by gamma every step_size epochs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_size` | int | Required | Period of learning rate decay |
| `gamma` | float | 0.1 | Multiplicative factor of decay |

**Example**:
```yaml
scheduler: "StepLR"
scheduler_kwargs:
  step_size: 30
  gamma: 0.1
```

#### MultiStepLR

Decays learning rate by gamma at specific milestones.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `milestones` | List[int] | Required | List of epoch indices for decay |
| `gamma` | float | 0.1 | Multiplicative factor of decay |

**Example**:
```yaml
scheduler: "MultiStepLR"
scheduler_kwargs:
  milestones: [30, 60, 90]
  gamma: 0.1
```

#### ExponentialLR

Decays learning rate by gamma every epoch.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | Required | Multiplicative factor of decay |

**Example**:
```yaml
scheduler: "ExponentialLR"
scheduler_kwargs:
  gamma: 0.95
```

#### ReduceLROnPlateau

Reduces learning rate when validation metric plateaus.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "min" | "min" or "max" - minimize or maximize metric |
| `factor` | float | 0.1 | Factor to reduce learning rate |
| `patience` | int | 10 | Number of epochs with no improvement to wait |
| `threshold` | float | 1e-4 | Threshold for measuring improvement |
| `cooldown` | int | 0 | Epochs to wait before resuming normal operation |
| `min_lr` | float | 0 | Minimum learning rate |

**Example**:
```yaml
scheduler: "ReduceLROnPlateau"
scheduler_kwargs:
  mode: "min"
  factor: 0.5
  patience: 10
  threshold: 0.001
scheduler_configs:
  monitor: "val_loss"  # Metric to monitor
  interval: "epoch"
  frequency: 1
```

#### OneCycleLR

Varies learning rate according to 1cycle policy.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lr` | float | Required | Upper learning rate bound |
| `total_steps` | int | Required | Total number of training steps |
| `pct_start` | float | 0.3 | Percentage of cycle spent increasing LR |
| `anneal_strategy` | str | "cos" | "cos" or "linear" |
| `div_factor` | float | 25.0 | Initial LR = max_lr/div_factor |
| `final_div_factor` | float | 1e4 | Final LR = max_lr/final_div_factor |

**Example**:
```yaml
scheduler: "OneCycleLR"
scheduler_kwargs:
  max_lr: 0.01
  total_steps: 10000  # epochs * steps_per_epoch
  pct_start: 0.3
  anneal_strategy: "cos"
scheduler_configs:
  interval: "step"  # Update every batch
  frequency: 1
```

## Scheduler Configuration

### Scheduler Configs

The `scheduler_configs` parameter controls how the scheduler is stepped:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | str | "epoch" | "epoch" or "step" - when to update LR |
| `frequency` | int | 1 | How often to update within interval |
| `monitor` | str | "train_loss" | Metric to monitor (for ReduceLROnPlateau) |

### Monitoring Metrics

Available metrics for scheduler monitoring:

- **Training**: `train_loss`, `train_accuracy`, `train_top5_accuracy`
- **Validation**: `val_loss`, `val_accuracy`, `val_top5_accuracy`
- **Custom**: Any metric logged via `self.log()` in training/validation steps

## Common Patterns

### Pattern 1: Warmup + Cosine Decay

Recommended for most training runs. Stabilizes initial training and smoothly reduces learning rate.

```yaml
optimizer: "AdamW"
learning_rate: 0.001
optimizer_kwargs:
  weight_decay: 0.01

scheduler: "LinearWarmupCosineAnnealingLR"
scheduler_kwargs:
  warmup_epochs: 10
  max_epochs: 100
  eta_min: 1.0e-6
```

### Pattern 2: SGD with Step Decay

Traditional approach, good for fine-tuning.

```yaml
optimizer: "SGD"
learning_rate: 0.01
optimizer_kwargs:
  momentum: 0.9
  weight_decay: 0.0001

scheduler: "MultiStepLR"
scheduler_kwargs:
  milestones: [30, 60, 90]
  gamma: 0.1
```

### Pattern 3: Adaptive Learning Rate

Automatically adjusts based on validation performance.

```yaml
optimizer: "Adam"
learning_rate: 0.001

scheduler: "ReduceLROnPlateau"
scheduler_kwargs:
  mode: "min"
  factor: 0.5
  patience: 10
scheduler_configs:
  monitor: "val_loss"
```

### Pattern 4: Fast Training with 1cycle

For rapid convergence in limited epochs.

```yaml
optimizer: "SGD"
learning_rate: 0.1  # Will be max_lr
optimizer_kwargs:
  momentum: 0.9

scheduler: "OneCycleLR"
scheduler_kwargs:
  max_lr: 0.1
  total_steps: 10000  # Calculate: epochs * len(train_loader)
  pct_start: 0.3
scheduler_configs:
  interval: "step"
```

## Parameter Groups

DynVision automatically creates parameter groups for different model components, allowing different learning rates:

```python
# Configured automatically based on model architecture
# Recurrent connections get recurrent_learning_rate_multiplier
# Other layers use base learning_rate
```

Override with `recurrent_learning_rate_multiplier`:
```yaml
learning_rate: 0.001
recurrent_learning_rate_multiplier: 0.1  # Recurrent weights learn at 0.0001
```

## Troubleshooting

### Learning rate too high
**Symptoms**: Loss is NaN, exploding gradients, unstable training
**Solutions**:

- Reduce `learning_rate` (try 0.0001 instead of 0.001)
- Add warmup: use `LinearWarmupCosineAnnealingLR` with `warmup_epochs: 5-10`
- Increase `weight_decay` for regularization

### Learning rate too low
**Symptoms**: Very slow convergence, training plateaus early
**Solutions**:

- Increase `learning_rate` (try 0.01 instead of 0.001)
- Use `OneCycleLR` for faster convergence
- Reduce `weight_decay`

### Training plateaus
**Symptoms**: Loss/accuracy stops improving
**Solutions**:

- Switch to `ReduceLROnPlateau` to automatically reduce LR
- Manually reduce learning rate: `learning_rate: 0.0001`
- Check if you need more epochs or if model has converged

### Scheduler not updating
**Symptoms**: Learning rate stays constant
**Solutions**:

- Check `scheduler_configs.interval` matches your use case ("epoch" vs "step")
- Verify `scheduler_kwargs` are correct for chosen scheduler
- Check logs for scheduler step messages

## Related Documentation

- [Configuration Reference](configuration.md) - Full configuration system
- [Model Base Reference](model-base.md) - Training infrastructure
- [Parameter Handling](../user-guide/parameter-handling.md) - Parameter system details

## External Resources

- [PyTorch Optimizers Documentation](https://pytorch.org/docs/stable/optim.html#algorithms)
- [PyTorch LR Schedulers Documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [PyTorch Lightning Optimization](https://lightning.ai/docs/pytorch/stable/common/optimization.html)
