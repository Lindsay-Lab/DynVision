# Loss Functions

This reference describes the loss functions available in DynVision and their behavior with temporal data and presentation patterns.

## Description

DynVision provides specialized loss functions designed for temporal neural networks. These losses handle timestep-wise computation, masked labels, and activity regularization. Multiple losses can be combined with configurable weights to balance different training objectives.

**Key Features:**

- Temporal normalization accounting for valid vs. invalid timesteps
- Presentation pattern-aware masking for reaction time windows
- Hook-based activity computation for efficient memory usage
- Flexible loss combination with weighted sums

## Available Loss Functions

### CrossEntropyLoss

Standard cross-entropy loss adapted for temporal sequences with masked timesteps.

**Location:** `dynvision.losses.CrossEntropyLoss`

**Purpose:** Classification loss that ignores invalid timesteps (null inputs, reaction windows) when computing loss and gradients.

**Parameters:**

- `reduction` (str, default="mean"): How to reduce the loss. Options: "mean", "sum", "none"
- `ignore_index` (int, default=-1): Target value to ignore when computing loss

**Normalization Behavior:**

- Computes element-wise cross-entropy with `reduction="none"`
- Creates validity mask excluding `ignore_index` entries
- Normalizes by **valid timestep count only** (excludes masked timesteps)
- Formula: `loss.sum() / num_valid_timesteps`

**Example:**
```python
from dynvision.losses import CrossEntropyLoss

# Create loss function
criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)

# Compute loss (outputs: [batch*timesteps, classes], targets: [batch*timesteps])
loss = criterion(outputs, targets)
# Only timesteps where targets != -1 contribute to the loss
```

**Use Cases:**

- Primary classification objective
- Training with variable-length sequences
- Excluding null inputs and reaction windows from supervision

---

### ActivityLoss

Regularization loss that penalizes total neural activity across all timesteps.

**Location:** `dynvision.losses.ActivityLoss`

**Purpose:** Compute total computational activity (neural activity) to encourage efficient, sparse representations.

**Parameters:**

- `reduction` (str, default="mean"): How to reduce the loss. Options: "mean", "sum", "none"
- `p` (int, default=1): Norm order for activity calculation (1=L1, 2=L2)

**Normalization Behavior:**

- Accumulates activity across **all timesteps** (including null inputs and reaction windows)
- Uses forward hooks to capture activations during the model's forward pass
- Normalizes by spatial dimensions, number of monitored modules, and total timesteps
- Formula: `sum_t(sum_modules(||activation||_p / n_units)) / (n_modules * n_timesteps)`

**Hook-Based Operation:**

- Registers forward hooks on monitored layers (Conv2d, Linear, ConvTranspose2d)
- Hooks fire once per layer per timestep during forward pass
- Activity accumulates across timesteps, then gets normalized when loss is computed
- Accumulators reset after each batch

**Example:**
```python
from dynvision.losses import ActivityLoss

# Create activity loss
activity_loss = ActivityLoss(reduction="mean", p=1)  # L1 norm

# Register hooks on model layers
activity_loss.register_hooks(model)

# During training, activity accumulates automatically via hooks
# Compute loss (outputs and targets are ignored for ActivityLoss)
loss = activity_loss(outputs=None, targets=None)
# Returns average absolute activity per unit per timestep per module
```

**Use Cases:**

- Encouraging sparse activations
- Biological plausibility (metabolic cost)
- Regularization to prevent overfitting
- Typically combined with CrossEntropyLoss

**Important Notes:**

- Must call `register_hooks(model)` before training
- Activity includes all timesteps (unlike CrossEntropyLoss which respects masking)
- Hooks automatically handle device transfers (CPU/GPU)
- Call `remove_hooks()` or rely on `__del__` for cleanup

### Expected Training Behavior

**Activity loss measures total network activity, not prediction quality.** During training, you should expect:

**Early Training (epochs 1-10):**

- Activity typically **increases** as the network learns stronger feature representations
- Weak random weights → small activations → low activity (~0.05-0.08)
- Learning requires stronger activations → activity rises (~0.10-0.15)

**Mid Training (epochs 10-50):**

- Activity **plateaus** at an operating point
- Network balances prediction accuracy (minimize CrossEntropy) with activity level (activity regularization)
- Activity stabilizes (~0.12-0.18) while CrossEntropy continues decreasing

**Late Training (epochs 50+):**

- Activity remains **stable** or slightly decreases
- Network has found efficient representations
- Activity may fluctuate slightly but should not grow unbounded

**This is normal and expected.** The activity regularization is working if:

- ✅ Activity stabilizes (doesn't continuously grow)
- ✅ CrossEntropy decreases (network is learning)
- ✅ Total loss decreases (activity weight is appropriate)
- ✅ Activity contribution to total loss is small (typically <5%)

**Warning Signs** (indicating actual problems):

- ❌ Activity continuously growing without plateau (raw activity >1.0)
- ❌ Activity dominating total loss (weighted_activity > CrossEntropy)
- ❌ Both activity and CrossEntropy increasing together
- ❌ Activation magnitudes >10 (check with monitoring)

**Example Training Curve:**
```
Epoch | CrossEntropy | Activity | Weighted Activity (0.05) | Total Loss
------|-------------|--------|------------------------|------------
  1   |    2.30     |  0.05  |        0.0025          |   2.3025
 10   |    1.50     |  0.12  |        0.0060          |   1.5060
 20   |    1.00     |  0.15  |        0.0075          |   1.0075
 50   |    0.50     |  0.15  |        0.0075          |   0.5075
100   |    0.30     |  0.14  |        0.0070          |   0.3070
```

**Key Insight**: Activity increasing from 0.05 to 0.15 while CrossEntropy decreases from 2.3 to 0.5 is **healthy training**. The regularization prevents unbounded growth while allowing the network to learn effective representations.

---

## Loss Combination

DynVision supports combining multiple losses with configurable weights.

**Configuration:**
```yaml
# In config file
criterion:

  - name: cross_entropy_loss
    weight: 1.0
    kwargs:
      reduction: mean
      ignore_index: -1

  - name: activity_loss
    weight: 0.05
    kwargs:
      reduction: mean
      p: 1
```

**Computation Flow:**

1. Each criterion computes its loss independently
2. Losses are multiplied by their respective weights
3. Weighted losses are summed: `total_loss = sum(weight_i * loss_i)`
4. Individual loss values are logged for monitoring

**Example:**
```python
# Manually combining losses
ce_loss = criterion_ce(outputs, targets)  # CrossEntropyLoss
activity = criterion_activity(None, None)      # ActivityLoss

# Weighted combination
total_loss = 1.0 * ce_loss + 0.05 * activity
```

---

## Temporal Masking and Presentation Patterns

Loss computation interacts with temporal data presentation and reaction time masking.

### Presentation Patterns

Data presentation patterns control which timesteps receive actual input vs. null (zero) input:

- Pattern `"1111"`: All timesteps receive input
- Pattern `"1011"`: Null input at timestep index 1
- Pattern `"10001000"`: Alternating stimulus and null blocks

**Label Masking:**
Labels for null input timesteps are set to `ignore_index` (default -1) so they don't contribute to CrossEntropyLoss.

### Reaction Time Masking

The `loss_reaction_time` parameter (in milliseconds) masks the initial portion of each stimulus presentation to account for neural processing delays.

**Behavior:**

- Converts reaction time to timesteps: `reaction_steps = ceil(loss_reaction_time / dt)`
- Detects stimulus onsets (rising edges in presentation pattern)
- Masks first `reaction_steps` of each stimulus chunk by setting labels to `ignore_index`
- Warnings emitted if reaction window exceeds chunk duration

**Example:**
```python
# Configuration
n_timesteps = 10
dt = 2  # ms per timestep
loss_reaction_time = 6  # ms
pattern = "1000111000"  # Two stimulus chunks

# Reaction masking:
# - reaction_steps = ceil(6/2) = 3
# - Chunk 1: timesteps [0,1,2,3] → mask [0,1,2]
# - Chunk 2: timesteps [4,5,6] → mask [4,5,6] (entire chunk masked, warning issued)
```

### Normalization Differences

| Loss Type | Normalization Base | Reaction Masking | Null Input Handling |
|-----------|-------------------|------------------|---------------------|
| CrossEntropyLoss | Valid timesteps only | Respects (via ignore_index) | Respects (via ignore_index) |
| ActivityLoss | All timesteps | Ignores (counts all) | Ignores (counts all) |

**Rationale:**

- **CrossEntropyLoss**: Evaluates prediction accuracy only when supervision is meaningful
- **ActivityLoss**: Measures total computational cost regardless of supervision availability

---

## Implementation Details

### BaseLoss

All loss functions inherit from `BaseLoss` which provides:

**Reduction Logic:**
```python
def apply_reduction(self, loss: torch.Tensor, num_valid_timesteps: Optional[int] = None) -> torch.Tensor:
    if self.reduction == "mean":
        if num_valid_timesteps is not None:
            return loss.sum() / float(num_valid_timesteps)
        return loss.mean()
    elif self.reduction == "sum":
        return loss.sum()
    return loss
```

**Valid Timestep Inference:**

- Automatically counts valid timesteps from targets when `ignore_index` is set
- Passes count to `apply_reduction()` for correct normalization
- Handles edge case of zero valid timesteps (returns zero loss)

### Hook Management (ActivityLoss)

**Hook Registration:**
```python
def register_hooks(self, model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            hook = module.register_forward_hook(
                lambda module, input, output, name=name: self._accumulate_activity(name, output)
            )
            self.hooks.append(hook)
```

**Activity Accumulation:**
```python
def _accumulate_activity(self, module_name: str, activation: torch.Tensor) -> None:
    batch_activity = torch.norm(activation, p=self.p, dim=tuple(range(1, activation.ndim)))

    if module_name not in self.batch_activity:
        self.batch_activity[module_name] = batch_activity
        self._hook_call_count[module_name] = 1
    else:
        # Handle device alignment for GPU/CPU transfers
        existing_activity = self.batch_activity[module_name]
        if existing_activity.device != batch_activity.device:
            existing_activity = existing_activity.to(batch_activity.device)
        self.batch_activity[module_name] = existing_activity + batch_activity
        self._hook_call_count[module_name] += 1
```

**Timestep Inference:**

- Infers `n_timesteps` from hook call counts
- All monitored modules should be called the same number of times
- Uses `max(call_counts)` as the timestep count

---

## Common Patterns

### Basic Training Setup

```python
from dynvision.losses import CrossEntropyLoss, ActivityLoss

# Classification loss
ce_loss = CrossEntropyLoss(reduction="mean", ignore_index=-1)

# Activity regularization
activity_loss = ActivityLoss(reduction="mean", p=1)
activity_loss.register_hooks(model)

# In training loop
def training_step(batch):
    outputs = model(inputs)

    # Flatten temporal dimension
    outputs_flat = outputs.view(-1, n_classes)
    targets_flat = targets.view(-1)

    # Compute losses
    ce = ce_loss(outputs_flat, targets_flat)
    activity = activity_loss(None, None)

    # Combine
    loss = ce + 0.05 * activity
    return loss
```

### Monitoring Individual Losses

```python
# Log individual components for tracking
self.log("loss/CrossEntropyLoss", ce_loss.item())
self.log("loss/ActivityLoss", activity_loss.item())
self.log("train_loss", total_loss.item())
```

### Cleanup

```python
# Explicit cleanup (automatic on deletion)
activity_loss.remove_hooks()
```

---

## Troubleshooting

### Issue: Loss is NaN

**Possible Causes:**

1. Learning rate too high
2. Gradient explosion
3. Invalid inputs (inf or NaN)
4. Division by zero in normalization

**Solutions:**

- Reduce learning rate
- Enable gradient clipping
- Check data for invalid values
- Verify valid timestep count > 0

### Issue: Activity loss not changing

**Possible Causes:**

1. Hooks not registered
2. Model not in training mode
3. Weight too small to affect optimization

**Solutions:**
```python
# Verify hooks are registered
activity_loss.register_hooks(model)

# Ensure model is in training mode
model.train()

# Increase activity loss weight
total_loss = ce_loss + 0.1 * activity_loss  # Try larger weight
```

### Issue: Warning about monitored key not found

**Possible Causes:**

- Validation runs less frequently than checkpointing
- Monitoring `val_loss` but validation hasn't run yet

**Solutions:**

- Use `train_loss` for checkpoint monitoring when `check_val_every_n_epoch > 1`
- System automatically handles this (see checkpoint callback configuration)

---

## Performance Considerations

### Memory Efficiency

**ActivityLoss:**

- Uses hooks to avoid storing full activation tensors
- Only accumulates scalar activity values per module
- Minimal memory overhead compared to standard forward pass

**CrossEntropyLoss:**

- Element-wise computation allows batch processing
- Masking done via multiplication (no tensor copying)

### Computation Efficiency

**Temporal Masking:**

- Fully vectorized using PyTorch broadcasting
- Zero GPU-CPU synchronization
- Pattern detection uses tensor operations only

**Device Handling:**

- Automatic device alignment in activity accumulation
- Supports mixed CPU/GPU training
- Preserves gradients across device transfers

---

## References

### Related Documentation
- [Base Model Classes](model-base.md) - Model architecture and training integration
- [Temporal Data Presentation](../user-guide/temporal-data-presentation.md) - Presentation patterns and reaction time
- [Configuration System](configuration.md) - Loss configuration syntax
- [Temporal Dynamics](../explanation/temporal_dynamics.md) - Conceptual understanding

### External Resources
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Forward Hooks Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
