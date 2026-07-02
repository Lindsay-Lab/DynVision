# Temporal Data Presentation & Loss Reaction Time - Implementation Notes

**User Documentation:** See [Temporal Data Presentation Guide](../../user-guide/temporal-data-presentation.md)

## Implementation Summary (2025-11-20 - 2025-11-21)

This document tracks the implementation of pattern-aware reaction time masking and temporal data presentation mechanisms.

### Status: ✅ Completed

**Key Changes:**

1. ✅ Moved `loss_reaction_time` and `_init_loss()` from LightningBase to TemporalBase
2. ✅ Implemented pattern-aware reaction masking in `_expand_timesteps()`
3. ✅ Added per-chunk reaction time masking with warnings
4. ✅ Implemented pattern shuffling that preserves chunk durations
5. ✅ Eliminated GPU-CPU synchronization in batch processing
6. ✅ Created comprehensive user-facing documentation

### Implementation Details

**Reaction Time Masking:**

- Detects stimulus onset (rising edges in presentation pattern)
- Masks first `ceil(loss_reaction_time / dt)` timesteps after each onset
- Warns when reaction window exceeds chunk duration
- Fully vectorized using PyTorch broadcasting (zero GPU-CPU sync)

**Pattern Shuffling:**

- Shuffles base pattern entries **before** resampling to `n_timesteps`
- Preserves chunk durations after shuffling
- Different random order per batch during training

**Performance Optimizations:**

- Vectorized tensor operations throughout
- Pre-allocated tensor caching with LRU eviction
- CUDA stream support for async operations
- Channels-last memory format for GPU efficiency

---

## Loss Calculation Implementation (2025-11-21)

### Summary
Reviewed and fixed the complete loss calculation pipeline to ensure correct handling of valid vs. invalid timesteps across multiple loss types.

**Key Findings:**

- ✅ **CrossEntropyLoss:** Normalizes by valid (non-masked) timesteps only
- ⚠️ **ActivityLoss:** Was overwriting instead of accumulating activity across timesteps
- ✅ **Loss Combination:** Correctly weights and sums individual losses

### Fixes Applied

1. **ActivityLoss Accumulation** - Modified `_accumulate_activity()` to accumulate activity across timesteps rather than overwrite
2. **ActivityLoss Timestep Normalization** - Infer `n_timesteps` from hook call counts and normalize by total timesteps
3. **Device Handling** - Added automatic device alignment for GPU/CPU transfers
4. **GPU-CPU Sync Elimination** - Rewrote `_compute_reaction_mask()` using fully vectorized operations
5. **Unit Tests** - Created `tests/losses/test_loss_normalization.py` with 6 comprehensive tests

**Result:** ActivityLoss now correctly measures average absolute activity per unit, per timestep, per module, per sample.

**Detailed documentation:** See [Loss Functions Reference](../../reference/losses.md)

---

## GPU Accelerator Auto-Detection Fix (2025-11-21)

### Issue
Training ran on CPU despite GPU being available. PyTorch Lightning showed:
```
GPU available: True (cuda), used: False
PossibleUserWarning: GPU available but not used. Set `accelerator` and `devices`
```

All model parameters were on CPU device instead of cuda:0.

### Root Cause
**Location:** `dynvision/params/trainer_params.py:644-649`

When using single-device training (non-distributed, `world_size=1`) with default `accelerator="auto"`:

1. The code only set `accelerator` if it wasn't "auto": `if self.accelerator != "auto"`
2. With the default config having `accelerator: auto`, the accelerator was never explicitly set
3. PyTorch Lightning then defaulted to CPU when accelerator wasn't specified

### Fix
**Modified:** `dynvision/params/trainer_params.py:644-657`

Changed the single-device training logic to:
```python
else:
    # For single device training
    if self.accelerator == "auto":
        # Auto-detect: use GPU if available, otherwise let Lightning default to CPU
        if self._detect_available_gpu_count() > 0:
            trainer_kwargs["accelerator"] = "gpu"
            # Use configured devices or default to 1 for single-device training
            trainer_kwargs["devices"] = self.devices if self.devices is not None else 1
        # If no GPU available, let Lightning default to CPU (don't set accelerator)
    else:
        # User explicitly specified accelerator
        trainer_kwargs["accelerator"] = self.accelerator
        if self.devices is not None:
            trainer_kwargs["devices"] = self.devices
```

**Behavior:**

- When `accelerator="auto"` in single-device mode:
  - Detects available GPUs using existing `_detect_available_gpu_count()` method
  - If GPUs available: sets `accelerator="gpu"` and `devices` (respects user config or defaults to 1)
  - If no GPUs: lets Lightning default to CPU (no accelerator set)
- When accelerator explicitly specified: uses user's configuration as before

**Result:** Single-device training now automatically uses GPU when available, matching user expectations for the "auto" setting.

_This document will be updated as implementation proceeds._
