# Troubleshooting Guide

Common issues and solutions when working with DynVision.

## Training Issues

### Training Runs on CPU Instead of GPU

**Symptoms:**
```
GPU available: True (cuda), used: False
TPU available: False, using: 0 TPU cores
All parameters on device: cpu
```

**Causes:**

1. `accelerator` set to `"auto"` in single-device mode without explicit GPU configuration
2. CUDA not properly installed
3. PyTorch built without CUDA support

**Solutions:**

1. **Explicit GPU configuration** (recommended):
```yaml
# In config_defaults.yaml
accelerator: "gpu"
devices: 1
```

2. **Command-line override**:
```bash
snakemake train_model --config accelerator="gpu" devices=1
```

3. **Verify CUDA installation**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

4. **Check PyTorch CUDA support**:
```bash
python -c "import torch; print(torch.__version__)"
# Should show +cu118 or similar (CUDA version)
```

---

### Loss is NaN or Infinite

**Symptoms:**

- Training loss suddenly becomes `nan`
- Loss values explode (> 1e10)
- Gradients are `nan` or `inf`

**Causes:**

1. Learning rate too high
2. Numerical instability in loss computation
3. Division by zero in custom operations
4. Gradient explosion in recurrent connections

**Solutions:**

1. **Reduce learning rate**:
```yaml
learning_rate: 0.0001  # Instead of 0.001
```

2. **Add gradient clipping**:
```yaml
gradient_clip_val: 1.0  # Clip gradients to max norm of 1.0
```

3. **Use warmup scheduler**:
```yaml
scheduler: "LinearWarmupCosineAnnealingLR"
scheduler_kwargs:
  warmup_epochs: 10
  max_epochs: 100
  warmup_start_lr: 0.0
```

4. **Check for NaN in data**:
```python
# Add to your dataset
assert not torch.isnan(data).any(), "NaN values in input data"
```

5. **Enable anomaly detection** (for debugging):
```python
torch.autograd.set_detect_anomaly(True)
```

---

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce batch size**:
```yaml
batch_size: 32  # Instead of 128
```

2. **Enable gradient accumulation**:
```yaml
batch_size: 32
accumulate_grad_batches: 4  # Effective batch size = 32 * 4 = 128
```

3. **Reduce number of timesteps**:
```yaml
n_timesteps: 10  # Instead of 20
```

4. **Disable response storage during training**:
```yaml
store_responses: 0  # Only store during testing
```

5. **Use mixed precision training**:
```yaml
precision: "bf16-mixed"  # Requires Ampere GPUs or newer
# or
precision: "16-mixed"  # For older GPUs
```

6. **Clear CUDA cache periodically**:
```python
# In training loop
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

---

### Training is Very Slow

**Symptoms:**

- Epochs take much longer than expected
- GPU utilization is low (< 50%)
- CPU bottleneck

**Solutions:**

1. **Use FFCV for data loading**:
```yaml
use_ffcv: true
```

First convert dataset to FFCV format:
```bash
snakemake project_paths.data.processed/imagenette/train_all/train.beton
```

2. **Increase num_workers**:
```yaml
num_workers: 8  # Match number of CPU cores
```

3. **Enable pin_memory**:
```yaml
pin_memory: true
```

4. **Use channels_last memory format**:
```yaml
use_channels_last: true
```

5. **Disable progress bar on cluster**:
```yaml
enable_progress_bar: false
```

6. **Check validation frequency**:
```yaml
check_val_every_n_epoch: 5  # Validate less frequently
```

---

## Model Issues

### Model Parameters Not Loading

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict for DyRCNNx4:
Missing key(s) in state_dict: "V1_conv.weight", ...
```

**Causes:**

1. Model architecture changed since checkpoint was saved
2. Loading checkpoint from different model variant
3. Mismatch in parameter names

**Solutions:**

1. **Check model variant matches**:
```bash
# Make sure model_name and model_args match exactly
model_name=DyRCNNx4
model_args="{rctype:full,dt:2,tau:8}"
```

2. **Use strict=False for partial loading**:
```python
model.load_state_dict(checkpoint, strict=False)
```

3. **Inspect checkpoint**:
```python
checkpoint = torch.load("model.pt")
print(checkpoint.keys())  # See what's in the checkpoint
```

---

### Recurrent Connections Not Working

**Symptoms:**

- Model behaves like feedforward CNN
- No temporal dynamics observed
- Responses identical across timesteps

**Causes:**

1. `n_timesteps = 1` (no temporal processing)
2. Recurrent weights initialized to zero
3. Time constant too large (tau >> dt * n_timesteps)

**Solutions:**

1. **Ensure adequate timesteps**:
```yaml
n_timesteps: 20  # Minimum 10-20 for observing dynamics
```

2. **Check temporal parameters**:
```yaml
dt: 2  # Time step in ms
tau: 8  # Time constant - should be 2-4x dt
```

3. **Verify recurrence type**:
```yaml
model_args: "{rctype:full}"  # Not "none" or "feedforward"
```

4. **Check delays**:
```yaml
t_feedforward: 2  # Should be ~= dt
t_recurrence: 2   # Should be ~= dt
```

---

## Data Issues

### Dataset Not Found

**Symptoms:**
```
FileNotFoundError: Dataset not found at path/to/dataset
```

**Solutions:**

1. **Download dataset first**:
```bash
snakemake project_paths.data.raw/cifar10/train
```

2. **Create dataset links**:
```bash
snakemake project_paths.data.interim/cifar10/train_all.ready
```

3. **Verify dataset path**:
```bash
ls -la data/raw/cifar10/
```

---

### FFCV Conversion Fails

**Symptoms:**
```
ImportError: cannot import name 'ffcv' from 'ffcv'
AttributeError: module 'ffcv' has no attribute 'Writer'
```

**Causes:**

1. FFCV not installed
2. Incompatible FFCV version
3. Missing dataset

**Solutions:**

1. **Install FFCV**:
```bash
conda install cupy pkg-config compilers libjpeg-turbo opencv numba -c conda-forge
pip install ffcv
```

2. **Check FFCV version**:
```bash
python -c "import ffcv; print(ffcv.__version__)"
# Should be >= 1.0.0
```

3. **Verify dataset exists before conversion**:
```bash
ls -la data/interim/cifar10/train_all/
```

---

### Double Temporal Expansion

**Symptoms:**

- Output shape is `(batch, n_timesteps, n_timesteps, channels, H, W)`
- Memory usage much higher than expected
- Unexpected tensor dimensions

**Cause:**
Both `data_timesteps` and model `n_timesteps` are > 1, causing double expansion.

**Solution:**
Choose one expansion method:

```yaml
# Option 1: DataLoader expansion (for testing)
data_timesteps: 20
n_timesteps: 1  # In model config

# Option 2: Model expansion (for training with patterns)
data_timesteps: 1
n_timesteps: 20  # In model config
```

---

## Configuration Issues

### Configuration Not Taking Effect

**Symptoms:**

- Changed config values but model still uses old values
- Parameters don't match what's in YAML

**Causes:**

1. Config file load order (later files override earlier)
2. Command-line args override config files
3. Model defaults override `None` values
4. **Config frozen at workflow start** (cluster execution only)

**Solutions:**

1. **Check config load order**:
```
config_defaults.yaml  (lowest priority)
config_data.yaml
config_visualization.yaml
config_experiments.yaml
config_workflow.yaml
Command-line --config args  (highest priority)
```

2. **Use explicit values**:
```yaml
# Instead of leaving commented (which becomes None):
# dt: 2
# Use explicit value:
dt: 2
```

3. **Check the frozen workflow config snapshot**:
```bash
# Find the most recent workflow config
ls -lt logs/configs/workflow_config_*.yaml | head -1

# View the frozen config used by the workflow
cat logs/configs/workflow_config_<timestamp>.yaml
```

4. **Understanding config freezing** (cluster execution):

When using `snakecharm.sh` for cluster execution, the configuration is **frozen** at workflow start for reproducibility. This means:

- ✅ Config changes **do not** affect running workflows (this is intentional!)
- ✅ All jobs in a workflow run see identical configuration
- ✅ To use updated configs, start a **new** workflow run
- ✅ The frozen snapshot is saved in `logs/configs/workflow_config_<timestamp>.yaml`

**Example scenario:**
```bash
# Start workflow
./dynvision/cluster/snakecharm.sh train_model

# While workflow running, you edit config_defaults.yaml
vim dynvision/configs/config_defaults.yaml  # Change learning_rate: 0.001 → 0.0001

# Jobs submitted AFTER the edit still use learning_rate: 0.001
# The workflow uses the frozen config from workflow start
```

**To apply config changes:**
```bash
# Option 1: Stop current workflow and start new one with updated config
# (Ctrl+C to stop, then restart)

# Option 2: Use CLI override instead (works immediately)
./dynvision/cluster/snakecharm.sh train_model --config learning_rate=0.0001
```

**Why freezing is necessary:**
Without freezing, changing config files mid-workflow would cause:

- Jobs submitted early vs. late to have different parameters
- Inconsistent results within a single experiment
- Irreproducible workflows

See [Parameter Processing Guide](../development/guides/parameter-processing.md#config-freezing-for-cluster-execution) for technical details.

---

### Model Naming Conflicts

**Symptoms:**
```
FileExistsError: Model file already exists
ValueError: Conflicting model configurations
```

**Cause:**
Model name doesn't uniquely identify the configuration.

**Solution:**

Include all varying parameters in `model_args`:
```yaml
model_name: DyRCNNx4
model_args: "{rctype:full,dt:2,tau:8,tsteps:20}"
seed: 0001
```

This creates unique filename:
```
DyRCNNx4:rctype=full+dt=2+tau=8+tsteps=20_0001_cifar10_trained.pt
```

---

## PyTorch Lightning Issues

### Validation Metrics Not Found

**Symptoms:**
```
WARNING: ModelCheckpoint(monitor='val_loss') could not find the monitored key
```

**Cause:**
Validation doesn't run on first epoch when `check_val_every_n_epoch > 1`.

**Solution:**
Automatically handled - trainer switches to `train_loss` monitoring until validation runs. You can also:

```yaml
check_val_every_n_epoch: 1  # Validate every epoch
# or
monitor: "train_loss"  # Monitor training loss instead
```

---

### Hook Signature Mismatches

**Symptoms:**
```
TypeError: on_train_batch_start() takes 3 positional arguments but 4 were given
```

**Cause:**
PyTorch Lightning updated hook signatures.

**Solution:**
This should be fixed in the current version. If you still see this:

1. Update DynVision to latest version
2. Check PyTorch Lightning version compatibility:
```bash
pip list | grep lightning
# Should be pytorch-lightning>=2.0.0
```

---

## Snakemake Issues

### Rule Not Found

**Symptoms:**
```
WorkflowError: Target rules may not contain wildcards
RuleException: Could not resolve wildcards
```

**Causes:**

1. Missing wildcards in target specification
2. Incorrect path format
3. Rule doesn't exist

**Solutions:**

1. **Use complete path with wildcards**:
```bash
# Good:
snakemake project_paths.models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar10_trained.pt

# Bad (missing seed):
snakemake project_paths.models/DyRCNNx4/DyRCNNx4:rctype=full_cifar10_trained.pt
```

2. **Use rule names for expansion**:
```bash
snakemake train_model --config model_name=DyRCNNx4 model_args="{rctype:full}"
```

3. **Check available rules**:
```bash
snakemake --list
```

---

### Cluster Job Failures

**Symptoms:**

- Jobs fail silently on cluster
- No output in log files
- SLURM job status shows FAILED

**Solutions:**

1. **Check cluster logs**:
```bash
cat logs/slurm/slurm-JOBID.out
cat logs/slurm/slurm-JOBID.err
```

2. **Verify resource requests**:
```yaml
# In cluster config
mem_mb: 32000  # Increase if OOM
runtime: 1440  # Increase for long jobs
gpus: 1        # Ensure GPU requested
```

3. **Test locally first**:
```bash
# Run without cluster submission
snakemake train_model --config model_name=DyRCNNx4
```

4. **Check environment**:
```bash
# Make sure conda environment is activated in cluster job
which python
conda env list
```

---

### Cluster Not Detected (ModuleNotFoundError)

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'numpy'
```

**Cause:**
Cluster execution not detected, so Python runs without singularity/conda wrapper.

**Diagnosis:**

Check workflow logs for detection message:
```bash
# Should see this on cluster:
[INFO] Cluster execution detected via: SLURM_JOB_ID

# If you see this instead:
[DEBUG] Local execution detected (no cluster scheduler variables)
# Then detection failed
```

**Solutions:**

1. **Verify scheduler environment variables**:
```bash
# On SLURM cluster, check:
echo $SLURM_JOB_ID

# On PBS cluster:
echo $PBS_JOBID

# On LSF cluster:
echo $LSB_JOBID
```

If these are not set, you're not running in a scheduler job context.

2. **Check executor wrapper configuration**:
```bash
# Verify executor_wrapper.sh exists and is configured
ls -la dynvision/cluster/executor_wrapper.sh

# Test manually
dynvision/cluster/executor_wrapper.sh python -c "import torch; print(torch.__version__)"
```

3. **Supported schedulers**:
DynVision detects:

- SLURM (via `SLURM_JOB_ID` or `SLURM_JOBID`)
- PBS/Torque (via `PBS_JOBID`)
- LSF (via `LSB_JOBID`)
- SGE/UGE (via `SGE_TASK_ID`)

If using a different scheduler, add its environment variable to `is_cluster_execution()` in `dynvision/workflow/snake_utils.smk`.

4. **Running outside scheduler**:
If running Snakemake directly (not via `snakecharm.sh`), ensure you're submitting via scheduler:
```bash
# Correct (via snakecharm):
./dynvision/cluster/snakecharm.sh train_model

# Wrong (direct execution on login node):
cd dynvision/workflow && snakemake train_model  # No cluster detected!
```

See: [Cluster Integration Guide](cluster-integration.md) for proper setup.

---

## Performance Debugging

### Identify Bottlenecks

**Tools:**

1. **PyTorch profiler**:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(batch)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

2. **Time individual operations**:
```python
import time

start = time.time()
output = model(input)
torch.cuda.synchronize()  # Wait for GPU
print(f"Forward pass: {time.time() - start:.3f}s")
```

3. **Monitor GPU usage**:
```bash
watch -n 0.5 nvidia-smi
```

4. **Profile data loading**:
```python
import time
for batch in dataloader:
    print(f"Load time: {time.time() - start:.3f}s")
    start = time.time()
```

---

## Getting Help

If you can't find a solution here:

1. **Check logs**:
      - Training logs: `logs/training/`
      - Cluster logs: `logs/slurm/`
      - Runtime config: `logs/config_runtime.yaml`

2. **Enable debug mode**:
   ```yaml
   log_level: "DEBUG"
   ```

3. **Search GitHub Issues**: Check if others have encountered the same problem

4. **Create detailed bug report** with:
      - Complete error message and stack trace
      - Configuration files used
      - DynVision version
      - PyTorch and CUDA versions
      - Steps to reproduce

5. **Contact**: robin.gutzen@nyu.edu

---

## Related Documentation

- [Installation Guide](installation.md) - Setup and dependencies
- [Configuration Reference](../reference/configuration.md) - Config system details
- [Optimizers and Schedulers](../reference/optimizers-schedulers.md) - Training optimization
- [Temporal Data Presentation](temporal-data-presentation.md) - Temporal expansion details
