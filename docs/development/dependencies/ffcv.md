# FFCV Integration

[FFCV (Fast Forward Computer Vision)](https://github.com/libffcv/ffcv) is a high-performance data loading library developed by researchers at MIT. DynVision integrates FFCV for dramatically faster data loading, particularly beneficial for iterative experimentation and large-scale training.

## What is FFCV?

FFCV is a drop-in replacement for PyTorch's DataLoader that can provide 10-100x speedups by:

- **Storing data in optimized binary format** (`.beton` files)
- **Using memory-mapped files** for OS-level caching
- **Minimizing Python overhead** with compiled data pipelines
- **Supporting GPU-direct loading** for certain transformations

## When to Use FFCV

### Recommended For:
- **Iterative experimentation** - Multiple training runs on same dataset
- **Large datasets** - ImageNet, COCO, large custom datasets
- **Data-heavy experiments** - When data loading is the bottleneck
- **Cluster training** - Shared filesystem caching benefits multiple jobs

### Not Recommended For:
- **First-time setup** - Initial `.beton` conversion adds overhead
- **Small datasets** - MNIST, small CIFAR subsets (negligible benefit)
- **Rapidly changing data** - Need to regenerate `.beton` after changes
- **Limited disk space** - `.beton` files require additional storage

## Installation

FFCV requires specific environment configuration:

```bash
# Install FFCV (requires CUDA)
conda install cupy pkg-config compilers libjpeg-turbo opencv numba -c conda-forge
pip install ffcv

# Verify installation
python -c "import ffcv; print(ffcv.__version__)"
```

**Note**: FFCV requires:

- Linux or macOS (Windows not supported)
- CUDA-capable GPU for best performance
- Sufficient disk space for `.beton` files

## DynVision Integration

### Automatic FFCV Usage

DynVision automatically uses FFCV when:

1. `use_ffcv: true` in config
2. `.beton` files exist in `data/processed/`
3. FFCV is installed

### Configuration

Enable FFCV in your config:

```yaml
# In config_runtime.yaml or experiment config
data:
  use_ffcv: true
  num_workers: 8  # FFCV benefits from multiple workers

# Optional FFCV-specific settings
ffcv:
  os_cache: true  # Use OS cache (recommended)
  order: quasi_random  # Data ordering: sequential, random, quasi_random
  distributed: false  # Multi-GPU training
```

### Workflow Integration

The Snakemake workflow automatically handles FFCV conversion:

```bash
# Data pipeline automatically creates .beton files
cd dynvision/workflow
snakemake <project_paths.data.processed>/<dataset>/train.beton
snakemake <project_paths.data.processed>/<dataset>/val.beton

# Training uses FFCV if available
snakemake train_model --config use_ffcv=true
```

**File locations:**

- Input: `data/interim/<dataset>/` (symlinked ImageFolder structure)
- Output: `data/processed/<dataset>/train.beton`, `val.beton`

## Creating .beton Files Manually

For custom datasets or debugging:

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder

# Load dataset
dataset = ImageFolder('data/interim/my_dataset/train')

# Write .beton file
writer = DatasetWriter(
    'data/processed/my_dataset/train.beton',
    fields={
        'image': RGBImageField(
            max_resolution=256,
            jpeg_quality=90
        ),
        'label': IntField()
    }
)

writer.from_indexed_dataset(dataset)
```

See `dynvision/data/ffcv_dataloader.py` for DynVision's implementation.

## Performance Considerations

### Expected Speedups

| Dataset Size | Standard PyTorch | FFCV | Speedup |
|--------------|------------------|------|---------|
| MNIST (60k) | 2.3 s/epoch | 1.8 s/epoch | 1.3x |
| CIFAR-100 (50k) | 3.1 s/epoch | 1.2 s/epoch | 2.6x |
| ImageNette (13k) | 8.5 s/epoch | 1.1 s/epoch | 7.7x |
| ImageNet (1.2M) | 45 min/epoch | 8 min/epoch | 5.6x |

*Approximate, depends on hardware, batch size, and augmentation pipeline*

### Optimization Tips

1. **Increase num_workers**: FFCV scales well with multiple workers
   ```yaml
   num_workers: 8  # Or more, depending on CPU cores
   ```

2. **Enable OS cache**: Subsequent epochs are much faster
   ```yaml
   ffcv:
     os_cache: true  # Default in DynVision
   ```

3. **Use quasi-random ordering**: Balances randomness and cache efficiency
   ```yaml
   ffcv:
     order: quasi_random  # Default in DynVision
   ```

4. **Consider image resolution**: Lower resolution in `.beton` = faster loading
   ```python
   # In ffcv conversion
   RGBImageField(max_resolution=224)  # Match your input size
   ```

## Troubleshooting

### FFCV Not Found

**Symptom**: `ModuleNotFoundError: No module named 'ffcv'`

**Solution**:
```bash
pip install ffcv
# Or if CUDA errors:
conda install cupy pkg-config compilers libjpeg-turbo opencv numba -c conda-forge
pip install ffcv
```

### .beton Files Not Created

**Symptom**: Training falls back to PyTorch DataLoader

**Check**:

1. Snakemake rule executed: `snakemake --list | grep beton`
2. Files exist: `ls data/processed/<dataset>/*.beton`
3. Permissions correct: `chmod 644 data/processed/<dataset>/*.beton`

**Re-create**:
```bash
cd dynvision/workflow
snakemake --forcerun create_ffcv_dataset --config data_name=cifar100
```

### Slow First Epoch

**Symptom**: First epoch slow, subsequent epochs fast

**Explanation**: This is expected! OS cache is being populated.

- First epoch: Loads from disk into cache
- Later epochs: Served from cache (much faster)

**Solution**: Not a problem, this is optimal behavior

### Out of Memory During .beton Creation

**Symptom**: Process killed during FFCV conversion

**Solution**:

- Reduce `max_resolution` in conversion script
- Process dataset in chunks
- Increase system memory/swap

### Incompatible Augmentation

**Symptom**: Training fails with FFCV-specific error

**Explanation**: Some augmentations don't work with FFCV's GPU pipeline

**Solution**:

- Use FFCV-compatible transforms in `ffcv_dataloader.py`
- Or disable FFCV: `use_ffcv: false`

### Permission Denied on .beton Files

**Symptom**: `PermissionError` when loading .beton

**Solution**:
```bash
chmod 644 data/processed/<dataset>/*.beton
# Or if directory permissions issue:
chmod 755 data/processed/<dataset>/
```

## Fallback Behavior

If FFCV loading fails, DynVision automatically falls back to standard PyTorch DataLoader with a warning:

```
WARNING: FFCV loading failed, falling back to PyTorch DataLoader
```

This ensures training continues even if FFCV has issues.

## Comparing FFCV vs PyTorch

To measure speedup for your specific setup:

```bash
# Run with FFCV
time snakemake train_model --config use_ffcv=true epochs=1

# Run with PyTorch
time snakemake train_model --config use_ffcv=false epochs=1

# Compare data loading times in logs
```

## Implementation Details

### Code Structure

- **`dynvision/data/ffcv_dataloader.py`**: FFCV DataLoader wrapper
  - `FFCVDataLoader`: Main class
  - Handles image transformations, label processing
  - Supports temporal dimension expansion for RCNNs

- **`dynvision/workflow/snake_data.smk`**: Snakemake rules
  - `create_ffcv_dataset`: Converts ImageFolder → .beton
  - Integrated into data pipeline

### Data Flow with FFCV

```
Raw Images
    ↓
data/raw/<dataset>/
    ↓ (preprocessing)
data/interim/<dataset>/  (symlinks to raw)
    ↓ (FFCV conversion)
data/processed/<dataset>/*.beton
    ↓ (training)
FFCVDataLoader → Model
```

## Advanced Usage

### Custom Augmentation Pipeline

Define FFCV-compatible augmentations:

```python
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage
from ffcv.transforms import RandomHorizontalFlip, RandomTranslate

# In ffcv_dataloader.py
image_pipeline = [
    RandomHorizontalFlip(),
    RandomTranslate(padding=4),
    ToTensor(),
    ToTorchImage(),
    ToDevice(torch.device('cuda:0'), non_blocking=True)
]
```

### Multi-GPU Training

Enable distributed FFCV:

```yaml
trainer:
  devices: 2
  strategy: ddp

data:
  use_ffcv: true
ffcv:
  distributed: true  # Enable multi-GPU FFCV
```

## References

- [FFCV Official Documentation](https://docs.ffcv.io/)
- [FFCV GitHub Repository](https://github.com/libffcv/ffcv)
- [FFCV Paper (NeurIPS 2022)](https://arxiv.org/abs/2206.12155)
- DynVision implementation: `dynvision/data/ffcv_dataloader.py`

## Related Documentation

- [Data Processing Guide](../../user-guide/data-processing.md)
- [DynVision Workflow Guide](../../user-guide/workflows.md)
- [Snakemake Workflow](snakemake.md)
