# Transform Configuration Reference

This reference describes how to configure data transforms and augmentation in DynVision experiments.

## Overview

DynVision uses a declarative YAML-based system for configuring data transforms. Transforms are automatically selected based on your dataset and training mode, and you can customize them through configuration files or command-line arguments.

## Quick Start

### Using Default Transforms

By default, DynVision selects appropriate transforms based on your dataset:

```bash
# Training with default transforms for Imagenette
python -m dynvision.models.train \
    --data_name imagenette \
    --train true
```

This automatically applies:

- Dataset-specific augmentations for training
- Minimal preprocessing for testing
- Backend-appropriate transforms (PyTorch or FFCV)

### Customizing Transforms via CLI

You can override the default transform preset:

```bash
# Use a custom augmentation preset
python -m dynvision.models.train \
    --data_name imagenette \
    --transform_preset heavy_augmentation
```

### Customizing Transforms via Config

Add transform settings to your experiment config file:

```yaml
# config/my_experiment.yaml
data:
  data_name: imagenette
  transform_preset: custom_augmentation
```

## Transform Parameters

### Core Parameters

DynVision automatically derives transform parameters from your experiment configuration:

| Parameter | Derived From | Description | Override |
|-----------|--------------|-------------|----------|
| `transform_backend` | `use_ffcv` | Transform library (`torch` or `ffcv`) | `--transform_backend` |
| `transform_context` | `train` | Context (`train` or `test`) | `--transform_context` |
| `transform_preset` | `data_name` | Preset name (e.g., `imagenette`, `mnist`) | `--transform_preset` |

**Example Derivation**:
```bash
# These flags...
--data_name imagenette --train true --use_ffcv false

# Automatically derive...
# transform_backend = "torch"
# transform_context = "train"
# transform_preset = "imagenette"
```

## Available Presets

### PyTorch Presets

#### Training Presets

**Base** (default fallback):

- `RandomRotation(10)`: Random rotation up to ±10 degrees
- `RandomAffine(0, translate=(0.1, 0.1))`: Random translation up to 10% in each direction
- `RandomHorizontalFlip()`: 50% chance of horizontal flip
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: Color augmentation

**MNIST**:

- `RandomRotation(10)`: Random rotation for digit recognition
- `RandomAffine(0, translate=(0.1, 0.1))`: Small translations
- `Grayscale(num_output_channels=1)`: Ensure grayscale format

**Imagenette** (and other ImageNet-derived datasets):

- `Resize(256)`: Resize shorter edge to 256 pixels
- `CenterCrop(224)`: Crop 224×224 patch from center
- `RandomRotation(10)`: Moderate rotation
- `RandomAffine(0, translate=(0.1, 0.1))`: Small translations
- `RandomHorizontalFlip()`: Horizontal flipping
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: Color augmentation

#### Test Presets

**Base**: No augmentation (empty list)

**Imagenette**:

- `Resize(256)`: Resize shorter edge to 256 pixels
- `CenterCrop(224)`: Crop 224×224 patch from center

### FFCV Presets

#### Training Presets

**Base**:

- `RandomHorizontalFlip()`: Horizontal flipping
- `RandomBrightness(0.2)`: Brightness jitter
- `RandomContrast(0.2)`: Contrast jitter
- `RandomSaturation(0.2)`: Saturation jitter
- `RandomTranslate(padding=22, fill=(0, 0, 0))`: Random translation

**MNIST**:

- `RandomBrightness(0.2)`: Brightness jitter
- `RandomContrast(0.2)`: Contrast jitter
- `RandomSaturation(0.2)`: Saturation jitter

**Imagenette**:

- `RandomHorizontalFlip()`: Horizontal flipping
- `RandomBrightness(0.2)`: Brightness jitter
- `RandomContrast(0.2)`: Contrast jitter
- `RandomSaturation(0.2)`: Saturation jitter
- `RandomTranslate(padding=22, fill=(0, 0, 0))`: Random translation

#### Test Presets

**Base**: No augmentation (empty list)

**Imagenette**: No augmentation (empty list)

> **Note**: FFCV handles resizing and cropping at the file creation stage, so these transforms are not included in the runtime preset.

## Creating Custom Presets

You can define your own transform presets by editing the configuration file.

### Location

Transform presets are defined in:
```
dynvision/configs/config_data.yaml
```

### Syntax

Add your custom preset under the appropriate backend and context:

```yaml
transform_presets:
  torch:
    train:
      my_custom_preset:
        - "Resize(128)"
        - "RandomCrop(96)"
        - "RandomHorizontalFlip()"
        - "ColorJitter(brightness=0.3, contrast=0.3)"
    test:
      my_custom_preset:
        - "Resize(128)"
        - "CenterCrop(96)"
```

### Transform String Format

Each transform is specified as a string with the following format:

**Bare module name** (uses default arguments):
```yaml
- "RandomHorizontalFlip"
```

**Module with positional arguments**:
```yaml
- "RandomRotation(10)"
- "Resize(256)"
```

**Module with keyword arguments**:
```yaml
- "ColorJitter(brightness=0.2, contrast=0.2)"
```

**Module with mixed arguments**:
```yaml
- "RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1))"
```

### Supported Transforms

DynVision uses the standard transform libraries:

**PyTorch**: [torchvision.transforms.v2](https://pytorch.org/vision/stable/transforms.html)

- All transforms from torchvision.transforms.v2 are supported
- Legacy v1 transforms available as fallback

**FFCV**: [FFCV Transforms](https://docs.ffcv.io/api/transforms.html)

- FFCV-specific transforms for optimized data loading
- Limited to FFCV's transform API

### Example: Heavy Augmentation Preset

```yaml
transform_presets:
  torch:
    train:
      heavy_augmentation:
        - "RandomResizedCrop(224, scale=(0.7, 1.0))"
        - "RandomHorizontalFlip()"
        - "RandomRotation(20)"
        - "ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)"
        - "RandomGrayscale(p=0.1)"
        - "RandomErasing(p=0.5)"
    test:
      heavy_augmentation:
        - "Resize(256)"
        - "CenterCrop(224)"
```

**Usage**:
```bash
python -m dynvision.models.train \
    --data_name imagenette \
    --transform_preset heavy_augmentation
```

## Transform Pipeline

### PyTorch Pipeline

Transforms are applied in this order:

1. **User-defined augmentation** (from YAML preset)
2. **PILToTensor**: Convert PIL image to tensor
3. **Dtype conversion**: Convert to specified precision
4. **Normalization**: Normalize using dataset statistics

**Example**:
```
Raw PIL Image
    ↓ RandomRotation(10)
    ↓ RandomHorizontalFlip()
    ↓ ColorJitter(...)
    ↓ PILToTensor()
    ↓ ConvertDtype(torch.float32)
    ↓ NormalizeRange(mean, std)
    → Normalized Tensor
```

### FFCV Pipeline

Transforms are applied in this order:

1. **FFCV decoder**: Decode compressed image from FFCV file
2. **User-defined augmentation** (from YAML preset)
3. **Normalization**: Normalize using dataset statistics
4. **ToTensor**: Convert to tensor format
5. **Dtype conversion**: Convert to specified precision
6. **ToDevice**: Move to GPU if available

**Example**:
```
FFCV File
    ↓ SimpleRGBImageDecoder()
    ↓ RandomHorizontalFlip()
    ↓ RandomBrightness(0.2)
    ↓ Normalize(mean, std)
    ↓ ToTensor()
    ↓ ToTorchDtype(torch.float32)
    ↓ ToDevice(cuda:0)
    → GPU Tensor
```

## Common Use Cases

### Case 1: Training with Default Settings

Use dataset-specific defaults for standard training:

```bash
python -m dynvision.models.train \
    --data_name imagenette \
    --train true
```

DynVision applies the `imagenette` training preset automatically.

### Case 2: Testing with Minimal Preprocessing

Test mode uses minimal preprocessing automatically:

```bash
python -m dynvision.models.validate \
    --data_name imagenette \
    --train false
```

DynVision applies the `imagenette` test preset (resize + center crop only).

### Case 3: Experimenting with Augmentation

Override the preset to try different augmentation strategies:

```bash
# Try base augmentation (simpler)
python -m dynvision.models.train \
    --data_name imagenette \
    --transform_preset base

# Try heavy augmentation (more aggressive)
python -m dynvision.models.train \
    --data_name imagenette \
    --transform_preset heavy_augmentation
```

### Case 4: Using FFCV Backend

FFCV backend is selected automatically when `use_ffcv=true`:

```bash
python -m dynvision.models.train \
    --data_name imagenette \
    --use_ffcv true
```

This uses FFCV-specific transforms from the FFCV preset section.

### Case 5: Custom Dataset with Custom Preset

For a new dataset, create a preset and reference it:

1. Add preset to `config_data.yaml`:
```yaml
transform_presets:
  torch:
    train:
      my_dataset:
        - "Resize(128)"
        - "RandomCrop(112)"
        - "RandomHorizontalFlip()"
```

2. Use in experiment:
```bash
python -m dynvision.models.train \
    --data_name my_dataset \
    --train true
```

DynVision automatically uses the `my_dataset` preset.

## Troubleshooting

### Transform Not Found

**Error**:
```
AttributeError: Transform 'CustomTransform' not found in torchvision.transforms.v2
```

**Solution**: Ensure the transform name exactly matches the torchvision or FFCV API. Check the documentation:

- [PyTorch Transforms](https://pytorch.org/vision/stable/transforms.html)
- [FFCV Transforms](https://docs.ffcv.io/api/transforms.html)

### Invalid Arguments

**Error**:
```
ValueError: Invalid transform string format: 'RandomRotation(10'
```

**Solution**: Check for matching parentheses and proper argument syntax. Valid examples:
```yaml
- "RandomRotation(10)"           # ✓ Correct
- "RandomRotation(10"            # ✗ Missing closing parenthesis
- "RandomRotation(degrees=10)"   # ✓ Correct with keyword
```

### Backend Mismatch

**Error**:
```
ValueError: Transform 'RandomBrightness' not found in torchvision.transforms.v2
```

**Solution**: Some transforms are FFCV-specific. Ensure you're using the correct backend:

- PyTorch: `ColorJitter`, `RandomRotation`, etc.
- FFCV: `RandomBrightness`, `RandomContrast`, `RandomTranslate`, etc.

### Preset Not Applied

**Problem**: Custom preset not being used.

**Solution**: Check derivation order. If `data_name` is set, it overrides the preset. Explicitly set the preset:
```bash
python -m dynvision.models.train \
    --data_name imagenette \
    --transform_preset my_custom_preset
```

## Configuration Examples

### Minimal Configuration

```yaml
# config/minimal.yaml
data:
  data_name: mnist
  train: true
  # transform_backend: "torch"     # Auto-derived from use_ffcv
  # transform_context: "train"     # Auto-derived from train
  # transform_preset: "mnist"      # Auto-derived from data_name
```

### Explicit Configuration

```yaml
# config/explicit.yaml
data:
  data_name: imagenette
  train: true
  use_ffcv: false
  transform_backend: torch        # Explicitly set (though auto-derived)
  transform_context: train        # Explicitly set (though auto-derived)
  transform_preset: imagenette    # Explicitly set (though auto-derived)
```

### Custom Preset Configuration

```yaml
# config/custom.yaml
data:
  data_name: imagenette
  train: true
  transform_preset: heavy_augmentation  # Override with custom preset
```

### FFCV Configuration

```yaml
# config/ffcv.yaml
data:
  data_name: imagenette
  train: true
  use_ffcv: true
  # transform_backend: "ffcv"      # Auto-derived from use_ffcv
  # transform_preset: "imagenette" # Auto-derived from data_name
```

## Best Practices

### 1. Use Dataset-Specific Presets

Create separate presets for each dataset to capture dataset-specific requirements:

```yaml
transform_presets:
  torch:
    train:
      small_images:      # For MNIST, CIFAR-10
        - "RandomRotation(10)"
        - "RandomAffine(0, translate=(0.1, 0.1))"
      large_images:      # For ImageNet, Imagenette
        - "Resize(256)"
        - "RandomResizedCrop(224)"
        - "RandomHorizontalFlip()"
```

### 2. Match Training and Test Preprocessing

Ensure test preprocessing matches training preprocessing (except augmentation):

```yaml
transform_presets:
  torch:
    train:
      imagenette:
        - "Resize(256)"
        - "CenterCrop(224)"
        - "RandomHorizontalFlip()"    # Augmentation
        - "ColorJitter(...)"           # Augmentation
    test:
      imagenette:
        - "Resize(256)"               # Same preprocessing
        - "CenterCrop(224)"           # Same preprocessing
        # No augmentation
```

### 3. Keep Presets Simple

Start with simple augmentation and increase complexity only if needed:

```yaml
# Start here
base:

  - "RandomHorizontalFlip()"

# Add if underfitting
moderate:

  - "RandomHorizontalFlip()"
  - "ColorJitter(brightness=0.2)"

# Add if still underfitting
heavy:

  - "RandomHorizontalFlip()"
  - "ColorJitter(brightness=0.3, contrast=0.3)"
  - "RandomRotation(15)"
```

### 4. Document Custom Presets

Add comments explaining the purpose of custom presets:

```yaml
transform_presets:
  torch:
    train:
      # Heavy augmentation for small datasets to prevent overfitting
      heavy_augmentation:
        - "RandomResizedCrop(224, scale=(0.7, 1.0))"
        - "RandomHorizontalFlip()"
        - "RandomRotation(20)"
        - "ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)"
```

### 5. Validate Before Training

Test your custom preset on a small batch before launching long training runs:

```python
from dynvision.params import DataParams

# Create params with your custom preset
params = DataParams(
    data_name="imagenette",
    transform_preset="my_custom_preset",
    train=True,
    use_ffcv=False,
    # ... other required params
)

# Check derived parameters
print(f"Backend: {params.transform_backend}")
print(f"Context: {params.transform_context}")
print(f"Preset: {params.transform_preset}")
```

## Related Documentation

- **Developer Guide**: [Transform System Architecture](../development/guides/transform-system.md)
- **How-to Guides**: [Data Processing](../user-guide/data-processing.md)
- **API Reference**: `dynvision.data.transforms`
- **External Documentation**:
  - [PyTorch Vision Transforms](https://pytorch.org/vision/stable/transforms.html)
  - [FFCV Transforms](https://docs.ffcv.io/api/transforms.html)
