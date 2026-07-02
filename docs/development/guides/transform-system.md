# Transform System Architecture

This guide provides a comprehensive overview of DynVision's YAML-driven transform configuration system for developers working on data loading, preprocessing, and augmentation pipelines.

## Overview

The transform system manages data augmentation and preprocessing transforms for both PyTorch and FFCV data loaders. It replaces the previous hardcoded Python dictionary approach with a declarative YAML-based configuration that supports:

- Backend-specific transforms (PyTorch torchvision.transforms.v2 vs FFCV transforms)
- Context-aware presets (training vs testing)
- Dataset-specific configurations with base fallbacks
- Safe AST-based parsing of transform strings
- Automatic parameter derivation from experiment configuration

## Architecture Components

### 1. YAML Registry (`config_data.yaml`)

The transform registry lives in `dynvision/configs/config_data.yaml` under the `transform_presets` section:

```yaml
transform_presets:
  <backend>:      # "torch" or "ffcv"
    <context>:    # "train" or "test"
      <preset>:   # "base", dataset name, or custom preset name
        - "TransformName()"
        - "TransformWithArgs(arg1, kwarg=value)"
```

**Structure**:

- **Backend**: Selects the transform library (`torch` → torchvision.transforms.v2, `ffcv` → ffcv.transforms)
- **Context**: Training contexts use augmentation; test contexts typically use minimal preprocessing
- **Preset**: Dataset-specific presets replace base presets when available (no implicit layering)

**Example**:
```yaml
transform_presets:
  torch:
    train:
      base:
        - "RandomRotation(10)"
        - "RandomHorizontalFlip()"
        - "ColorJitter(brightness=0.2, contrast=0.2)"
      imagenette:
        - "Resize(256)"
        - "CenterCrop(224)"
        - "RandomHorizontalFlip()"
        - "ColorJitter(brightness=0.2, contrast=0.2)"
    test:
      base: []
      imagenette:
        - "Resize(256)"
        - "CenterCrop(224)"
```

### 2. Transform Parser (`dynvision/data/transforms.py`)

The parser module provides safe conversion from YAML strings to callable transform objects.

#### Key Functions

**`parse_transform_string(transform_str: str, backend: Backend) -> Optional[Callable]`**

Parses a single transform string using AST to safely handle mixed positional and keyword arguments:

```python
# Bare module name
parse_transform_string("RandomHorizontalFlip", backend="torch")

# With positional args
parse_transform_string("RandomRotation(10)", backend="torch")

# With keyword args
parse_transform_string("ColorJitter(brightness=0.2, contrast=0.2)", backend="torch")

# Mixed args
parse_transform_string("RandomAffine(0, translate=(0.1, 0.1))", backend="torch")
```

**Implementation Details**:

1. Splits transform string into module name and arguments
2. Uses `ast.parse()` to create a Call node from the full expression
3. Extracts positional args via `call_node.args` and keyword args via `call_node.keywords`
4. Uses `ast.literal_eval()` on individual arguments for safe evaluation
5. Dynamically imports the transform class from the backend module
6. Instantiates the transform with parsed arguments

**`parse_transform_list(transform_strings: List[str], backend: Backend) -> List[Callable]`**

Parses a list of transform strings, raising errors if any individual transform fails:

```python
transforms = parse_transform_list(
    ["RandomHorizontalFlip()", "RandomRotation(10)"],
    backend="torch"
)
```

**`validate_transform_string(transform_str: str, backend: Backend) -> Tuple[bool, Optional[str]]`**

Validates a transform string without instantiating it, useful for configuration validation:

```python
is_valid, error = validate_transform_string("RandomHorizontalFlip()", backend="torch")
if not is_valid:
    logger.error(f"Invalid transform: {error}")
```

### 3. Preset Resolution (`dynvision/data/transforms.py`)

**`resolve_transform_preset(backend: Backend, context: str, dataset_or_preset: Optional[str]) -> List[str]`**

Resolves the appropriate preset with fallback logic:

1. Look for `transform_presets[backend][context][dataset_or_preset]`
2. If not found, fall back to `transform_presets[backend][context]["base"]`
3. Raise `ValueError` if backend or context is invalid

```python
# Dataset-specific preset
transforms = resolve_transform_preset(
    backend="torch",
    context="train",
    dataset_or_preset="imagenette"
)

# Falls back to base when dataset preset doesn't exist
transforms = resolve_transform_preset(
    backend="torch",
    context="train",
    dataset_or_preset="unknown_dataset"  # → uses "base"
)
```

### 4. Parameter Management (`dynvision/params/data_params.py`)

`DataParams` automatically derives transform parameters from experiment configuration:

**New Fields**:
```python
transform_backend: Optional[Literal["torch", "ffcv"]] = None
transform_context: Optional[Literal["train", "test"]] = None
transform_preset: Optional[str] = None
target_data_name: Optional[str] = None
target_data_group: Optional[str] = None
```

**Automatic Derivation** (in `validate_transforms()` validator):
```python
# Backend from use_ffcv flag
if transform_backend is None:
    transform_backend = "ffcv" if use_ffcv else "torch"

# Context from train flag
if transform_context is None:
    transform_context = "train" if train else "test"

# Preset from data_name
if transform_preset is None:
    transform_preset = data_name

# Target parameters
if target_data_name is None:
    target_data_name = data_name
if target_data_group is None:
    target_data_group = "all" if train else data_group
```

**CLI Overrides**: All parameters can be explicitly set via CLI or config files:
```bash
# Use a custom preset
python -m dynvision.models.train --transform_preset custom_augmentation

# Override backend (not recommended)
python -m dynvision.models.train --transform_backend torch --use_ffcv false
```

### 5. Loader Integration

**PyTorch Loader** (`dynvision/data/datasets.py`):
```python
def get_dataset(
    data_path: Path,
    transform_backend: str = "torch",
    transform_context: str = "train",
    transform_preset: Optional[str] = None,
    target_data_name: Optional[str] = None,
    target_data_group: str = "all",
    ...
):
    # Get augmentation transforms
    additional_transforms = get_data_transform(
        backend=transform_backend,
        context=transform_context,
        dataset_or_preset=transform_preset,
    ) or []

    # Compose: augmentation → PILToTensor → dtype → normalize
    all_transforms = additional_transforms + [
        tv2.PILToTensor(),
        ConvertDtype(dtype=dtype),
        NormalizeRange(pixel_range, dtype=dtype, data_stats=data_stats),
    ]
    transform = tv2.Compose(all_transforms)

    # Get target transforms
    target_transform = get_target_transform(
        data_name=target_data_name or data_name,
        data_group=target_data_group,
    )
```

**FFCV Loader** (`dynvision/data/ffcv_dataloader.py`):
```python
def get_ffcv_dataloader(
    path: Union[str, Path],
    transform_backend: str = "ffcv",
    transform_context: str = "train",
    transform_preset: Optional[str] = None,
    target_data_name: Optional[str] = None,
    target_data_group: str = "all",
    ...
):
    # Get augmentation transforms
    data_transform = get_data_transform(
        backend=transform_backend,
        context=transform_context,
        dataset_or_preset=transform_preset,
    ) or []

    # Compose: decoder → augmentation → normalize → ToTensor → dtype → device
    image_pipeline = [
        SimpleRGBImageDecoder(),
        *data_transform,
        # ... normalize, ToTensor, dtype, ToDevice
    ]

    # Get target transforms
    target_transform = get_target_transform(
        data_name=target_data_name,
        data_group=target_data_group,
    ) or []
```

## Data Flow

### Training Pipeline (PyTorch)
```
YAML preset → parse_transform_list() → augmentation transforms
                                      ↓
Raw image → augmentation → PILToTensor → dtype → normalize → model
```

### Training Pipeline (FFCV)
```
YAML preset → parse_transform_list() → augmentation transforms
                                      ↓
FFCV file → decoder → augmentation → normalize → ToTensor → dtype → device → model
```

### Test Pipeline
```
YAML preset (minimal) → preprocessing only (resize, crop)
                       ↓
Raw image → preprocessing → normalization → model
```

## Design Decisions

### 1. Non-Compositional Lookup

**Decision**: Dataset-specific presets fully replace base presets; no implicit layering.

**Rationale**:

- Explicit configuration is easier to understand and debug
- Avoids hidden dependencies between base and dataset transforms
- Users can copy-paste base transforms into dataset presets if needed

**Example**:
```yaml
# imagenette preset REPLACES base, not extends it
torch:
  train:
    base: ["RandomHorizontalFlip()"]
    imagenette: ["Resize(256)", "CenterCrop(224)"]  # No RandomHorizontalFlip
```

### 2. Separation of Concerns

**Decision**: YAML presets only manage augmentation; normalization/dtype remain in loader code.

**Rationale**:

- Normalization depends on dataset statistics (mean/std) computed at runtime
- Dtype conversion depends on precision settings
- Device placement is a deployment concern, not a transform concern
- Clearer separation between data augmentation (user-configurable) and technical preprocessing (automatic)

### 3. AST-Based Parsing

**Decision**: Use `ast.parse()` and `ast.literal_eval()` instead of `eval()` or manual parsing.

**Rationale**:

- **Security**: No arbitrary code execution; only literal values allowed
- **Robustness**: Handles complex argument structures (nested tuples, mixed args)
- **Error Handling**: Clear error messages when parsing fails
- **Maintainability**: Leverages Python's built-in AST tools

**Rejected Alternatives**:

- `eval()`: Security risk, allows arbitrary code execution
- Manual regex parsing: Fragile, complex, error-prone for nested structures
- JSON-like syntax: Requires users to learn non-Python syntax

### 4. Torchvision v2 API

**Decision**: Prefer torchvision.transforms.v2 over legacy v1 API.

**Rationale**:

- v2 is the recommended API per PyTorch documentation
- Better support for mixed input types (PIL, tensors, videos)
- Consistent interface with better composition semantics
- Backward compatibility maintained via automatic fallbacks in parser

**Implementation**: Parser tries v2 first, falls back to v1 if module not found.

### 5. Automatic Parameter Derivation

**Decision**: `DataParams` automatically derives transform parameters from `use_ffcv`, `train`, and `data_name`.

**Rationale**:

- Reduces boilerplate configuration
- Ensures consistency (FFCV flag automatically sets FFCV backend)
- Supports CLI overrides for experimentation
- Explicit logging shows derived values for transparency

## Adding New Datasets

To add support for a new dataset:

1. **Add YAML presets** in `config_data.yaml`:
```yaml
transform_presets:
  torch:
    train:
      my_new_dataset:
        - "Resize(128)"
        - "RandomCrop(96)"
        - "RandomHorizontalFlip()"
    test:
      my_new_dataset:
        - "Resize(128)"
        - "CenterCrop(96)"
  ffcv:
    train:
      my_new_dataset:
        - "RandomHorizontalFlip()"
        - "RandomBrightness(0.2)"
    test:
      my_new_dataset: []
```

2. **Add dataset registration** in `config_data.yaml` data_stats section (if needed for normalization):
```yaml
data_stats:
  my_new_dataset:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

3. **Test the configuration**:
```python
from dynvision.params import DataParams

params = DataParams(
    data_name="my_new_dataset",
    train=True,
    use_ffcv=False,
    # ... other required params
)

# Verify derived parameters
assert params.transform_preset == "my_new_dataset"
assert params.transform_backend == "torch"
```

## Adding Custom Transform Presets

Users can define custom presets for experiment-specific augmentation:

1. **Add to YAML**:
```yaml
transform_presets:
  torch:
    train:
      heavy_augmentation:
        - "RandomRotation(20)"
        - "RandomAffine(10, translate=(0.2, 0.2), scale=(0.8, 1.2))"
        - "RandomHorizontalFlip()"
        - "RandomVerticalFlip()"
        - "ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)"
        - "RandomErasing(p=0.5)"
```

2. **Use via CLI**:
```bash
python -m dynvision.models.train --transform_preset heavy_augmentation
```

3. **Use in config**:
```yaml
data:
  transform_preset: heavy_augmentation
```

## Logging and Debugging

The transform system provides extensive logging at multiple levels:

### Parameter Derivation Logging

`DataParams.validate_transforms()` logs all derived parameters:
```
INFO: Derived transform_backend='torch' from use_ffcv=False
INFO: Derived transform_context='train' from train=True
INFO: Derived transform_preset='imagenette' from data_name='imagenette'
```

### Transform Resolution Logging

`resolve_transform_preset()` logs preset selection:
```
DEBUG: Resolved preset: backend='torch', context='train', preset='imagenette'
DEBUG: Found 6 transform strings in preset
```

### Transform Parsing Logging

`parse_transform_list()` logs each parsed transform:
```
DEBUG: Parsed transform 1/6: RandomRotation(degrees=10)
DEBUG: Parsed transform 2/6: RandomHorizontalFlip(p=0.5)
```

### Dataset Creation Logging

`get_dataset()` and `get_ffcv_dataloader()` log final transform composition:
```
INFO: Data transforms: [RandomRotation, RandomHorizontalFlip, ColorJitter, PILToTensor, ConvertDtype, NormalizeRange]
INFO: Target transforms: None
```

## Testing

The transform system has comprehensive test coverage (51 tests across two test files):

### `tests/data/test_transforms.py` (35 tests)
- `TestParseTransformString`: Parsing bare names, args, mixed args, error handling
- `TestParseTransformList`: List parsing and error propagation
- `TestValidateTransformString`: Validation without instantiation
- `TestResolveTransformPreset`: Preset resolution and fallback logic
- `TestGetDataTransform`: End-to-end data transform retrieval
- `TestGetTargetTransform`: Target transform retrieval
- `TestTransformIntegration`: Complete workflows for different datasets/backends

### `tests/data/test_data_params_transforms.py` (16 tests)
- `TestTransformDerivation`: Automatic parameter derivation
- `TestTransformKwargs`: Parameter inclusion in dataloader kwargs
- `TestTransformScenarios`: Realistic configuration scenarios

**Running Tests**:
```bash
# Run all transform tests
pytest tests/data/test_transforms.py tests/data/test_data_params_transforms.py -v

# Run with coverage
pytest tests/data/ --cov=dynvision.data.transforms --cov-report=html
```

## Known Issues and Limitations

### 1. Transform Compatibility

**Issue**: Not all torchvision v1 transforms have direct v2 equivalents.

**Workaround**: Parser automatically falls back to v1 when v2 module not found.

**Long-term**: Migrate all presets to v2-compatible transforms.

### 2. FFCV Transform Limitations

**Issue**: FFCV transforms have different APIs and fewer options than torchvision.

**Impact**: Some advanced augmentations (e.g., AutoAugment, RandAugment) not available for FFCV backend.

**Workaround**: Use simpler augmentations for FFCV, or preprocess with torchvision before FFCV conversion.

### 3. Preset Validation Timing

**Issue**: Invalid transform strings are only detected when dataset is created, not when config is loaded.

**Impact**: Configuration errors discovered late in experiment setup.

**Future Enhancement**: Add early validation in `DataParams.validate_transforms()` to parse and validate presets at parameter initialization.

## Migration Guide

### From Legacy String-Based System

**Old API**:
```python
# Legacy compound strings
data_transform = "ffcv_test_imagenette"
target_transform = "imagenette_one"

transforms = get_data_transform(
    transform=data_transform,
    data_name="imagenette"
)
```

**New API**:
```python
# Explicit parameters (auto-derived in DataParams)
transform_backend = "ffcv"
transform_context = "test"
transform_preset = "imagenette"
target_data_name = "imagenette"
target_data_group = "one"

transforms = get_data_transform(
    backend=transform_backend,
    context=transform_context,
    dataset_or_preset=transform_preset
)

target_transforms = get_target_transform(
    data_name=target_data_name,
    data_group=target_data_group
)
```

**Breaking Changes**:

1. `DataParams.data_transform` → removed (use `transform_backend`, `transform_context`, `transform_preset`)
2. `DataParams.target_transform` → removed (use `target_data_name`, `target_data_group`)
3. `get_data_transform(transform, data_name)` → `get_data_transform(backend, context, dataset_or_preset)`
4. `get_target_transform(transform)` → `get_target_transform(data_name, data_group)`

## Related Documentation

- **User Guide**: [Transform Configuration Reference](../../reference/transform-configuration.md)
- **Planning Document**: [Transform Roadmap](../planning/transforms.md)
- **API Reference**: `dynvision.data.transforms`, `dynvision.params.data_params`
- **Dependencies**: [PyTorch Vision Transforms](https://pytorch.org/vision/stable/transforms.html), [FFCV Transforms](https://docs.ffcv.io/api/transforms.html)
