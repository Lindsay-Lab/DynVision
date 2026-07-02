# DynVision Transform Configuration Roadmap

_Last updated: 2025-11-19_

## 1. Context & Motivation
- Recent parameter-processing refactor surfaces hidden coupling between `DataParams.data_transform`, `transforms.py`, and loader backends. Example failure: `ffcv_test_imagenette` raised `ValueError` because the name was auto-derived yet missing from `transform_presets`.
- Transform logic is semi-hardcoded in `dynvision/data/transforms.py` with ad-hoc dataset-specific overrides. Users cannot easily inspect/modify presets through configuration files, and the composition rules differ between PyTorch and FFCV loaders.
- Transform selection must become transparent, traceable, and documented within the same config-driven workflow used for other experiment parameters.

## 2. Current Flow Snapshot
| Stage | File(s) | Notes |
| --- | --- | --- |
| Config defaults | `dynvision/configs/config_data.yaml` | Defines dataset stats/resolution but no transform metadata. |
| Parameter resolution | `dynvision/params/data_params.py` | Derives `data_transform` / `target_transform` strings based on `use_ffcv`, `train`, `data_name`. CLI overrides allowed. |
| Dataset creation (torch) | `dynvision/data/datasets.py` | Calls `get_data_transform(transform=data_transform, data_name=data_name)` before tensor/dtype/normalize steps. Logs composed sequence. |
| Dataset creation (ffcv) | `dynvision/data/ffcv_dataloader.py` | Calls `get_data_transform(data_transform)` without `data_name`; backend-specific decoder/normalize handled separately. |
| Transform registry | `dynvision/data/transforms.py` | Small dict (`transform_presets`) mixing backend, context, dataset names. String lookup uses substring matching; dataset-specific presets skipped when `"ffcv" in transform` string. |
| Target transforms | `dynvision/data/operations.py` (`IndexToLabel`) | Already driven by YAML `data_groups`. |
| Logging | `dynvision/data/datamodule.py` | Logs dataset/dataloader kwargs but cannot explain how transform names map to actual ops. |

### Pain Points
1. **Implicit Composition**: PyTorch path prepends dataset-specific transforms when `data_transform` lacks `ffcv`, but FFCV path ignores `data_name`. Presets cannot be layered (e.g., "base train" + "imagenette" overlay) without manual duplication.
2. **Naming Fragility**: DataParams generates names such as `ffcv_test_imagenette`, yet the registry stores only `ffcv_test` and `imagenette`; substring lookup stops at the first empty match, causing errors.
3. **Opaque Configuration**: Users must inspect Python files to know what augmentations run; there is no YAML reflection or logging that lists available presets.
4. **Limited Extensibility**: Adding new datasets/backends requires editing code; there are no hooks for experiment-specific overrides or inheritance.

## 3. Requirements & Constraints
- **Axes**: Selection still depends on `(backend ∈ {torch, ffcv}, context ∈ {train,test}, dataset_name)`.
- **Non-compositional lookup**: Backend/context pick a transform list; if a dataset-specific entry exists it fully replaces the base list, otherwise the base list is used. No implicit layering.
- **Preset overrides**: Users can select named presets (e.g., `auto`, `contrast_heavy`) through params/config. Presets replace backend/context auto-selection when specified.
- **Declarative syntax**: YAML stores literal constructor strings such as `"RandomAffine(0, translate=(0.1, 0.1))"` or bare module names (`"RandomAffine"`). Strings are parsed into torchvision/FFCV calls directly.
- **Library alignment**: Torch presets should rely on torchvision.v2 modules per <https://docs.pytorch.org/vision/main/transforms.html>. FFCV presets must follow <https://docs.ffcv.io/api/transforms.html> recommendations.
- **Separation of concerns**: Normalization and dtype/device conversions remain in `get_dataset()` / `ffcv_dataloader.py`; the preset system only manages augmentation transforms.
- **Validation & logging**: `DataParams` must verify that every `(backend, context, dataset/preset)` combination resolves and log the parsed transform strings.

## 4. Proposed Design
### 4.1 Transform Registry Structure (YAML)
Add a `transform_presets` section to `config_data.yaml`:
```yaml
transform_presets:
  torch:
    train:
      base: ["RandomHorizontalFlip", "RandomBrightness(0.2)", "RandomContrast(0.2)"]
      imagenette: ["Resize(256)", "CenterCrop(224)", "RandomHorizontalFlip", "RandomBrightness(0.2)", "RandomContrast(0.2)"]
      mnist: ["Grayscale(1)", "RandomAffine(0, translate=(0.1, 0.1))"]
      auto: ["AutoAugment"]
    test:
      base: []
      imagenette: ["Resize(256)", "CenterCrop(224)"]
  ffcv:
    train:
      base: ["RandomHorizontalFlip", "RandomBrightness(0.2)", "RandomContrast(0.2)"]
      imagenette: ["Resize(256)", "CenterCrop(224)","RandomHorizontalFlip", "RandomBrightness(0.2)", "RandomContrast(0.2)"]
    test:
      base: []
```
- Values are strings mirroring torchvision/FFCV constructor syntax; parentheses indicate explicit arguments.
- Backend/context choose either a dataset key or fall back to `base`. Optional named presets (like `auto`) can be selected directly via params/CLI.
- YAML remains augmentation-only; normalization and dtype conversion continue to live in loader code for clarity.

### 4.2 Transform Parsing Layer
- Introduce `dynvision/data/transform_parser.py` (or a helper in `dynvision/utils/`) that:
  - Detects backend (torch vs FFCV) and imports preferred APIs (torchvision.transforms.v2 where possible per PyTorch docs).
  - Parses literal strings via `ast.parse`/`ast.literal_eval`, supporting forms like `"RandomAffine"` (no args) and `"RandomAffine(0, translate=(0.1, 0.1))"`.
  - Maps module names to actual classes/functions, including compatibility shims for legacy torchvision v1 ops until we finish migration.
- `transforms.py` loads the YAML, selects the relevant preset list, and converts each entry via the parser while emitting detailed debug logs.
- Keep the current Python dictionary temporarily, but wrap it in compatibility glue that converts existing configs into the new schema with deprecation warnings.

Implementation detail: expose a reusable parser helper so other modules (e.g., metadata validators or CLI tools) can share the same logic.

### 4.3 Selection Flow Updates
1. `DataParams` exposes `transform_backend`, `transform_context`, and `transform_preset`. Defaults derive backend from `use_ffcv`, context from `train`, and preset from `data_name` (falling back to `base`). CLI/config can override preset to named values like `auto`.
2. `get_dataset` / `get_ffcv_dataloader` call `resolve_transforms(backend, context, dataset, preset=None)`; the resolver chooses either the requested preset or the dataset entry, falling back to base.
3. `transforms.py` logs the YAML key path, parsed module strings, and any parsing issues. `DataParams` logging should surface the resolved preset name and the parsed sequence so users see the exact transforms applied.
4. Normalization/dtype remain appended inside loader builders, keeping augmentation presets focused and easier to audit.

### 4.4 Target Transform Alignment
- Leave target transforms unchanged (already YAML-driven), but document how `IndexToLabel` integrates with the new registry for symmetry.

## 5. Implementation Roadmap
### Phase 0 – Documentation & Guardrails ✅ COMPLETED
**Investigation findings (2025-11-20):**

- Confirmed naming bug: `DataParams.validate_transforms` generates `ffcv_test_imagenette`, but `get_data_transform` substring matching stops at first empty match
- Current presets dict in `transforms.py:20-54` uses torchvision v1 APIs (`RandomRotation`, `ColorJitter`)
- Dependencies verified: `torchvision >= 0.16.0` supports v2 transforms
- Transform composition paths identified:
  - PyTorch (datasets.py:278-306): augmentation → PILToTensor → dtype → normalize
  - FFCV (ffcv_dataloader.py:145-153): augmentation → normalize → ToTensor → dtype → device
- Logging infrastructure in `DataParams.log_dataloader_creation` and `DataModule._log_dataloader_creation` ready for extension
- User decisions: Maintain roadmap ✅, Test-Last approach ✅

**Actions completed:**

- Traced complete flow from `DataParams` → `get_data_transform` → loaders
- Cataloged existing infrastructure (logging, validation, composition)
- Analyzed dependencies and confirmed torchvision v2 availability

### Phase 1 – Registry Foundations ✅ COMPLETED (2025-11-20)
**Implementation details:**

1. ✅ Extended `config_data.yaml` with `transform_presets` section following YAML schema from planning doc
   - Converted existing presets to declarative string format
   - Preserved all current augmentation logic (torch: base/mnist/imagenet/imagenette, ffcv: base/mnist)
2. ✅ Implemented `dynvision/data/transform_parser.py` module
   - `parse_transform_string()`: Converts strings like `"RandomHorizontalFlip()"` to callable objects
   - Supports both bare module names and parameterized calls
   - Uses `ast.literal_eval` for safe argument parsing
   - Backend-specific module selection (torchvision.transforms.v2 vs ffcv.transforms)
3. ✅ Rewrote `dynvision/data/transforms.py` with new API
   - `_load_transform_presets()`: YAML loading with caching
   - `resolve_transform_preset()`: Selection logic (dataset/preset → base fallback)
   - `get_data_transform()`: New interface (backend, context, dataset_or_preset) + legacy compatibility layer
   - `_get_data_transform_legacy()`: Backward compatibility for existing callers
4. ✅ Validation built into parser (`validate_transform_string()` function)
   - Raises AttributeError if module doesn't exist in backend
   - Raises ValueError if string format is invalid or arguments can't be parsed

### Phase 2 – Param & Loader Integration ✅ COMPLETED (2025-11-20)
**Implementation details:**

1. ✅ Updated `DataParams` with new fields:
   - `transform_backend`, `transform_context`, `transform_preset` - derived automatically from `use_ffcv`, `train`, `data_name`
   - `target_data_name`, `target_data_group` - derived from `data_name` and `data_group`/`train`
   - Removed legacy `data_transform` and `target_transform` string parameters entirely
   - Updated summary sections to display new fields
2. ✅ Updated `datasets.py` and `ffcv_dataloader.py`:
   - Both now use new interface: `get_data_transform(backend, context, dataset_or_preset)`
   - Target transforms use: `get_target_transform(data_name, data_group)`
   - Preserved normalization/dtype-after-augment pattern
   - All legacy code paths removed
3. ✅ Updated `transforms.py` to simplified interface:
   - Removed all legacy transform string parsing (`_get_data_transform_legacy`)
   - `get_target_transform()` now takes explicit `data_name` and `data_group` parameters
   - Clean, type-safe API with no compound string generation

### Phase 3 – Bug Fixes & Cleanup ✅ COMPLETED (2025-11-20)
**Bug fixes:**

1. ✅ **Merged transform_parser.py into transforms.py**
   - Consolidated parser functions directly into transforms.py
   - Removed separate transform_parser.py file
   - Cleaner module structure with all transform logic in one place

2. ✅ **Fixed precision validation bug**
   - Root cause: PyTorch Lightning 1.9.5 only accepted `'64', '32', '16', 'bf16'` (no `-mixed` variants)
   - TrainerParams and DataParams validators were allowing `'16-mixed'` and `'bf16-mixed'` which Lightning rejected
   - Solution: Updated `pyproject.toml` to require `pytorch-lightning >=2.0.0`
   - Updated both validators to support Lightning 2.0+ precision values including `-mixed` variants
   - Error message now clearly indicates valid precision values

**Tests completed:**

- ✅ Created comprehensive test suite (51 tests total)
  - `tests/data/test_transforms.py` - 35 tests covering parsing, resolution, and integration
  - `tests/data/test_data_params_transforms.py` - 16 tests covering parameter derivation
- ✅ All tests passing with 100% success rate
- Test coverage includes:
  - Transform string parsing (bare names, with args, mixed args)
  - Transform list parsing and validation
  - Preset resolution (backend, context, dataset fallbacks)
  - Data and target transform retrieval
  - DataParams automatic derivation
  - Integration workflows for all backends and contexts

### Phase 4 – Documentation Deliverables ✅ COMPLETED (2025-11-20)
**Documentation completed:**

1. ✅ **Developer Guide** (`docs/development/guides/transform-system.md`):
   - Complete architecture overview with all components
   - Data flow diagrams for PyTorch and FFCV pipelines
   - Design decisions and rationale
   - Implementation details for adding new datasets and presets
   - Comprehensive logging and debugging guide
   - Testing documentation with test organization
   - Migration guide from legacy system
   - Known issues and limitations

2. ✅ **User Reference** (`docs/reference/transform-configuration.md`):
   - Quick start with common use cases
   - Complete parameter reference table
   - Available presets for all backends and contexts
   - Transform string format syntax guide
   - Step-by-step custom preset creation
   - Transform pipeline explanations
   - Troubleshooting section with common errors
   - Best practices and configuration examples
   - Cross-references to related documentation

**Documentation follows Diátaxis framework:**

- Developer guide: Explanation-oriented (understanding architecture)
- User reference: Information-oriented (technical descriptions)

**Remaining optional tasks:**

1. Consider adding CLI helper to list available presets (enhancement, not required)

## 6. Open Questions
1. Should the YAML schema be validated via Pydantic (early) or is `DataParams`-level validation sufficient?
2. If future experiments need combined behavior (dataset + mode), how do we support it without reintroducing implicit composition?
3. What is the deprecation schedule for legacy `data_transform` strings, and how noisy should warnings be?
4. How aggressively should we enforce migration to torchvision v2 APIs, and do we need shims for legacy semantics?

### Current Decisions
- Validation stays within `DataParams`; no extra Pydantic schema needed at load time.
- Future experiment customizations should rely on specialized dataloaders rather than multi-layer transform composition, so no additional mechanism is required now.
- Legacy `data_transform` syntax will not be preserved—configs must migrate to the new preset names immediately once this work lands.
- Prefer torchvision v2 transforms everywhere; introduce substitutions wherever a one-to-one replacement exists (reserve shims only for cases without v2 parity).

---
This document is the authoritative roadmap for transform work. Update sections as decisions are made or implementation progresses.
