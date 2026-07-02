# Python 3.12 Compatibility

**Last updated**: 2026-06-23
**Status**: Python 3.12 is **not yet tested or supported**. Upgrade dependencies pending FFCV verification.

## Current Status

DynVision currently targets Python 3.11 (`python_requires = ">=3.11,<3.14"`). Python 3.12 support is planned but blocked on FFCV compatibility testing.

## Dependency Compatibility Matrix

| Dependency | Current Spec | 3.12 Compatible? | Minimum 3.12-Compatible Version |
|---|---|---|---|
| `scipy` | `==1.12` | ✅ Yes | No bump needed |
| `snakemake` | `~=9.1.5` | ✅ Yes | No bump needed |
| `numba` | `~=0.61.0` | ✅ Yes | No bump needed |
| `ruamel.yaml` | `~=0.18.3` | ✅ Yes | No bump needed |
| `python-dotenv` | `~=1.0.1` | ✅ Yes | No bump needed |
| `scikit-image` | `~=0.20.0` | ✅ Yes | No bump needed |
| `matplotlib` | `~=3.5.1` | ✅ Yes | No bump needed |
| `seaborn` | `~=0.13.0` | ✅ Yes | No bump needed |
| `pillow` | `==10.0.0` | ✅ Yes | No bump needed |
| `tqdm` | `~=4.66.0` | ✅ Yes | No bump needed |
| `torchsummary` | `~=1.5.0` | ✅ Yes | No bump needed |
| `jupyterlab` | `~=4.0.0` | ✅ Yes | No bump needed |
| `ipympl` | `~=0.7.0` | ✅ Yes | No bump needed |
| `wandb` | (unpinned) | ✅ Yes | No bump needed |
| `tensorboard` | (unpinned) | ✅ Yes | No bump needed |
| **`torch`** | `>=2.2.0` | ❌ No | **`>=2.4.0`** |
| **`torchvision`** | `>=0.16.0` | ❌ No | **`>=0.19.0`** |
| **`pytorch-lightning`** | `>=2.0.0` | ❌ No | **`>=2.4.0`** |
| **`scikit-learn`** | `>=1.2.0,<2` | ❌ No (needs 1.4+) | **`>=1.4.0`** |
| **`ffcv`** | `~=1.0.2` | ⚠️ Unclear | **Testing required** |

## Key Risk: FFCV

FFCV 1.0.2 was last released on 2023-03-05, before Python 3.12 was released (2023-10-02). Key concerns:

- **No official Python 3.12 wheels or CI** for FFCV
- The Ubuntu/Debian package was removed for Python 3.12 due to reported incompatibility
- FFCV depends on `numba`, which historically had Python 3.12 issues (resolved in numba 0.59.0+)
- The upstream GitHub repository ([libffcv/ffcv](https://github.com/libffcv/ffcv)) has not released a Python 3.12-compatible version

### Mitigation Options

If FFCV proves incompatible with Python 3.12:

1. **Stay on Python 3.11** for the 0.1 release — safest option, no code changes needed
2. **Make FFCV optional** — allow fallback to standard PyTorch DataLoader when FFCV is unavailable
3. **Replace FFCV** — alternatives include:
      - [WebDataset](https://github.com/webdataset/webdataset) — streaming I/O for large datasets
      - [NVIDIA DALI](https://github.com/NVIDIA/DALI) — GPU-accelerated data loading
      - Standard PyTorch DataLoader with appropriate tuning

## Version Bumps Required for 3.12

To update `pyproject.toml` for Python 3.12 support, the following changes are needed:

```diff
 dependencies = [
     ...

-    "torch >=2.2.0",
-    "torchvision >=0.16.0",
+    "torch >=2.4.0",
+    "torchvision >=0.19.0",
     ...

-    "pytorch-lightning >=2.0.0",
+    "pytorch-lightning >=2.4.0",
     ...

-    "scikit-learn >=1.2.0,<2",
+    "scikit-learn >=1.4.0,<2",
     ...
     "ffcv ~= 1.0.2",    # ⚠️ Test Python 3.12 compatibility
 ]
```

And update the Python version constraint:

```diff
- python_requires = ">=3.11,<3.14"
+ python_requires = ">=3.12,<3.14"
+ classifiers = [
+     "Programming Language :: Python :: 3.12",
+ ]
```

## Testing Plan

1. Create a Python 3.12 conda environment
2. Install DynVision with updated dependencies
3. Run the test suite: `pytest tests/ -v`
4. Test FFCV data loading specifically:
   ```python
   from ffcv.loader import Loader
   # Verify basic loading works on 3.12
   ```
5. If FFCV fails, implement fallback option or document 3.11-only for 0.1

## Recommendation for 0.1 Release

**Keep Python 3.11 as the minimum requirement for 0.1.** The risk from FFCV incompatibility with 3.12 is too high to resolve without thorough testing. Add explicit 3.12 support in a point release (0.1.1 or 0.2.0) after FFCV testing is complete.

Set `python_requires = ">=3.11,<3.14"` to allow forward compatibility while clearly documenting the 3.12 testing status.
