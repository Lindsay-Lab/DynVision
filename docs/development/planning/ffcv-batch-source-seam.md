# Collapsing `use_ffcv` Branches Into a `BatchSource` Seam

**Status:** ✅ IMPLEMENTED
**Created:** 2026-09-08
**Issue:** [Lindsay-Lab/DynVision#16](https://github.com/Lindsay-Lab/DynVision/issues/16)
**Branch:** `fix/16-ffcv-batch-source-seam`

---

## 1. Problem

Commit `8d66967` ("disable ffcv") removed `ffcv` from `pyproject.toml`, but module-scope
`ffcv` imports remained in several files. The most severe of these has since been
partially patched (`dynvision/params/data_params.py` now guards its
`from ffcv.loader import OrderOption` import behind a `try/except ImportError` with a
stand-in `OrderOption` IntEnum), so `import dynvision.params` no longer crashes without
ffcv installed.

The regression is still live one level down, in `dynvision/data/datamodule.py`:

```
dynvision/data/datamodule.py:30  import ffcv
dynvision/data/datamodule.py:43  from dynvision.data.ffcv_dataloader import get_ffcv_dataloader
```

Both are **module-scope, unguarded**. `ffcv_dataloader.py` in turn imports
`ffcv_operations.py`, which does `from ffcv.pipeline.operation import Operation` at
module scope. So any import of `dynvision.data.datamodule` — which `runtime/train_model.py`,
`runtime/init_model.py`, and `runtime/test_model.py` all do unconditionally — requires a
fully working ffcv install, *even when `use_ffcv=False`*.

Verified locally: with only a partial/stub `ffcv` package on `sys.path` (providing
`ffcv.loader.OrderOption` but not `ffcv.pipeline`), `import dynvision.params` succeeds but
`import dynvision.data.datamodule` fails with
`ModuleNotFoundError: No module named 'ffcv.pipeline.operation'`.

`dynvision/data/transforms.py` already guards its own `import ffcv.transforms` behind
`try/except ImportError`, so no change is needed there.

## 2. Root cause

The torch/ffcv split is duplicated as parallel files (`dataloader.py` / `ffcv_dataloader.py`,
`datasets.py` / `ffcv_datasets.py`, `operations.py` / `ffcv_operations.py`) selected by
`use_ffcv` branches inside `DataModule`. Those branches are already fairly well
localized (3 methods: `create_preview_loader`, `setup`, `_create_ffcv_loader`), but the
`import` that wires the ffcv branch in is eager and module-scope, so simply *having
`use_ffcv=False`* doesn't save you — the interpreter still has to resolve the ffcv
import graph to load the module at all.

## 3. Options considered

1. **Guard the two datamodule.py imports with a local `try/except`, matching the
   `data_params.py` pattern.** Minimal diff, fixes the reported crash. But it leaves the
   `use_ffcv` branching implicit and scattered, and doesn't give a clear, actionable error
   when someone actually sets `use_ffcv=True` without ffcv installed (they'd hit an
   `AttributeError`/`NameError` deep inside `_create_ffcv_loader` instead of an
   explanatory message at setup time).
2. **Full `BatchSource` interface + `TorchAdapter`/`FFCVAdapter`, collapsing all three
   file pairs (`dataloader`/`ffcv_dataloader`, `datasets`/`ffcv_datasets`,
   `operations`/`ffcv_operations`) into the seam**, as originally suggested in the issue.
   Most thorough, but reshapes working, tested torch-path code for no functional gain —
   the torch path was never broken.
3. **Scoped `BatchSource` seam**: introduce `dynvision/data/batch_source.py` with a
   `BatchSource` interface (`TorchAdapter`, `FFCVAdapter`) that owns *only* the loader
   selection point (`get_batch_source(use_ffcv) -> BatchSource`, `.create_loader(path,
   **config)`, `.loader_class`), and make it the single place ffcv is imported lazily
   (inside adapter methods, not at module scope). `DataModule` delegates to the adapter
   instead of branching on `use_ffcv` directly for loader construction. Leaves
   `dataloader.py` / `ffcv_dataloader.py` / `datasets.py` / `ffcv_datasets.py` /
   `operations.py` / `ffcv_operations.py` untouched internally — they already only get
   imported when the adapter that needs them is actually instantiated.

**Decision: Option 3.** It directly fixes the live regression (import of `datamodule.py`
no longer requires ffcv), realizes the "one seam" recommendation from the issue without
rewriting already-working, already-tested torch-path modules, and gives a single point to
emit a clear, actionable error (`FFCVAdapter.__init__` raises `ImportError` with an
install hint) instead of a crash at random depth. It also lets `data_params.py`'s
`OrderOption` stand-in be deduplicated into the seam module instead of living as a second
copy.

## 4. Design

```
dynvision/data/batch_source.py
├── OrderOption            # re-exported from ffcv.loader if available, else IntEnum stand-in
├── BatchSource(ABC)       # .loader_class property, .create_loader(path, **config)
├── TorchAdapter(BatchSource)   # imports dynvision.data.dataloader lazily
├── FFCVAdapter(BatchSource)    # imports dynvision.data.ffcv_dataloader lazily;
│                                 raises ImportError with install guidance if ffcv missing
└── get_batch_source(use_ffcv: bool) -> BatchSource
```

`dynvision/data/datamodule.py`:
- Drop `import ffcv` and `from dynvision.data.ffcv_dataloader import get_ffcv_dataloader`
  at module scope.
- Add `from dynvision.data.batch_source import get_batch_source`.
- `_create_ffcv_loader` becomes `_create_backend_loader`, delegating to
  `get_batch_source(self.config.data.use_ffcv).create_loader(...)`. Both the former
  `_create_ffcv_loader`/PyTorch preview path and the FFCV path route through the same
  adapter selection, so the `use_ffcv` check now happens in exactly one place
  (`get_batch_source`) instead of being duplicated inside every call site.

`dynvision/params/data_params.py`:
- Replace the local `OrderOption` try/except stand-in with
  `from dynvision.data.batch_source import OrderOption`, removing the duplicate
  definition. `dynvision.data.batch_source` has no eager ffcv dependency, so this does
  not reintroduce the import-time crash.

## 5. Tests (written first, TDD)

`tests/data/test_batch_source.py`:
- `import dynvision.data.batch_source` succeeds even when `ffcv` is not importable
  (simulated via monkeypatching `sys.modules`/`builtins.__import__`).
- `get_batch_source(False)` returns a `TorchAdapter`; `.loader_class` resolves to
  `dynvision.data.dataloader.get_data_loader` without touching ffcv.
- `get_batch_source(True)` returns an `FFCVAdapter`; when ffcv is unavailable,
  instantiating/using it raises a clear `ImportError` mentioning `use_ffcv` and how to
  install ffcv (not a bare `ModuleNotFoundError` from deep inside `ffcv_dataloader.py`).

`tests/data/test_datamodule_import.py`:
- Regression test for the reported issue: `import dynvision.data.datamodule` (and
  `runtime.train_model`/`init_model`/`test_model`, if importable in the test env) succeeds
  when `ffcv` is not importable.

## 6. Non-goals

- Not rewriting `ffcv_dataloader.py`/`ffcv_datasets.py`/`ffcv_operations.py` internals.
- Not adding a fake/in-memory `BatchSource` adapter for `DynVisionDataModule` testing —
  flagged in the issue as a future win, out of scope for this fix.
