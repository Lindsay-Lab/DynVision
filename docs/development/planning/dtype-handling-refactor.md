# Dtype handling refactor

**Status:** in progress
**Branch:** `refactor/dtype-policy`
**Scope:** Level 2 — central `DtypePolicy` with a single precision→dtype resolution source.

## Problem

A cluster test run crashed with `RuntimeError: Input type (c10::Half) and bias type
(float) should be the same` inside the first feedforward `nn.Conv2d`. The model ran in
float32 (precision `"32"`) while the test data pipeline emitted float16, with no autocast
to reconcile them.

Root cause: dtype handling is split across many independent, mutually inconsistent
resolution sites, and the training path coordinates component dtypes while the testing
path does not.

## Root-cause chain (the bug that started this)

1. `config_defaults.yaml:148` sets unscoped `precision: "32"`.
2. `TestingParams.get_aliases()` maps `"precision" -> "trainer.precision"`
   (`testing_params.py:376`). Alias resolution rewrites the bare key to
   `trainer.precision` and deletes the bare `precision` key before component routing.
3. `data.precision` therefore stays `None`.
4. `DataParams.resolve_dtype_precision_compatibility` calls
   `get_effective_dtype_from_precision(None)`, whose default fallback is the string
   `"float16"` → `data.dtype = torch.float16`.
5. Dataset applies `ConvertImageDtype(float16)`; the fp32 model receives Half input
   → mismatch at the first conv.

Training did not crash because `TrainingParams` aliases only `"prec"` (so unscoped
`precision` routes to both trainer and data) and `TrainingParams.coordinate_component_dtypes`
force-aligns data to trainer. `TestingParams` has neither.

## Design decisions (approved)

| # | Decision |
|---|----------|
| D1 | `precision` stays canonical on `TrainerParams` only (Lightning concept). `dtype` (torch.dtype) is the derived cross-component value. |
| D2 | `DataParams.precision` deleted. `ModelParams.target_dtype` deleted. `config_defaults.yaml` `target_dtype` line deleted. |
| D3 | Unknown/unmapped precision → **warning + default float32** (no silent float16 fallback). |
| D4 | Resolved dtype = **stored-parameter dtype** (Lightning semantics): `16-mixed`/`bf16-mixed` keep fp32 weights → data fp32. Only full modes (`16`/`bf16`/`32`/`64`) pre-cast. |
| D5 | Expose only `data.dtype` as a user-settable override now; trainer/model/checkpoint slots remain internal (extensible later). |
| D6 | If user sets `data.dtype` conflicting with trainer effective dtype → **warn + force-align**. |
| D7 | Refactor covers `DtypeDeviceCoordinator` (its job is dtype handling). |
| D8 | TDD + feature branch + this planning doc. |

## Canonical precision → stored-param dtype map

| precision | stored-param dtype |
|-----------|--------------------|
| `32` / `"32"` / `"32-true"` | `torch.float32` |
| `16` / `"16"` / `"16-true"` | `torch.float16` |
| `64` / `"64"` / `"64-true"` | `torch.float64` |
| `bf16` / `"bf16"` / `"bf16-true"` | `torch.bfloat16` |
| `16-mixed` | `torch.float32` (master weights) |
| `bf16-mixed` | `torch.float32` (master weights) |
| unknown / None / int 32 / int 16 / int 64 | **warn → torch.float32** |

## New central module

`dynvision/utils/dtype_policy.py`:

- `PRECISION_TO_DTYPE: Dict[str, torch.dtype]` — the single canonical map above.
- `normalize_precision(precision) -> str` — accepts str/int/torch.dtype/bool/None.
- `resolve_dtype(precision, *, warn_fallback=True) -> torch.dtype` — returns torch.dtype,
  warns + returns float32 on unknown.
- `DtypePolicy` frozen dataclass with slots `trainer_dtype`, `model_dtype`, `data_dtype`,
  `checkpoint_dtype` + `from_precision(precision, *, data_dtype=None)`.
- `coordinate_component_dtypes(config)` — shared helper: derive trainer dtype, force-align
  `data.dtype` with a warning on divergence.

## Per-file changes

- `dynvision/utils/dtype_policy.py` — new.
- `dynvision/utils/torch_utils.py` — `get_effective_dtype_from_precision` becomes a thin
  delegate returning `torch.dtype`.
- `dynvision/params/trainer_params.py` — `_effective_dtype` stored as `torch.dtype`;
  `get_effective_dtype()` returns `torch.dtype`.
- `dynvision/params/data_params.py` — delete `precision` field + validator +
  `resolve_dtype_precision_compatibility`; keep `dtype` as Optional override.
- `dynvision/base/coordination.py` — delete private `dtype_map`; route
  `_determine_dtype_from_lightning` through `resolve_dtype`; fallback float16→float32.
- `dynvision/params/training_params.py` — replace `coordinate_component_dtypes` body with
  shared helper.
- `dynvision/params/testing_params.py` — delete `"precision"` alias; add
  `coordinate_component_dtypes` after-validator (shared helper) + `get_coordinated_dtype`.
- `dynvision/params/model_params.py` — delete `target_dtype`.
- `dynvision/configs/config_defaults.yaml` — delete `target_dtype` line.
- `dynvision/data/dataloader.py`, `dynvision/data/ffcv_dataloader.py` — dtype default
  float16 → float32.
- `dynvision/utils/__init__.py` — export `PRECISION_TO_DTYPE`, `resolve_dtype`,
  `DtypePolicy`.

## Tests

`tests/params/test_dtype_policy.py`:

- `resolve_dtype` mapping: `16`→fp16, `16-mixed`→fp32, `bf16-mixed`→fp32, `32`→fp32,
  unknown→warn+fp32.
- Regression: load `TestingParams` and `TrainingParams` from an unscoped `precision`
  config and assert `data.dtype == trainer.get_effective_dtype()`.
