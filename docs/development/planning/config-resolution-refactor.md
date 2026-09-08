# Splitting Config Resolution from Config Validation

**Status:** 🚧 IN PROGRESS
**Created:** 2026-09-08
**Issue:** [Lindsay-Lab/DynVision#13](https://github.com/Lindsay-Lab/DynVision/issues/13)
**Branch:** `refactor/split-config-resolution` (stacked on `refactor/dtype-policy`)
**Related:** `docs/development/planning/dtype-handling-refactor.md`

---

## 1. Problem

Config resolution and pydantic validation are fused into a single interface.
`CompositeParams.from_cli_and_config` is the only way to exercise resolution, and it
ends by constructing a fully-validated params object. There is no seam between
"here is the merged parameter dict" and "now validate it".

### 1.1 Actual call structure

The issue describes five sequential phases on `from_cli_and_config`. The real
structure is slightly different and worth recording, because it makes the fusion
tighter than the issue text suggests:

```
from_cli_and_config(config_path, override_kwargs, args)
├── _get_config_and_cli_params_separate()      # I/O: CLI parse + config file load
│   └── _load_config_file()                    # ALSO does mode-section merging:
│       ├── _deep_merge()
│       ├── _remove_conflicting_base_keys()
│       └── _flatten_component_sections()
├── _resolve_mode_overrides()                  # ModeRegistry toggles -> flattened patches
├── _strip_mode_toggle_keys()                  # in-place mutation of both source dicts
├── _separate_component_configs_two_sources()  # <-- FUSION POINT
│   ├── _separate_single_source()  (per source)
│   │   └── _resolve_aliases_with_precedence()
│   ├── merge sources (config -> modes -> cli)
│   ├── apply preprocessors
│   └── INSTANTIATE pydantic components        # validation happens here
└── cls(**separated)                           # validation of the composite
```

Two corrections to the issue text:

- `_resolve_aliases_with_precedence` is **not** a top-level phase 5. It is called
  from inside `_separate_single_source`, once per source. It cannot be reached
  without going through source separation.
- Component instantiation (pydantic validation) happens inside
  `_separate_component_configs_two_sources`, *before* the outer `cls(**separated)`.
  So validation is interleaved with resolution at two different levels, not appended
  after it.

Neither correction changes the diagnosis — they make it worse. The seam is buried
one level deeper than the issue assumed.

### 1.2 Why this matters — the `c10::Half` crash

`TestingParams` declared an alias `precision` → `trainer.precision`.
`_resolve_aliases_with_precedence` rewrites the alias to its target and **deletes the
alias key in every branch**. By the time component field routing ran, the unscoped
`precision` key no longer existed, so `DataParams.precision` was never populated and
silently stayed `None` → float16 data tensors fed into an fp32 model → `c10::Half`
runtime crash.

No test could reach that intermediate state. Inspecting the dict between alias
resolution and component instantiation required constructing a fully valid
`TestingParams`, which requires torch, real dataset paths, and a real model
checkpoint. The bug was only observable as a crash at training time.

### 1.3 Consequences

- **Depth without a seam.** `from_cli_and_config` is a ~40-line method that transitively
  runs ~600 lines of resolution logic and then validates. Its only observable output is
  a fully-constructed params object.
- **Untestable precedence.** Alias precedence, mode precedence, and scoped/unscoped
  precedence are three interacting rule sets with no direct test surface.
- **Import weight.** Testing any resolution rule pulls in pydantic, torch, and
  `dynvision.utils` — resolution itself needs none of these.
- **Poor locality.** A precedence bug spans a classmethod chain, a `_load_config_file`
  override, and a field validator across four files.

---

## 2. Constraints

Discovered by auditing callers before designing:

1. **Existing tests call the private helpers directly.**
   `tests/params/test_params_mode_config_merging.py` calls `CompositeParams._deep_merge`,
   `._remove_conflicting_base_keys`, and `._flatten_component_sections` as classmethods
   (10+ call sites). These must keep working.
2. **Subclasses override resolution hooks.**
   `_handle_unscoped_param` is overridden in `InitParams` (→ model),
   `TestingParams` (→ model *and* data), and `TrainingParams` (→ model).
   `get_component_preprocessors` is overridden in `InitParams`, `TestingParams`,
   `TrainingParams`. `get_component_assignment_order` is overridden in `InitParams`.
   The resolver must accept these as injected behaviour, not inherit them.
3. **`get_aliases()` is a classmethod chain.** Aliases are assembled via `super()`
   calls across `BaseParams` → each composite subclass.
4. **`ProvenanceRecord` must survive resolution.** Provenance is attached per key,
   re-scoped during routing (`with_scope`), and tagged during preprocessing
   (`add_mutation("derived")`).
5. **`ProvenanceRecord`/`ParamsDict` currently live in `base_params.py`**, which imports
   pydantic. A pydantic-free resolver cannot import from there.
6. **Baseline is green.** 191 passed / 3 skipped, plus 3 pre-existing `ffcv` failures in
   `tests/data/test_transforms.py` (unrelated; ffcv is disabled on the parent branch).

---

## 3. Options considered

### Level 1 — Extract phases as module-level functions

Move the phase bodies into `resolution.py` as free functions; keep the existing
classmethods as thin delegating wrappers.

- ✅ Cheap, low risk, satisfies constraint 1 for free.
- ❌ Functions still take `cls` or a pydantic class to read `model_fields`, so they still
  import pydantic. No real seam — the interface has not shrunk, it has only moved.
- ❌ Does not deliver the issue's stated win ("no pydantic dependency").

### Level 2 — `resolve(sources, schema) -> ResolvedConfig` (**recommended, chosen**)

Introduce a plain-data description of the target (`ConfigSchema`) and a plain-data
result (`ResolvedConfig`). Resolution becomes a pure function over dicts.
`CompositeParams` builds the schema from its pydantic classes, calls `resolve()`, then
instantiates. Existing classmethods stay as delegating wrappers.

- ✅ One interface: `resolve()`. The seam callers cross is the seam tests cross.
- ✅ Resolver imports stdlib + `mode_registry` + `provenance` only — no pydantic, no torch.
- ✅ Precedence rules concentrate in one module (locality).
- ✅ One resolver serves `TrainingParams`, `TestingParams`, `InitParams` (leverage).
- ✅ Subclass hooks become explicit fields on `ConfigSchema` (constraint 2) instead of
  implicit inheritance — the extension points become visible in the type.
- ⚠️ Requires moving `ProvenanceRecord`/`ParamsDict` to their own module (constraint 5).
- ⚠️ Larger diff than Level 1.

### Level 3 — Level 2 + extract source loading behind an adapter

Also pull CLI parsing and config-file loading behind a `SourceLoader` protocol, making
`from_cli_and_config` a three-line orchestrator.

- ✅ Fully separates I/O from computation.
- ❌ CLI parsing reads `model_fields` to build the argparse spec — it is genuinely
  coupled to the pydantic classes. Decoupling it means duplicating the field
  introspection, which trades one coupling for a worse one.
- ❌ Out of scope for #13, which is about resolution vs. validation, not I/O vs. compute.

**Decision: Level 2.** It is exactly what the issue specifies, it delivers the testability
win, and it stops short of the speculative Level 3 boundary. Source *gathering* stays on
`CompositeParams` (it is an adapter over argparse/YAML); source *interpretation* moves out.

---

## 4. Target design

### 4.1 New module: `dynvision/params/provenance.py`

Holds `ProvenanceRecord` and `ParamsDict`, moved verbatim from `base_params.py`.
`base_params.py` re-exports both, so every existing import path keeps working.

Dependencies: stdlib only.

### 4.2 New module: `dynvision/params/resolution.py`

Dependencies: stdlib + `dynvision.params.provenance` + `dynvision.params.mode_registry`.
**No pydantic, no torch, no `dynvision.utils`.** Enforced by a test.

```python
@dataclass(frozen=True)
class ConfigSchema:
    """Plain-data description of the composite the resolver targets."""
    component_fields: Mapping[str, frozenset[str]]   # component -> field names
    base_fields: frozenset[str]                      # composite-only field names
    aliases: Mapping[str, str]                       # alias -> dotted target
    mode_name: Optional[str] = None                  # e.g. "test"
    component_order: tuple[str, ...] = ()            # assignment precedence order
    preprocessors: Mapping[str, Preprocessor] = ...  # component -> callable
    unscoped_handler: Optional[UnscopedHandler] = None

@dataclass(frozen=True)
class ConfigSource:
    """One labelled parameter source with its provenance."""
    label: str                                   # "config" | "modes" | "cli"
    params: Mapping[str, Any]
    provenance: Mapping[str, ProvenanceRecord] = ...

@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved, *unvalidated* configuration."""
    components: Dict[str, Dict[str, Any]]                      # ready for cls(**kwargs)
    composite: Dict[str, Any]
    component_provenance: Dict[str, Dict[str, ProvenanceRecord]]
    composite_provenance: Dict[str, ProvenanceRecord]

def resolve(sources: Sequence[ConfigSource], schema: ConfigSchema) -> ResolvedConfig: ...
```

Supporting pure functions, all individually testable:

| Function | Replaces |
| --- | --- |
| `deep_merge(base, override)` | `CompositeParams._deep_merge` |
| `remove_conflicting_base_keys(config, overrides)` | `CompositeParams._remove_conflicting_base_keys` |
| `flatten_component_sections(config, component_names)` | `CompositeParams._flatten_component_sections` |
| `merge_mode_sections(config, mode_name, component_names)` | inline block in `_load_config_file` |
| `resolve_aliases(params, provenance, aliases)` | `_resolve_aliases_with_precedence` |
| `separate_source(source, schema)` | `_separate_single_source` |
| `flatten_nested_payload(payload)` | `_flatten_nested_payload` |
| `gather_mode_toggle_values(...)` / `flatten_mode_patches(...)` | mode helpers |

### 4.3 `CompositeParams` after the refactor

```python
@classmethod
def build_config_schema(cls) -> ConfigSchema:
    """Project the pydantic class graph onto a plain schema. The adapter."""

@classmethod
def from_cli_and_config(cls, config_path=None, override_kwargs=None, args=None):
    config_params, cli_params = cls._get_config_and_cli_params_separate(...)   # I/O
    mode_params, mode_resolution = cls._resolve_mode_overrides(...)            # I/O-ish
    cls._strip_mode_toggle_keys(config_params, cli_params)

    resolved = resolve(sources, cls.build_config_schema())                     # PURE
    return cls._instantiate_resolved(resolved, mode_resolution)                # VALIDATION
```

Every pre-existing classmethod remains as a thin delegation to the new functions, so
constraints 1–3 hold and no caller — test or subclass — has to change.

### 4.4 What stays in the params classes

Field declarations, field/model validators (including `coordinate_component_dtypes`),
alias tables, preprocessor callables, and the `_handle_unscoped_param` hooks. These are
*declarations of intent*; the resolver consumes them via `ConfigSchema`.

---

## 5. Plan (TDD)

1. **Planning doc** (this file). ✔
2. **Tests first** — `tests/params/test_config_resolution.py`, written against the
   not-yet-existing `resolve()` interface:
   - `test_resolution_module_imports_no_pydantic_or_torch` — the architectural invariant.
   - Regression for §1.2: an alias that captures an unscoped key diverts it away from
     other components that declare the same field — assertable on a plain dict, with no
     torch, no checkpoint, no dataset.
   - Alias precedence: same-scope (alias wins), cross-scope (scoped target wins),
     cross-scope (scoped alias wins over unscoped target).
   - Source precedence: config → modes → cli.
   - Scoped/unscoped precedence within a source.
   - Mode section merging and `mode.component.param` routing.
   - `unscoped_handler` hook and composite-base propagation.
   - Provenance survives resolution with correct scope/mutation tags.
3. **Extract `provenance.py`**, re-export from `base_params.py`.
4. **Implement `resolution.py`** until the new tests pass.
5. **Rewire `CompositeParams`** onto `resolve()`; keep delegating wrappers.
6. **Verify** the full suite matches baseline (191 passed, 3 pre-existing ffcv failures).
7. **Update** `docs/development/guides/parameter-processing.md`.

## 6. Success criteria

- [ ] `resolve()` is callable with plain dicts and asserts without pydantic or torch.
- [ ] The `c10::Half` alias-shadowing shape has a direct regression test.
- [ ] All pre-existing params tests pass unchanged.
- [ ] Full suite matches the recorded baseline.
- [ ] Docs guide reflects the new structure.

## 7. Decision log

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-09-08 | Level 2 over Level 1 | Level 1 moves code without shrinking the interface; the pydantic dependency — the thing that makes resolution untestable — survives. |
| 2026-09-08 | Level 2 over Level 3 | CLI parsing is genuinely coupled to `model_fields`; decoupling it duplicates introspection. Out of scope for #13. |
| 2026-09-08 | Keep all private classmethods as delegating wrappers | 10+ existing test call sites and 3 subclass override points. Deleting them is a separate, unrelated breaking change. |
| 2026-09-08 | Move `ProvenanceRecord`/`ParamsDict` to `provenance.py` | Prerequisite for a pydantic-free resolver; re-exported so no import path breaks. |
| 2026-09-08 | Subclass hooks become `ConfigSchema` fields | Makes the extension points visible in the type instead of implicit in the MRO. |
| 2026-09-08 | Branch stacked on `refactor/dtype-policy` | #13 explicitly generalizes the pattern that branch established, and the motivating bug's fix lives there. |
