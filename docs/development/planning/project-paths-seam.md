# Place a Seam Under `project_paths`

**Status:** 🚧 IN PROGRESS
**Created:** 2026-09-09
**Issue:** [Lindsay-Lab/DynVision#14](https://github.com/Lindsay-Lab/DynVision/issues/14)
**Branch:** `fix/14-project-paths-seam`

---

## 1. Problem

`dynvision/project_paths.py` instantiates a module-level singleton at import time:

```python
project_paths = project_paths_class()
```

Constructing that singleton has side effects that run the moment anything imports it:

- `os.popen("hostname").read()` inside `iam_on_cluster()` — a shell-out, run twice per
  construction (once in `__init__`'s first branch check, again in the second).
- `os.environ["WANDB_DIR"] = str(self.large_logs.resolve())` — mutates process
  environment unconditionally, even for code paths that never touch wandb.
- A hardcoded personal fallback, `working_dir = Path("/home/rgutzen/01_PROJECTS/Modeling_Dynamical_Vision")`,
  and `user_name = "rg5022"`, baked into the class body.

`dynvision/project_paths.py` is gitignored as a personal file in principle
(`.gitignore:12`), but is in fact tracked in this fork (`git ls-files` confirms it), with
`project_paths_template.py` as a near-duplicate checked-in template for other users. This
issue is about the *seam*, not about removing the personalization — the personal values
stay; only how they're wired in changes.

8 modules import `from dynvision.project_paths import project_paths` and use it as an
already-constructed value (`project_paths.data.raw`, `project_paths.models`,
`project_paths.scripts.configs`, `project_paths.iam_on_cluster()`, etc.):

- `dynvision/data/operations.py`
- `dynvision/model_components/temp_base.py`
- `dynvision/models/cordsnet.py`
- `dynvision/params/mode_registry.py`
- `dynvision/runtime/train_model.py`
- `dynvision/runtime/test_model.py` (commented out)
- `dynvision/utils/data_utils.py`

None of these call sites construct `project_paths_class` themselves — they all rely on
the import-time singleton. There is no way to substitute a test layout (e.g. `tmp_path`)
without either running on the real cluster/local filesystem or monkeypatching
`os.popen`/`os.environ` around the import.

## 2. Root cause

Local and cluster are already two adapters — they're just implemented as
`if self.iam_on_cluster():` branches inside `__init__`, rather than as a seam
(`environment -> layout` is a pure function, but it's inlined into a constructor that
also has an environment-detection side effect and an environment-mutation side effect).
Three separate concerns are fused into one `__init__`:

1. **detecting** the environment (hostname shell-out),
2. **computing** a directory layout from that environment (pure, adapter selection), and
3. **mutating** process state from the result (`WANDB_DIR`).

Only (2) needs to run at construction time. (1) only needs to run once, lazily, before
first use. (3) shouldn't run at construction at all — it belongs at the runtime entry
point that actually starts a wandb run.

## 3. Options considered

1. **Minimal: guard the shell-out, keep everything else.** Wrap
   `os.popen("hostname")` in a cache so it only runs once, leave `WANDB_DIR` mutation and
   the singleton-at-import pattern in place. Cheapest diff, but doesn't address the actual
   ask — imports still have a side effect (env mutation), and there's still no way to
   substitute a test layout without hitting the real filesystem/hostname.
2. **Full seam per the issue: `PathLayout.for_environment(env: Environment) -> PathLayout`.**
   Introduce `dynvision/path_layout.py` with:
   - `Environment` — a small explicit value (`is_cluster: bool`, `hostname: str`),
     constructed either by `detect_environment()` (the hostname shell-out, called
     explicitly instead of at import) or directly (`Environment.local()` /
     `Environment.cluster()`) in tests.
   - `PathLayout` — holds the resolved attributes (`.data.raw`, `.models`, `.figures`,
     etc., same names as today) and is built via `PathLayout.for_environment(env,
     working_dir=None, toolbox_dir=None)`. Local and cluster become two private adapter
     methods (`_local_layout`, `_cluster_layout`) selected by `env.is_cluster` — the `if`
     branches move out of `__init__` into the one place that already needs to choose
     between them.
   - `set_wandb_dir(layout)` — the only place `os.environ["WANDB_DIR"]` is set; called
     explicitly from runtime entry points that start wandb (`train_model.py`), not from
     construction.

   `dynvision/project_paths.py` and `project_paths_template.py` keep their personal
   overrides (`project_name`, `user_name`, hardcoded local fallback path) but stop doing
   construction work themselves — they become thin personalization points on top of
   `path_layout.py`, and `project_paths` becomes a **lazily-constructed proxy**: the
   underlying `PathLayout` is built on first attribute access, not at import. This keeps
   all 8 call sites unchanged (`project_paths.data.raw` still works exactly as before) and
   removes the import-time side effects, without introducing a global constructor call
   that anyone has to remember to make.
3. **Full seam + eager DI everywhere: rewrite all 8 call sites to accept a `PathLayout`
   parameter instead of importing the singleton.** Removes the module singleton entirely
   in favor of explicit dependency injection. Most "pure," but churns 8 files including
   `runtime/train_model.py`'s public API for no immediate testing win — none of today's
   tests actually need `train_model.py` or `cordsnet.py` to take a `PathLayout` argument;
   they need `path_layout.py` itself to be substitutable, which option 2 already provides.
   Out of scope here — flagged as a possible future follow-up if a call site actually
   needs it.

**Decision: Option 2.** It's exactly what the issue asks for (the
`PathLayout.for_environment` seam, same attribute names, `WANDB_DIR` as an explicit call),
it makes `path_layout.py` fully substitutable for tests (`PathLayout.for_environment(Environment.local(),
working_dir=tmp_path, toolbox_dir=tmp_path)` needs no monkeypatching), and it does so
without touching any of the 8 call sites — only how `project_paths` is constructed
changes, not how it's used. Option 3's call-site rewrite is real DI but isn't earned yet;
nothing today needs it, and it can be layered on later without revisiting this module.

## 4. Design

```
dynvision/path_layout.py
├── Environment(is_cluster: bool, hostname: str = "")   # frozen dataclass
│   ├── Environment.local()                             # explicit local env, no hostname
│   └── Environment.cluster()                           # explicit cluster env
├── detect_environment() -> Environment                 # the hostname shell-out, called
│                                                          explicitly, not at import
├── PathLayout                                           # replaces project_paths_class's
│   │                                                      path-computation logic
│   ├── PathLayout.for_environment(env, working_dir=None, toolbox_dir=None) -> PathLayout
│   ├── _local_layout(working_dir, toolbox_dir)          # adapter 1 (today's `else` branch)
│   ├── _cluster_layout(working_dir, toolbox_dir)        # adapter 2 (today's `if` branch,
│   │                                                       scratch-partition overrides)
│   └── iam_on_cluster() -> bool                          # kept for the 2 call sites that
│                                                            use it; reads self._environment
└── set_wandb_dir(layout: PathLayout) -> None             # the only os.environ mutation,
                                                             called explicitly, not from
                                                             construction
```

`dynvision/project_paths.py` (and `project_paths_template.py`, kept in sync as the
checked-in template):

```python
class PersonalPathLayout(PathLayout):
    project_name = "DynVision_Working"      # personalization stays here
    toolbox_name = "DynVision"
    user_name = "rg5022"

    @classmethod
    def _local_layout(cls, working_dir, toolbox_dir):
        if working_dir is None:
            working_dir = Path("/home/rgutzen/01_PROJECTS/Modeling_Dynamical_Vision")
        return super()._local_layout(working_dir, toolbox_dir)


class _LazyPathLayout:
    """Defers construction (and the hostname shell-out inside it) to first use."""

    def __getattr__(self, name):
        layout = self._ensure()
        return getattr(layout, name)


project_paths = _LazyPathLayout()   # no side effects at import time
```

`dynvision/runtime/train_model.py`: add an explicit `set_wandb_dir(project_paths)` call
at the point training starts (before `pl.loggers.WandbLogger` is constructed), instead of
relying on it having already happened as a side effect of importing `project_paths`.

## 5. Tests (written first, TDD)

`tests/test_path_layout.py`:

- `import dynvision.path_layout` has no observable side effects: no `hostname` call, no
  `os.environ` mutation (verified by monkeypatching `os.popen` to raise if called, and by
  asserting `WANDB_DIR` is untouched after import).
- `PathLayout.for_environment(Environment.local(), working_dir=tmp_path, toolbox_dir=tmp_path)`
  resolves `.data.raw`, `.models`, `.figures`, etc. under `tmp_path`, with no filesystem or
  hostname dependency — this is the "substitutable test layout" the issue asks for.
- `PathLayout.for_environment(Environment.cluster(), working_dir=tmp_path, toolbox_dir=tmp_path)`
  overrides `.data.raw`/`.data.interim`/`.data.processed`/`.data.external`/`.models`/
  `.reports`/`.large_logs` onto the scratch-partition pattern, matching today's cluster
  branch behavior.
- `detect_environment()` calls `os.popen("hostname")` exactly once and classifies known
  cluster hostname substrings (`hpc`, `log-`, `greene`, `slurm`, `compute`, `node`,
  `cluster`) as `is_cluster=True`, matching today's `iam_on_cluster()` behavior exactly
  (regression test against the existing marker list).
- `set_wandb_dir(layout)` sets `os.environ["WANDB_DIR"]` to `str(layout.large_logs.resolve())`
  and does nothing else (no other env mutation, doesn't touch `hostname`).

`tests/test_project_paths.py` (regression tests for the lazy proxy):

- `import dynvision.project_paths` does not call `os.popen` and does not set
  `WANDB_DIR` (monkeypatch `os.popen` to raise if called during import).
- First attribute access on `project_paths` (e.g. `project_paths.data.raw`) constructs
  the underlying layout exactly once; a second access reuses the cached layout (assert
  `os.popen` mock call count stays at 1 across two attribute accesses).
- All attributes previously exposed by `project_paths_class` (`.data.raw`, `.data.interim`,
  `.data.processed`, `.data.external`, `.models`, `.notebooks`, `.references`, `.reports`,
  `.figures`, `.logs`, `.large_logs`, `.benchmarks`, `.scripts.*`, `.project_name`,
  `.iam_on_cluster()`) remain accessible through the proxy unchanged.

## 6. Non-goals

- Not rewriting the 8 call sites to take `PathLayout` as an explicit parameter (option 3
  above) — they keep using `from dynvision.project_paths import project_paths` unchanged.
- Not removing the personal hardcoded values (`rg5022`, the local working-dir fallback)
  from `project_paths.py` — the issue explicitly scopes personalization as out of scope.
- Not adding a third `TmpLayout` adapter class — `PathLayout.for_environment(Environment.local(),
  working_dir=tmp_path, toolbox_dir=tmp_path)` already gives tests a fully substitutable
  layout without one, since `working_dir`/`toolbox_dir` were already parameters.
