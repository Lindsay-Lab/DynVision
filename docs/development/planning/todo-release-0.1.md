# DynVision 0.1 Release Preparation Roadmap

**Created**: 2026-06-23
**Status**: ✅ All 10 tasks completed. Remaining: git tag v0.1.0.
**Goal**: Prepare DynVision for its 0.1 release and accompany the manuscript submission.
**Current commit**: `babbf28 feat: 0.1 release preparation`

---

## Task Summary

| # | Task | Priority | Effort | Dependencies |
|---|------|----------|--------|-------------|
| 1 | Separate manuscript files to manuscript repo | 🔥 Critical | 🟡 Medium (4-6h) | None |
| 2 | Clean up old/deprecated/backup files | 🔥 Critical | 🟢 Small (<1h) | None |
| 3 | Fix project naming inconsistencies | 🔥 Critical | 🟢 Small (1-2h) | None |
| 4 | Relax scikit-learn version constraint | ⚡ High | 🟢 Small (<1h) | None |
| 5 | Assess and document Python 3.12 compatibility | ⚡ High | 🟡 Medium (2-4h) | Depends on task 4 |
| 6 | Fill documentation gaps | ⚡ High | 🟡 Medium (3-5h) | None |
| 7 | Update README for release (badges, citation, etc.) | ⚡ High | 🟢 Small (1h) | Depends on tasks 1,2,3 |
| 8 | Add/verify test coverage for core modules | ⭐ Medium | 🟡 Medium (4-8h) | None |
| 9 | Standardize logging and improve error messages | ⭐ Medium | 🟡 Medium (2-4h) | None |
| 10 | Update version to 0.1.0 and add release notes | 🔥 Critical | 🟢 Small (<1h) | Blocks release |

---

## Detailed Task Specifications

### Task 1: Separate Manuscript Files to Manuscript Repo 🔥

**Rationale**: Manuscript-specific scripts, figures, and Snakemake rules should live in the manuscript repository (`/home/rgutzen/01_PROJECTS/Modeling_Dynamical_Vision/`), not in the DynVision toolbox. This keeps the toolbox general-purpose and reduces maintenance burden.

**Files to MOVE** (8 files, ~6,974 lines total):

| File | Lines | Reason |
|------|-------|--------|
| `dynvision/workflow/snake_manuscript.smk` | 831 | Manuscript-only Snakemake workflow (13 rules: manuscript_figures, plot_stability, plot_performance_manuscript, plot_dynamics_manuscript, plot_all_dynamics_manuscript, plot_unrolling, plot_reference_models, dataloader, imagenet, benchmark_training, benchmark_dagger_reruns, process_all_wandb_data, current_figure) |
| `dynvision/visualization/plot_all_dynamics_manuscript.py` | 1626 | Composite 3-experiment dynamics figure with Groen comparison |
| `dynvision/visualization/plot_performance_manuscript.py` | 1424 | 4-panel performance + Jang benchmarks figure |
| `dynvision/visualization/plot_dynamics_with_groen.py` | 1484 | Dynamics with Groen et al. 2022 human V1 data overlay |
| `dynvision/visualization/plot_reference_models.py` | 1039 | Reference model comparison (CorNetRT, CordsNet, DyRCNNx8) |
| `dynvision/visualization/fetch_benchmarks.py` | 745 | W&B benchmarking data fetch for resource table |
| `dynvision/visualization/count_params.py` | 154 | Parameter counting for benchmark table |
| `dynvision/visualization/weight_caption_metrics.py` | 471 | Weight distribution metrics for figure captions |

**Dependencies within manuscript files**:

- `plot_all_dynamics_manuscript.py` imports from `plot_dynamics_with_groen.py`
- `plot_performance_manuscript.py` imports from `plot_performance.py` (general) — will need path adjustment
- `plot_dynamics_with_groen.py` imports from `plot_dynamics.py` (general) — will need path adjustment
- `fetch_benchmarks.py` imports and calls `count_params.py`
- `snake_manuscript.smk` orchestrates all manuscript rules; references `snake_visualizations.smk` for `plot_responses_tripytch`

**Borderline files — KEEP in DynVision**:

- `plot_response_tripytch.py` — used by general `snake_visualizations.smk`, not just manuscript
- `plot_performance.py`, `plot_dynamics.py`, `plot_responses.py`, `plot_training.py` — general-purpose visualization modules
- `snake_visualizations.smk` — general workflow rules

**Split procedure**:

1. Create branch `manuscript-split` in DynVision repo
2. Move the 8 files to a `scripts/` directory in the manuscript repo
3. Adjust import paths in moved files (they currently import from `dynvision.visualization.*`)
   - Option A: Add `sys.path` manipulation at top of each script
   - Option B: Install DynVision as editable dependency, change imports to use package paths
   - Option C: Copy necessary utility functions into manuscript scripts (avoids DynVision dependency)
4. Remove manuscript-only Snakemake rules from DynVision
5. Update the manuscript repo's workflow to reference DynVision as a dependency
6. Merge `manuscript-split` into `dev`

---

### Task 2: Clean Up Old/Deprecated/Backup Files 🔥

**Rationale**: Remove files that are superseded or erroneous, reducing repository size and confusion.

**Files to DELETE** (7 files, ~8,939 lines total):

| File | Lines | Superseded By |
|------|-------|---------------|
| `dynvision/visualization/plot_performance_manuscript_backup.py` | 890 | `plot_performance_manuscript.py` (in manuscript repo after Task 1) |
| `dynvision/visualization/plot_response_tripytch_old.py` | 1497 | `plot_response_tripytch.py` |
| `dynvision/visualization/plot_response_tripytch_old_old.py` | 1497 | `plot_response_tripytch.py` |
| `dynvision/visualization/plot_training_old.py` | 1381 | `plot_training.py` |
| `dynvision/visualization/plot_responses_old.py` | 1755 | `plot_responses.py` |
| `dynvision/models/cordsnet_old.py` | 265 | Current CordsNet in `dynvision/models/cordsnet.py` |
| `dynvision/data/noise_old.py` | 652 | Current noise implementation in `dynvision/data/` |

**Procedure**:

1. Verify none of these files are imported by other code
2. Delete files
3. Add `*_old.py`, `*_backup.py` patterns to `.gitignore`
4. Commit

**Verification**: `grep -r "old\.py\|backup\.py" --include='*.py'` returns no references in remaining code.

---

### Task 3: Fix Project Naming Inconsistencies 🔥

**Rationale**: Multiple naming schemes create confusion for users and developers.

**Issues Found**:

| Location | Current | Should Be | Notes |
|----------|---------|-----------|-------|
| `Makefile:6` | `PROJECT_ABRV = rva` | `dvn` or remove | `rva` = "rhythmic visual attention" (old name) |
| `Makefile:121` | `flake8 rhythmic_visual_attention` | `flake8 dynvision` | References old directory name |
| `Makefile:122` | `black --check --config pyproject.toml rhythmic_visual_attention` | `black --check --config pyproject.toml dynvision` | Same |
| `Makefile:128` | `black --config pyproject.toml rhythmic_visual_attention` | `black --config pyproject.toml dynvision` | Same |
| `Makefile:165` | `rhythmic_visual_attention/data/make_dataset.py` | `dynvision/data/make_dataset.py` (verify this script exists) | May need update or removal |
| `dynvision/project_paths.py:13` | `project_name = "Modeling_Dynamical_Vision"` | Should be `"DynVision"` or documented as separate from toolbox | Mixed naming |
| `dynvision/project_paths.py:14` | `toolbox_name = "DynVision"` | OK | But inconsistent with `project_name` |
| `dynvision/project_paths.py:23` | `working_dir = Path("/home/rgutzen/01_PROJECTS/Modeling_Dynamical_Vision")` | User-specific path — should default to `Path.home()` or be configurable via env var | Hard-coded absolute path |

**Procedure**:

1. Update `Makefile` to use `dynvision` consistently
2. Either remove `PROJECT_ABRV` or rename to `dvn`
3. Reconcile `project_paths.py` naming — decide whether `project_name` should be "DynVision" (and if the working directory should be separate)
4. Remove hard-coded user paths or make them configurable via environment variables (e.g., `DYNVISION_WORKING_DIR`)
5. Update the Claude guide's "Known Issues" section after fixes

**Verification**: `grep -ri "rhythmic_visual_attention\|rva" Makefile` returns no matches. `make lint` and `make format` work correctly.

---

### Task 4: Relax scikit-learn Version Constraint ⚡

**Rationale**: The current constraint `scikit-learn ~=1.1.0` is too narrow and may not install on Python 3.11 (only 1.1.3+ has 3.11 wheels). DynVision only uses `sklearn.metrics.confusion_matrix` in one file, an API that has been stable since scikit-learn 0.18.

**Recommended change in `pyproject.toml`**:
```diff
- "scikit-learn ~=1.1.0",
+ "scikit-learn >=1.2.0,<2",
```

**Justification**:

- The only scikit-learn API used is `confusion_matrix(y_true, y_pred)` in `dynvision/visualization/plot_confusion_matrix.py`
- This API has not changed across all scikit-learn 1.x versions (1.2 through 1.9)
- scikit-learn 1.2+ provides full Python 3.11 support in CI
- No breaking changes in any 1.x version affect this code

**Procedure**:

1. Edit `pyproject.toml` line 30
2. Verify no other scikit-learn usage exists: `grep -r "sklearn\|scikit" dynvision/ --include='*.py'`
3. Test: `pip install -e ".[dev]"` with the new constraint

---

### Task 5: Assess and Document Python 3.12 Compatibility ⚡

**Rationale**: Python 3.12 was released 2023-10 and is becoming the default in many environments. Users and collaborators will expect compatibility.

**Compatibility Summary** (from @librarian research):

| Dependency | Current Spec | 3.12 Compatible? | Minimum 3.12 Version |
|---|---|---|---|
| `scipy` | `==1.12` | ✅ Yes | No bump needed |
| `snakemake` | `~=9.1.5` | ✅ Yes | No bump needed |
| `snakemake-executor-plugin-cluster-generic` | `~=1.0.0` | ✅ Yes (inherits from snakemake) | No bump needed |
| `numba` | `~=0.61.0` | ✅ Yes | No bump needed |
| `torch` | `>=2.2.0` | ❌ No | `>=2.4.0` |
| `torchvision` | `>=0.16.0` | ❌ No | `>=0.19.0` (paired with torch 2.4) |
| `pytorch-lightning` | `>=2.0.0` | ❌ No | `>=2.4.0` |
| `scikit-learn` | `~=1.1.0` | ❌ No | `>=1.4.0` (recommended after Task 4) |
| `ffcv` | `~=1.0.2` | ⚠️ Unclear | **Test required** — last released before 3.12 |

**Key risk: FFCV**
FFCV 1.0.2 (last released 2023-03-05) predates Python 3.12. The Ubuntu package was removed for 3.12 due to numba incompatibility. With numba 0.61.0 supporting Python 3.12, FFCV *may* work but needs explicit testing. If FFCV fails:

- Option A: Stay on Python 3.11 for the 0.1 release
- Option B: Document FFCV as optional, provide fallback to standard PyTorch DataLoader
- Option C: Replace FFCV with an alternative (WebDataset, DALI)

**Recommended approach for 0.1 release**:

1. **Keep `python_requires = ">=3.11,<3.14"`** (forward-compatible but non-breaking)
2. Document that 3.12+ is not yet officially supported
3. Add Python 3.12 testing as a post-release item
4. If time allows, test FFCV on 3.12 with numba 0.61+

**Procedure**:

1. Update `python_requires` to `>=3.11,<3.14` (allows future Python versions)
2. Update classifiers to include `Programming Language :: Python :: 3.12` (forward-looking)
3. Create `docs/development/python-3.12-compatibility.md` documenting the research above
4. Add a note in README installation instructions about Python version requirements

---

### Task 6: Fill Documentation Gaps ⚡

**Rationale**: The `todo-docs.md` file lists 20+ documentation issues. Several high-priority items must be addressed before release.

**High-priority fixes**:

| # | Issue | Source | Fix |
|---|-------|--------|-----|
| 1 | ✅ Base class docs mismatched (Mixin vs base) | `docs/reference/model-base.md` | Verified 2026-07-01: docs already show `*Mixin` variants matching `base/__init__.py` |
| 2 | ✅ Broken training.md link | `docs/user-guide/model-testing.md` | Fixed 2026-07-01: created `user-guide/training.md`; link repointed and added to nav |
| 3 | ✅ Missing recurrence type images | `docs/reference/recurrence-types.md` | Fixed: `recurrency_types.png` from manuscript figures |
| 4 | ✅ Missing dynamics equation image | `docs/reference/dynamics-solvers.md` | Fixed: replaced with inline LaTeX |
| 5 | ✅ Data loader naming (no mismatch) | docs | Verified 2026-07-01: `get_data_loader_class` appends `DataLoader` automatically, so bare alias and full class name are both valid; docs use each in the correct context |
| 6 | ✅ Solver naming inconsistency | `docs/reference/dynamics-solvers.md` | Verified 2026-07-01: docs already document config strings `euler`/`rk4` (per `model_params.py` `Literal`) vs classes `EulerStep`/`RungeKuttaStep` |

**Medium-priority fixes (status as of 2026-07-01)**:

| # | Issue | Source | Status |
|---|-------|--------|--------|
| 7 | Parameter system integration unclear | `docs/user-guide/parameter-handling.md` | ⬜ Document how Pydantic params + @alias_kwargs work together |
| 8 | Config mode detection incomplete | `docs/user-guide/parameter-handling.md` | ⬜ Document complete mode detection decision tree |
| 9 | GitHub URLs placeholder | `docs/index.md` | ✅ Fixed in docs-website overhaul (2026-06-30) |
| 10 | FFCV troubleshooting missing | Multiple files | ⬜ Add FFCV troubleshooting section |
| 11 | Mixed precision guidance missing | References in configs | ⬜ Add best practices for bf16-mixed |

**Procedure**:

1. ✅ Fix items 1-6 (completed 2026-06-30 / 2026-07-01)
2. Fix items 7, 8, 10, 11 (remaining medium-priority)
3. Run `mkdocs build` or equivalent to verify docs build cleanly
4. Review all doc links for broken references

---

### Task 7: Update README for Release 🔥

**Rationale**: The README is the first thing users see. It needs to be polished for a public release.

**Changes needed**:

1. **Badges**: Update Python version badge, add PyPI badge (if publishing), add test coverage badge
2. **Citation**: Uncomment and fill in the citation section with the actual paper DOI
3. **Installation**: Update Python version recommendation, clarify conda vs pip
4. **Quick Start**: Ensure the code example imports correctly and works
5. **Roadmap link**: Link to this release roadmap
6. **Contributors**: Add contributors section if applicable
7. **Remove "Comprehensive Model Zoo"**: The current model count is modest (AlexNet, CorNetRT, ResNet variants, DyRCNN, CordsNet). Either expand or adjust language.

**Procedure**:

1. Edit README.md sections
2. Verify all links work
3. Test the Quick Start code snippet in a fresh environment

---

### Task 8: Add/Verify Test Coverage ⭐

**Rationale**: The test suite exists but has gaps. Before release, core functionality should have basic test coverage.

**Existing tests** (20 test files):

- `tests/base/`: idle timesteps, temporal masking (2 tests)
- `tests/params/`: literal string conversion, mode registry, config merging, precedence (5 tests + 1 conftest)
- `tests/data/`: data params transforms (2 tests)
- `tests/workflow/`: hash compression (1 test)
- `tests/cluster/`: cluster detection (1 test)
- `tests/losses/`: loss normalization (1 test)

**Gaps** (from roadmap #29, #30, #31):

- No tests for `DataBuffer` (circular buffer correctness)
- No tests for delay propagation
- No tests for recurrence types (full, self, depthwise, pointdepthwise)
- No tests for ODE solver accuracy (Euler vs RK4)

**Minimal viable additions for 0.1 release**:

1. `tests/base/test_data_buffer.py`: Basic circular buffer operations
2. `tests/base/test_dynamics_solver.py`: Euler vs RK4 on simple ODE
3. `tests/model_components/test_recurrence.py`: Smoke test each recurrence type

**Procedure**:

1. Write minimal tests for the three areas above
2. Run `pytest tests/` and fix any failing existing tests
3. Document current test coverage in `docs/development/testing.md`

---

### Task 9: Standardize Logging and Error Messages ⭐

**Rationale**: Inconsistent logging levels and cryptic error messages degrade user experience.

**Issues** (from roadmap #36):

- Logging levels vary across modules (some use `print`, others `logging.info`, others `logger.warning`)
- Error messages lack scientific context (e.g., shape mismatch without explaining which layer)
- No structured logging for cluster debugging

**Procedure**:

1. Audit logging across `dynvision/base/`, `dynvision/models/`, `dynvision/data/`
2. Standardize: use `logging.getLogger(__name__)` consistently
3. Add scientific context to common error messages
4. Ensure `log_level` config parameter controls verbosity correctly

---

### Task 10: Update Version to 0.1.0 and Add Release Notes 🔥

**Rationale**: The current version is `0.0.1` with development status "Planning". The release should reflect the actual state.

**Changes**:

1. Update `pyproject.toml`: `version = "0.1.0"`
2. Update classifiers: `"Development Status :: 4 - Beta"` (from `"1 - Planning"`)
3. Create `CHANGELOG.md` summarizing changes from the initial codebase
4. Create a git tag: `git tag -a v0.1.0 -m "DynVision 0.1.0"`
5. Consider publishing to PyPI (if desired)

---

## Dependency Graph

```
Task 2 (cleanup) ─┐
Task 3 (naming)  ─┤
Task 4 (sklearn) ─┼──> Task 7 (README) ──> Task 10 (release)
Task 1 (split)   ─┘        │
                            └──> Task 5 (Python 3.12 assessment)
Task 6 (docs)    ────────────> (independent)
Task 8 (tests)   ────────────> (independent)
Task 9 (logging) ────────────> (independent)
```

---

## Execution Order (Recommended)

### Phase 1: Foundation Cleanup (Day 1)
1. **Task 2**: Delete old/deprecated files
2. **Task 3**: Fix project naming
3. **Task 4**: Relax scikit-learn constraint

### Phase 2: Manuscript Separation + Docs (Day 1-2)
4. **Task 1**: Move manuscript files to manuscript repo (parallel with Task 6)
5. **Task 6**: Fix documentation gaps

### Phase 3: Release Polish (Day 2)
5. **Task 5**: Document Python 3.12 compatibility status
6. **Task 7**: Update README
7. **Task 9**: Standardize logging

### Phase 4: Quality Assurance (Day 2-3)
8. **Task 8**: Add/verify test coverage
9. **Task 10**: Version bump + release notes + tag

---

## Success Criteria for 0.1 Release

- [ ] Repository is free of old/deprecated/backup files (Task 2)
- [ ] Manuscript-specific code lives in manuscript repo, not in DynVision (Task 1)
- [ ] All internal naming is consistent — DynVision, no rva/rva references (Task 3)
- [ ] scikit-learn constraint is relaxed (`>=1.2.0,<2`) (Task 4)
- [ ] Python version compatibility is documented (Task 5)
- [ ] README is polished — correct badges, citation, links, code examples (Task 7)
- [x] Documentation has no user-facing broken links and uses correct class names (Task 6 — high-priority fixes verified)
- [x] Thin reference pages fleshed out, code-vs-doc mismatches resolved (Task 6 — 2026-07-01 sweep)
- [ ] Core modules have basic test coverage — DataBuffer, dynamics solver, recurrence (Task 8)
- [ ] Version is `0.1.0` with `Development Status :: 4 - Beta` (Task 10)
- [ ] CHANGELOG.md documents changes (Task 10)
- [ ] Git tag `v0.1.0` is created (Task 10)

---

## Post-Release Items (not blocking 0.1)

These items from the existing roadmap should be deferred:

| Roadmap # | Item | Reasoning |
|-----------|------|-----------|
| #25 | Pre-commit hooks | Important but not release-blocking |
| #27 | Dev env setup script | Nice-to-have |
| #32 | CI/CD pipeline | Requires GitHub Actions setup; post-release |
| #34 | Configuration validator | High-impact but can be added in 0.1.1 |
| #38 | Benchmarking suite | Requires significant effort; post-release |
| #41 | Quick-start Jupyter notebook | High-value but can follow release |
| #44 | API reference auto-generation | Requires mkdocstrings setup; post-release |

---

## References

- [todo-docs.md](todo-docs.md): Documentation fixes and implementation mismatches
- [todo-roadmap.md](todo-roadmap.md): Long-term development roadmap
- [Claude Code Guide](../guides/claude-guide.md): Architecture overview
- Python 3.12 compatibility data: from @librarian tasks (2026-06-23)
- scikit-learn compatibility data: from @librarian tasks (2026-06-23)
- Manuscript file analysis: from @explorer tasks (2026-06-23)
