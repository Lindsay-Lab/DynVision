# Documentation and Implementation TODO

> **Updated 2026-07-01**: Outdated after the 2026-06-30 docs-website overhaul + 2026-07-01 todo sweep. Entries flagged with ✅/⬜ are verified current; unmarked sections may be stale. See [todo-release-0.1.md](todo-release-0.1.md) for release-prep tasks.

This file tracks inconsistencies between documentation and implementation, areas needing improvement, and future work items.

## Docs-Website Overhaul Completed (2026-06-30)

The following high-level documentation tasks were completed during the feature/docs-website branch:

### Structure & Layout
- ✅ Diátaxis framework enforced: quadrant badges via template overrides, per-section accent colours
- ✅ Header: logo-only (site_name empty), dark/light mode adaptive via partials/logo.html
- ✅ Home page: logo hero, Diátaxis-structured nav, citation block
- ✅ Index pages: YAML frontmatter titles instead of h1 headings (prevents 'Index' fallback)
- ✅ Diátaxis CSS: per-quadrant accent colours, Tutorial single-column, Explanation wider line-height

### Content Fixes
- ✅ 40+ broken links fixed (tutorial→tutorials, evaluation→model-testing, docs/ prefix, code-of-contact→code-of-conduct)
- ✅ 575+ blank-line-before-list fixes across 52 files (Python-Markdown requirement)
- ✅ README.md: fixed Quick Start params, docs badge URL, stale paths
- ✅ Installation: added `pip install dynvision` to all install guides (README, getting-started, installation)
- ✅ Stale 'yourusername' placeholder replaced with Lindsay-Lab in installation.md

### Configuration
- ✅ mkdocs.yml: site_name empty, `pymdownx.tasklist` extension added
- ✅ pyproject.toml: added mike to [doc] extra
- ✅ favicon: thumbnail.png → logo.svg (transparent thumbnail was invisible)

### Figures
- ✅ All 28 manuscript figures copied to docs/assets/
- ✅ Non-manuscript PNGs removed (overview, tau, ordering, dynamical_systems_equation, local_recurrence)
- ✅ recurrence_types.png → recurrency_types.png (match manuscript spelling)
- ✅ 17 previously-unused manuscript figures wired into pages with proper captions
- ✅ Image paths normalised to `../../assets/` prefix for depth-2 pages

### New Pages Created (structured placeholders)
- ✅ `explanation/comparison-to-neural-data.md` — ECoG comparison, noise robustness, two-regime dissociation
- ✅ `explanation/engineering-vs-biological-time.md` — delay conversion formulas
- ✅ `reference/layer-operations.md`, `reference/skip-feedback-connections.md`, `reference/integration-strategies.md`
- ✅ `reference/evaluation-metrics.md`, `reference/benchmarking.md`

### Outstanding from the Overhaul (status as of 2026-07-01)
- ✅ Write full prose for thin reference pages:
  `benchmarking.md`, `evaluation-metrics.md`, `skip-feedback-connections.md`,
  `layer-operations.md`, `integration-strategies.md` — completed in 2026-07-01 sweep
- ✅ Add `docs/assets/recurrency_types.png` to main README
- ✅ Convert tutorial/index.md commented links to visible roadmap note
- ✅ Convert user-guide/index.md commented links to visible roadmap note
- ⬜ Code-of-conduct.md is in `not_in_nav` — decide whether it should be surfaced

### Remaining Broken Links (status as of 2026-07-01)
- ✅ `user-guide/training.md` referenced in `reference/model-base.md` — file created, link resolved
- ℹ️ Tutorial/user-guide index aspirational links converted to visible roadmap notes (invisible to readers)
- ℹ️ Any remaining broken links are in development/planning guides (known, non-blocking per `--strict` decision)

### Missing Pages / Sections
- ⬜ **Visualization gallery** — `user-guide/visualization.md` is bare; needs screenshot examples of plot types
- ⬜ **Monitoring callbacks** — no docs for what metrics are logged, where, and how to add custom ones
- ⬜ **StorageBuffer API** — mentioned in model-base but no dedicated reference page
- ⬜ **Parameter override examples** — CLI / YAML / Snakemake override patterns not documented
- ⬜ **Transforms reference** — complete list of available transforms with parameters
- ⬜ **Model naming conventions** — how to name models, variants, checkpoints
- ⬜ **Performance tips cheat sheet** — quick wins for faster training
- ⬜ **Complete parameter reference** — exhaustive list of all parameters by component
- ⬜ **FFCV troubleshooting** — installation issues, when beneficial vs overhead, fallback behavior
- ⬜ **Mixed-precision best practices** — GPU requirements, numerical stability, benchmarks
- ⬜ **Visualization tutorial** — example notebooks showing common plots
- ⬜ **Migration guide** — for users of previous versions
- ⬜ **Custom models template files** — README mentions them, none exist in repo

### Code Modules Without Reference Docs
The following code modules have no corresponding reference page. Not all need one
(some are internal), but high-value candidates are flagged:

- **High priority** (user-facing): `recurrence`, `temporal`, `dynamics_solver`,
  `integration_strategy`, `transforms`, `callbacks`, `datamodule`
- **Medium priority**: `dataloader`, `datasets`, `ffcv_*`, `noise`, `retina`,
  `supralinearity`, `bias`
- **Low priority** (internal/developer): `*_utils`, `*_params`, `project_paths`,
  `snake*`, `mode_registry`

### Reference Pages That Are Grouping / Index Files (not 1:1 with a code module)
These are intentional: `organization`, `models`, `losses`, `model-components`,
`recurrence-types`, `dynamics-solvers`, `configuration`, `workflow`,
`optimizers-schedulers`, `model-base`, `model-architecture`.

### Sections That Could Use a Dedicated Page
- **Recurrence** — currently part of recurrence-types.md but the code module is
  `model_components/recurrence.py`; a separate API reference may help
- **Temporal data** — `temporal.py` is partiality covered in
  temporal-data-presentation.md but not in reference
- **Transforms** — `transforms.py` has no dedicated doc page, only mentioned in
  transform-configuration.md
- **Callbacks** — `callbacks.py` has no reference page
- **Data module** — `datamodule.py`, `dataloader.py`, `datasets.py` have no
  dedicated API reference

## Critical Issues

### 1. Project Naming Inconsistencies

**Issue**: Multiple naming schemes used throughout the project
- **Location**: `Makefile`, `project_paths.py`, documentation
- **Problem**:
  - Makefile uses `rhythmic_visual_attention` instead of `dynvision`
  - `project_paths.py` has both `project_name = "Modeling_Dynamical_Vision"` and `toolbox_name = "DynVision"`
  - Default `working_dir` points to a user-specific absolute path
- **Impact**: Confusion for new users, inconsistent commands
- **Fix Required**:
  - Update Makefile targets to use `dynvision`
  - Reconcile project naming in `project_paths.py`
  - Update documentation to reflect single canonical name

### 2. Broken Links: user-guide/training.md ✅ RESOLVED (2026-07-01)

**Issue**: `model-base.md` referenced a non-existent `user-guide/training.md`.
**Fix**: Created `user-guide/training.md` how-to guide, repointed link, added to nav and `user-guide/index.md`.

### 3. Base Class Documentation Mismatch ✅ VERIFIED — ALREADY CORRECT

**Issue** (original): Documentation suspected of describing wrong base classes

- **Location**: `docs/reference/model-base.md`
- **Finding (2026-07-01)**: `model-base.md` already documents `BaseModel` as
  inheriting from `TemporalBase, LightningBase, StorageBufferMixin,
  MonitoringMixin, DtypeDeviceCoordinatorMixin`, which matches the actual
  definition in `dynvision/base/__init__.py` (lines 41–47) exactly.
- **Impact**: None.
- **Fix Required**: ✅ None (verified consistent)

### 4. Parameter System Documentation vs Implementation

**Issue**: Extensive parameter handling documentation but unclear integration

- **Location**: `docs/user-guide/parameter-handling.md`
- **Problem**:
  - Documents sophisticated Pydantic-based parameter system in `dynvision/params/`
  - System exists in code but documentation doesn't explain how it integrates with:
    - `@alias_kwargs` decorator system used throughout models
    - Snakemake config system
    - PyTorch Lightning's hyperparameter tracking

  - Unclear which system has precedence when both are used
- **Impact**: Developers unsure which parameter system to use
- **Fix Required**:
  - Document integration points between systems
  - Show examples of both systems working together
  - Clarify when to use Pydantic params vs @alias_kwargs

## Recent Updates (2025-11-22)

### Completed Documentation Tasks
1. ✅ **Created losses.md** - Comprehensive reference for CrossEntropyLoss and ActivityLoss
2. ✅ **Created temporal-data-presentation.md** - Complete user guide for temporal features
3. ✅ **Updated model-base.md** - Added temporal presentation and loss configuration sections
4. ✅ **Updated configuration.md** - Added temporal parameters documentation
5. ✅ **Fixed cross-references** - Corrected temporal_dynamics.md references (hyphen→underscore)
6. ✅ **Updated tutorials** - Fixed BaseModel imports, added temporal examples
7. ✅ **Updated user-guide/index.md** - Added temporal-data-presentation.md entry

### New Documentation Gaps Identified
- **Missing**: API reference for monitoring and storage systems
- **Missing**: Comprehensive scheduler documentation (only mentioned briefly)
- **Missing**: Performance benchmarking guide
- **Missing**: Debugging guide for common errors
- **Missing**: Migration guide for users of previous versions

## Documentation Gaps

### 5. Missing Implementation Details ✅ FIXED

**Issue**: Documentation describes features but lacks implementation examples

- **Location**: `docs/user-guide/custom-models.md`
- **Status**: Fixed (2025-11-23)
  - ✅ Completed "Training Configurations" section with:
    - Optimizer configuration examples
    - Learning rate scheduling
    - Custom PyTorch Lightning callbacks
    - Advanced training options

  - ✅ Completed "Troubleshooting Guide" section with:
    - Debug mode usage (3 activation methods)
    - Common issues and solutions (NaN, recurrence, performance, OOM)
    - Debugging tools (anomaly detection, logging, profiling)
    - Response inspection techniques

- **Impact**: Users now have comprehensive implementation guidance
- **Fix Required**: ✅ Complete

### 6. Recurrence Type Images Missing ✅ FIXED

**Issue**: Documentation references images that don't exist

- **Location**: `docs/reference/recurrence-types.md`
- **Status**: Fixed (2026-06-30)
  - ✅ `recurrency_types.png` exists in docs/assets (sourced from manuscript figures)
  - ✅ Individual recurrence-type diagrams (`self_recurrence.png`, etc.) are manuscript figures
    referenced where appropriate and will be added as they become available

- **Impact**: Visual explanations now available
- **Fix Required**: ✅ Complete

### 7. Dynamics Equation Image Missing ✅ FIXED

**Issue**: Reference to equation image that doesn't exist

- **Location**: `docs/reference/dynamics-solvers.md`
- **Status**: Fixed (2026-06-30)
  - ✅ Uses inline LaTeX for equations via `pymdownx.arithmatex`
  - ✅ Removed reference to non-existent `dynamical_systems_equation.png`
- **Impact**: Equations now properly rendered
- **Fix Required**: ✅ Complete

### 8. Custom Models Template Files Missing

**Issue**: Documentation mentions template files that don't exist

- **Location**: README.md mentions "template files and guides"
- **Problem**: No templates found in repository
- **Impact**: New users can't easily bootstrap custom models
- **Fix Required**: Create template files or remove references

## Minor Inconsistencies

### 9. Model Initialization Sequence

**Issue**: Documentation lists slightly different initialization orders

- **Location**:
  - `docs/reference/model-base.md` (lines 86-88)
  - Actual implementation in `base/temporal.py`
- **Discrepancy**: Documentation missing `_verify_architecture()` step
- **Impact**: Minor - sequence is mostly correct
- **Fix Required**: Add missing step to documentation

### 10. Experiment Configuration Wildcards

**Issue**: Documentation uses different wildcard formats in examples

- **Location**:
  - `docs/user-guide/workflows.md` shows one format
  - Claude Code Guide shows expanded format with data_loader and data_args
- **Problem**: Not clear which wildcards are optional vs required
- **Impact**: Users may specify incomplete paths
- **Fix Required**: Standardize wildcard documentation with clear required/optional markers

### 11. Config Mode Detection

**Issue**: Documentation incomplete on mode detection logic

- **Location**: `docs/user-guide/parameter-handling.md`
- **Problem**:
  - States debug mode triggered by `log_level="DEBUG"` OR `epochs <= 5`
  - Doesn't explain what happens if both conditions are met
  - Doesn't explain mode override precedence
- **Impact**: Unpredictable mode activation
- **Fix Required**: Document complete decision tree for mode detection

### 12. GitHub URLs Placeholder ✅ FIXED

**Issue**: Documentation has placeholder GitHub URLs

- **Location**: `docs/index.md` (line 51)
- **Status**: Fixed (2026-06-30) — updated to actual `https://github.com/Lindsay-Lab/dynvision/issues`
- **Fix Required**: ✅ Complete

### 13. Repository Citation Missing ✅ FIXED

**Issue**: Citation section commented out in README

- **Location**: `README.md` (lines 95-109)
- **Status**: Fixed (2026-06-30)
  - ✅ Active preprint citation with DOI in README
  - ✅ Software citation (Zenodo) in HTML comment pending DOI assignment
  - ✅ Citation block on docs/index.md home page
- **Fix Required**: Uncomment Zenodo citation once DOI assigned

## Code vs Documentation Mismatches

### 14. Data Loader Names ✅ VERIFIED — NO MISMATCH

**Issue** (original): Suspected inconsistent naming of data loaders

- **Location**: `docs/explanation/temporal_dynamics.md` vs `dynvision/data/dataloader.py`
- **Finding (2026-07-01)**: Both forms are valid and used correctly.
  - The Python class names in `dynvision/data/dataloader.py` are
    `StimulusDurationDataLoader`, `StimulusIntervalDataLoader`,
    `StimulusContrastDataLoader` (see `DATALOADER_CLASSES`, lines 1026–1033).
  - `get_data_loader_class()` (line 1077) appends `"DataLoader"` automatically
    when it is absent, so the bare form (`StimulusDuration`) is a valid alias
    used in config/CLI `data_loader=` contexts.
  - Docs use each form in the correct context: full class name in Python
    import/instantiation examples (`custom-model.md`,
    `temporal-data-presentation.md`), bare alias in config/CLI examples
    (`model-testing.md`, `data-processing.md`, `workflow.md`).
    `getting-started.md` documents the alias convention explicitly.
- **Impact**: None — examples work as written.
- **Fix Required**: ✅ None (verified consistent)

### 15. Operation Sequence Names ✅ FIXED (2026-07-01)

**Issue**: Documentation uses different operation names than code

- **Location**: `docs/reference/layer-operations.md`
- **Finding (2026-07-01)**: The actual DyRCNN `layer_operations`
  (`dynvision/models/dyrcnn.py` lines 232–243) is:
  `["layer", "addext", "addskip", "addfeedback", "tstep", "nonlin", "supralin",
  "record", "delay", "pool"]`. The base default in
  `dynvision/base/temporal.py` (lines 319–331) is the same but adds `"norm"` at
  the end. The doc previously listed `["rconv", "addskip", "addfeedback",
  "tstep", "nonlin", "record", "delay", "pool"]`, i.e. it used `"rconv"` instead
  of `"layer"` and omitted `"addext"` and `"supralin"`.
- **Impact**: Users misled about the real operation pipeline.
- **Fix Required**: Update `layer-operations.md` to the verified operation list.

### 16. Solver Naming Inconsistency ✅ VERIFIED — ALREADY CORRECT (2026-07-01)

**Issue** (original): Documentation suspected of using different solver names

- **Location**: `docs/reference/dynamics-solvers.md`
- **Finding (2026-07-01)**: Docs correctly document config strings `euler`/`rk4`
  (per `model_params.py` `Literal["euler", "rk4"]`) vs class names
  `EulerStep`/`RungeKuttaStep`.
- **Impact**: None.
- **Fix Required**: ✅ None (verified consistent)

## Performance and Optimization

### 17. FFCV Integration Documentation

**Issue**: Documentation mentions FFCV but lacks troubleshooting

- **Location**: Multiple files mention `use_ffcv: true`
- **Problem**: No documentation on:
  - FFCV installation issues
  - When FFCV is beneficial vs overhead
  - Fallback behavior if FFCV fails
- **Impact**: Users may have setup issues
- **Fix Required**: Add FFCV troubleshooting guide

### 18. Mixed Precision Documentation

**Issue**: Mixed precision mentioned but not fully documented

- **Location**: References to `precision: "bf16-mixed"` in configs
- **Problem**:
  - No guidance on GPU requirements
  - No discussion of numerical stability implications
  - No benchmarks showing speedup
- **Impact**: Users may enable inappropriately
- **Fix Required**: Add mixed precision best practices guide

## Future Enhancements

### 19. Test Suite Needed

**Issue**: No formal test suite exists

- **Current State**: No `tests/` directory, no pytest configuration
- **Needed Tests**:
  - Temporal dynamics correctness
  - Recurrence integration
  - Data loading pipelines
  - Parameter validation
  - Model initialization
  - Coordination network building
- **Impact**: No automated validation of code changes
- **Fix Required**: Implement comprehensive test suite

### 20. Visualization Examples Needed

**Issue**: Visualization code exists but lacks examples

- **Location**: `dynvision/visualization/` exists but not well documented
- **Needed**:
  - Example notebooks showing common visualizations
  - Gallery of possible plots
  - Customization guide
- **Impact**: Users may not leverage visualization capabilities
- **Fix Required**: Create visualization tutorial and gallery

### 21. Model Zoo Expansion

**Issue**: README mentions "Comprehensive Model Zoo" but limited models

- **Current Models**: AlexNet, CorNetRT, ResNet variants, CordsNet, DyRCNN
- **Potential Additions**: VGG, EfficientNet, Vision Transformers with recurrence
- **Impact**: Users may expect more pre-built models
- **Fix Required**: Either expand model zoo or adjust marketing language

### 22. Cluster Integration Documentation ✅ FIXED

**Issue**: Cluster integration exists but not well documented

- **Location**: `cluster/` directory and `docs/user-guide/cluster-integration.md`
- **Status**: Fixed (2026-06-30)
  - ✅ Complete cluster-integration.md with basic + advanced execution methods
  - ✅ Script table, snake-env setup, SLURM profile configuration
  - ✅ Common troubleshooting section
  - ✅ Added cluster path setup from `dynvision/cluster/README.md`
- **Fix Required**: ✅ Complete

## Documentation Style Issues

### 23. Inconsistent Code Example Style

**Issue**: Code examples use different formatting

- **Locations**: Throughout user guides
- **Variations**:
  - Some use full class paths, others don't
  - Some show imports, others assume them
  - Inconsistent comment styles
- **Impact**: Confusing for learners
- **Fix Required**: Establish and follow code example style guide

### 24. Missing Type Hints in Examples

**Issue**: Documentation examples often lack type hints

- **Location**: Throughout tutorials and guides
- **Problem**: Code uses type hints extensively, but examples don't
- **Impact**: Users may not understand expected types
- **Fix Required**: Add type hints to all code examples

## Priority Recommendations (updated 2026-07-01)

**High Priority** (User-blocking issues):

1. Fix project naming inconsistencies (Issue #1)
2. Clarify parameter system integration (Issue #4)
3. Add missing type hints to code examples (Issue #24)

**Medium Priority** (Quality improvements):

4. Document mode detection logic completely (Issue #11)
5. Document FFCV setup and troubleshooting (Issue #17)
6. Create visualization examples and gallery (Issue #20 — also Missing Pages)
7. Standardize experiment-config wildcard documentation (Issue #10)

**Low Priority** (Nice to have):

8. Update model initialization sequence docs (Issue #9)
9. Establish code example style guide (Issue #23)
10. Expand model zoo (Issue #21) or adjust marketing language
11. Add comprehensive test suite (Issue #19)

**Completed since last review:**

- ✅ Issues #2, #3, #5, #6, #7, #12, #13, #14, #15, #16, #22 — all resolved/verified/fixed

## Low-Hanging Fruits (Quick Wins)

These are documentation tasks that can be completed quickly and provide immediate value:

### Medium Additions (1-2 hours each)
11. ✅ **Complete TODO sections in custom-models.md** - COMPLETED (2025-11-23)
    - Training configurations with optimizer, scheduler, callback examples
    - Troubleshooting guide with debug mode and common issues

12. **Create visualization gallery** - Screenshot examples of available plots
13. ✅ **Add inline LaTeX for dynamics equations** - COMPLETED (2026-06-30)
14. **Create parameter override examples** - Show CLI, YAML, and Snakemake override patterns
15. **Document monitoring callbacks** - What metrics are logged and when

### Reference Completions (2-4 hours each)
16. **Document StorageBuffer API** - Methods, usage patterns, memory considerations
17. **Document Monitoring API** - Logging methods, custom metrics, integration
18. **Create transforms reference** - Complete list of available transforms with parameters
19. **Document visualization utilities** - All plot types, customization options
20. **Create complete parameter reference** - Exhaustive list of all parameters by component

## Notes for Contributors

When updating documentation:

- Always verify class names, method signatures, and file paths against actual code
- Include working code examples that have been tested
- Use consistent terminology (check existing docs)
- Add type hints to code examples
- Consider adding diagrams for complex concepts
- Link to related documentation sections

When updating code:

- Update relevant documentation when changing APIs
- Add docstrings following existing style
- Consider backward compatibility for documented features
- Update [Claude Code Guide](../guides/claude-guide.md) if architecture changes significantly
