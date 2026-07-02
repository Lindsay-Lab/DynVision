# Parallel Experiment Processing: Splitting process_test_data

****Status:** ✅ COMPLETED
**Created:** 2025-12-12
**Authors:** Robin Gutzen, Claude (AI Assistant)
**Branch:** `feature/parallel-experiment-processing`

---

## Overview

Split the monolithic `process_test_data` rule into two stages to enable:

1. **Parallel processing** of individual test outputs immediately after generation
2. **Reduced disk pressure** by compressing large test files sooner
3. **Better scalability** when running multiple testing experiments concurrently

## Problem Statement

### Current Bottleneck

The `process_test_data` rule currently:

- **Input:** Multiple test output tuples (test_responses.pt, test_outputs.csv, test_outputs.csv.config.yaml)
- **Processing:** Processes all tests for an experiment sequentially in a single batch
- **Output:** Single aggregated `test_data.csv` file per experiment

**Critical Issues:**

1. **Disk Space Exhaustion:** Large test_responses.pt files (30GB+) accumulate faster than they can be processed
2. **Sequential Processing:** Even though the script processes files one at a time, Snakemake can't start processing until ALL inputs are ready
3. **Memory Constraints:** Current implementation already handles large files carefully, but batching is limited
4. **Workflow Bottleneck:** Cannot leverage parallel execution across multiple tests

### Root Causes

From `snake_runtime.smk:284-368`:

- Rule waits for all category sweep values to complete testing
- Uses `expand()` to collect all test outputs before processing begins
- Single monolithic output prevents incremental progress

From `process_test_data.py`:

- Already designed to process files in batches (line 826: `--batch_size`)
- Already memory-optimized with incremental layer loading
- But entire script must run to completion before freeing disk space

## Proposed Solution

### Two-Stage Architecture

#### Stage 1: Per-Test Processing (New)
**Rule:** `process_single_test`
**Script:** `dynvision/visualization/process_single_test.py`

```python
# Input (per individual test)
input:
    test_responses = .../test_responses.pt
    test_outputs = .../test_outputs.csv
    test_config = .../test_outputs.csv.config.yaml

# Output (per individual test)
output:
    test_data = .../test_data.csv

# Processing
- Process test performance (add first_label_index, accuracy)
- Calculate classifier metrics (confidence, top-k accuracy)
- Calculate layer metrics (response_avg, response_std, etc.)
- Save individual test_data.csv at SAMPLE LEVEL (no metadata, no resolution applied)
- Optionally delete test_responses.pt if --remove_input_responses=True
```

**Key Characteristics:**

- Runs immediately after each test_model execution completes
- Parallel execution across all tests in experiment
- Can delete large test_responses.pt after processing
- Memory-efficient (already implemented in current script)

#### Stage 2: Experiment Aggregation (New)
**Rule:** `aggregate_experiment_data`
**Script:** `dynvision/visualization/aggregate_experiment_data.py`

```python
# Input (all processed tests in experiment)
input:
    test_data_files = expand(.../{test_identifier}/test_data.csv)

# Output (experiment-level aggregated data)
output:
    experiment_data = .../experiment_test_data.csv

# Processing
- Load all test_data.csv files
- Extract metadata from corresponding .config.yaml files (parameter, category, additional_parameters)
- Add metadata columns to each dataframe
- Concatenate into single dataframe
- Apply resolution (sample-level OR class-level aggregation)
- Handle missing measures gracefully (if some tests lack certain metrics)
- Validate consistency (columns, data types)
- Sort by relevant keys
- Save aggregated CSV
```

**Key Characteristics:**

- Lightweight (no heavy computation, just concatenation)
- Only runs when all per-test processing complete
- Fast execution (small CSV files vs large PT files)

---

## Design Questions

### 1. File Cleanup Strategy

**Question:** Should we automatically delete the large input files after successful processing?

**Options:**

- **A) Delete after Stage 1 (per-test):**
  - Delete `test_responses.pt` after creating `test_data.csv`
  - Keep `test_outputs.csv` and `test_outputs.csv.config.yaml` for potential reprocessing
  - **Pros:** Maximum disk space savings, immediate cleanup
  - **Cons:** Cannot reprocess layer metrics without re-running tests

- **B) Delete after Stage 2 (aggregation):**
  - Keep all files until experiment aggregation complete
  - **Pros:** Can reprocess individual tests if needed
  - **Cons:** Disk space pressure remains until full experiment complete

- **C) Configurable via flag:**
  - Add `--remove_input_responses` flag (already exists in current script)
  - User controls cleanup policy per workflow
  - **Pros:** Flexibility for different use cases
  - **Cons:** Requires user decision

**Current Recommendation:** Option A with safety check - delete `test_responses.pt` after Stage 1 success, but only if Stage 1 completes without errors.

---

### 2. Output File Naming and Organization

**Question:** How should we organize the per-test and aggregated output files?

**Current Structure:**
```
reports/
  {experiment}/
    {model_identifier}/
      {data_name}:{data_group}_{status}/
        {test_identifier}/
          test_responses.pt      # Large (30GB+)
          test_outputs.csv       # Small
          test_outputs.csv.config.yaml  # Small
```

**Option A - Add per-test data alongside current outputs:**
```
reports/
  {experiment}/
    {model_identifier}/
      {data_name}:{data_group}_{status}/
        {test_identifier}/
          test_responses.pt      # Large - delete after processing
          test_outputs.csv       # Small - keep
          test_outputs.csv.config.yaml  # Small - keep
          test_data.csv          # NEW - Medium size, processed output
        experiment_test_data.csv  # NEW - Aggregated across all test_identifiers
```

**Option B - Separate processed data directory:**
```
reports/
  {experiment}/
    {model_identifier}/
      {data_name}:{data_group}_{status}/
        raw/
          {test_identifier}/
            test_responses.pt
            test_outputs.csv
            test_outputs.csv.config.yaml
        processed/
          {test_identifier}/
            test_data.csv
        experiment_test_data.csv
```

**Option C - Flat structure (current approach for aggregated data):**
```
reports/
  {experiment}/
    {model_identifier}/
      {data_name}:{data_group}_{status}/
        {test_identifier}/
          test_responses.pt
          test_outputs.csv
          test_outputs.csv.config.yaml
          test_data.csv          # NEW - per-test processed
        test_data.csv            # Current - aggregated (RENAME TO experiment_test_data.csv?)
```

**Current Recommendation:** Option A - simplest migration path, minimal disruption to existing visualization rules that consume aggregated data.

---

### 3. Measure Configuration

**Question:** Should both stages use identical measure configurations, or allow different measures per stage?

**Context:** Current `--measures` parameter specifies:

- Layer metrics: `response_avg`, `response_std`, `spatial_variance`, `feature_variance`
- Confidence measures: `guess_confidence`, `label_confidence`, `first_label_confidence`
- Top-k accuracies: `accuracy_top3`, `accuracy_top5`
- Classifier activations: `classifier_top5`

**Options:**

- **A) Identical measures across both stages:**
  - Stage 1 computes all measures specified in experiment config
  - Stage 2 simply concatenates (no recomputation)
  - **Pros:** Simple, consistent
  - **Cons:** Cannot change measures without reprocessing

- **B) Stage 1 computes all possible measures:**
  - Stage 1 always computes full measure set
  - Stage 2 filters to requested measures during aggregation
  - **Pros:** Flexibility for different analyses
  - **Cons:** Higher computation/storage cost per test

- **C) Configurable per stage:**
  - Allow different measure lists for each stage
  - **Pros:** Maximum flexibility
  - **Cons:** Complex configuration, easy to misconfigure

**Current Recommendation:** Option A - keeps implementation simple, maintains current behavior.

---

### 4. Resolution Handling

**Question:** Should resolution (sample vs class) be applied in Stage 1 or Stage 2?

**Context:** Current `--sample_resolution` parameter controls:

- `sample`: Output at (sample_index, times_index) level
- `class`: Output at (first_label_index, times_index) level with aggregation

**Options:**

- **A) Apply resolution in Stage 1:**
  - Each test_data.csv is already at final resolution
  - Stage 2 just concatenates
  - **Pros:** Maximum disk savings (class-level is smaller)
  - **Cons:** Cannot change resolution without reprocessing tests

- **B) Apply resolution in Stage 2:**
  - Stage 1 always outputs sample-level data
  - Stage 2 aggregates to class-level if requested
  - **Pros:** Flexibility to generate both resolutions from same data
  - **Cons:** Larger per-test files, more computation in Stage 2

- **C) Hybrid approach:**
  - Stage 1 outputs sample-level
  - Stage 2 can apply class-level aggregation OR keep sample-level
  - **Pros:** Flexibility with reasonable storage
  - **Cons:** Stage 2 becomes more complex

**Current Recommendation:** Option A - apply resolution in Stage 1 to minimize storage, consistent with goal of reducing disk pressure.

---

### 5. Error Handling and Partial Results

**Question:** How should we handle failures in individual tests?

**Context:** Current script has `--fail_on_missing_inputs` flag (default: True)

**Options:**

- **A) Strict mode (current default):**
  - Stage 1 fails if any test processing fails
  - Stage 2 fails if any test_data.csv is missing
  - **Pros:** No partial/corrupted results
  - **Cons:** One bad test blocks entire experiment

- **B) Permissive mode:**
  - Stage 1 logs errors but continues with other tests
  - Stage 2 aggregates available test_data.csv files only
  - Add metadata tracking which tests succeeded/failed
  - **Pros:** Partial results still useful
  - **Cons:** Silent failures possible

- **C) Configurable with clear warnings:**
  - Keep `--fail_on_missing_inputs` flag
  - Add clear logging about skipped tests
  - Stage 2 reports which tests are missing in output metadata
  - **Pros:** Flexibility with safety
  - **Cons:** Requires careful user attention

**Current Recommendation:** Option C - maintain existing flag, enhance logging, add missing test report to aggregation output.

---

### 6. Backward Compatibility

**Question:** Should we maintain the old `process_test_data` rule or deprecate it?

**Options:**

- **A) Complete replacement:**
  - Remove old rule and script entirely
  - All workflows must use new two-stage approach
  - **Pros:** Clean codebase, forces best practice
  - **Cons:** Breaks existing workflows immediately

- **B) Deprecation period:**
  - Keep old rule with deprecation warning
  - New rules recommended but old still works
  - Remove after 1-2 releases
  - **Pros:** Smooth transition
  - **Cons:** Maintenance burden, code duplication

- **C) Both approaches supported:**
  - Keep both rules indefinitely
  - Users choose based on use case
  - **Pros:** Maximum flexibility
  - **Cons:** Permanent maintenance overhead

**Current Recommendation:** Option A - complete replacement. The new approach is strictly superior for all use cases where parallel execution is possible.

---

### 7. Additional Parameters Handling

**Question:** How should `--additional_parameters` (currently line 807-811) be handled?

**Context:** Allows extraction of extra parameters beyond the main `parameter` and `category` from config files (e.g., `epoch` in current usage line 337).

**Current Recommendation:** Maintain identical behavior - Stage 1 extracts all additional parameters from config file, Stage 2 preserves them during concatenation.

---

## Implementation Plan

### Phase 1: Preparation ✅ CURRENT
- [x] Create feature branch
- [x] Create roadmap document
- [ ] Get user approval on design decisions
- [ ] Finalize design based on feedback

### Phase 2: Stage 1 - Per-Test Processing
- [ ] Create `dynvision/visualization/process_single_test.py`
  - Extract relevant functions from `process_test_data.py`
  - Simplify to handle single test tuple
  - Add error handling for individual test failures
  - Implement cleanup logic for test_responses.pt
- [ ] Create Snakemake rule `process_single_test`
  - Input: Single test tuple (responses, outputs, config)
  - Output: Single test_data.csv
  - Wildcards: Must match test_model output structure
  - Priority: Higher than current process_test_data

### Phase 3: Stage 2 - Experiment Aggregation
- [ ] Create `dynvision/visualization/aggregate_experiment_data.py`
  - Simple concatenation logic
  - Column validation
  - Missing test reporting
  - Consistent sorting
- [ ] Create Snakemake rule `aggregate_experiment_data`
  - Input: Expand to collect all test_data.csv for experiment
  - Output: experiment_test_data.csv
  - Replaces current process_test_data output

### Phase 4: Integration
- [ ] Update downstream rules that depend on aggregated data
  - Search for rules using `test_data.csv` as input
  - Update paths to `experiment_test_data.csv` if needed
- [ ] Update experiment configuration if needed
- [ ] Test workflow with sample experiment

### Phase 5: Testing and Validation
- [ ] Test single-test processing
- [ ] Test aggregation with multiple tests
- [ ] Test error handling (missing files, failed tests)
- [ ] Test cleanup functionality
- [ ] Verify memory usage remains acceptable
- [ ] Compare output with current implementation

### Phase 6: Documentation and Cleanup
- [ ] Update workflow documentation
- [ ] Add migration notes for existing users
- [ ] Update developer guide with new pattern
- [ ] Remove old `process_test_data` rule and script

---

## Technical Details

### Data Flow

```
test_model (per test)
    ↓
    produces: test_responses.pt (30GB+)
              test_outputs.csv (small)
              test_outputs.csv.config.yaml (small)
    ↓
process_single_test (NEW - parallel across tests)
    ↓
    produces: test_data.csv (medium ~100MB)
    deletes:  test_responses.pt (30GB+ freed immediately)
    ↓
aggregate_experiment_data (NEW - runs once per experiment)
    ↓
    produces: experiment_test_data.csv (aggregated)
    ↓
visualization rules (existing - may need path updates)
```

### Wildcard Structure

**Stage 1 (process_single_test):**
```python
{experiment}/{model_name}{model_identifier}/{data_name}:{data_group}_{status}/{test_identifier}/test_data.csv
```

**Stage 2 (aggregate_experiment_data):**
```python
{experiment}/{model_name}{args1}{category}=*{args2}_{seed}/{data_name}:{data_group}_{status}/experiment_test_data.csv
```

### Key Functions to Extract/Adapt

From `process_test_data.py`:

- `build_measure_config()` - reuse as-is
- `_extract_metadata()` - reuse as-is
- `_load_responses()` - reuse as-is
- `_append_classifier_metrics()` - reuse as-is
- `_apply_resolution()` - reuse as-is
- `_append_layer_metrics()` - reuse as-is
- `process_single_batch_optimized()` - ADAPT to handle single file tuple instead of batch

---

## Open Questions for User

### Critical Decisions Needed:

1. **File Cleanup (Question 1):** Should we delete test_responses.pt after Stage 1, Stage 2, or make it configurable?

2. **Output Organization (Question 2):** Prefer Option A (alongside current), Option B (separate directories), or Option C (flat)?

3. **Resolution Application (Question 4):** Apply resolution in Stage 1 (saves disk) or Stage 2 (more flexibility)?

4. **Error Handling (Question 5):** Strict (fail fast) or permissive (skip failed tests)?

### Lower Priority Questions:

5. **Measure Configuration (Question 3):** Identical measures both stages, or allow differences?

6. **Backward Compatibility (Question 6):** Complete replacement or deprecation period?

7. **Output Naming:** Should aggregated file be named `test_data.csv` (current) or `experiment_test_data.csv` (more explicit)?

---

## Risk Assessment

### Low Risk
- ✅ Code reuse: Most logic already exists and is tested
- ✅ Memory efficiency: Already handled in current implementation
- ✅ Parallel execution: Snakemake handles this natively

### Medium Risk
- ⚠️ Wildcard complexity: Need to ensure wildcards resolve correctly for both stages
- ⚠️ Downstream dependencies: Need to identify and update all rules that consume aggregated data

### High Risk
- ❌ None identified - this is primarily a refactoring/reorganization task

---

## Success Criteria

1. **Parallel Execution:** Multiple tests can be processed simultaneously
2. **Disk Space:** test_responses.pt files deleted after successful processing
3. **Correctness:** Output data matches current implementation exactly
4. **Performance:** Total processing time should decrease when running parallel tests
5. **Robustness:** Failed individual tests don't block experiment completion (if permissive mode chosen)

---

## Notes

- Current `process_test_data.py` is already well-structured for this split (line 668: `process_single_batch_optimized` already handles batches of files)
- Memory monitoring infrastructure already exists (`MemoryMonitor` class)
- Config file metadata extraction already implemented
- Main work is creating Snakemake rules and adapting argument parsing

---

## Approved Design Decisions

**Decision Date:** 2025-12-12

### 1. File Cleanup Strategy
**Chosen:** Option C - Configurable via `--remove_input_responses` flag

Stage 1 script will accept `--remove_input_responses` flag (boolean). If True, deletes `test_responses.pt` after successful processing. Default behavior to be determined by user workflow needs.

### 2. Output File Organization
**Chosen:** Option A - Per-test data alongside current outputs

```
reports/{experiment}/{model_identifier}/{data_name}:{data_group}_{status}/
  {test_identifier}/
    test_responses.pt      # Deleted after Stage 1 if flag=True
    test_outputs.csv       # Kept
    test_outputs.csv.config.yaml  # Kept
    test_data.csv          # NEW - Stage 1 output (sample-level, no metadata)
  test_data.csv            # NEW - Stage 2 output (aggregated with metadata)
```

### 3. Measure Configuration
**Chosen:** Option A with graceful handling of missing data

Both stages use identical measure configurations. Stage 2 handles missing measures gracefully - if some tests lack certain metrics, those columns will have NaN values for those tests.

### 4. Resolution Handling
**Chosen:** Option C - Hybrid approach

- **Stage 1:** Always outputs sample-level data (sample_index, times_index resolution)
- **Stage 2:** Can apply class-level aggregation (first_label_index, times_index) OR keep sample-level based on `--sample_resolution` parameter
- **Rationale:** Provides flexibility to generate both resolutions without reprocessing

### 5. Error Handling
**Chosen:** Option C - Configurable with enhanced reporting

Maintain existing `--fail_on_missing_inputs` flag:

- If True: Fail if any test processing fails or files missing
- If False: Skip problematic tests, continue with available data
- Stage 2 adds metadata reporting which tests succeeded/failed

### 6. Backward Compatibility
**Chosen:** Option B - Deprecation period

Keep old `process_test_data` rule with deprecation warning for 1-2 releases. New two-stage approach is recommended but old rule still functional. Plan removal after transition period.

### 7. Metadata Extraction ⚠️ IMPORTANT DESIGN CHANGE
**Chosen:** Metadata extraction happens in Stage 2 only

- **Stage 1:** Pure data processing
  - Input: test_responses.pt, test_outputs.csv
  - Processing: Calculate all metrics (layer, classifier, performance)
  - Output: test_data.csv with NO metadata columns (no parameter, category, additional_parameters)

- **Stage 2:** Metadata extraction + aggregation
  - Input: All test_data.csv files + corresponding .config.yaml files
  - Processing: Extract parameter, category, and additional_parameters from config files
  - Output: Aggregated CSV with metadata columns added

**Rationale:**

- Cleaner separation of concerns
- Metadata extraction happens once (Stage 2) instead of N times (Stage 1)
- More flexible: can change which parameters to extract without reprocessing data
- Stage 1 focuses purely on computationally expensive operations

---

## Implementation Status

### Phase 1: Preparation ✅ COMPLETED
- [x] Create feature branch `feature/parallel-experiment-processing`
- [x] Create roadmap document
- [x] Get user approval on design decisions
- [x] Finalize design based on feedback

### Phase 2: Stage 1 - Per-Test Processing ✅ COMPLETED
- [x] Create `dynvision/visualization/process_single_test.py`
  - Extracts functions from `process_test_data.py`
  - Handles single test tuple (test_responses.pt, test_outputs.csv)
  - Computes all metrics at sample-level
  - NO metadata extraction (done in Stage 2)
  - Implements cleanup logic for test_responses.pt
- [x] Create Snakemake rule `process_single_test`
  - Input: Single test tuple (responses, outputs)
  - Output: Single test_data.csv (sample-level, no metadata)
  - Priority: 4 (higher than aggregation)
  - Parameters: measures, memory_limit_gb, remove_input_responses

### Phase 3: Stage 2 - Experiment Aggregation ✅ COMPLETED
- [x] Create `dynvision/visualization/aggregate_experiment_data.py`
  - Concatenation logic
  - Metadata extraction from .config.yaml files
  - Column validation and missing data handling
  - Resolution transformation (sample → class if requested)
  - Consistent sorting
- [x] Create Snakemake rule `aggregate_experiment_data`
  - Input: Expand to collect all test_data.csv + config files
  - Output: test_data.csv (aggregated, same name for compatibility)
  - Priority: 3 (lower than process_single_test)
  - Parameters: parameter, category, additional_parameters, sample_resolution

### Phase 4: Integration ✅ COMPLETED
- [x] Add deprecation warning to old `process_test_data` rule
- [ ] Update downstream rules if needed (visualization rules use same output path)
- [ ] Update experiment configuration if needed

### Phase 5: Testing and Validation 🔄 PENDING
- [ ] Test single-test processing with sample data
- [ ] Test aggregation with multiple tests
- [ ] Test error handling (missing files, failed tests)
- [ ] Test cleanup functionality
- [ ] Verify memory usage remains acceptable
- [ ] Compare output with current implementation
- [ ] Test parallel execution behavior

### Phase 6: Documentation and Cleanup 🔄 PENDING
- [ ] Update workflow documentation
- [ ] Add migration notes for existing users
- [ ] Update developer guide with new pattern
- [ ] Plan removal timeline for old `process_test_data` rule

---

## Status Updates

**2025-12-12 10:00:** Initial roadmap created, awaiting design decision approval.

**2025-12-12 11:30:** All design decisions approved. Updated roadmap. Beginning implementation.

**2025-12-12 12:00:** ✅ Core implementation completed!

- Created `process_single_test.py` (Stage 1 script)
- Created `aggregate_experiment_data.py` (Stage 2 script)
- Added `process_single_test` Snakemake rule
- Added `aggregate_experiment_data` Snakemake rule
- Added deprecation warning to old `process_test_data` rule
- Ready for testing
