# Hierarchical File Organization and Model Identifier Hashing
**Status:** ⚠️ ARCHIVED — Decision was NOT to implement model hashing. Kept for reference.

**Status:** ❌ SUPERSEDED - Model hashing was NOT implemented
**Created:** 2025-12-06
**Superseded:** 2025-12-07
**Authors:** Robin Gutzen, Claude (AI Assistant)

---

**IMPORTANT NOTE:** This document describes a PROPOSED system that was never fully implemented.

**What WAS implemented:**

- ✅ Hierarchical file organization
- ✅ Models organized as `{model_name}/{model_identifier}/{data_name}/`
- ✅ Experiment-based grouping for test outputs
- ✅ Checkpoint coordination for model dependencies

**What was NOT implemented:**

- ❌ Model identifier hashing and symlinks
- ❌ Polymorphic `{model_identifier}` wildcard matching full/hash forms
- ❌ `.hash` documentation files for models

**What WAS implemented instead:**

- ✅ Optional test identifier compression (not model identifiers)
- ✅ Config files (`.config.yaml`) for parameter preservation
- ✅ Config-based parameter extraction in `process_test_data.py`

**See instead:**

- [Current Workflow Reference](../../reference/workflow.md)
- [Current Developer Workflow Guide](../guides/workflow.md)

---

## Original Proposal (for historical reference)

This section documents the original proposal. The actual implementation differs significantly.

## Overview

Reorganize DynVision file structure to:

1. **Solve filesystem limitations** through model identifier hashing
2. **Improve conceptual clarity** by hierarchical separation of concerns
3. **Enable scalability** for large parameter sweeps

## Problem

**Filesystem errors** when parameter strings exceed 255-byte limit:
```
OSError: [Errno 36] File name too long: '/home/.../logs/slurm/rule_test_model/...'
```

**Root causes:**

- Flat structure mixes model and test attributes in single filename
- Long parameter combinations exceed filesystem limits
- Difficult to navigate and query

## Solution: Hierarchical File Organization

### Directory Structure

#### Old Flat File Structure

**Models: `{model_identifier}/{data_name}/{status}.pt`**
```
models/
	{model_name}/
		{model_name}:{model_args}_{seed}_{data_name}_{status}.pt
        
		checkpoints/
			{model_name}:{model_args}_{seed}_{data_name}_{status}-<ckpt-info>.ckpt
```

**Test Results: `{data_loader}/{model_name}{model_args}_{seed}_{data_name}_{status}_{data_group}_{data_loader}{data_args}_{data_group}/test_outputs.csv`**
```
reports/
	{data_loader}/
		{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}/
			test_outputs.csv
			test_responses.pt
```

**Processed Experiment Data: `{experiment}/{model_name}:{model_args}_{seed}_{data_name}_{status}_{data_group}/test_data.csv`**
```
reports/
	{experiment}/
		{model_name}:{model_args}_{seed}_{data_name}_{status}_{data_group}/
			test_data.csv
```

**Visualization: `{experiment}/{model_name}:{model_args}_{seed}_{data_name}_{status}_{data_group}/{plot}.png`**
```
figures/
	{experiment}/
		{model_name}:{model_args}_{seed}_{data_name}_{status}_{data_group}/
			performance.png
			responses.png
```

#### New Hierarchical File Structure

Note {model_args} and {data_args} are either empty or start with a colon ':'

**Models: `{model_identifier}/{data_name}/{status}.pt`**
```
models/
  {model_name}/                                      	← Model (level 1)
	{model_name}{model_args}_{seed}/         			← Model identifier (full: args_seed)
		{data_name}/                           			← Training data specification
		├── init.pt                         			← Status: initialization
		├── trained.pt                              	← Status: trained
		└── a7f3c9d4.hash                             	← Hash documentation

	{model_name}:hash={hash_id}/                      	← Model identifier (hashed)
		→ symlink to {model_name}{model_args}_{seed}/   ← Symlink at model level

	checkpoints/
		{model_name}{model_args}_{seed}/         		← Model identifier (full: args_seed)
			{data_name}/                           		← Training data specification
			├── trained-<ckpt-info>.ckpt   				← Checkpoint files
			└── trained-<ckpt-info>.ckpt	  			  ckpt-info e.g. "epoch-149-2.30"
```

**Test Results: `{experiment}/{model_identifier}/{train+test_spec}/{data_loader}{data_args}/test_outputs.csv`**
```
reports/
  {experiment}/                                      	← Experiment (level 1)
    {model_name}:hash={hash_id}/                        ← Model identifier (level 2)
      {data_name}:{data_group}_{status}/  				← Train+test spec (level 3)
        {data_loader}{data_args}/ 						← Data loader + params (level 4)
          ├── test_outputs.csv                       	← Test results
          └── test_responses.pt                      	← Layer responses
```

**Processed Experiment Data: `{experiment}/{model_identifier}/{train+test_spec}/test_data.csv`**
```
reports/
  {experiment}/                                      	← Experiment (level 1)
    {model_name}{model_args}_{seed}/                	← Model identifier (level 2)
      {data_name}:{data_group}_{status}/  				← Train+test spec (level 3)
        └── test_data.csv                           	← Processed test data
```

**Visualization: `{experiment}/{model_identifier}/{train+test_spec}/{plot}.png`**
```
figures/
  {experiment}/                                      	← Experiment (level 1)
    {model_name}{model_args}_{seed}/                	← Model identifier (level 2)
      {data_name}:{data_group}_{status}/  				← Train+test spec (level 3)
        ├── performance.png                         	← Performance plot
        └── responses.png                           	← Response plot
```

### Key Principles

1. **Model identifier**: `{model_name}{model_args}_{seed}` (data_name in subfolder)
2. **Hash computation**: `compute_hash(model_args, seed)` - excludes data_name
3. **Symlink level**: At model folder, not data subfolder
4. **Experiment grouping**: All test/report outputs under `{experiment}/`
5. **Train+test spec**: `{data_name}:{data_group}_{status}` (simplified from previous)
6. **Polymorphic wildcard**: `{model_identifier}` matches full or hash form

### Hash Function

**Location:** `dynvision/workflow/snake_utils.smk`

```python
def compute_hash(*args, length: int = 8) -> str:
    """Compute deterministic hash from model_args and seed.

    Args:
        *args: Components to hash (model_args, seed)
        length: Hash length in hex characters (default: 8)

    Returns:
        Hash string (e.g., ':hash=a7f3c9d4')

    Notes:
        - Idempotent: returns input unchanged if already a hash
        - Uses MD5 for speed (not cryptographic)
        - 8 hex chars = ~4 billion combinations
    """
    import hashlib

    # Idempotent check
    for arg in args:
        if 'hash=' in str(arg):
            return str(arg)

    # Combine and hash
    combined = '_'.join(str(arg).lstrip(':') for arg in args)
    hash_val = hashlib.md5(combined.encode()).hexdigest()[:length]
    return f':hash={hash_val}'
```

## Implementation

### Phase 1: Core Utilities

**File:** `dynvision/workflow/snake_utils.smk`

1. Implement `compute_hash()` function
2. Run unit tests: `pytest tests/workflow/test_hash_compression.py`

### Phase 2: Model Rules

**File:** `dynvision/workflow/snake_runtime.smk`

#### `init_model`
```python
output:
    project_paths.models / "{model_name}{model_args}_{seed}" / "{data_name}" / "init.pt"
```

#### `train_model` (checkpoint)
```python
checkpoint train_model:
    input:
        project_paths.models / "{model_name}{model_args}_{seed}" / "{data_name}" / "init.pt"

    params:
        model_folder = lambda w: project_paths.models / f"{w.model_name}{w.model_args}_{w.seed}",
        symlink_folder = lambda w: project_paths.models / f"{w.model_name}:{compute_hash(w.model_args, w.seed)}",
        hash_file = lambda w: project_paths.models / f"{w.model_name}{w.model_args}_{w.seed}" / w.data_name / f"{compute_hash(w.model_args, w.seed)}.hash"

    output:
        project_paths.models / "{model_name}{model_args}_{seed}" / "{data_name}" / "trained.pt"

    shell:
        """
        # Training command...

        # Document hash
        echo "{wildcards.model_args}_{wildcards.seed}" > {params.hash_file}

        # Create symlink at model level
        ln -s {params.model_folder} {params.symlink_folder}
        """
```

#### `test_model`
```python
input:
    project_paths.models / "{model_name}:{model_identifier}" / "{data_name}" / "{status}.pt"

output:
    project_paths.reports
    / "{experiment}"
    / "{model_name}:{model_identifier}"
    / "{data_name}:{data_group}_{status}"
    / "{data_loader}{data_args}"
    / "test_outputs.csv"
```

**Note:** `{model_identifier}` matches either:

- Full: `tsteps=20+dt=2+...._42`
- Hash: `hash=a7f3c9d4`

#### `process_test_data`
```python
input:
    # Full-form models (triggers checkpoint)
    models = expand(
        project_paths.models / "{{model_name}}:{{args1}}{category}={{value}}{{args2}}_{{seed}}" / "{{data_name}}" / "{status}.pt",
        ...
    ),

    # Hashed test outputs
    test_outputs = expand(
        project_paths.reports / "{{experiment}}" / "{{model_name}}:{hash_id}" / "{{data_name}}:{{data_group}}_{{status}}" / "{data_loader}{data_args}" / "test_outputs.csv",
        hash_id = lambda w: compute_hash(f"{{args1}}{category}={{value}}{{args2}}", w.seed),
        ...
    )
params:
    # when model_identifier is hash, we need to look up the category values to pass them to the script
    cat_values = lambda w: config.experiment_config['categories'].get(w.category, ''),
output:
    project_paths.reports / "{experiment}" / "{model_name}:{args1}{category}=*{args2}_{seed}" / "{data_name}:{data_group}_{status}" / "test_data.csv"
```

### Phase 3: Visualization Rules

**File:** `dynvision/workflow/snake_visualizations.smk`

**All plotting rules** follow pattern:
```python
input:
    project_paths.reports / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "test_data.csv"

output:
    project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png"
```

### Phase 4: Experiment Rules

**File:** `dynvision/workflow/snake_experiments.smk`

#### Helper Functions

```python
def model_path(..., data_name=DATA_NAME, status=STATUS):
    return [(project_paths.models / f"{model_name}{args}_{seed}" / data_name / f"{status}.pt")
            for seed in seeds for args in args_product(arg_dict)]

def result_path(experiment, ..., plot=None):
    folder = project_paths.reports if plot is None else project_paths.figures
    file = "test_data.csv" if plot is None else f"{plot}.png"
    return [folder / exp / f"{model_name}{args}_{seed}" / f"{data_name}:{data_group}_{status}" / file
            for ...]
```

#### All Experiment Rules

Pattern for all experiment rules (idle, feedback, skip, tsteps, etc.):
```python
input:
    expand(
        project_paths.reports / "{experiment}" / "{model_name}:{params}_{seed}" / "{data_name}:{data_group}_{status}" / "test_data.csv",
        ...
    )
```

### Phase 5: Snakefile

```python
rule all:
    input:
        expand(
            project_paths.figures / '{experiment}' / '{model_name}{model_args}_{seed}' / '{data_name}:{data_group}_{status}' / '{plot}.png',
            ...
        )
```

## Path Transformations Summary

```
OLD: models/{model_name}/{model_name}{args}_{seed}_{data}_{status}.pt
NEW: models/{model_name}{args}_{seed}/{data}/{status}.pt
     models/{model_name}:hash=XXX/  → symlink

OLD: reports/{data_loader}/{model}{args}_{seed}_{data}_{status}_{loader}{args}_{group}/test_outputs.csv
NEW: reports/{experiment}/{model}:{id}/{data}:{group}_{status}/{loader}{args}/test_outputs.csv

OLD: reports/{experiment}/{exp}_{model}{args}_{seed}_{data}_{status}_{group}/test_data.csv
NEW: reports/{experiment}/{model}{args}_{seed}/{data}:{group}_{status}/test_data.csv

OLD: figures/{experiment}/{exp}_{model}{args}_{seed}_{data}_{status}_{group}/{plot}.png
NEW: figures/{experiment}/{model}{args}_{seed}/{data}:{group}_{status}/{plot}.png
```

**Key changes:**

1. Data name in subfolder (not part of model identifier)
2. Hash excludes data_name
3. Symlink at model level
4. Experiment grouping for all outputs
5. Simplified train+test spec: `{data}:{group}_{status}`
6. No redundant prefixes

## Testing

**Unit tests:** `tests/workflow/test_hash_compression.py`

- Determinism
- Idempotence
- Variadic arguments
- Hash format
- Collision resistance

**Integration testing:**
```bash
# Dry run
snakemake --config experiment=rctarget -n

# Test checkpoint
snakemake models/DyRCNNx8:tsteps=20+dt=2_42/imagenette/trained.pt -f

# Verify symlink
ls -la models/DyRCNNx8:hash=*/
readlink models/DyRCNNx8:hash=*/

# Test polymorphic wildcard
snakemake reports/uniformnoise/DyRCNNx8:hash=*/imagenette:all_trained/StimulusNoise:*/test_outputs.csv -n
```

## Migration

**Backward compatibility:**

- Old and new structures can coexist
- No need to migrate existing data
- New runs automatically use new structure
- Optional cleanup script if needed

**Rollout:**

1. Implement `compute_hash()` + tests
2. Update model rules (init, train, test, process)
3. Update visualization rules
4. Update experiment rules
5. Integration testing
6. Full deployment

## Change Log

- 2025-12-06: Initial planning document
- 2025-12-06: Major restructure - comprehensive hierarchy
- 2025-12-06: Updated to match temp.smk - simplified train+test spec, data_name in subfolder
- 2025-12-06: **Condensed document** - removed verbose examples, streamlined for implementation
- 2025-12-05: **✅ IMPLEMENTATION COMPLETE** - All 5 phases implemented successfully:
  - Phase 1: compute_hash() utility and unit tests
  - Phase 2: Model rules (init, train, test, process) with checkpoint and symlinks
  - Phase 3: All visualization rules updated
  - Phase 4: Experiment helper functions (model_path, result_path)
  - Phase 5: Snakefile rule all updated
  - Total commits: 10+ (see git log for details)
