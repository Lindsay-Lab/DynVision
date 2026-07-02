# Snakecharm Config Stability Issue - Analysis and Fix

**Date**: 2025-11-22
**Issue**: Config changes during running workflow affect subsequent jobs

## Problem Statement

When running a workflow via `snakecharm.sh` on a compute cluster with the snakemake-executor-plugin:

1. Workflow starts and generates a fixed `workflow_config_<timestamp>.yaml`
2. User modifies config files (e.g., `config_defaults.yaml`) while workflow is running
3. **Subsequent jobs submitted by Snakemake see the UPDATED config, not the original**
4. This breaks reproducibility and can cause inconsistent results within a single workflow run

## Root Cause Analysis

### Current Implementation

**File**: `dynvision/workflow/snake_utils.smk` (lines 37-42, 301-303)

```python
# Lines 37-42: Config files loaded DYNAMICALLY via Snakemake's configfile directive
configfile: project_paths.scripts.configs / 'config_defaults.yaml'
configfile: project_paths.scripts.configs / 'config_data.yaml'
configfile: project_paths.scripts.configs / 'config_visualization.yaml'
configfile: project_paths.scripts.configs / 'config_experiments.yaml'
configfile: project_paths.scripts.configs / 'config_modes.yaml'
configfile: project_paths.scripts.configs / 'config_workflow.yaml'

# Lines 301-303: Snapshot created AFTER config loading
_raw_config = config.__dict__.copy() if isinstance(config, SimpleNamespace) else dict(config)
WORKFLOW_CONFIG_PATH = _write_base_config_file(_raw_config)
config = SimpleNamespace(**_raw_config)
```

### The Problem

**Snakemake's `configfile` directive behavior**:

1. `configfile: path/to/config.yaml` tells Snakemake to **load** that file at workflow parsing time
2. **BUT** Snakemake re-parses the workflow for EACH submitted job in cluster mode
3. Each time Snakemake parses the workflow (for each job submission), it re-reads the `configfile:` directives
4. If the config files have changed on disk, the **new values** are loaded

### Timeline of Events

```
T=0: User runs snakecharm.sh
  ├─> Snakemake parses Snakefile + includes
  ├─> Reads config files (current state)
  ├─> Creates workflow_config_20251122-143000.yaml (snapshot)
  ├─> Submits job #1 (init_model)
  └─> Job #1 uses WORKFLOW_CONFIG_PATH correctly ✓

T=5min: Job #1 completes, Job #2 ready to submit
  ├─> Snakemake re-parses workflow for job #2
  ├─> Re-reads configfile: directives (gets CURRENT disk state)
  ├─> config dict now has NEW values
  ├─> But WORKFLOW_CONFIG_PATH still points to old snapshot ✓
  └─> **Problem**: Lambda functions in rules use config.* directly!

T=10min: User edits config_defaults.yaml (changes learning_rate)

T=15min: Job #3 ready to submit
  ├─> Snakemake re-parses workflow for job #3
  ├─> Re-reads configfile: directives
  ├─> config dict now has UPDATED learning_rate ✗
  ├─> Rules that reference config.learning_rate see NEW value ✗
  └─> Job #3 submitted with MIXED config (snapshot + new values) ✗
```

### Where Config Is Used

**Protected paths** (use WORKFLOW_CONFIG_PATH):

- Runtime scripts receive `--config_path {WORKFLOW_CONFIG_PATH}` ✓
- These are SAFE - they read the frozen snapshot

**Vulnerable paths** (use config.* directly):
Lines in snake_runtime.smk:
```python
# Line 91-95: Direct config access in lambda functions
resolution = lambda w: config.data_resolution[w.data_name],
normalize = lambda w: json.dumps((
    config.data_mean[w.data_name],
    config.data_std[w.data_name]
)),

# Line 165-169: Another instance
normalize = lambda w: (
    # Allow override via --config normalize=null
    (config.data_mean[w.data_name], config.data_std[w.data_name])
    if config.normalize != "null" else None
),
```

**These lambda functions are evaluated WHEN THE JOB IS SUBMITTED**, not when the workflow starts!

### Concrete Example

```yaml
# Initial config_data.yaml (T=0)
data_mean:
  imagenet: [0.485, 0.456, 0.406]
data_std:
  imagenet: [0.229, 0.224, 0.225]

# User edits while workflow running (T=10min)
data_mean:
  imagenet: [0.500, 0.500, 0.500]  # Changed!
```

**Result**:

- Jobs submitted before T=10min: Use mean=[0.485, 0.456, 0.406] ✓
- Jobs submitted after T=10min: Use mean=[0.500, 0.500, 0.500] ✗
- **Same workflow run has inconsistent normalization!**

## Why WORKFLOW_CONFIG_PATH Doesn't Fully Solve This

The `WORKFLOW_CONFIG_PATH` snapshot is correctly used for runtime scripts, but:

1. **Lambda functions in `params:`** are evaluated at job submission time
2. **They access `config.*` which is re-loaded from disk each parse**
3. **They don't read from WORKFLOW_CONFIG_PATH**

## Solution Options

### Option 1: Freeze Config in Memory (Recommended)

**Approach**: Load config files ONCE into a frozen dict, don't use Snakemake's `configfile:` directive

**Implementation**:

```python
# dynvision/workflow/snake_utils.smk

# REMOVE these lines (37-42):
# configfile: project_paths.scripts.configs / 'config_defaults.yaml'
# ...

# REPLACE with manual loading (new lines 37-50):
def _load_frozen_config() -> Dict[str, Any]:
    """Load config files once and freeze them for entire workflow."""
    config_files = [
        'config_defaults.yaml',
        'config_data.yaml',
        'config_visualization.yaml',
        'config_experiments.yaml',
        'config_modes.yaml',
        'config_workflow.yaml',
    ]

    merged_config = {}
    for config_file in config_files:
        config_path = project_paths.scripts.configs / config_file
        if config_path.exists():
            with config_path.open('r') as f:
                file_config = yaml.safe_load(f) or {}
                merged_config.update(file_config)

    # Merge with any --config args from Snakemake CLI
    merged_config.update(config)

    return merged_config

# Load config ONCE and freeze it
_frozen_config = _load_frozen_config()

# Lines 301-305 become:
_raw_config = _frozen_config.copy()
WORKFLOW_CONFIG_PATH = _write_base_config_file(_raw_config)
config = SimpleNamespace(**_raw_config)
```

**Benefits**:

- ✅ Config loaded ONCE at workflow start
- ✅ Subsequent re-parses see same frozen values
- ✅ Changes to disk files don't affect running workflow
- ✅ Lambda functions see consistent values
- ✅ Minimal code changes

**Drawbacks**:

- Need to handle Snakemake CLI `--config` overrides carefully

### Option 2: Load Config from Snapshot in Lambda Functions

**Approach**: Make lambda functions read from WORKFLOW_CONFIG_PATH instead of config.*

**Implementation**:

```python
# Load snapshot config once
def _load_snapshot_config():
    with WORKFLOW_CONFIG_PATH.open('r') as f:
        return yaml.safe_load(f)

_SNAPSHOT_CONFIG = _load_snapshot_config()

# In rules, change:
# OLD:
normalize = lambda w: config.data_mean[w.data_name]

# NEW:
normalize = lambda w: _SNAPSHOT_CONFIG['data_mean'][w.data_name]
```

**Benefits**:

- ✅ Explicitly uses frozen snapshot
- ✅ Clear separation between live and frozen config

**Drawbacks**:

- ❌ Must update every lambda function in all .smk files
- ❌ More invasive changes
- ❌ Easy to miss some references

### Option 3: Document and Accept Limitation

**Approach**: Document that users should not modify configs during workflow runs

**Implementation**: Add warning to documentation and workflow start message

**Benefits**:

- ✅ No code changes needed

**Drawbacks**:

- ❌ Doesn't actually solve the problem
- ❌ Users will still encounter issues
- ❌ Hard to enforce / easy to forget

## Recommended Solution

**Implement Option 1: Freeze Config in Memory**

This is the cleanest solution that:

1. Prevents the issue at its source
2. Requires minimal code changes
3. Makes workflow behavior more predictable
4. Aligns with principle of workflow reproducibility

## Implementation Plan

### Phase 1: Freeze Config Loading

```python
# dynvision/workflow/snake_utils.smk

import yaml
from typing import Dict, Any, Optional

def _load_and_freeze_config(cli_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration files once and freeze them for the entire workflow.

    This prevents mid-workflow config changes from affecting running jobs
    when using cluster execution with snakemake-executor-plugin.

    Args:
        cli_config: Optional dictionary of CLI config overrides from Snakemake --config

    Returns:
        Merged configuration dictionary
    """
    config_files = [
        'config_defaults.yaml',
        'config_data.yaml',
        'config_visualization.yaml',
        'config_experiments.yaml',
        'config_modes.yaml',
        'config_workflow.yaml',
    ]

    merged_config = {}
    configs_dir = project_paths.scripts.configs

    for config_file in config_files:
        config_path = configs_dir / config_file
        if config_path.exists():
            pylogger.debug(f"Loading config: {config_path}")
            with config_path.open('r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
                merged_config.update(file_config)
        else:
            pylogger.warning(f"Config file not found: {config_path}")

    # Merge with any --config overrides from Snakemake CLI
    if cli_config:
        pylogger.info(f"Applying {len(cli_config)} CLI config overrides: {list(cli_config.keys())}")
        merged_config.update(cli_config)

    pylogger.info(f"Config frozen at workflow start with {len(merged_config)} keys")
    return merged_config


# IMPORTANT: Remove configfile: directives to prevent dynamic reloading
# configfile: project_paths.scripts.configs / 'config_defaults.yaml'  # REMOVE
# configfile: project_paths.scripts.configs / 'config_data.yaml'      # REMOVE
# ... etc

# Load and freeze config ONCE
# Snakemake injects 'config' dict into global scope before parsing workflow files
try:
    _frozen_config = _load_and_freeze_config(cli_config=config)
except NameError:
    # If config doesn't exist (e.g., when testing modules in isolation)
    pylogger.warning("Snakemake config not found - CLI overrides will not be applied")
    _frozen_config = _load_and_freeze_config(cli_config=None)
```

### Phase 2: Update Config Snapshot Creation

```python
# Lines ~301-305 (adjust line numbers after changes above)

# Use frozen config for all downstream processing
_raw_config = _frozen_config.copy()

# Write snapshot to disk for runtime scripts
WORKFLOW_CONFIG_PATH = _write_base_config_file(_raw_config)

# Convert to SimpleNamespace for dot notation access
config = SimpleNamespace(**_raw_config)

# Log the snapshot location
pylogger.info(f"Workflow config snapshot: {WORKFLOW_CONFIG_PATH}")
```

### Phase 3: Add Validation

```python
def _write_base_config_file(config_payload: Dict[str, Any]) -> Path:
    """Persist the fully merged Snakemake config for reuse by runtime scripts."""

    config_dir = project_paths.large_logs / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_config_path = config_dir / f"workflow_config_{timestamp}.yaml"

    header = [
        "# DynVision workflow base configuration",
        f"# Generated at: {timestamp}",
        "#",
        "# WARNING: This config is FROZEN for the duration of this workflow run.",
        "# Changes to source config files will NOT affect this workflow.",
        "# To use updated configs, start a new workflow run.",
    ]

    with base_config_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n\n")
        yaml.safe_dump(config_payload, handle, default_flow_style=False, sort_keys=False)

    pylogger.info(f"Persisted FROZEN workflow config to {base_config_path}")
    return base_config_path
```

### Phase 4: Update Documentation

Update `docs/development/guides/parameter-processing.md`:

```markdown
## Workflow Config Freezing

When using `snakecharm.sh` for cluster execution, the configuration is frozen
at workflow start to prevent inconsistencies from mid-workflow config changes.

### How It Works

1. **Workflow Start**: All config files are loaded and merged ONCE
2. **Snapshot Created**: Merged config written to `logs/configs/workflow_config_<timestamp>.yaml`
3. **Frozen for Duration**: Subsequent job submissions see the same frozen config
4. **Runtime Scripts**: Read from the frozen snapshot via `--config_path`

### Important Notes

- **Config changes during workflow run are IGNORED** (this is intentional!)
- To use updated configs: Start a new workflow run
- The frozen snapshot is preserved in logs for reproducibility
- Direct config.* accesses in rules use the frozen version

### Why Freezing Is Necessary

Without freezing, when using cluster execution:

- Snakemake re-parses workflow for each job submission
- Config files are re-read from disk each time
- Mid-workflow changes would cause inconsistent parameters across jobs
- Results would not be reproducible

With freezing:

- Config loaded once at workflow start
- All jobs in the run see identical configuration
- Workflow run is self-contained and reproducible
```

## Testing

### Test Case 1: Config Stability

```bash
# Start workflow
./dynvision/cluster/snakecharm.sh train_model

# While running, modify config
echo "learning_rate: 0.999" >> dynvision/configs/config_defaults.yaml

# Check that jobs use original config
# Grep job logs for learning_rate parameter
# Should all show original value, NOT 0.999
```

### Test Case 2: CLI Override Still Works

```bash
# Start workflow with CLI override
./dynvision/cluster/snakecharm.sh train_model --config learning_rate=0.005

# Verify all jobs use 0.005 (CLI override wins)
```

### Test Case 3: Snapshot Persists

```bash
# Check snapshot file exists and contains correct values
cat logs/configs/workflow_config_<timestamp>.yaml
```

## Migration Notes

- **No breaking changes**: Rules continue to access config.* as before
- **Automatic**: Users don't need to change their workflows
- **Transparent**: Freezing happens automatically at workflow start
- **Logged**: Clear logging when config is frozen and where snapshot is saved

## Related Files

- `dynvision/workflow/snake_utils.smk`: Config loading and freezing
- `dynvision/workflow/snake_runtime.smk`: Rules using config.*
- `dynvision/cluster/snakecharm.sh`: Workflow entry point
- `docs/development/guides/parameter-processing.md`: Documentation

## References

- Snakemake `configfile` directive: https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html
- Cluster execution: https://snakemake.readthedocs.io/en/stable/executing/cluster.html
- Parameter processing system: `docs/development/guides/parameter-processing.md`
