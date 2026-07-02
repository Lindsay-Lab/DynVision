# Workflow Architecture and File Organization

This guide explains DynVision's hierarchical workflow structure, file organization patterns, and the technical implementation details for developers working with the Snakemake workflow system.

## Overview

DynVision uses a hierarchical file organization system designed to:

1. **Improve conceptual clarity** through hierarchical separation of concerns
2. **Enable scalability** for large parameter sweeps
3. **Solve filesystem limitations** through optional test identifier compression
4. **Preserve parameter information** via configuration files

The workflow uses Snakemake's checkpoint system to enable efficient data-dependent execution. Test identifier compression handles long test protocol specifications when needed, with parameters preserved in `.config.yaml` files.

## Hierarchical File Structure

### Design Principles

1. **Model identifier**: `{model_name}{model_args}_{seed}` (data_name in subfolder)
2. **Full model paths**: Models always use full parameter strings (no hashing)
3. **Experiment grouping**: All test/report outputs under `{experiment}/`
4. **Train+test spec**: `{data_name}:{data_group}_{status}`
5. **Test identifier compression**: Optional hashing for long test protocol specifications
6. **Config file preservation**: All parameters saved in `.config.yaml` files

### Directory Structure

#### Models

Models are organized with data_name as a subfolder, enabling multiple training runs on different datasets for the same model configuration:

```
models/
  {model_name}/                                      ← Model architecture (level 1)
    {model_name}{model_args}_{seed}/                ← Model identifier (level 2)
      {data_name}/                                  ← Training dataset (level 3)
        ├── init.pt                                 ← Initialization state
        ├── trained.pt                              ← Trained state
        ├── trained-epoch=149.pt                    ← Intermediate checkpoint
        └── trained-best.pt                         ← Best checkpoint
```

**Example:**
```
models/DyRCNNx8/
  DyRCNNx8:tsteps=20+dt=2_0042/
    imagenette/
      ├── init.pt
      ├── trained.pt
      ├── trained-epoch=149.pt
      └── trained-best.pt
```

#### Reports (Test Outputs)

Test outputs are organized hierarchically by experiment, model, training/test specification, and test protocol:

```
reports/
  {experiment}/                                      ← Experiment name (level 1)
    {model_name}{model_args}_{seed}/                ← Model identifier (level 2)
      {data_name}:{data_group}_{status}/            ← Train+test spec (level 3)
        {test_identifier}/                          ← Test protocol (level 4)
          ├── test_outputs.csv                      ← Test results
          ├── test_responses.pt                     ← Layer responses
          └── test_outputs.csv.config.yaml          ← Run configuration
```

**Example:**
```
reports/uniformnoise/
  DyRCNNx8:tsteps=20+dt=2_0042/                     # Full model identifier
    imagenette:all_trained/
      StimulusNoise:noisetype=uniform+noiselevel=0.2/
        ├── test_outputs.csv
        ├── test_responses.pt
        └── test_outputs.csv.config.yaml
```

**Key Points:**

- `{experiment}`: Inferred from data_loader and arguments (e.g., "uniformnoise")
- `{model_name}{model_args}_{seed}`: Full model identifier
- `{data_name}:{data_group}_{status}`: Combined training/test specification
- `{test_identifier}`: Either `{data_loader}{data_args}` or compressed hash if very long

#### Processed Data

Processed experiment data aggregates results across parameter sweeps:

```
reports/
  {experiment}/                                      ← Experiment name (level 1)
    {model_name}{model_args}_{seed}/                ← Model identifier (level 2)
      {data_name}:{data_group}_{status}/            ← Train+test spec (level 3)
        └── test_data.csv                           ← Aggregated results
```

**Note:** Processed data uses full model identifier (not hash) since it aggregates across wildcards.

#### Figures

Visualizations follow the same hierarchical pattern as processed data:

```
figures/
  {experiment}/                                      ← Experiment name (level 1)
    {model_name}{model_args}_{seed}/                ← Model identifier (level 2)
      {data_name}:{data_group}_{status}/            ← Train+test spec (level 3)
        ├── responses.png                           ← Response plots
        ├── performance.png                         ← Performance plots
        └── {custom_plot}.png                       ← Experiment-specific plots
```

## Test Identifier Compression

### Purpose

Long test protocol specifications can exceed filesystem path limits (255 bytes). Test identifier compression solves this by:

1. Creating short, fixed-length identifiers (8 hex characters) for test protocols
2. Preserving full parameter specification in `.config.yaml` files
3. Enabling transparent parameter extraction by processing scripts

**Note:** Model identifiers are NEVER compressed - they always use full parameter strings for clarity.

### Implementation

**Location:** `dynvision/workflow/snake_utils.smk`

Test identifiers can be optionally compressed using the `parse_test_identifier()` function:

```python
def parse_test_identifier(test_id: str) -> Tuple[str, str]:
    """Parse test identifier into data_loader and data_args.

    Handles both:
    - Uncompressed: 'StimulusNoise:noisetype=uniform+noiselevel=0.2'
    - Compressed: 'abc123ef' (hash)

    Returns:
        (data_loader, data_args) tuple
    """
    # Implementation details in snake_utils.smk
```

### Configuration File Preservation

**Every test run creates a config file:**

**Location:** `{test_output_dir}/test_outputs.csv.config.yaml`

**Content:** All resolved parameters with namespace prefixes:
```yaml
data.noiselevel: 0.2
data.noisetype: uniform
data.stim: 50
model.tsteps: 20
model.dt: 2
seed: 42
# ... etc
```

**Usage by process_test_data.py:**
```python
# Read config file
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Extract parameters
param_value = config[f'data.{param_name}']
model_value = config[f'model.{param_name}']
```

This approach ensures that even when test identifiers are compressed, all parameter information is preserved and accessible.

## Test Identifier Wildcards

### Concept

The `{test_identifier}` wildcard matches **either**:

- Uncompressed form: `StimulusNoise:noisetype=uniform+noiselevel=0.2`
- Compressed form: `abc123ef` (hash)

This enables rules to work with both forms transparently.

### Implementation Pattern

**In test_model rule outputs:**
```python
output:
    test_outputs = project_paths.reports / "{experiment}" / "{model_name}{model_identifier}" / "{data_name}:{data_group}_{status}" / "{test_identifier}" / "test_outputs.csv",
    test_responses = ...,
    meta_data = project_paths.reports / "{experiment}" / "{model_name}{model_identifier}" / "{data_name}:{data_group}_{status}" / "{test_identifier}" / "test_outputs.csv.config.yaml"
```

**In process_test_data rule:**
```python
input:
    test_configs = expand(
        project_paths.reports / "{{experiment}}" / "{{model_name}}{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" / "{{data_name}}:{{data_group}}_{status}" / "{test_identifier}" / "test_outputs.csv.config.yaml",
        test_identifier=lambda w: get_test_specs_for_experiment(w.experiment),
        ...
    )
```

### Parameter Extraction

The `process_test_data.py` script extracts parameters from config files:

```python
def _extract_metadata(config_file, data_arg_key, category, extra_parameters):
    """Extract metadata from config YAML file."""
    import yaml

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract from data namespace
    param_value = config[f'data.{data_arg_key}']

    # Extract category from path (wildcard)
    cat_value = extract_param_from_string(str(config_file.parent.parent), key=category)

    # Extract extra parameters from multiple namespaces
    extra_values = {}
    for param in extra_parameters:
        value = config.get(f'data.{param}') or config.get(f'model.{param}') or config.get(param)
        extra_values[param] = str(value) if value else None

    return FileMetadata(param_value, cat_value, extra_values)
```

This approach eliminates the need for complex path parsing and works regardless of whether test identifiers are compressed.

## Snakemake Checkpoints

### Purpose

Checkpoints enable data-dependent DAG re-evaluation. DynVision uses checkpoints to:

1. Ensure models are trained before testing
2. Dynamically expand test outputs based on experiment configuration
3. Coordinate dependencies across parameter sweeps

### train_model Checkpoint

**The train_model rule is a checkpoint:**

```python
checkpoint train_model:
    input:
        model_state = project_paths.models / "{model_name}{model_args}_{seed}" / "{data_name}" / "init.pt",
        dataset_ready = project_paths.data.interim / "{data_name}" / "train_all.ready",
        script = SCRIPTS / "runtime" / "train_model.py"

    params:
        checkpoint_dir = lambda w: project_paths.models / w.model_name / f"{w.model_name}{w.model_args}_{w.seed}" / w.data_name,
        ...

    output:
        model_state = project_paths.models / "{model_name}{model_args}_{seed}" / "{data_name}" / "trained.pt"

    shell:
        """
        {params.execution_cmd} \\
            --checkpoint_dir {params.checkpoint_dir:q} \\
            --input_model_state {input.model_state:q} \\
            --output_model_state {output.model_state:q} \\
            ...
        """
```

**Key features:**

- Creates trained model with full parameter specification in path
- Saves intermediate checkpoints to same directory
- Enables downstream rules to depend on trained models

### process_test_data with Checkpoint Dependency

**The process_test_data rule depends on the checkpoint:**

```python
rule process_test_data:
    input:
        # Models with full parameter strings (triggers checkpoint)
        models = expand(
            project_paths.models / "{{model_name}}{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" / "{{data_name}}" / "{status}.pt",
            cat_value=lambda w: config.experiment_config["categories"].get(w.category, []),
            status=lambda w: config.experiment_config[w.experiment].get("status", w.status),
        ),

        # Test outputs (may use compressed test identifiers)
        test_configs = expand(
            project_paths.reports / "{{experiment}}" / "{{model_name}}{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" / "{{data_name}}:{{data_group}}_{status}" / "{test_identifier}" / "test_outputs.csv.config.yaml",
            test_identifier=lambda w: get_test_specs_for_experiment(w.experiment),
            ...
        ),
        ...

    output:
        test_data = project_paths.reports / "{experiment}" / "{model_name}{args1}{category}=*{args2}_{seed}" / "{data_name}:{data_group}_{status}" / "test_data.csv"
```

**Execution flow:**

1. User requests: `reports/exp/DyRCNNx8:tsteps=20_42/mnist:all_trained/test_data.csv`
2. Snakemake expands `models` input: triggers `train_model` checkpoint if needed
3. After training completes, DAG re-evaluates
4. Test outputs are generated (with possible test identifier compression)
5. `process_test_data` reads config files to extract parameters
6. Aggregated results are created

### Benefits

- ✅ Full model parameters always visible in paths
- ✅ Test identifier compression handled automatically when needed
- ✅ Config files preserve all parameter information
- ✅ Checkpoint ensures proper execution order
- ✅ No manual hash tracking required

## Experiment Name Inference

### Purpose

The `{experiment}` level organizes related test outputs. The workflow infers experiment names from data loader configurations to automatically group related tests.

### Implementation

**Location:** `dynvision/workflow/snake_utils.smk`

Experiment names are inferred based on:

1. Data loader type (e.g., `StimulusNoise`, `StimulusDuration`)
2. Data loader arguments (e.g., `noisetype=uniform`)
3. Model status (e.g., `trained`, `init`)

**Examples:**

- `StimulusDuration` + any args → `response`
- `StimulusNoise` + `noisetype=uniform` → `uniformnoise`
- `StimulusNoise` + `noisetype=gaussianblur` → `gaussianblurnoise`

### Configuration

Experiment inference rules are defined in workflow configuration:

```yaml
experiments:
  response:
    data_loader: StimulusDuration
    description: "Temporal response characterization"

  uniformnoise:
    data_loader: StimulusNoise
    data_args:
      noisetype: uniform
    description: "Uniform noise robustness"
```

## Path Construction Patterns

### Model Paths

```python
# Initialization
project_paths.models / "{model_name}{model_args}_{seed}" / "{data_name}" / "init.pt"

# Training (checkpoint)
project_paths.models / "{model_name}{model_args}_{seed}" / "{data_name}" / "trained.pt"

# Symlink (created by checkpoint)
project_paths.models / "{model_name}:{hash_id}" / "{data_name}" / "{status}.pt"
```

### Report Paths

```python
# Raw test outputs (hash form)
project_paths.reports / "{experiment}" / "{model_name}:{hash_id}" / "{data_name}:{data_group}_{status}" / "{data_loader}{data_args}" / "test_outputs.csv"

# Processed data (full form)
project_paths.reports / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "test_data.csv"
```

### Figure Paths

```python
# Visualization outputs (full form)
project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png"
```

## Workflow Rule Dependencies

### Core Rules

1. **init_model**: Create initialized model
      - Input: Configuration, scripts
      - Output: `models/{model_name}{model_args}_{seed}/{data_name}/init.pt`

2. **train_model** (checkpoint): Train model and create symlink
      - Input: Initialized model, training data
      - Output: `models/{model_name}{model_args}_{seed}/{data_name}/trained.pt`
      - Side effects: Create symlink, hash file

3. **test_model**: Evaluate model on test protocol
      - Input: Model (via polymorphic wildcard), test data
      - Output: `reports/{experiment}/{model_name}:{hash_id}/{data_name}:{data_group}_{status}/{data_loader}{data_args}/test_outputs.csv`

4. **process_test_data**: Aggregate test results
      - Input: Multiple test outputs (checkpoint-dependent)
      - Output: `reports/{experiment}/{model_name}{model_args}_{seed}/{data_name}:{data_group}_{status}/test_data.csv`

5. **plot_***: Generate visualizations
      - Input: Processed data
      - Output: `figures/{experiment}/{model_name}{model_args}_{seed}/{data_name}:{data_group}_{status}/{plot}.png`

### Dependency Graph

```
init_model
    ↓
train_model (checkpoint)
    ↓ (creates symlink)
    ├→ test_model (uses hash via symlink)
    │     ↓
    │  test_outputs.csv
    │     ↓
    └→ process_test_data (checkpoint-dependent)
          ↓
       test_data.csv
          ↓
       plot_*
          ↓
       figures/*.png
```

## Migration from Legacy Structure

For existing projects with the old flat structure, a migration script is provided:

**Script:** `dynvision/migrate_to_hierarchical_layout.py`

**Usage:**
```bash
# Dry run to preview changes
python3 dynvision/migrate_to_hierarchical_layout.py --dry-run

# Execute migration
python3 dynvision/migrate_to_hierarchical_layout.py
```

**Migration performs:**

1. Reorganizes model files into hierarchical structure
2. Creates hash documentation files
3. Creates symlinks for hashed identifiers
4. Reorganizes reports by experiment
5. Reorganizes figures by experiment
6. Cleans up empty directories

**See also:** `docs/development/planning/hash_model_args.md`

## Debugging and Troubleshooting

### Viewing the Dependency Graph

```bash
# Full DAG visualization
snakemake --dag | dot -Tpdf > dag.pdf

# Rule-specific graph
snakemake --dag reports/exp/DyRCNNx8:tsteps=20_42/mnist:all_trained/test_data.csv | dot -Tpdf > dag.pdf
```

### Checking Config Files

```bash
# List config files for an experiment
find reports/uniformnoise/ -name "*.config.yaml"

# View config file contents
cat reports/uniformnoise/DyRCNNx8:tsteps=20_42/mnist:all_trained/abc123ef/test_outputs.csv.config.yaml

# Search for specific parameter value
grep -r "data.noiselevel: 0.2" reports/uniformnoise/
```

### Force Re-execution

```bash
# Force specific rule
snakemake --forcerun train_model models/DyRCNNx8:tsteps=20_42/mnist/trained.pt

# Force checkpoint and downstream
snakemake --forcerun train_model reports/exp/DyRCNNx8:tsteps=20_42/mnist:all_trained/test_data.csv
```

### Common Issues

**Issue:** "File not found" for test outputs

- **Cause:** Test hasn't run yet or test identifier is incorrect
- **Solution:** Check if test_model rule completed successfully, verify experiment configuration

**Issue:** Cannot find parameters in compressed test identifier

- **Cause:** Test identifier is hashed
- **Solution:** Check the `.config.yaml` file in the same directory as test outputs

**Issue:** process_test_data fails with parameter extraction error

- **Cause:** Config file missing or doesn't contain expected parameters
- **Solution:** Verify test_model completed successfully and created config file, check parameter namespacing (data.* vs model.*)

## Best Practices

### For Rule Development

1. **Always use full model identifiers** in paths - no hashing needed
2. **Create config files** for all test outputs via `persist_resolved_config()`
3. **Read parameters from config files** in processing scripts, not from paths
4. **Test with compressed test identifiers** to ensure config file handling works correctly

### For Experiment Design

1. **Group related tests** under consistent experiment names
2. **Use descriptive data_loader arguments** for clarity
3. **Document experiment configurations** in workflow config
4. **Keep model parameter names concise** but meaningful

### For File Management

1. **Never manually edit config files** - they're auto-generated
2. **Preserve config files** when copying test outputs for reproducibility
3. **Use consistent naming** for model and data arguments
4. **Document custom experiments** in workflow configuration files

## References

- [Snakemake Checkpoints Documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#data-dependent-conditional-execution)
- [DynVision Planning Document](../planning/hash_model_args.md)
- [User Workflow Guide](../../reference/workflow.md)
- [Configuration Reference](../../reference/configuration.md)
