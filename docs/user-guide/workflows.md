# Workflow Management with Snakemake

DynVision uses Snakemake to manage complex workflows. This guide explains how to use Snakemake to run experiments, parameter sweeps, scale and port between environments with DynVision.

## Core Concepts

[Snakemake](https://www.snakemake.io/) manages DynVision's workflows through three key mechanisms:

### 1. Rule Dependencies
Each computational step is defined as a rule that transforms input files into output files. Snakemake automatically builds a dependency graph by matching output files of one rule to input files of another.

When you request a trained model:
```bash
snakemake models/DyRCNNx4_0000_cifar100_trained.pt
```

Snakemake builds a dependency graph by:

1. Finding the rule that creates this file (`train_model`)
2. Checking what input files it needs (`DyRCNNx4_0000_cifar100_init.pt`)
3. Finding rules that create those inputs (`init_model`)
4. Running rules in the correct order

You can visualize the graph:
```bash
snakemake --dag models/DyRCNNx4_0000_cifar100_trained.pt | dot -Tpdf > workflow.pdf
```

The `all` rule defines what happens when no specific target is given:
```python
rule all:
    input:
        # Run experiments from config
        expand("reports/experiment_{name}.done",
               name=config.experiment)
```

This rule:

- Serves as the default target when running `snakemake`
- Uses `expand()` to generate multiple targets
- Typically requests experiment completion flags


### 2. Smart Execution

DynVision uses Snakemake's timestamp tracking to avoid redundant work:
```python
rule train_model:
    input: "models/{name}_init.pt"       # Input file
    output: "models/{name}_trained.pt"   # Output file
```

The rule only runs when:

- Output files are missing
- Input files are newer than outputs
- Explicitly requested with `--forcerun`

### 3. Wildcards and Patterns

DynVision uses wildcards to create flexible rules:
```python
# Basic model training with configuration
models/{model_name}{model_args}_{seed}_{data_name}_{status}.pt
```

This enables:

- Parameter sweeps (`model_args`)
- Multiple seeds for validation
- Consistent file organization

For more details, see [Snakemake Documentation](https://snakemake.readthedocs.io/).

## Workflow Organization

DynVision organizes its workflow into specialized components:

```
dynvision/workflow/
├── Snakefile                # Main entry point and targets
└── snake_*.smk              # Specialized rule files
```

Each component handles specific tasks:

1. **Snakefile**: The main entry point that includes the other files and defines the top-level targets.
2. **snake_utils.smk**: Utility functions, path management, and configuration loading.
3. **snake_data.smk**: Rules for dataset acquisition, organization, and preprocessing.
4. **snake_runtime.smk**: Rules for model initialization, training, and evaluation.
5. **snake_experiments.smk**: Rules for running suites of tests
5. **snake_visualizations.smk**: Rules for visualizing model responses and analyzing results.

See [Organization](../reference/organization.md) for detailed structure.

## Basic to Advanced Usage

DynVision workflows support a progression from simple to complex use cases:

### 1. Single Experiment
Run a predefined experiment with default settings:
```bash
# Basic experiment execution
snakemake --config experiment=contrast
```

### 2. Custom Configuration
Override default parameters for specific needs:
```bash
# Configure model and dataset
snakemake --config \
  model_name=DyRCNNx4 \
  data_name=cifar100 \
  model_args="{rctype: full}"
```

### 3. Parameter Sweeps

Run experiments with multiple parameter combinations:

```bash
# Test different recurrence types
snakemake --config \
  experiment=contrast \
  model_args="{rctype: [full, self, depthpointwise]}"
```

Snakemake will:

- Create separate output files for each combination
- Run jobs in parallel (limited by -j parameter)
- Skip combinations that are already complete

### 4. Model Comparison
Evaluate different architectures:
```bash
# Compare model architectures
snakemake --config \
  model_name="[AlexNet, DyRCNNx4]" \
  experiment=contrast
```

### 5. Result Analysis
Generate comprehensive visualizations:
```bash
# Create analysis plots
snakemake plot_experiments_on_models
```

For more complex patterns and best practices, see:

- [Configuration Reference](../reference/configuration.md)

## Rule Implementation

DynVision implements Snakemake rules with consistent patterns. Each rule:

- Takes input files and parameters
- Produces output files
- Uses wildcards for flexibility

Example rule structure:

```python
rule test_model:
    """Evaluate a trained model on test data."""
    input:
        # Required input files
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}.pt',
        dataset_ready = project_paths.data.interim \
            / '{data_name}' \
            / 'test_{data_group}.ready',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        # Additional parameters
        config_path = CONFIGS,
        dataset_path = lambda w: project_paths.data.interim / w.data_name / f'test_{w.data_group}',
        batch_size = config.batch_size,
        store_responses = config.store_test_responses
    output:
        # Generated output files
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        results = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv'
    shell:
        # Command to execute
        """
        python {input.script:q} \
            --input_model_state {input.model_state:q} \
            --output_results {output.results:q} \
            --dataset_path {params.dataset_path:q} \
            --batch_size {params.batch_size}
        """
```

## Working with Wildcards

Snakemake uses wildcards to generalize rules. In DynVision, wildcards are extensively used to enable flexible workflows. Common wildcards include:

- `{model_name}`: Name of the model (e.g., DyRCNNx4, AlexNet)
- `{data_name}`: Name of the dataset (e.g., cifar100, mnist)
- `{model_args}`: Model arguments (e.g., `:rctype=full+tsteps=20`)
- `{data_loader}`: Data loader name (e.g., StimulusDuration)
- `{data_args}`: Data loader arguments (e.g., `:tsteps=100+stim=5`)
- `{data_group}`: Named subset of classes for testing
- `{seed}`: Random seed for reproducibility
- `{status}`: Either `init` or `trained` 
- `{experiment}`: Experiment name (e.g., contrast, duration)

These wildcards are used to specify which files to generate and how to connect the different steps of the workflow.

## Configuring Workflows

DynVision workflows are configured through YAML files and command-line overrides:

1. **YAML Configuration**:
      - `config_defaults.yaml`: Default parameters for all components
      - `config_data.yaml`: Dataset-specific configurations
      - `config_experiments.yaml`: Experiment-specific settings
      - `config_workflow.yaml`: Workflow execution parameters

2. **Command-Line Overrides**:
      - Directly override parameters with `--config key=value`
      - Specify complex parameters using Python-like syntax: `--config model_args="{rctype:full}"`

See the [Configuration Reference](../reference/configuration.md) for detailed information about configuration parameters.

## Output Organization

DynVision organizes outputs in a consistent hierarchy:

```
project_root/
├── data/
│   ├── raw/          # Original datasets
│   ├── interim/      # Processed datasets
│   └── processed/    # FFCV-optimized datasets
├── models/
│   └── {model_name}/
│       ├── *_init.pt      # Initialized models
│       ├── *_trained.pt   # Trained models
│       └── *_responses.pt # Model responses
├── reports/
│   └── {model_name}/
│       ├── *_outputs.csv  # Evaluation results
│       └── figures/       # Generated plots
└── logs/
    ├── training/     # Training logs
    └── slurm/        # Cluster execution logs
```

and a consistent naming pattern:

- **Models**: `/models/{model_name}/{model_name}{model_args}_{seed}_{data_name}_{status}.pt`
- **Responses**: `/models/{model_name}/{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt`
- **Results**: `/reports/{model_name}/{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv`
- **Figures**: `/reports/figures/{experiment}/{experiment}_{model_name}{model_args}_{seed}_{data_name}_{status}_{data_group}/{plot}.png`

This organization ensures that outputs can be easily located and associated with the parameters that generated them.

## Running Workflows on Clusters

DynVision workflows can scale seamlessly from laptops to high-performance computing clusters. The integration with Snakemake's cluster support provides:

- Automatic job distribution and scheduling
- Resource management (CPU, GPU, memory)
- Job dependency handling
- Logging and monitoring

Basic cluster execution:
```bash
# Run experiments on a cluster
./cluster/snakecharm.sh -j100 all_experiments

# Run specific experiment with cluster resources
./cluster/snakecharm.sh -j100 experiment --config \
    model_name=DyRCNNx4 \
    data_name=cifar100
```

For detailed setup instructions and advanced usage, see the [Cluster Integration Guide](cluster-integration.md).

## Advanced Workflow Usage

### Dry Runs

To see what Snakemake would do without actually executing commands:

```bash
snakemake -n all_experiments
```

### Creating Workflow Graphs

Generate a visual representation of the workflow:

```bash
snakemake --dag all_experiments | dot -Tpdf > workflow.pdf
```

### Resuming Interrupted Workflows

If a workflow is interrupted, you can resume from where it left off:

```bash
snakemake all_experiments --rerun-incomplete
```

### Force Rerunning Rules

To force Snakemake to rerun a particular rule:

```bash
snakemake all_experiments --forcerun test_model
```

## Custom Workflow Extensions

You can extend DynVision's workflows by adding new rules to the existing Snakemake files or creating new ones.

### Adding a New Experiment Type

To add a new experiment type:

1. Define the experiment in `config_experiments.yaml`:
   ```yaml
   experiment_config:
     my_experiment:
       status: trained
       parameter: my_parameter
       data_loader: MyDataLoader
       data_args:
         tsteps: 100
         my_parameter: [1, 2, 3, 4]
   ```

2. Create the corresponding data loader in `dataloader.py`

3. Add visualization rules if needed

### Creating Custom Analysis Workflows

You can create custom analysis workflows by defining new rules:

```python
rule my_custom_analysis:
    input:
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        script = SCRIPTS / 'analysis' / 'my_custom_analysis.py'
    output:
        results = project_paths.reports \
            / 'analysis' \
            / '{model_name}' \
            / 'my_custom_analysis_{model_name}{model_args}_{seed}_{data_name}_{status}_{data_group}.csv'
    shell:
        """
        python {input.script:q} \
            --responses {input.responses:q} \
            --output {output.results:q}
        """
```


## Troubleshooting Workflows

### Missing Input Files

If Snakemake reports missing input files, check:

1. If the dataset has been downloaded (`get_data` rule)
2. If the data paths are correct in `project_paths.py`
3. If all required symbolic links have been created

### Rule Execution Errors

If a rule fails to execute:

1. Check the error message in the log file
2. Ensure all dependencies are installed
3. Try running the individual script with the same parameters

### Performance Issues

If workflows are running slowly:

1. Enable FFCV data loading with `use_ffcv: True`
2. Adjust the number of threads (`-j` parameter)
3. Use mixed precision training with `precision: "16-mixed"`

## Further Reading

For more information about Snakemake, see the [official Snakemake documentation](https://snakemake.readthedocs.io/).
