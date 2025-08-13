# Snakemake Background
Snakemake is a workflow management system designed to create reproducible and scalable data analyses. It's based on a Python-like language that defines how to create output files from input files through rules. According to the provided materials, the DynVision toolbox leverages Snakemake "to facilitate the parameter handling, scalability, and portability of workflows" alongside yaml config scripts for high-level, human-readable interfaces to crucial parameters.

## Key Features of Snakemake Relevant to DynVision
### Rule-based Workflow Definition
Workflows in Snakemake are composed of rules that specify:

- Input files
- Output files
- Shell commands, scripts, or code to transform inputs into outputs

DynVision likely organizes its computational pipeline into discrete steps using this rule-based approach.

### Automatic Dependency Resolution
Snakemake automatically determines dependencies between rules by matching file names. This creates a directed acyclic graph (DAG) of jobs where the edges represent dependencies, making it perfect for complex pipelines like neural network training and evaluation.

### Wildcards for Flexible Rules
Snakemake allows generalizing rules using named wildcards, which are placeholders that get replaced with appropriate values. This enables DynVision to process multiple samples, model configurations, or experimental variations with the same workflow definition.

### Configuration Management
Configuration of a workflow can be handled via config files and tabular configuration like sample sheets. This aligns with DynVision's approach of using YAML config scripts for parameter management.
Execution and Scaling Capabilities

### Parallel Execution
Snakemake can automatically determine which parts of the workflow can be run in parallel. By specifying available cores, it can solve a binary knapsack problem to optimize job scheduling. This is crucial for DynVision's computationally intensive neural network operations.

### Resource Management
Resources like threads and memory can be defined and overwritten per rule. DynVision likely uses this to optimize compute resources for different stages of model training and evaluation.

### Cluster and Cloud Integration
Non-local execution on cluster or cloud infrastructure is implemented via plugins, allowing DynVision workflows to scale from laptops to high-performance computing environments seamlessly.

## Best Practices Relevant to DynVision

### Code Quality
It's recommended to use the linter for quality checking, keep filenames short but informative, and separate Python helper functions from rules. These practices would improve DynVision's codebase readability and maintainability.

### Portability
Annotating rules with versioned Conda or container-based software environment definitions ensures consistent software stacks across systems. This is important for reproducing DynVision's neural network experiments.

### Organization
The toolbox likely follows a cookiecutter-based structure to aid with clear folder organization, complementing Snakemake's workflow management capabilities.
How DynVision Likely Uses Snakemake
Based on the documentation:

- Parameter Management: YAML config scripts provide a high-level interface to crucial parameters
- Workflow Organization: DynVision emphasizes modularity, adaptability, reproducibility, reusability, and versatility in its design
- Resource Allocation: PyTorch lightning is used to organize training procedures and allocate them on GPU or CPU resources, which likely interfaces with 

## Snakemake's resource management
Data Pipeline Management: FFCV dataloader is used to optimize computational bottlenecks in data loading and transfer, which Snakemake would organize in the overall workflow

Understanding Snakemake is crucial for effectively using, extending, and documenting the DynVision toolbox, as it forms the backbone of how experimental workflows are defined, configured, and executed in this computational neuroscience framework.


# Snakemake in DynVision: Comprehensive Knowledge Reference

## 1. Core Architecture and Organization

### 1.1 Design Philosophy

DynVision leverages Snakemake as its workflow management system to create reproducible, scalable neural network experiments with biologically plausible temporal dynamics. The implementation follows key principles:

- **Modularity**: Separates concerns into specialized Snakefiles
- **Reproducibility**: Ensures consistent execution across environments
- **Scalability**: Scales from laptops to HPC clusters seamlessly
- **Extensibility**: Facilitates adding new models, datasets, and analyses
- **Discoverability**: Self-documents through structured rules and comments

### 1.2 File Structure Organization

The DynVision workflow is organized into specialized Snakefiles:

```
dynvision/
├── worklfow/
		├── Snakefile              # Main entry point and top-level rules
		├── snake_utils.smk        # Utility functions and configuration management
		├── snake_data.smk         # Dataset preparation and processing pipeline
		├── snake_runtime.smk      # Model training and evaluation pipeline
		└── snake_visualizations.smk # Result visualization and analysis
├── configs/           # YAML configuration files
├── data/              # Data processing scripts
├── runtime/           # Training and evaluation scripts
├── visualization/     # Visualization generation scripts
├── utils/             # Helper utilities
└── project_paths.py       # Centralized path management
```

### 1.3 Dependency Resolution

DynVision uses Snakemake's automated dependency resolution to manage complex workflow graphs:

```
rule all_experiments:
    input:
        expand(project_paths.reports / 'experiment_{experiment}.done',
            experiment = config.experiment)
```

This creates a directed acyclic graph (DAG) of jobs where experiments depend on model evaluations, which depend on trained models, which depend on initialized models and processed datasets.

## 2. Configuration Management System

### 2.1 Hierarchical Configuration

DynVision implements a layered configuration system with priority-based loading:

```python
# Priority-based configuration loading (higher overrides lower)
configfile: project_paths.scripts.configs / 'config_defaults.yaml'
configfile: project_paths.scripts.configs / 'config_data.yaml'
configfile: project_paths.scripts.configs / 'config_workflow.yaml'
configfile: project_paths.scripts.configs / 'config_experiments.yaml'

# Convert to SimpleNamespace for dot notation access
config = SimpleNamespace(**config)

# Save runtime configuration for debugging and reproducibility
runtime_config = project_paths.scripts.configs / 'config_runtime.yaml'
with open(runtime_config, 'w') as f:
    f.write("# This is an automatically compiled file. Do not edit manually!\n")
    json.dump(config.__dict__, f, indent=4)
```

This approach allows:
- Setting sensible defaults in `config_defaults.yaml`
- Overriding with specific configurations in specialized files
- Command-line overrides for experimentation
- Runtime compilation for reproducibility

### 2.2 Configuration Structure

Key configuration categories include:

1. **Model Configuration**:
   - Architecture parameters (recurrence type, skip connections)
   - Initialization options (pretrained weights, seed)
   - Hyperparameters (learning rate, batch size, epochs)

2. **Data Configuration**:
   - Dataset definitions and locations
   - Data group mappings (category subsets)
   - Preprocessing parameters (resolution, transforms)

3. **Experiment Configuration**:
   - Parameter sweeps (contrast, duration, interval)
   - Evaluation protocols
   - Visualization settings

### 2.3 Parameter Expansion

DynVision includes sophisticated parameter expansion for hyperparameter sweeps:

```python
def args_product(
    args_dict: Optional[Dict] = None,
    delimiter: str = '+',
    assigner: str = '=',
    prefix: str = ':'
) -> List[str]:
    """Generate product of argument combinations.
    
    Args:
        args_dict: Dictionary of argument options
        delimiter: Character separating arguments
        assigner: Character separating key and value
        prefix: Prefix character for argument string
    
    Returns:
        List of argument combination strings
    """
    if not args_dict:
        return ['']

    # Convert single values to lists
    args_dict = {
        key: [value] if not isinstance(value, list) else value
        for key, value in args_dict.items()
    }

    # Generate combinations
    args_combinations = product(*args_dict.values())
    return [
        prefix + delimiter.join(
            f'{key}{assigner}{value}'
            for key, value in zip(args_dict.keys(), combo)
        )
        for combo in args_combinations
    ]
```

This enables:
- Compact representation of parameter combinations
- Systematic exploration of hyperparameter space
- Consistent filename generation for experiment outputs

## 3. Data Management Pipeline

### 3.1 Dataset Acquisition and Organization

DynVision manages datasets through a series of organized steps:

```python
rule get_data:
    """Download and prepare standard datasets."""
    input:
        script = SCRIPTS / 'data' / 'get_data.py'
    params:
        raw_data_path = lambda w: project_paths.data.raw / f'{w.data_name}',
        output = lambda w: project_paths.data.raw \
            / f'{w.data_name}' \
            / f'{w.data_subset}' \
            / 'folder.link',
        # Additional parameters
    output:
        flag = directory(project_paths.data.raw \
            / '{data_name}' \
            / '{data_subset}' )
    shell:
        """
        python {input.script:q} \
            --output {params.output:q} \
            --data_name {params.data_name} \
            --raw_data_path {params.raw_data_path:q} \
            --subset {wildcards.data_subset} \
            --ext {params.ext} \
        """
```

Key features include:
- Support for standard datasets (CIFAR10, CIFAR100, MNIST)
- Support for external datasets (ImageNet)
- Efficient symbolic linking for dataset organization
- Dataset group definitions for experimental conditions

### 3.2 Symbolic Linking System

DynVision uses symbolic links to efficiently organize dataset subsets:

```python
rule symlink_data_subsets:
    """Create symlinks for dataset subsets."""
    input:
        get_data_location
    params:
        parent = lambda wildcards, output: Path(output.flag).parent,
        source = lambda wildcards, output: Path(output.flag).with_suffix(''),
    output:
        flag = project_paths.data.interim \
            / '{data_name}' \
            / '{data_subset}_{data_group}' \
            / '{category}.link'
    shell:
        """
        (mkdir -p {params.parent:q} && \
        ln -sf {input:q} {params.source:q} && \
        touch {output.flag:q}) 
        """
```

This provides:
- Storage efficiency (no data duplication)
- Flexible dataset organization
- Quick experiment reconfiguration

### 3.3 FFCV Integration for Performance

DynVision integrates FFCV for optimized data loading:

```python
rule build_ffcv_datasets:
    """Build FFCV datasets for faster data loading."""
    input:
        script = SCRIPTS / 'data' / 'ffcv_datasets.py',
        data = project_paths.data.interim / '{data_name}' / 'train_all' / 'folder.link'
    params:
        train_ratio = config.train_ratio,
        max_resolution = lambda w: config.data_resolution[w.data_name],
    output:
        train = project_paths.data.processed / '{data_name}' / 'train_all' / 'train.beton',
        val = project_paths.data.processed / '{data_name}' / 'train_all' / 'val.beton'
    shell:
        """
        python {input.script:q} \
            --input {input.data:q} \
            --output_train {output.train:q} \
            --output_val {output.val:q} \
            --train_ratio {params.train_ratio} \
            --data_name {wildcards.data_name} \
            --max_resolution {params.max_resolution} \
        """
```

Benefits include:
- Significantly faster data loading (50%+ speedup)
- Reduced CPU bottlenecks
- Optimized preprocessing and augmentation
- Improved GPU utilization

## 4. Model Lifecycle Management

### 4.1 Model Initialization

DynVision provides structured model initialization:

```python
rule init_model:
    """Initialize a model with specified configuration."""
    input:
        script = SCRIPTS / 'runtime' / 'init_model.py',
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all' \
            / 'folder.link'
    params:
        config_path = CONFIGS,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        init_with_pretrained = config.init_with_pretrained,
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt'
    shell:
        """
        python {input.script:q} \
            --config_path {params.config_path:q} \
            --model_name {wildcards.model_name} \
            --dataset {input.dataset:q} \
            --data_name {wildcards.data_name} \
            --seed {wildcards.seed} \
            --output {output.model_state:q} \
            --init_with_pretrained {params.init_with_pretrained} \
            {params.model_arguments} \
        """
```

Features include:
- Support for various model architectures (AlexNet, CorNetRT, CordsNet, DyRCNNx4)
- Parameter customization via model_args
- Pretrained weight initialization
- Seed control for reproducibility

### 4.2 Training Process

The training pipeline integrates with PyTorch Lightning:

```python
rule train_model:
    """Train a model on specified dataset."""
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt',
        dataset_train = project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'train.beton',
        dataset_val = project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'val.beton',
        script = SCRIPTS / 'runtime' / 'train_model.py'
    params:
        # Extensive parameter list
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_trained.pt'
    shell:
        """
        python {input.script:q} \
            --config_path {params.config_path:q} \
            --input_model_state {input.model_state:q} \
            --model_name {wildcards.model_name} \
            --dataset_train {input.dataset_train:q} \
            --dataset_val {input.dataset_val:q} \
            --data_name {wildcards.data_name} \
            --output_model_state {output.model_state:q} \
            --learning_rate {params.learning_rate} \
            --epochs {params.epochs} \
            --batch_size {params.batch_size} \
            --seed {wildcards.seed} \
            --check_val_every_n_epoch {params.check_val_every_n_epoch} \
            --log_every_n_steps {params.log_every_n_steps} \
            --accumulate_grad_batches {params.accumulate_grad_batches} \
            --resolution {params.resolution} \
            --precision {params.precision} \
            --profiler {params.profiler} \
            --store_responses {params.store_responses} \
            --enable_progress_bar {params.enable_progress_bar} \
            --use_ffcv {params.use_ffcv} \
            --loss {params.loss} \
            --n_timesteps {params.n_timesteps} \
            {params.model_arguments} \
        """
```

Key capabilities:
- Configurable training hyperparameters
- PyTorch Lightning integration for training loop management
- FFCV dataloader integration for performance
- Automatic checkpoint management
- Validation during training
- Response storage for analysis

### 4.3 Model Evaluation

DynVision provides a comprehensive model evaluation pipeline:

```python
rule test_model:
    """Evaluate a trained model on test data."""
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}.pt',
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'test_{data_group}' \
            / 'folder.link',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        # Various parameters
    output:
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        results = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv'
    shell:
        """
        python {input.script:q} \
            # Command-line parameters
        """
```

Features include:
- Testing on various data groups and conditions
- Storage of detailed neural responses
- Comprehensive result metrics
- Support for specialized data loaders
- Parameter sweep evaluation

## 5. Visualization and Analysis Pipeline

### 5.1 Analysis Categories

**Weight Analysis**:
```python
rule plot_weight_distributions:
    """Visualize model weight distributions."""
    input:
        state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{data_identifier}_{status}.pt',
        script = SCRIPTS / 'visualization' / 'plot_weight_distributions.py'
    output:
        plot = project_paths.figures \
            / 'weight_distributions' \
            / '{model_name}{data_identifier}_{status}_weights.{format}'
    # ...
```

**Classifier Response Analysis**:
```python
rule plot_classifier_responses:
    """Analyze and visualize classifier responses."""
    input:
        dataframe = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{data_identifier}_test_outputs.csv',
        script = SCRIPTS / 'visualization' / 'plot_classifier_responses.py'
    output:
        directory(project_paths.figures \
            / 'classifier_response' \
            / '{model_name}{data_identifier}')
    # ...
```

**Adaptation Analysis**:
```python
checkpoint plot_adaption:
    """Analyze and visualize model adaptation."""
    input:
        responses = expand(project_paths.models \
            / '{{model_name}}' \
            / '{{model_name}}:{{args1}}{{category}}={category_value}{{args2}}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}_test_responses.pt',
            category_value = lambda w: config.experiment_config['categories'][w.category],
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        # ...
    params:
        measures = ['power', 'peak_height', 'peak_time'],
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
    output:
        flag = project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / '{plot}.flag'
    # ...
```

### 5.2 Comparative Analysis

DynVision supports cross-model and cross-parameter experiment comparisons:

```python
rule plot_experiments_on_models:
    """Generate comparative visualizations across models."""
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}{{model_args}}_{seed}_{data_name}_{status}_{data_group}' / 'adaption.flag',
            experiment = config.experiment,
            model_name = config.model_name,
            seed = config.seed,
            data_name = config.data_name,
            status = config.status,
            data_group = config.data_group,
        )
    output:
        temp(project_paths.figures / 'plot_experiments_on_models{model_args}.done')
    # ...
```

This enables systematic analysis of:
- Performance differences between model architectures
- Effects of recurrence type on neural dynamics
- Parameter sensitivity (contrast, duration, interval)
- Temporal response characteristics

## 6. PyTorch Lightning Integration

### 6.1 Complementary Responsibilities

DynVision integrates Snakemake with PyTorch Lightning for efficient workflow management:

- **Snakemake Responsibilities**:
  - Workflow orchestration and dependency management
  - Parameter management and configuration
  - File management and organization
  - Experiment definition and tracking
  - Resource allocation and scheduling

- **PyTorch Lightning Responsibilities**:
  - Training loop implementation
  - Device management (CPU/GPU)
  - Precision handling (16/32-bit)
  - Checkpoint management
  - Validation and testing
  - Logging and monitoring

### 6.2 Parameter Passing

Snakemake passes configuration to Lightning via command-line arguments:

```python
shell:
    """
    python {input.script:q} \
        --config_path {params.config_path:q} \
        --input_model_state {input.model_state:q} \
        --model_name {wildcards.model_name} \
        --dataset_train {input.dataset_train:q} \
        --dataset_val {input.dataset_val:q} \
        --data_name {wildcards.data_name} \
        --output_model_state {output.model_state:q} \
        --learning_rate {params.learning_rate} \
        # ... additional parameters
    """
```

The Python script then configures a LightningModule:

```python
# Inside train_model.py
model = get_model(args.model_name, **model_args)
model.save_hyperparameters()  # Lightning hyperparameter logging
trainer = pl.Trainer(
    max_epochs=args.epochs,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    # ... other trainer parameters
)
trainer.fit(model, train_loader, val_loader)
```

### 6.3 Environment Detection

DynVision adapts to different execution environments:

```python
# Progress bar disabled on compute clusters
enable_progress_bar = not project_paths.iam_on_cluster()

# Batch size adjusted for development vs. production
batch_size = config.batch_size if project_paths.iam_on_cluster() else 3

# Path management for mounted datasets
def get_data_base_dir(wildcards: Any) -> Path:
    """Get the base directory for a dataset."""
    data_name = wildcards.data_name
    if data_name in config.mounted_datasets and project_paths.iam_on_cluster():
        return Path(f'/{data_name}')
    return project_paths.data.raw / data_name
```

## 7. Error Handling and Debugging

### 7.1 Logging System

DynVision implements structured logging:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
pylogger = logging.getLogger('workflow.utils')
```

Specialized loggers are created for each workflow component:

```python
logger = logging.getLogger('workflow.data')
logger = logging.getLogger('workflow.runtime')
logger = logging.getLogger('workflow.visualizations')
```

### 7.2 Benchmarking

Rules include benchmarking for performance analysis:

```python
benchmark:
    project_paths.benchmarks / 'train_model_{model_name}{model_args}_{seed}_{data_name}.txt'
```

### 7.3 Temporary Flags

The workflow uses temporary flag files to mark completion:

```python
output:
    temp(project_paths.reports / 'experiment_{experiment}.done')
shell:
    """
    touch {output:q}
    """
```

## 8. Execution Strategies

### 8.2 Cluster Execution

The workflow supports execution on compute clusters:

```python
params:
    executor_start = config.executor_start if config.use_executor else '',
    executor_close = config.executor_close if config.use_executor else ''
shell:
    """
    {params.executor_start}
    python {input.script:q} \
        # ... parameters
    {params.executor_close}
    """
```

This wrapper allows for cluster-specific execution commands.
To customize the execution to the requirements of your specific cluster environment, edit the executor_start and executor_close entries in config_defaults.yaml.

## 9. Practical Usage Examples

### 9.1 Basic Model Training

using the exisitng snakemake rule structure and tuning config settings

```bash
# Train a DyRCNNx4 model with full recurrence
snakemake \
  --config model_name=DyRCNNx4 seed=0001 model_args="{rctype:full, tsteps=:20}"
```

or by explicitly requesting a certain output file

```bash
snakemake \
  /path/to/model_dir/DyRCNNx4/DyRCNNx4:tsteps=20+rctype:full_0001_cifar100_trained.pt"
```

### 9.2 Parameter Sweep

```bash
# Run experiments with different recurrence types
snakemake -j4 all_experiments \
  --config experiment=adaption model_args="{rctype: [full, self, pointdepthwise, depthpointwise]}"
```

### 9.3 Visualizing Results

```bash
# Generate visualizations for all models
snakemake plot_experiments_on_models
```

### 9.4 Cluster Execution

```bash
# Run on SLURM cluster
snakemake -j100 all_experiments --cluster "sbatch -p {params.partition} -t {params.time}"
```

## 10. Extension Patterns

### 10.1 Adding New Models

To add a new model architecture:

1. Create model implementation in DynVision codebase
2. Add model name to configuration files
3. No changes to Snakemake workflow required

### 10.2 Adding New Analysis Rules

To add a new visualization:

1. Create visualization script in scripts/visualization/
2. Add new rule to snake_visualizations.smk following existing patterns
3. Optionally add to comprehensive visualization targets

### 10.3 Adding New Experiment Types

To define a new experiment:

1. Add experiment configuration to config_experiments.yaml
2. No changes to workflow required for standard experiments
3. For specialized experiments, add appropriate rule patterns

## 11. Best Practices

### 11.1 Code Organization

- Keep all rule definitions clear with descriptive docstrings
- Separate implementation details into Python scripts
- Use consistent naming patterns for inputs and outputs
- Group related rules logically in specialized Snakefiles

### 11.2 Dependency Management

- Use explicit wildcard constraints to prevent ambiguity
- Define rule order when output patterns could conflict
- Use lambda wildcards for dynamic input resolution
- Ensure all file paths use project_paths for consistency

### 11.3 Documentation Standards

- Include docstrings for all rules explaining purpose
- Document parameters and their impact
- Add example command lines in comments
- Use type annotations and utility functions

### 11.4 Resource Management

- Define compute requirements explicitly
- Group visualization tasks to prevent excessive parallelism