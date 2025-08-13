"""Model training and evaluation workflow.

This workflow handles all model-related operations including:
- Model initialization
- Training configuration
- Model training
- Model evaluation
- Result collection
"""

logger = logging.getLogger('workflow.runtime')

rule init_model:
    """Initialize a model with specified configuration.

    Input:
        script: Model initialization script
        dataset: Training dataset for shape inference
    
    Output:
        Initialized model state dict

    Parameters:
        config_path: Path to configuration file
        model_arguments: Model-specific arguments
        init_with_pretrained: Whether to initialize from pretrained weights
    """
    input:
        script = SCRIPTS / 'runtime' / 'init_model.py',
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all' \
            / 'folder.link'
    params:
        config_path = CONFIGS,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt'
    benchmark:
        project_paths.benchmarks / 'init_model_{model_name}{model_args}_{seed}_{data_name}.txt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.config_path:q} \
            --model_name {wildcards.model_name} \
            --dataset {input.dataset:q} \
            --data_name {wildcards.data_name} \
            --seed {wildcards.seed} \
            --output {output.model_state:q} \
            {params.model_arguments} 
        """

rule train_model:
    """Train a model on specified dataset.

    Input:
        model_state: Initial model state
        dataset_train: Training dataset
        dataset_val: Validation dataset
        script: Training script
    
    Output:
        Trained model state dict

    Parameters:
        config_path: Path to configuration file
        epochs: Number of training epochs
        batch_size: Training batch size
        model_arguments: Model-specific arguments
        learning_rate: Training learning rate
        loss: Loss function configuration
        resolution: Input resolution 
        check_val_every_n_epoch: Validation frequency
        accumulate_grad_batches: Gradient accumulation steps
        precision: Training precision
        store_responses: Number of responses to store
        profiler: Training profiler configuration
        enable_progress_bar: Whether to show progress bar
        execution_cmd: Complete execution command with conditional wrappers
    """
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt',
        dataset_link = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all' \
            / 'folder.link',
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
        config_path = CONFIGS,
        data_group = "all",
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        resolution = lambda w: config.data_resolution[w.data_name],
        normalize = lambda w: json.dumps((
            config.data_statistics[w.data_name]['mean'],
            config.data_statistics[w.data_name]['std']
        )),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=get_param('use_distributed_mode', False)(w),
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_trained.pt'
    benchmark:
        project_paths.benchmarks / 'train_model_{model_name}{model_args}_{seed}_{data_name}.txt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.config_path:q} \
            --input_model_state {input.model_state:q} \
            --output_model_state {output.model_state:q} \
            --model_name {wildcards.model_name} \
            --dataset_link {input.dataset_link:q} \
            --dataset_train {input.dataset_train:q} \
            --dataset_val {input.dataset_val:q} \
            --data_name {wildcards.data_name} \
            --data_group {params.data_group} \
            --seed {wildcards.seed} \
            --resolution {params.resolution} \
            --normalize {params.normalize:q} \
            {params.model_arguments} 
        """

use rule train_model as train_model_distributed with:
    output:
        # todo: find more general fix to automatically switch slurm resource requests for distributed mode
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name,imagenet}_trained.pt'

rule test_model:
    """Evaluate a trained model on test data.

    Input:
        model_state: Trained model state
        dataset: Test dataset
        script: Evaluation script
    
    Output:
        responses: Model responses
        results: Evaluation results

    Parameters:
        config_path: Path to configuration file
        batch_size: Evaluation batch size
        data_group: Data grouping configuration
        model_arguments: Model-specific arguments
        data_arguments: Data-specific arguments
        loss: Loss function configuration
        store_responses: Number of responses to store
        enable_progress_bar: Whether to show progress bar
    """
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
        config_path = CONFIGS,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        data_arguments = lambda w: parse_arguments(w, 'data_args'),
        normalize = lambda w: json.dumps((
            config.data_statistics[w.data_name]['mean'],
            config.data_statistics[w.data_name]['std']
        )),
        enable_progress_bar = True,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        results = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv'
    benchmark:
        project_paths.benchmarks / 'test_model_{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}.txt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.config_path:q} \
            --input_model_state {input.model_state:q} \
            --output_results {output.results:q} \
            --output_responses {output.responses:q} \
            --model_name {wildcards.model_name} \
            --data_name {wildcards.data_name} \
            --dataset {input.dataset:q} \
            --data_loader {wildcards.data_loader} \
            --data_group {wildcards.data_group} \
            --seed {wildcards.seed} \
            --normalize {params.normalize:q} \
            --enable_progress_bar {params.enable_progress_bar} \
            {params.model_arguments} \
            {params.data_arguments}
        """

logger.info("Model workflow initialized")
logger.info(f"Model directory: {project_paths.models}")
logger.info(f"Reports directory: {project_paths.reports}")