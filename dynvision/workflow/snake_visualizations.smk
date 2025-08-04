"""Visualization workflow for model analysis and results.

This workflow handles all visualization-related tasks including:
- Confusion matrix generation
- Classifier response analysis
- Weight distribution visualization
- Experiment result plotting
- Adaptation analysis
- Interactive notebook generation
- Enhanced model visualization
- Model architecture visualization
- Enhanced weight analysis
- Temporal dynamics visualization
- Interactive notebook generation

Usage:
    # Generate confusion matrix
    snakemake plot_confusion_matrix model_name=DyRCNNx4

    # Analyze classifier responses
    snakemake plot_classifier_responses model_name=DyRCNNx4
"""

logger = logging.getLogger('workflow.visualizations')


rule plot_confusion_matrix:
    """Generate and save confusion matrix visualization.

    Input:
        test_results: Model evaluation results
        dataset: Test dataset for class information
        script: Plotting script
    
    Output:
        Confusion matrix plot

    Parameters:
        palette: Color palette for visualization
    """
    input:
        test_results = project_paths.reports / '{path}_{data_name}_testing_results.csv',
        dataset = project_paths.data.interim / '{data_name}_test' / 'folder.link',
        script = SCRIPTS / 'visualization' / 'plot_confusion_matrix.py'
    params:
        palette = 'cividis',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        plot = project_paths.figures / '{path}_{data_name}_confusion.{format}'
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.test_results:q} \
            --output {output.plot:q} \
            --dataset {input.dataset} \
            --palette {params.palette} \
            --format {wildcards.format} \
        """

rule plot_classifier_responses:
    """Analyze and visualize classifier responses.

    This rule generates visualizations of model responses
    across different conditions and stimuli.

    Input:
        dataframe: Model output data
        script: Visualization script
    
    Output:
        Directory of response visualizations
    """
    input:
        dataframe = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{data_identifier}_test_outputs.csv',
        script = SCRIPTS / 'visualization' / 'plot_classifier_responses.py'
    params:
        n_units = 10,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        directory(project_paths.figures \
            / 'classifier_response' \
            / '{model_name}{data_identifier}')
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.dataframe:q} \
            --output {output:q} \
            --n_units {params.n_units}
        """

rule plot_weight_distributions:
    """Visualize model weight distributions.

    This rule analyzes and plots the distribution of weights
    across different layers of the model.

    Input:
        state: Model state dictionary
        script: Plotting script
    
    Output:
        Weight distribution plot
    """
    input:
        state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{data_identifier}_{status}.pt',
        script = SCRIPTS / 'visualization' / 'plot_weight_distributions.py'
    params:
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        plot = project_paths.figures \
            / 'weight_distributions' \
            / '{model_name}{data_identifier}_{status}_weights.{format}'
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.state:q} \
            --output {output.plot:q} \
            --format {wildcards.format} 
        """

rule plot_experiment_outputs:
    """Visualize experiment results.

    This rule creates comprehensive visualizations of
    experiment outputs and comparisons.

    Input:
        test_outputs: Experiment results
        script: Visualization script
    
    Output:
        Experiment visualization
    """
    input:
        test_outputs = expand(project_paths.reports \
            / '{{model_name}}' \
            / '{{model_name}}:{{category}}={category_value}{args}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}test_outputs.csv',
            category_value = lambda w: config.model_args[w.category],
            args = lambda w: args_product(dict_poped(config.model_args, w.category), prefix=','),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'plot_experiment_outputs.py'
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        plot = project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{category}=*_{seed}_{data_name}_{status}_{data_group}' / 'experiment_outputs_label{label_target}.{format}'
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --test_outputs {input.test_outputs:q} \
            --output {output.plot:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --format {wildcards.format} \
            --style {params.style} 
        """

checkpoint plot_adaption:
    """Analyze and visualize model adaptation.

    This rule generates visualizations of how models
    adapt to different conditions over time.

    Input:
        responses: Model responses
        test_outputs: Test results
        script: Analysis script
    
    Output:
        Adaptation analysis plots
    """
    input:
        responses = expand(project_paths.models \
            / '{{model_name}}' \
            / '{{model_name}}:{{args1}}{{category}}={category_value}{{args2}}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}_test_responses.pt',
            category_value = lambda w: config.experiment_config['categories'][w.category],
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        test_outputs = expand(project_paths.reports \
            / '{{model_name}}' \
            / '{{model_name}}:{{args1}}{{category}}={category_value}{{args2}}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}_test_outputs.csv',
            category_value = lambda w: config.experiment_config['categories'][w.category],
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'plot_{plot}.py'
    params:
        measures = ['power', 'peak_height', 'peak_time'],
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        flag = project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / '{plot}.flag'
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --responses {input.responses:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output.flag:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --measures {params.measures} 
        """

rule plot_experiments:
    """Collect and organize experiment visualizations.

    This rule aggregates visualizations from multiple
    experiments for comparison and analysis.
    """
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / 'adaption.flag',
            experiment = config.experiment,
            model_name = config.model_name,
            category = list(config.model_args.keys())[0],
            args1 = "tsteps=20+",
            args2 = args_product(dict_poped(config.model_args, [list(config.model_args.keys())[0], "tsteps"]), prefix='+'),
            seed = config.seed,
            data_name = config.data_name,
            status = config.status,
            data_group = config.data_group,
        )
    # group: "visualization"

rule plot_experiments_on_models:
    """Generate comparative visualizations across models.

    This rule creates visualizations that compare
    experiment results across different model architectures.
    """
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
    shell:
        """
        touch {output:q} 
        """

# Log workflow initialization
logger.info("Visualization workflow initialized")
logger.info(f"Figure directory: {project_paths.figures}")