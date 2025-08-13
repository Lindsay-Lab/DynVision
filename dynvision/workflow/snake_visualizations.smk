"""Visualization workflow for model analysis and results.
"""

logger = logging.getLogger('workflow.visualizations')


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
    group: "visualization"
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
    group: "visualization"
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
    group: "visualization"
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

rule process_plotting_data:
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
        script = SCRIPTS / 'visualization' / 'process_plotting_data.py'
    params:
        measures = ['power', 'peak_height', 'peak_time'],
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / 'layer_power.csv',
    shell:
        """
        {params.execution_cmd} \
            --responses {input.responses:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --measures {params.measures} 
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
        data = project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / 'layer_power.csv',
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
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / '{plot}.flag',
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q} \
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
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / 'adaption.csv',
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
    group: "visualization"


rule plot_experiments_on_models:
    """Generate comparative visualizations across models.

    This rule creates visualizations that compare
    experiment results across different model architectures.
    """
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}{{model_args}}_{seed}_{data_name}_{status}_{data_group}' / 'adaption.csv',
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

