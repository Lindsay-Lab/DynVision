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
        dataset_ready = project_paths.data.interim / '{data_name}' / 'test_all.ready',
        script = SCRIPTS / 'visualization' / 'plot_confusion_matrix.py'
    params:
        palette = 'cividis',
        dataset_path = lambda w: project_paths.data.interim / w.data_name / 'test_all',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        plot = project_paths.figures / '{path}_{data_name}_confusion.{format}'
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.test_results:q} \
            --output {output.plot:q} \
            --dataset_path {params.dataset_path:q} \
            --palette {params.palette} \
            --format {wildcards.format} 
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
            / '{data_loader}' \
            / '{model_name}{data_identifier}' \
            / 'test_outputs.csv',
        script = SCRIPTS / 'visualization' / 'plot_classifier_responses.py'
    params:
        n_units = 10,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
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
        model = expand(project_paths.models \
            / '{{model_name}}' \
            / '{{model_name}}{{args1}}{{category}}={cat_value}{{args2}}_{seeds}'
            / '{{data_name}}' \
            / '{{status}}.pt',
            cat_value = lambda w: config.experiment_config['categories'].get(w.category, []),
            seeds=lambda w: w.seeds.split('.')
            ),
        script = SCRIPTS / 'visualization' / 'plot_weight_distributions.py'
    params:
        row = 'connection_type',  # status
        column = None,
        hue = 'category',  # connection_type
        x_axis = 'layer',
        category_key = lambda w: w.category,
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        log_scale = lambda w: '--log-scale' if w.logscale else '',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        figure = project_paths.figures / 'weights' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}_{status}' / 'weights{logscale}.png',
        summary = project_paths.figures / 'weights' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}_{status}' / 'weights_summary{logscale}.csv',
    wildcard_constraints:
        logscale = r'(_logscale)?',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.model:q} \
            --output {output.figure:q} \
            --row {params.row} \
            --column {params.column} \
            --hue {params.hue} \
            --x_axis {params.x_axis} \
            --category-key {params.category_key} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --summary-output {output.summary:q} \
            {params.log_scale}
        """


rule plot_performance:
    """Plot performance metrics (hierarchical structure).

    Input/Output follow new pattern:
    - Input: {experiment}/{model_identifier}/{data_name}:{data_group}_{status}/test_data.csv
    - Output: {experiment}/{model_identifier}/{data_name}:{data_group}_{status}/performance.png
    """
    input:
        data = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seed=lambda w: w.seeds.split('.'),
        ),
        # data_ffonly = expand(
        #     project_paths.reports
        #     / '{{experiment}}ffonly'
        #     / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seed}'
        #     / '{{data_name}}:{{data_group}}_{{status}}'
        #     / 'test_data.csv',
        #     seed=lambda w: w.seeds.split('.'),
        # ),
        script = SCRIPTS / 'visualization' / 'plot_performance.py'
    params:
        data_ffonly = lambda w: [project_paths.reports / f'{w.experiment}ffonly' / f'{w.model_name}{w.args1}{w.category}=*{w.args2}_{seed}' / f'{w.data_name}:{w.data_group}_{w.status}' / 'test_data.csv' for seed in w.seeds.split('.')],
        row = None,
        subplot = 'parameter',
        hue = 'category',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category,
        experiment = lambda w: ['uniformnoise', 'poissonnoise', 'gaussiannoise', 'gaussiancorrnoise'] if w.experiment == 'noise' else w.experiment,
        confidence_measure = getattr(config, 'plot_confidence_measure', "first_label_confidence"),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        subplot_filter = [], #[0.2, 0.6, 1.0], #[0.1, 0.5, 1.0], #lambda w: config.experiment_config[w.experiment].get('subplot_filter', []),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.figures / '{experiment}' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}:{data_group}_{status}' / 'performance.png',
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --data-ffonly {params.data_ffonly:q} \
            --output {output:q} \
            --row {params.row} \
            --subplot {params.subplot} \
            --hue {params.hue} \
            --parameter-key {params.parameter} \
            --category-key {params.category} \
            --experiment {params.experiment} \
            --confidence-measure {params.confidence_measure} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --subplot-filter {params.subplot_filter}
        """
            # --plot-individual-seeds


rule plot_training:
    """Plot training metrics (hierarchical structure)."""
    input:
        test_data = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seeds}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seeds=lambda w: w.seeds.split('.'),
        ),
        script = SCRIPTS / 'visualization' / 'plot_training.py'
    params:
            # / f'{w.model_name}{w.args1}{w.category}=*{w.args2}_{".".join(config.seed)}_accuracy.csv',
        accuracy_csv = lambda w: project_paths.reports \
            / 'wandb' \
            / f'{w.model_name}{w.args1}{w.category}=*{w.args2}_7000.7001.7002_accuracy.csv',
        loss_csv= lambda w: project_paths.reports \
            / 'wandb' \
            / f'{w.model_name}{w.args1}{w.category}=*{w.args2}_7000.7001.7002_loss.csv',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
        validation_frequency = 10,
        column = getattr(config, 'column', 'parameter'),
        subplot = getattr(config, 'subplot', 'layers'),
        hue = getattr(config, 'hue', 'category'),
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category,
        confidence_measure = getattr(config, 'plot_confidence_measure', "first_label_confidence"),
        accuracy_measure = getattr(config, 'plot_accuracy_measure', "accuracy"),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
    output:
        project_paths.figures / '{experiment}' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}:{data_group}_{status}' / 'training.png',
    shell:
        """
        {params.execution_cmd} \
            --test_data {input.test_data:q} \
            --accuracy_csv {params.accuracy_csv:q} \
            --loss_csv {params.loss_csv:q} \
            --column {params.column} \
            --subplot {params.subplot} \
            --hue {params.hue} \
            --category-key {params.category} \
            --parameter-key {params.parameter} \
            --confidence-measure {params.confidence_measure} \
            --accuracy-measure {params.accuracy_measure} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --output {output:q} \
            --validation-frequency {params.validation_frequency}
        """


rule plot_dynamics:
    """Plot dynamics (hierarchical structure)."""
    input:
        data = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seeds}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seeds=lambda w: w.seeds.split('.'),
        ),
        script = SCRIPTS / 'visualization' / 'plot_dynamics.py'
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.figures / '{experiment}' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}:{data_group}_{status}' / 'dynamics_{focus_layer}.png',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --experiment {wildcards.experiment} \
            --category {wildcards.category} \
            --focus-layer {wildcards.focus_layer} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """


rule plot_responses:
    """Plot responses (hierarchical structure)."""
    input:
        data = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}:{{args1}}{{category}}=*{{args2}}_{seeds}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seeds=lambda w: w.seeds.split('.'),
        ),
        script = SCRIPTS / 'visualization' / 'plot_responses.py'
    params:
        column = getattr(config, 'column', 'parameter'),  # first_label_index, epoch
        subplot = getattr(config, 'subplot', 'layers'),  # classifier_topk
        hue = getattr(config, 'hue', 'category'),
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category,
        confidence_measure = getattr(config, 'plot_confidence_measure', "first_label_confidence"),
        accuracy_measure = getattr(config, 'plot_accuracy_measure', "accuracy"),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.figures / '{experiment}' / '{model_name}:{args1}{category}=*{args2}_{seeds}' / '{data_name}:{data_group}_{status}' / 'responses.png',
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q} \
            --column {params.column} \
            --subplot {params.subplot} \
            --hue {params.hue} \
            --parameter-key {params.parameter} \
            --category-key {params.category} \
            --experiment {wildcards.experiment} \
            --confidence-measure {params.confidence_measure} \
            --accuracy-measure {params.accuracy_measure} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --panel-labels
        """

def _get_triptych_value(category: str) -> str:
    """Resolve the fixed value for one triptych axis.

    For "seed", returns "-" placeholder since all seeds are used (no default value).
    This placeholder won't match any actual seed, so no asterisk marking is added.
    """
    if category == "seed":
        return "-"

    model_args = getattr(config, "model_args", {})

    if isinstance(model_args, dict) and category in model_args:
        return str(model_args[category])

    if hasattr(model_args, category):
        return str(getattr(model_args, category))

    categories = getattr(config, "experiment_config", {}).get("categories", {})
    if category in categories and categories[category]:
        return str(categories[category][0])

    raise ValueError(f"No default value available for triptych category '{category}'.")


def _build_triptych_identifier(wildcards, star_slot: int) -> str:
    """Build the model identifier with wildcards for glob matching.

    Categories set to "seed" are skipped since seed is in the path suffix,
    not the model identifier. However, the prefix before seed (which may
    contain other parameters) is preserved.
    """
    cats = [wildcards.cat1, wildcards.cat2, wildcards.cat3]
    prefixes = [wildcards.a1, wildcards.a2, wildcards.a3]
    values = [
        '*' if star_slot == i + 1 else _get_triptych_value(cat)
        for i, cat in enumerate(cats)
    ]

    parts = []
    for prefix, cat, value in zip(prefixes, cats, values):
        if cat == "seed":
            # When skipping seed, keep the prefix content (e.g., "+skip=true+feedback=false+")
            # but strip the trailing '+' that would have connected to seed=value
            trimmed_prefix = prefix.rstrip('+')
            if trimmed_prefix:
                parts.append(trimmed_prefix)
            continue
        parts.append(f"{prefix}{cat}={value}")

    return "".join(parts) + wildcards.a4


def _get_triptych_data_inputs(wildcards, star_slot: int) -> list:
    """Get input data files for a triptych column.

    Returns empty list when the category is "seed" since seed variation
    is already present in the other data files (they expand over seeds).
    """
    cats = [wildcards.cat1, wildcards.cat2, wildcards.cat3]
    if cats[star_slot - 1] == "seed":
        return []

    return expand(
        project_paths.reports
        / wildcards.experiment
        / f'{wildcards.model_name}:{{model_identifier}}_{{seed}}'
        / f'{wildcards.data_name}:{wildcards.data_group}_{wildcards.status}'
        / 'test_data.csv',
        model_identifier=_build_triptych_identifier(wildcards, star_slot),
        seed=wildcards.seeds.split('.'),
    )


rule plot_responses_tripytch:
    input:
        data1 = lambda w: _get_triptych_data_inputs(w, star_slot=1),
        data2 = lambda w: _get_triptych_data_inputs(w, star_slot=2),
        data3 = lambda w: _get_triptych_data_inputs(w, star_slot=3),
        script = SCRIPTS / 'visualization' / 'plot_response_tripytch.py'
    params:
        accuracy1 = lambda w: project_paths.reports / 'wandb' / f"{w.model_name}:{_build_triptych_identifier(w, star_slot=1)}_7000.7001.7002_accuracy.csv",
        accuracy2 = lambda w: project_paths.reports / 'wandb' / f"{w.model_name}:{_build_triptych_identifier(w, star_slot=2)}_7000.7001.7002_accuracy.csv",
        accuracy3 = lambda w: project_paths.reports / 'wandb' / f"{w.model_name}:{_build_triptych_identifier(w, star_slot=3)}_7000.7001.7002_accuracy.csv",
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
        # Build data arguments conditionally - skip empty inputs (seed categories)
        data1_arg = lambda w, input: f"--data {' '.join(shlex.quote(str(f)) for f in input.data1)}" if input.data1 else "",
        data2_arg = lambda w, input: f"--data2 {' '.join(shlex.quote(str(f)) for f in input.data2)}" if input.data2 else "",
        data3_arg = lambda w, input: f"--data3 {' '.join(shlex.quote(str(f)) for f in input.data3)}" if input.data3 else "",
        category = lambda w: ' '.join([w.cat1, w.cat2, w.cat3]),
        default_category_values = lambda w: ' '.join([_get_triptych_value(cat) for cat in [w.cat1, w.cat2, w.cat3]]),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        confidence_measure = getattr(config, 'plot_confidence_measure', "first_label_confidence"),
        accuracy_measure = getattr(config, 'plot_accuracy_measure', "accuracy"),
    output:
        project_paths.figures / '{experiment}' / '{model_name}:{a1}{cat1}=*{a2}{cat2}=*{a3}{cat3}=*{a4}_{seeds}' / '{data_name}:{data_group}_{status}' / 'responses_tripytch.png',
    wildcard_constraints:
        a1 = r'([a-z,;\+\-=\d\.]*?)',
        a2 = r'([a-z,;\+\-=\d\.]*?)',
        a3 = r'([a-z,;\+\-=\d\.]*?)',
        a4 = r'([a-z,;\+\-=\d\.]+|\s?)',
        cat1 = r'[a-z]+',
        cat2 = r'[a-z]+',
        cat3 = r'[a-z]+',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            {params.data1_arg} \
            {params.data2_arg} \
            {params.data3_arg} \
            --accuracy1 {params.accuracy1:q} \
            --accuracy2 {params.accuracy2:q} \
            --accuracy3 {params.accuracy3:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --experiment {wildcards.experiment} \
            --category {params.category:q} \
            --default-category-values {params.default_category_values:q} \
            --confidence-measure {params.confidence_measure} \
            --accuracy-measure {params.accuracy_measure} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """
