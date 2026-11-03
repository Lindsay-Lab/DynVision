# Snakefile for output processing rules
# Included from the main Snakefile

ruleorder: aggregate_experiment_data > aggregate_experiment_data_single


rule process_single_test:
    """Process individual test output to extract metrics (Stage 1 of experiment processing).

    This rule runs immediately after each test_model execution completes,
    enabling parallel processing and reducing disk pressure.

    Input:
        test_responses: Model layer responses (large, 30GB+)
        test_outputs: Test performance metrics (small)
        script: Processing script

    Output:
        test_data: Processed metrics at sample-level (no metadata, no resolution applied)

    Processing:
        - Calculate layer metrics (response_avg, response_std, etc.)
        - Calculate classifier metrics (confidence, top-k accuracy)
        - Process test performance (add first_label_index, accuracy)
        - Add the run's time base (dt, idle_steps, time_ms) from
          test_outputs.csv.config.yaml, which test_model writes beside test_outputs.csv
        - Save sample-level CSV (metadata added in Stage 2)
        - Optionally delete test_responses.pt to free disk space

    Parameters:
        measures: List of metrics to compute
        memory_limit_gb: Soft memory limit
        remove_input_responses: Whether to delete test_responses.pt after processing
    """
    input:
        test_responses = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{test_identifier}' / 'test_responses.pt',
        test_outputs = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{test_identifier}' / 'test_outputs.csv',
        # ancient: once test_responses.pt is removed, a rerun needs the GPU
        # test again, so a newer script must not outdate existing test_data.csv.
        # Stage 2 adds the time base to Stage 1 files written before it existed.
        script = ancient(SCRIPTS / 'processing' / 'process_single_test.py')
    params:
        measures = ['response_avg', 'response_std', 'guess_confidence', 'first_label_confidence', 'accuracy_top3', 'accuracy_top5'],
        memory_limit_gb = 68.0,
        remove_input_responses = True,  # Delete large test_responses.pt after processing
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 40,  # Higher than aggregation
    output:
        test_data = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{test_identifier}' / 'test_data.csv',
    shell:
        """
        {params.execution_cmd} \
            --response {input.test_responses:q} \
            --test_output {input.test_outputs:q} \
            --output {output.test_data:q} \
            --measures {params.measures} \
            --memory_limit_gb {params.memory_limit_gb} \
            --remove_input_responses {params.remove_input_responses}
        """


rule aggregate_experiment_data:
    """Aggregate individual test data into experiment-level dataset (Stage 2 of experiment processing).

    This rule runs after all process_single_test rules for an experiment complete.
    It performs lightweight aggregation and metadata extraction.

    Input:
        test_data_files: All test_data.csv files from Stage 1 (across category sweep)
        test_configs: Corresponding .config.yaml files for metadata extraction
        script: Aggregation script

    Output:
        experiment_data: Aggregated CSV with metadata columns added

    Processing:
        - Load all test_data.csv files
        - Extract metadata from .config.yaml files (parameter, category, additional_parameters)
        - If 'status' in additional_parameters: extract status from file paths (with auto-epoch extraction)
        - Add metadata columns to each dataframe
        - Add each run's time base (dt, idle_steps, time_ms = (times_index + 1) * dt)
          and ms columns for legacy step parameters (stim -> tstim, ...)
        - Concatenate into single dataframe
        - Optionally apply class-level aggregation
        - Sort and save

    Parameters:
        parameter: Parameter key to extract (e.g., 'dsteps', 'stim')
        category: Category key to extract (e.g., 'rctype', 'tsteps')
        additional_parameters: Extra parameters to extract (if 'status' included, extracts from path; if 'epoch' included, auto-extracts from status string)
        sample_resolution: 'sample' or 'class'
        fail_on_missing_inputs: Error handling mode
    """
    input:
        test_data = expand(project_paths.reports \
            / '{{experiment}}' \
            / "{{model_name}}:{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{test_identifier}' / 'test_data.csv',
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            cat_value = lambda w: config.experiment_config['categories'].get(w.category, []),
            test_identifier = lambda w: get_test_specs_for_experiment(w.experiment),
        ),
        test_configs = expand(project_paths.reports \
            / '{{experiment}}' \
            / "{{model_name}}:{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{test_identifier}' / 'test_outputs.csv.config.yaml',
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            cat_value = lambda w: config.experiment_config['categories'].get(w.category, []),
            test_identifier = lambda w: get_test_specs_for_experiment(w.experiment),
        ),
        script = SCRIPTS / 'processing' / 'aggregate_experiment_data.py'
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        additional_parameters = ['seed', 'status', 'epoch', 'activityloss'],  # config keys ('status' and 'epoch' are extracted from file path); the time base (dt, idle_steps, time_ms) is always added
        sample_resolution = 'sample',  # 'sample' or 'class'
        fail_on_missing_inputs = False,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 3,  # Lower than process_single_test
    output:
        experiment_data = project_paths.reports \
            / '{experiment}' \
            / '{model_name}:{args1}{category}=*{args2}_{seed}' \
            / '{data_name}:{data_group}_{status}' \
            / 'test_data.csv',
    shell:
        """
        {params.execution_cmd} \
            --test_data {input.test_data:q} \
            --test_configs {input.test_configs:q} \
            --output {output.experiment_data:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --additional_parameters {params.additional_parameters} \
            --sample_resolution {params.sample_resolution} \
            --fail_on_missing_inputs {params.fail_on_missing_inputs}
        """


rule aggregate_experiment_data_single:
    """Aggregate test data for a single model configuration (no category sweep).

    Aggregates across test_identifiers (experiment variations) for one model config.
    Used when model_args contains no wildcard (*).

    Input:
        test_data_files: All test_data.csv files from Stage 1 (across test_identifiers)
        test_configs: Corresponding .config.yaml files for metadata extraction
        script: Aggregation script

    Output:
        experiment_data: Aggregated CSV with metadata columns added

    Processing:
        Same as aggregate_experiment_data but without category extraction.
        Category parameter is passed as empty string.
    """
    input:
        test_data = expand(project_paths.reports \
            / '{{experiment}}' \
            / '{{model_name}}{{model_args}}_{{seed}}' \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{test_identifier}' / 'test_data.csv',
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            test_identifier = lambda w: get_test_specs_for_experiment(w.experiment),
        ),
        test_configs = expand(project_paths.reports \
            / '{{experiment}}' \
            / '{{model_name}}{{model_args}}_{{seed}}' \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{test_identifier}' / 'test_outputs.csv.config.yaml',
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            test_identifier = lambda w: get_test_specs_for_experiment(w.experiment),
        ),
        script = SCRIPTS / 'processing' / 'aggregate_experiment_data.py'
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        additional_parameters = ['seed', 'status', 'epoch', 'activityloss'],
        sample_resolution = 'sample',
        fail_on_missing_inputs = False,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 3,
    wildcard_constraints:
        model_args = r'(:[a-z,;:\+=\d\.]+)?',  # No * allowed (excludes \*)
    output:
        experiment_data = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{model_args}_{seed}' \
            / '{data_name}:{data_group}_{status}' \
            / 'test_data.csv',
    shell:
        """
        {params.execution_cmd} \
            --test_data {input.test_data:q} \
            --test_configs {input.test_configs:q} \
            --output {output.experiment_data:q} \
            --parameter {params.parameter} \
            --category "" \
            --additional_parameters {params.additional_parameters} \
            --sample_resolution {params.sample_resolution} \
            --fail_on_missing_inputs {params.fail_on_missing_inputs}
        """


rule process_wandb_data:
    input:
        data = Path('{path}.csv'),
        script = SCRIPTS / 'processing' / 'process_wandb_data.py'
    params:
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        Path('{path}_summary.csv'),
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q}
        """
