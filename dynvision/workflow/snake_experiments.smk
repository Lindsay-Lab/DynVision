rule experiment:
    input:
        lambda w: expand(project_paths.reports \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv',
            model_name = config.model_name,
            seed = config.seed,
            model_args = args_product(config.model_args),
            data_name = config.data_name,
            data_group = config.data_group,
            status = config.experiment_config[w.experiment]["status"],
            data_loader = config.experiment_config[w.experiment]["data_loader"],
            data_args = args_product(config.experiment_config[w.experiment]["data_args"]))
    output:
        temp(project_paths.reports / 'experiment_{experiment}.done')
    shell:
        """
        touch {output:q}
        """