"""Data preparation and processing workflow.

This workflow handles all data-related operations including:
- Dataset creation and preprocessing
- Data organization and linking
- FFCV dataset building
- Data validation and verification

The workflow supports multiple dataset types:
- Standard datasets (MNIST, CIFAR10, CIFAR100, TinyImageNet)
- External datasets (ImageNet)
"""

logger = logging.getLogger('workflow.data')


rule get_data:
    """Download and prepare standard datasets.
    
    This rule handles the download and initial preparation of
    standard datasets like MNIST, CIFAR10, and CIFAR100.

    Input:
        script: Python script for dataset preparation
    
    Output:
        Directory containing prepared dataset

    Parameters:
        data_name: Name of the dataset
        data_subset: Dataset split (train/test)
        ext: Image file extension
    """
    input:
        script = SCRIPTS / 'data' / 'get_data.py'
    params:
        raw_data_path = lambda w: project_paths.data.raw / f'{w.data_name}',
        output = lambda w: project_paths.data.raw \
            / f'{w.data_name}' \
            / f'{w.data_subset}' \
            / 'folder.link',
        ext = 'png',
        data_name = lambda w: ''.join([c.upper() if c.isalpha() else c for c in w.data_name]),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        flag = directory(project_paths.data.raw \
            / '{data_name}' \
            / '{data_subset}' )
    # wildcard_constraints:
    #     data_name = r"cifar10|cifar100|mnist"
    benchmark:
        project_paths.benchmarks / 'get_data_{data_name}_{data_subset}.txt'
    shell:
        """
        {params.execution_cmd} \
            --output {params.output:q} \
            --data_name {params.data_name} \
            --raw_data_path {params.raw_data_path:q} \
            --subset {wildcards.data_subset} \
            --ext {params.ext}
        """

rule symlink_data_subsets:
    """Create symlinks for dataset subsets.

    This rule organizes dataset subsets by creating appropriate
    symbolic links in the interim data directory.

    Input:
        Source data location from get_data_location function
    
    Output:
        Symlink flag file
    """
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
    group: "dataprep"
    shell:
        """
        (mkdir -p {params.parent:q} && \
        ln -sf {input:q} {params.source:q} && \
        touch {output.flag:q}) 
        """

rule symlink_data_groups:
    """Create symlinks for data groups.

    This rule organizes data into groups by looking up group definitions in 
    config_data.yaml and creating appropriate symbolic links.

    Input:
        Category symlinks from symlink_data_subsets
    
    Output:
        Group symlink flag file
    """
    input:
        lambda wildcards: expand(project_paths.data.interim \
            / '{{data_name}}' \
            / '{{data_subset}}_{{data_group}}' \
            / '{category}.link',
            category=get_category(wildcards.data_name, wildcards.data_group)
            )
    output:
        flag = project_paths.data.interim \
            / '{data_name}' \
            / '{data_subset}_{data_group}' \
            / 'folder.link'
    group: "dataprep"
    shell:
        """
        touch {output.flag:q} 
        """

rule build_ffcv_datasets:
    """Build FFCV datasets for faster data loading.

    This rule converts prepared datasets into FFCV format
    for improved training performance.

    Input:
        script: Python script for FFCV conversion
        data: Prepared dataset directory
    
    Output:
        train: Training dataset in FFCV format
        val: Validation dataset in FFCV format

    Parameters:
        train_ratio: Split ratio for training data
        max_resolution: Maximum image resolution
    """
    input:
        script = SCRIPTS / 'data' / 'ffcv_datasets.py',
        data = project_paths.data.interim / '{data_name}' / 'train_all' / 'folder.link'
    params:
        train_ratio = get_param('train_ratio'),
        max_resolution = lambda w: config.data_resolution[w.data_name],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),    
    output:
        train = project_paths.data.processed / '{data_name}' / 'train_all' / 'train.beton',
        val = project_paths.data.processed / '{data_name}' / 'train_all' / 'val.beton'
    benchmark:
        project_paths.benchmarks / 'build_ffcv_datasets_{data_name}.txt'
    shell:
        """
        {params.execution_cmd} \
            --input {input.data:q} \
            --output_train {output.train:q} \
            --output_val {output.val:q} \
            --train_ratio {params.train_ratio} \
            --data_name {wildcards.data_name} \
            --max_resolution {params.max_resolution} 
        """

# Log workflow initialization
logger.info("Data workflow initialized")
logger.info(f"Data directory: {project_paths.data}")