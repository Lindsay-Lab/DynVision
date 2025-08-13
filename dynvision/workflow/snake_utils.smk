"""Utility functions and configuration management for the workflow.

This module provides common utility functions and configuration management for
the workflow, including:
- Path management and system configuration
- Argument parsing and string manipulation
- Dataset class management
- Environment detection and setup

The utilities in this module are used by other workflow components to ensure
consistent behavior and reduce code duplication.
"""

import sys
import inspect
import logging
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
from itertools import product
import json
import os
import subprocess
import re
from typing import Dict, List, Optional, Union, Any

# Add the parent directory to the system path
package_dir = Path(inspect.getfile(lambda: None)).parents[2].resolve()
sys.path.insert(0, str(package_dir))

from dynvision.project_paths import project_paths
from dynvision.workflow.mode_manager import ConfigModeManager

pylogger = logging.getLogger('workflow.utils')

# Load configuration files in priority order
configfile: project_paths.scripts.configs / 'config_defaults.yaml'
configfile: project_paths.scripts.configs / 'config_data.yaml'
# configfile: project_paths.scripts.configs / 'config_visualization.yaml'
configfile: project_paths.scripts.configs / 'config_experiments.yaml'
configfile: project_paths.scripts.configs / 'config_modes.yaml'
configfile: project_paths.scripts.configs / 'config_workflow.yaml'

CONFIGS = project_paths.scripts.configs / 'config_runtime.yaml'
SCRIPTS = project_paths.scripts_path

config = SimpleNamespace(**config)
    
wildcard_constraints:
    model_name = r'[a-zA-Z0-9]+',
    data_name = r'[a-z0-9]+',
    data_subset = r'[a-z]+',
    data_group = r'[a-z0-9]+',
    data_loader = r'[a-zA-Z]+',
    status = r'[a-z]+',
    seed = r'\d+',
    condition = r'(withwave_|nowave_|\s?)',
    category = r'(?!folder)[a-z0-9]+',
    model_args = r'(:[a-z,;:\+=\d\.\*]+|\s?)',
    data_args = r'(:[a-zTF,;:\+=\d\.]+|\s?)',
    args = r'([a-z,;:\+=\d\.]+|\s?)',
    args1 = r'([a-z,;:\+=\d\.]+[,;:\+]|\s?)',
    args2 = r'([a-z,;:\+=\d\.]+|\s?)',
    parameter = r'(contrast|duration|interval)',
    experiment = r'[a-z]+',
    layer_name = r'(layer1|layer2|V1|V2|V4|IT)'

localrules: all, symlink_data_subsets, symlink_data_groups, experiment
ruleorder: symlink_data_groups > symlink_data_subsets > train_model_distributed > train_model

def run_mode_manager():
    """Initialize and apply mode manager after CLI config overrides"""
    global mode_manager, config
    
    working_config = config.__dict__.copy() if isinstance(config, SimpleNamespace) else config.copy()

    # Initialize mode manager with final config (including CLI overrides)
    mode_manager = ConfigModeManager(working_config, local=(not project_paths.iam_on_cluster()))
    mode_manager.apply_modes()
    mode_manager.log_modes()
    mode_manager.save_config(path=CONFIGS)
    mode_manager.log_config()
    
    # Get the processed config
    processed_config = mode_manager.get_config(return_namespace=True)   
    if isinstance(processed_config, SimpleNamespace):
        config.__dict__.clear()
        config.__dict__.update(processed_config.__dict__)
    elif isinstance(processed_config, dict):
        config.__dict__.clear()
        config.__dict__.update(processed_config)
    else:
        raise TypeError("Processed config must be a SimpleNamespace or dict")

# Set up logging
def setup_logger():
    # create logger
    logger = logging.getLogger('project')
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    
onstart:
    run_mode_manager()
    setup_logger()

def get_param(key, default=None) -> callable:
    """Get a parameter value from the config.

    Args:
        key: Parameter key to retrieve

    Returns:
        Value of the parameter or None if not found
    """
    return lambda w: getattr(config, key, default)

def get_imagenet_classes(tiny: bool = False) -> tuple[list, dict]:
    """Get ImageNet class information.

    Args:
        tiny: Whether to use TinyImageNet classes

    Returns:
        Tuple containing:
            - List of class names
            - Dictionary mapping indices to class information
    """
    index_file = "tinyimagenet_class_index" if tiny else "imagenet_class_index"
    try:
        with open(project_paths.references / f"{index_file}.json") as f:
            class_dict = json.load(f)
            imagenet_classes = [v[0] for k, v in class_dict.items()]
        return imagenet_classes
    except FileNotFoundError:
        raise ValueError(f"Class index file not found: {index_file}")

def get_category(data_name: str, data_group: str) -> List[str]:
    """Get category information for a dataset.

    Args:
        data_name: Name of the dataset
        data_group: Group within the dataset

    Returns:
        List of category names

    Raises:
        ValueError: If data_name is unknown
    """
    category_handlers = {
        'imagenet': lambda: get_imagenet_classes(tiny=('tiny' in data_name)),
        'imagenette': lambda: ['n01440764',  'n02979186',  'n03028079',  'n03417042',  'n03445777', 'n02102040',  'n03000684',  'n03394916',  'n03425413',  'n03888257'],
        'cifar10': lambda: [str(i) for i in range(10)],
        'cifar100': lambda: [str(i) for i in range(100)],
        'mnist': lambda: [str(i) for i in range(10)],
    }

    if data_name not in category_handlers:
        raise ValueError(f"Unknown dataset: {data_name}")

    if data_group == 'all':
        return category_handlers[data_name]()
    
    try:
        group_indices = config.data_groups[data_name][data_group]
        if 'imagenet' in data_name:
            return [category_handlers['imagenet']()[i] for i in group_indices]
        else:
            return group_indices
    except KeyError:
        raise ValueError(f"Unknown data group '{data_group}' for {data_name}")

def get_data_location(wildcards: Any) -> Path:
    """Get the data directory for given wildcards.

    Args:
        wildcards: Snakemake wildcards object

    Returns:
        Path to data directory
    """
    data_name = wildcards.data_name
    data_subset = 'val' if data_name == 'imagenet' and wildcards.data_subset == 'test' else wildcards.data_subset
    category = wildcards.category

    base_dir = get_data_base_dir(wildcards)
    return base_dir / data_subset / category

def get_data_base_dir(wildcards: Any) -> Path:
    """Get the base directory for a dataset.

    Args:
        wildcards: Snakemake wildcards object

    Returns:
        Path to base directory
    """
    data_name = wildcards.data_name
    if data_name in config.mounted_datasets and project_paths.iam_on_cluster():
        return Path(f'/{data_name}')
    return project_paths.data.raw / data_name

def parse_arguments(
    wildcards: Any,
    args_key: str = 'model_args',
    delimiter: str = '+',
    assigner: str = '=',
    prefix: str = ":"
) -> str:
    """Parse argument string into command line arguments.

    Args:
        wildcards: Snakemake wildcards object
        args_key: Key for arguments in wildcards
        delimiter: Character separating arguments
        assigner: Character separating key and value
        prefix: Prefix character for argument string

    Returns:
        Formatted command line arguments string
    """
    args = getattr(wildcards, args_key, '')
    args = args.lstrip(prefix).split(delimiter)

    if len(args) == 1 and not args[0]:
        return ""

    args_dict = {
        arg.split(assigner)[0]: arg.split(assigner)[1]
        for arg in args
    }

    return " ".join(f"--{key} {value}" for key, value in args_dict.items())

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

def replace_param_in_string(
    s: str,
    key: str = "contrast",
    value_type: Optional[type] = None,
    new_value: str = "*"
) -> str:
    """Replace parameter value in string with new value.

    Args:
        s: Input string
        key: Parameter key to replace
        value_type: Type of value to match
        new_value: New value to insert

    Returns:
        Modified string

    Raises:
        ValueError: If parameter not found or invalid value type
    """
    patterns = {
        int: rf"{key}=(\d+)",
        float: rf"{key}=(\d+(\.\d+)?)",
        str: rf"{key}=([a-z]+)",
        None: rf"{key}=([\da-z\.]+)"
    }

    if value_type not in patterns:
        raise ValueError(f"Invalid value type: {value_type}")

    pattern = patterns[value_type]
    match = re.search(pattern, s)
    
    if not match:
        raise ValueError(f"No {key} value found in string: {s}")
        
    return s.replace(match.group(0), f"{key}={new_value}")

def dict_poped(d: Dict, keys: Union[str, List[str]]) -> Dict:
    """Create new dictionary with specified keys removed.

    Args:
        d: Input dictionary
        keys: Key or list of keys to remove

    Returns:
        New dictionary without specified keys
    """
    dc = d.copy()
    if isinstance(keys, list):
        for key in keys:
            dc.pop(key, None)
    else:
        dc.pop(keys, None)
    return dc

def get_conda_env() -> tuple[Optional[str], str]:
    """Get information about the current conda environment.

    Returns:
        Tuple containing:
            - Environment name (or None if not in conda)
            - Environment status string
    """
    try:
        env_name = os.environ['CONDA_DEFAULT_ENV']
        env_status = subprocess.check_output(
            ['conda', 'info', '--envs']
        ).decode('utf-8')
        return env_name, env_status
    except (KeyError, subprocess.CalledProcessError):
        return None, "No conda environment active"

# Initialize environment information
env_name, env_status = get_conda_env()
pylogger.info(f"Conda environment: {env_name or 'None'}")

rule checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries."""
    input:
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    params:
        checkpoint_dir = lambda w: project_paths.models / f"{w.model_name}" / 'checkpoints',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        temp(project_paths.models / '{model_name}' / '{model_identifier}.ckpt2pt'),
    shell:
        """
        {params.execution_cmd} \
            --checkpoint_dir {params.checkpoint_dir:q} \
            --output {output:q}
        """


def build_execution_command(script_path, use_distributed=False, use_executor=False):
    """
    Build the execution command with conditional wrappers.
    
    Args:
        script_path: Path to the Python script to execute
        use_distributed: Whether to use distributed setup
        use_executor: Whether to use executor wrapper
    
    Returns:
        String containing the complete execution command
    """
    cmd_parts = []
    
    if use_distributed:
        setup_script = SCRIPTS / 'cluster' / 'setup_distributed_execution.sh'
        cmd_parts.append(f"source {setup_script} &&")

    # Add executor wrapper if enabled
    if use_executor:
        executor_script = SCRIPTS / 'cluster' / 'executor_wrapper.sh'
        cmd_parts.append(str(executor_script))
    
    if use_distributed:
        cmd_parts.append(
            "torchrun "
            "--nproc_per_node=${GPU_PER_NODE:-2} "
            "--nnodes=${NUM_NODES:-1} "
            "--node_rank=${NODE_RANK:-0} "
            "--master_addr=${MASTER_ADDR} "
            "--master_port=${MASTER_PORT} "
            f"{script_path}"
        )
    else:
        cmd_parts.append(f"python {script_path}")
    
    return "\\\n        ".join(cmd_parts)