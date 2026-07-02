"""Configuration utility functions for the DynVision toolbox.

This module provides configuration-related utilities:
- YAML config loading
- Command line argument parsing
- Keyword argument handling
- Function parameter aliasing
"""

import argparse
import inspect
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file does not exist
        yaml.YAMLError: If file is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        raise


def parse_kwargs(kwargs: List[str]) -> Dict[str, Any]:
    """Parse command line arguments into parameter dictionary.

    Converts arguments of the form:
    --param1 value1 --param2 value2
    into:
    {'param1': 'value1', 'param2': 'value2'}

    Args:
        kwargs: List of command line arguments

    Returns:
        Dictionary of parameter names and values

    Example:
        parse_kwargs(['--model', 'resnet', '--layers', '18', '34'])
        -> {'model': 'resnet', 'layers': ['18', '34']}
    """
    params = {}
    key = None
    for element in kwargs:
        if element.startswith("-"):
            key = element.lstrip("-")
            params[key] = []
        elif key is not None:
            params[key] += [element]
        else:
            pass
    for k, v in params.items():
        if len(v) == 1:
            params[k] = v[0]
    return params


def filter_kwargs(
    function: Union[Callable, type], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by function.

    Handles:
    - Class constructors
    - Regular functions
    - Decorated functions with aliases

    Args:
        function: Function or class to filter kwargs for
        kwargs: Dictionary of keyword arguments

    Returns:
        Filtered kwargs dictionary

    Example:
        def func(a, b=2): pass
        filter_kwargs(func, {'a': 1, 'b': 2, 'c': 3})
        -> {'a': 1, 'b': 2}
    """
    if not len(kwargs):
        logger.warning("No kwargs provides to filter_kwargs function.")
        return kwargs

    if inspect.isclass(function):
        known, unknown = {}, {}
        for base in reversed(inspect.getmro(function)):
            if base == object:
                continue

            base_known, base_unknown = filter_kwargs(base.__init__, kwargs)
            known.update(base_known)
            unknown.update(base_unknown)

        # Remove entries in `unknown` that are already in `known`
        unknown = {k: v for k, v in unknown.items() if k not in known}

        return known, unknown

    else:
        function_args = set(inspect.signature(function).parameters.keys())
        if hasattr(function, "__wrapped__"):
            wrapped_func = function.__wrapped__
            if (
                hasattr(wrapped_func, "__annotations__")
                and "aliases" in wrapped_func.__annotations__
            ):
                aliases = wrapped_func.__annotations__.get("aliases", {})
                # overwrite parameter values with alias values when given
                for alias, original in aliases.items():
                    if alias in kwargs:
                        function_args.add(alias)
                        function_args.discard(original)

        known = {k: v for k, v in kwargs.items() if k in function_args}
        unknown = {k: v for k, v in kwargs.items() if k not in function_args}

        return known, unknown


def update_config_with_kwargs(
    config: Dict[str, Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Update configuration dictionary with command line arguments.

    Args:
        config: Base configuration dictionary
        kwargs: Keyword arguments to update with

    Returns:
        Updated configuration dictionary

    Example:
        update_config_with_kwargs({'a': 1}, {'b': 2, 'c': None})
        -> {'a': 1, 'b': 2}
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    config.update(kwargs)
    return config


def parse_parameters(
    parser: argparse.ArgumentParser,
    config_path_arg: str = "config_path",
    return_namespace: bool = True,
) -> Union[Dict[str, Any], argparse.Namespace]:
    """Parse command line arguments and configuration file.

    Combines arguments from:
    - Command line arguments
    - YAML configuration file
    - Unknown arguments

    Args:
        parser: ArgumentParser instance
        config_path_arg: Name of config path argument
        return_namespace: Whether to return Namespace object

    Returns:
        Parsed parameters as dictionary or Namespace

    Example:
        parser = ArgumentParser()
        parser.add_argument('--model')
        config = parse_parameters(parser)
    """
    args, unknown = parser.parse_known_args()
    kwargs = parse_kwargs(unknown)

    if hasattr(args, config_path_arg) and getattr(args, config_path_arg) is not None:
        with open(args.config_path, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    config = update_config_with_kwargs(config, vars(args) | kwargs)
    if return_namespace:
        config = argparse.Namespace(**config)

    return config


def alias_kwargs(**aliases: Dict[str, str]) -> Callable:
    """Decorator to support parameter aliases.

    Allows using alternative parameter names that map to original names.
    The alias takes precedence over the original name.

    Args:
        **aliases: Mapping from alias to original parameter names

    Returns:
        Decorated function with alias support

    Example:
        @alias_kwargs(n='num_layers')
        def func(num_layers=2):
            pass

        func(n=3)  # num_layers will be 3
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for alias, original in aliases.items():
                # alias takes precedence over original
                if alias in kwargs:
                    kwargs[original] = kwargs.pop(alias)
            return func(*args, **kwargs)

        if not hasattr(wrapper, "__annotations__"):
            wrapper.__annotations__ = {}
        wrapper.__annotations__["aliases"] = aliases

        return wrapper

    return decorator
