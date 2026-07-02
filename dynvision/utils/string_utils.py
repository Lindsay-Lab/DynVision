"""String utility functions for the DynVision toolbox.

This module provides string manipulation utilities:
- Path parsing
- Parameter extraction
- String-to-dictionary conversion
- Pattern matching and replacement
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from .type_utils import guess_type


def path_to_index(path: Union[str, Path], non_index: int = -1) -> int:
    """Extract index from path filename.

    Args:
        path: Path or string containing index in filename
        non_index: Value to return if index not found

    Returns:
        Extracted index or non_index if invalid

    Example:
        path_to_index('file_123.txt') -> 123
    """
    if not isinstance(path, str) or isinstance(path, Path):
        warnings.warn(f"Invalid path type: {path}")
        return non_index
    name = Path(path).name
    index = int(name.split("_")[-1])
    return index


def extract_param_from_string(
    s: str,
    key: str,
    value_type: Optional[type] = None,
    assigner: str = "=",
) -> Union[int, float, str, bool, None]:
    """Extract parameter value from string.

    Args:
        s: String containing parameter
        key: Parameter key to extract
        value_type: Expected value type

    Returns:
        Extracted parameter value

    Raises:
        ValueError: If parameter not found or invalid type

    Example:
        extract_param_from_string("model_contrast=0.5", "contrast", float)
        -> 0.5
    """
    if value_type == int:
        match = re.search(rf"{key}{assigner}(\d+)", s)
    elif value_type == float:
        match = re.search(rf"{key}{assigner}(\d+(\.\d+)?)", s)
    elif value_type == str:
        match = re.search(rf"{key}{assigner}([a-z]+)", s)
    elif value_type is None:
        match = re.search(rf"{key}{assigner}([\da-z\.]+)", s)
        value_type = guess_type
    else:
        raise ValueError(f"Invalid value type: {value_type}")
    if match:
        return value_type(str(match.group(1)))
    else:
        raise ValueError(f"No {key} value found in the string!")


def replace_param_in_string(
    s: str,
    key: str,
    value_type: Optional[type] = None,
    new_value: str = "*",
) -> str:
    """Replace parameter value in string.

    Args:
        s: String containing parameter
        key: Parameter key to replace
        value_type: Expected value type
        new_value: New parameter value

    Returns:
        String with replaced parameter

    Raises:
        ValueError: If parameter not found or invalid type

    Example:
        replace_param_in_string("model_contrast=0.5", "contrast", float, "0.8")
        -> "model_contrast=0.8"
    """
    if value_type == int:
        match = re.search(rf"{key}=(\d+)", s)
    elif value_type == float:
        match = re.search(rf"{key}=(\d+(\.\d+)?)", s)
    elif value_type == str:
        match = re.search(rf"{key}=([a-z]+)", s)
    elif value_type is None:
        match = re.search(rf"{key}=([\da-z\.]+)", s)
    else:
        raise ValueError(f"Invalid value type: {value_type}")
    if match:
        return s.replace(match.group(0), f"{key}={new_value}")
    else:
        raise ValueError(f"No {key} value found in the string!")


def str2dict(string: str, assigner: str = ":", separator: str = ",") -> Dict[str, Any]:
    """Convert string to dictionary.

    Handles:
    - Basic key-value pairs
    - List values
    - Tuple values
    - Nested structures

    Args:
        string: String to convert
        assigner: Key-value separator
        separator: Item separator

    Returns:
        Converted dictionary

    Example:
        str2dict("a:1,b:[2,3]") -> {'a': 1, 'b': [2, 3]}
    """
    if string[0] == "{":
        string = string[1:]
    if string[-1] == "}":
        string = string[:-1]

    if not len(string):
        return {}

    my_dict = {}
    # list or tuple values
    brackets = [delimiter for delimiter in ["[", "]", "(", ")"] if delimiter in string]
    if len(brackets):
        for kv in string.split(f"{brackets[1]}{separator}"):
            k, v = kv.split(assigner)
            v = v.replace(brackets[0], "").replace(brackets[1], "")
            values = [guess_type(val) for val in v.split(separator)]
            if len(values) == 1:
                values = values[0]
            my_dict[k.strip()] = values
    # scalar values
    else:
        for kv in string.split(separator):
            k, v = kv.split(assigner)
            my_dict[k.strip()] = guess_type(v.strip())
    return my_dict


def parse_string2dict(
    kwargs_str: Union[str, List[str]], **kwargs: Any
) -> Dict[str, Any]:
    """Parse string representation of dictionary.

    Handles:
    - Single string
    - List of strings
    - Nested dictionaries
    - Lists and tuples

    Args:
        kwargs_str: String or list of strings to parse
        **kwargs: Additional keyword arguments

    Returns:
        Parsed dictionary

    Example:
        parse_string2dict("a:1,b:[2,3]")
        -> {'a': 1, 'b': [2, 3]}
    """
    if isinstance(kwargs_str, list):
        if len(kwargs_str) == 0:
            return {}
        elif len(kwargs_str) == 1:
            kwargs = kwargs_str[0]
        else:
            kwargs = "".join(kwargs_str)[1:-1]
    else:
        kwargs = str(kwargs_str)

    if guess_type(kwargs) is None:
        return {}

    kwargs = kwargs.strip("{}")
    kwargs = kwargs.replace("'", "")

    my_dict = {}

    # match all nested dicts
    pattern = re.compile(r"[\w\s]+:{[^}]*},*")
    for match in pattern.findall(kwargs):
        nested_dict_name, nested_dict = match.split(":{")
        nested_dict = nested_dict[:-1]
        my_dict[nested_dict_name] = str2dict(nested_dict)
        kwargs = kwargs.replace(match, "")

    # match entries with word value, list value, or tuple value
    pattern = re.compile(r"[\w\s]+:(?:[\w\.\s\/\-\&\+]+|\[[^\]]+\]|\([^\)]+\))")
    for match in pattern.findall(kwargs):
        my_dict.update(str2dict(match))

    return my_dict
