"""Type utility functions for the DynVision toolbox.

This module provides type-related utilities:
- Type definitions and constants
- Type conversion functions
- Type inference functions
- Type mapping dictionaries
"""

from typing import Optional, Union

# Type definitions
types_int = [
    "id",
    "index",
    "sample_index",
    "times_index",
    "class_index",
    "label_index",
    "guess_index",
]

types_float = ["response"]

types_bool = []

types_str = ["label_set"]

# Type mapping dictionary
dtypes = (
    dict.fromkeys(types_int, int)
    | dict.fromkeys(types_float, float)
    | dict.fromkeys(types_bool, bool)
    | dict.fromkeys(types_str, str)
)

# Type conversion sets
_true_set = {"yes", "true", "t", "y", "1"}
_false_set = {"no", "false", "f", "n", "0"}


def str_to_bool(value: Union[str, bool], raise_exc: bool = False) -> Optional[bool]:
    """Convert string to boolean value.

    Handles common boolean string representations:
    - True: 'yes', 'true', 't', 'y', '1'
    - False: 'no', 'false', 'f', 'n', '0'

    Args:
        value: String or boolean value to convert
        raise_exc: Whether to raise exception for invalid values

    Returns:
        Boolean value or None if invalid and not raising

    Raises:
        ValueError: If value is invalid and raise_exc is True
    """
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False
    elif isinstance(value, bool):
        return value
    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def guess_type(string: str) -> Union[int, float, str, bool, None]:
    """Infer type from string value.

    Attempts to convert string to:
    - Integer
    - Float
    - Boolean
    - None
    - String (fallback)

    Args:
        string: String to convert

    Returns:
        Converted value

    Example:
        guess_type("123") -> 123
        guess_type("3.14") -> 3.14
        guess_type("True") -> True
    """
    try:
        out = int(string)
    except Exception:
        try:
            out = float(string)
        except Exception:
            out = str(string)
            if out.lower() == "none":
                out = None
            elif out.lower() == "true":
                out = True
            elif out.lower() == "false":
                out = False
    return out
