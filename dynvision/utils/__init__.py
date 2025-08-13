"""DynVision Utilities Package

This package provides utility functions organized into focused modules:

type_utils: Type conversion and validation
string_utils: String and path parsing operations
config_utils: Configuration and argument parsing
data_utils: Data loading and manipulation
torch_utils: PyTorch device/dtype management
model_utils: Model weight checking and stability
visualization_utils: Visualization and plotting utilities
"""

from .type_utils import (
    # Type conversion and constants
    str_to_bool,
    guess_type,
    dtypes,
    types_int,
    types_float,
    types_bool,
    types_str,
)

from .string_utils import (
    # Path parsing
    path_to_index,
    # Parameter extraction
    extract_param_from_string,
    replace_param_in_string,
    # Dictionary string parsing
    str2dict,
    parse_string2dict,
)

from .config_utils import (
    load_config,
    parse_parameters,
    parse_kwargs,
    filter_kwargs,
    update_config_with_kwargs,
    alias_kwargs,
)

from .data_utils import (
    load_df,
    identity,
    tqdm,
)

from .torch_utils import (
    ensure_same_device,
    ensure_same_dtype,
    on_same_device,
    set_seed,
    apply_parametrization,
    get_effective_dtype_from_precision,
    determine_target_dtype,
    calculate_conv_out_dim,
)

from .model_utils import (
    check_stability,
    check_weights,
    handle_errors,
    load_model_and_weights,
)

from .visualization_utils import (
    layer_power,
    peak_time,
    peak_height,
    peak_ratio,
    calculate_accuracy,
    load_responses,
    load_responses_in_batches,
    save_plot,
)

# For backward compatibility, expose all utilities at package level
__all__ = [
    # Type utilities
    "str_to_bool",
    "guess_type",
    "dtypes",
    "types_int",
    "types_float",
    "types_bool",
    "types_str",
    # String utilities (including path parsing)
    "path_to_index",
    "extract_param_from_string",
    "replace_param_in_string",
    "str2dict",
    "parse_string2dict",
    # Config utilities
    "load_config",
    "parse_parameters",
    "parse_kwargs",
    "filter_kwargs",
    "update_config_with_kwargs",
    "alias_kwargs",
    # Data utilities
    "load_df",
    "identity",
    "tqdm",
    # Torch utilities
    "ensure_same_device",
    "ensure_same_dtype",
    "on_same_device",
    "set_seed",
    "apply_parametrization",
    "get_effective_dtype_from_precision",
    "determine_target_dtype",
    "calculate_conv_out_dim",
    # Model utilities
    "check_stability",
    "check_weights",
    "handle_errors",
    "load_model_and_weights",
    # Visualization utilities
    "layer_power",
    "peak_time",
    "peak_height",
    "peak_ratio",
    "calculate_accuracy",
    "load_responses",
    "load_responses_in_batches",
    "save_plot",
]
