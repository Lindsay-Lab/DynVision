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

from .logging_utils import (
    configure_logging,
    log_section,
    log_section_table,
    format_value,
    SummaryItem,
    build_section,
    resolve_signature_defaults,
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
    _torch_load_with_retries,
)

from .dtype_policy import (
    PRECISION_TO_DTYPE,
    DEFAULT_DTYPE,
    normalize_precision,
    resolve_dtype,
    DtypePolicy,
    coordinate_component_dtypes,
)

from .model_utils import (
    check_stability,
    check_weights,
    handle_errors,
    load_model_and_weights,
)

from .visualization_utils import (
    aggregate_plot_data,
    calculate_accuracy,
    calculate_label_indicator,
    find_classifier_meta_columns,
    find_classifier_value_columns,
    find_layer_response_columns,
    layer_response_avg,
    layer_response_std,
    load_responses,
    load_responses_in_batches,
    parse_error_type,
    peak_height,
    peak_ratio,
    peak_time,
    time_averaged_response,
    resolve_measure_columns,
    save_plot,
    standardize_category_value,
    standardize_series,
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
    # Logging utilities
    "configure_logging",
    "log_section",
    "log_section_table",
    "format_value",
    "SummaryItem",
    "build_section",
    "resolve_signature_defaults",
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
    "_torch_load_with_retries",
    # Dtype policy
    "PRECISION_TO_DTYPE",
    "DEFAULT_DTYPE",
    "normalize_precision",
    "resolve_dtype",
    "DtypePolicy",
    "coordinate_component_dtypes",
    # Model utilities
    "check_stability",
    "check_weights",
    "handle_errors",
    "load_model_and_weights",
    # Visualization utilities
    "standardize_category_value",
    "standardize_series",
    "parse_error_type",
    "aggregate_plot_data",
    "find_layer_response_columns",
    "find_classifier_value_columns",
    "find_classifier_meta_columns",
    "resolve_measure_columns",
    "layer_response_avg",
    "layer_response_std",
    "peak_time",
    "peak_height",
    "peak_ratio",
    "time_averaged_response",
    "calculate_accuracy",
    "calculate_label_indicator",
    "load_responses",
    "load_responses_in_batches",
    "save_plot",
]
