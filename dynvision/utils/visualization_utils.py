"""Visualization utility functions for the DynVision toolbox.

This module provides visualization-related utilities:
- Layer response analysis
- Peak detection and analysis
- Accuracy calculation
- Response data loading
- Plot saving
- Common plotting functions
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data_utils import load_df
from .string_utils import extract_param_from_string


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array with proper dtype handling.

    Args:
        tensor: PyTorch tensor in any precision format

    Returns:
        NumPy array in float32 format
    """
    return tensor.cpu().float().numpy()


logger = logging.getLogger(__name__)

# Parameters that need dt conversion from timesteps to milliseconds
DT_CONVERT_PARAMS = ["tau", "trc", "tsk", "lossrt", "duration", "interval"]


def format_parameter_value(value, parameter, dt=None):
    """Format parameter values with appropriate units.

    Args:
        value: Parameter value to format
        parameter: Parameter name
        dt: Optional temporal resolution for converting timesteps to ms

    Returns:
        Formatted string with value and unit
    """
    if dt is not None and parameter.lower() in DT_CONVERT_PARAMS:
        return f"{int(float(value) * dt)} ms"
    elif str(value).lower() in ["true", "false"]:
        return str(value).capitalize()
    elif parameter.lower() in ["contrast", "noiselevel"]:
        return f"{float(value):.2f}"
    else:
        return str(value).capitalize()


def standardize_category_value(value: Any) -> str:
    """Normalize categorical labels (booleans, whitespace, casing)."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    value_str = str(value).strip()
    lowered = value_str.lower()
    if lowered in {"true", "false"}:
        return lowered
    return value_str


def standardize_series(series: pd.Series) -> pd.Series:
    """Vectorized wrapper around :func:`standardize_category_value`."""

    return series.astype(str).map(standardize_category_value)


def parse_error_type(error_type: Optional[str]) -> str:
    """Return a normalized error type identifier."""

    if not error_type:
        return "std"

    normalized = error_type.strip().lower()
    if normalized in {"none", "std", "sem", "ci95"}:
        return normalized
    raise ValueError(
        f"Unsupported errorbar type '{error_type}'. Choose from none, std, sem, ci95"
    )


def _error_function_factory(kind: str) -> Callable[[pd.Series], float]:
    if kind == "std":
        return lambda s: float(s.std(ddof=0))
    if kind == "sem":
        return lambda s: float(s.sem(ddof=0))
    if kind == "ci95":
        return lambda s: float(s.sem(ddof=0) * 1.96)
    raise ValueError(f"Unknown error function kind '{kind}'")


def aggregate_plot_data(
    df: pd.DataFrame,
    group_keys: Sequence[str],
    mean_columns: Sequence[str],
    error_type: str = "std",
    min_count: int = 1,
    extra_aggs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Aggregate large plotting tables to the desired granularity.

    Args:
        df: Source dataframe (already filtered to required columns).
        group_keys: Columns to group by (order preserved for sorting).
        mean_columns: Numeric columns that should be averaged (and get errors).
        error_type: One of ``none``, ``std``, ``sem``, ``ci95``.
        min_count: Drop groups with fewer than this many rows.
        extra_aggs: Optional mapping of ``column -> aggregation``
            (e.g., ``{"label_valid": "max"}``).

    Returns:
        Aggregated dataframe containing the group keys, averaged columns,
        and optional error/extra aggregation columns.
    """

    if not group_keys:
        raise ValueError("At least one group key is required for aggregation")

    observed = True
    grouped = df.groupby(list(group_keys), observed=observed, sort=True)

    frames: List[pd.DataFrame] = []

    if mean_columns:
        mean_df = grouped[list(mean_columns)].mean()
        frames.append(mean_df)

        normalized_error = parse_error_type(error_type)
        if normalized_error != "none":
            error_fn = _error_function_factory(normalized_error)
            err_frames = []
            for column in mean_columns:
                err_series = grouped[column].agg(error_fn)
                err_frames.append(err_series.rename(f"{column}_err"))
            if err_frames:
                frames.append(pd.concat(err_frames, axis=1))

    if extra_aggs:
        extra_df = grouped[list(extra_aggs.keys())].agg(extra_aggs)
        frames.append(extra_df)

    if not frames:
        result = grouped.size().to_frame("__count")
    else:
        result = pd.concat(frames, axis=1)

    counts = grouped.size().rename("__count")
    result = result.join(counts)
    if min_count > 1:
        result = result[result["__count"] >= min_count]

    result = result.reset_index().sort_values(list(group_keys)).reset_index(drop=True)
    return result.drop(columns=["__count"], errors="ignore")


def find_layer_response_columns(
    columns: Sequence[str], suffix: str = "_response_avg"
) -> List[str]:
    """Return all layer response columns matching the expected suffix."""

    return [col for col in columns if isinstance(col, str) and col.endswith(suffix)]


def find_classifier_value_columns(
    columns: Sequence[str], prefix: str = "classifier_top"
) -> List[str]:
    """Return classifier activation columns (excluding metadata columns)."""

    return [
        col
        for col in columns
        if isinstance(col, str) and col.startswith(prefix) and not col.endswith("_id")
    ]


def find_classifier_meta_columns(
    columns: Sequence[str], prefix: str = "classifier_top", suffix: str = "_id"
) -> List[str]:
    """Return classifier metadata columns (e.g., unit ids)."""

    return [
        col
        for col in columns
        if isinstance(col, str) and col.startswith(prefix) and col.endswith(suffix)
    ]


def resolve_measure_columns(
    columns: Sequence[str],
    requested: Sequence[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Resolve requested metric names to actual dataframe columns.

    Returns a tuple ``(resolved, missing)`` where ``resolved`` is a list of
    ``(requested_name, actual_column)`` pairs and ``missing`` contains names that
    could not be matched.
    """

    if not requested:
        return [], []

    available = set(columns)
    resolved: List[Tuple[str, str]] = []
    missing: List[str] = []

    for base in requested:
        candidates = [base]
        if not base.endswith("_avg"):
            candidates.append(f"{base}_avg")
        if not base.endswith("_mean"):
            candidates.append(f"{base}_mean")

        match = next((cand for cand in candidates if cand in available), None)
        if match:
            resolved.append((base, match))
        else:
            missing.append(base)

    return resolved, missing


def layer_response_avg(response: torch.Tensor) -> torch.Tensor:
    """Calculate mean response of layer activations.

    Args:
        response: Layer response tensor

    Returns:
        Mean absolute response tensor of shape [batch_size, n_timesteps]
    """
    return response.abs().mean(dim=list(range(2, response.dim())))


def layer_response_std(response: torch.Tensor) -> torch.Tensor:
    """Calculate standard deviation of layer activations.

    Args:
        response: Layer response tensor

    Returns:
        Standard deviation of absolute response tensor of shape [batch_size, n_timesteps]
    """
    return response.abs().std(dim=list(range(2, response.dim())))


def time_averaged_response(mean_response: torch.Tensor) -> torch.Tensor:
    """Calculate time-averaged response for each sample.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of response sums for each sample
    """
    return mean_response.mean(dim=1)


def normalized_temporal_summation(
    response_dict: dict,
    parameter_name: str,
    subtract_baseline: bool = False,
) -> dict:
    """Calculate normalized summed response for temporal summation analysis.

    Matches methodology from Groen et al. (2022) for human ECoG data:
    1. Sum broadband power over time window
    2. (Optional) Subtract minimum summed value as baseline
    3. Normalize to maximum summed value

    This reveals compressive temporal integration (subadditive summation).

    Args:
        response_dict: Dictionary with structure:
            {
                parameter_value: {
                    'response_tensor': torch.Tensor (samples × time),
                    'entity_values': array of sample indices
                }
            }
        parameter_name: Name of the parameter (for logging)
        subtract_baseline: If True, subtract minimum response value before
            normalization. Set to False to match Groen et al. exactly
            (they don't have duration=0 condition). Default True.

    Returns:
        Dictionary mapping (parameter_value, entity) -> normalized metric value
    """
    metrics_map = {}

    # Get all parameter values
    param_values = sorted(response_dict.keys())

    # Process each parameter value: sum over time
    summed_responses = {}  # {entity: {param_val: summed_value}}

    for param_val in param_values:
        param_data = response_dict[param_val]
        param_traces = param_data["response_tensor"]  # (n_samples, n_timepoints)
        param_entities = param_data["entity_values"]

        for i, entity in enumerate(param_entities):
            entity = int(entity)

            if entity not in summed_responses:
                summed_responses[entity] = {}

            # Sum over time
            param_trace = param_traces[i, :]
            summed_value = param_trace.sum().item()
            summed_responses[entity][param_val] = summed_value

    # Normalize each entity independently
    for entity, param_sums in summed_responses.items():
        # Get all summed values for this entity
        sum_values = list(param_sums.values())

        if subtract_baseline:
            # Find minimum summed response (baseline)
            baseline = min(sum_values)

            # Subtract baseline from all values
            baseline_corrected = {
                param_val: summed_val - baseline
                for param_val, summed_val in param_sums.items()
            }

            # Find maximum after baseline correction
            max_sum = max(baseline_corrected.values())

            # Normalize: (sum - baseline) / (max - baseline)
            for param_val, corrected_val in baseline_corrected.items():
                normalized_value = corrected_val / max_sum
                metrics_map[(param_val, entity)] = normalized_value
        else:
            # Groen et al. approach: normalize to max without baseline subtraction
            max_sum = max(sum_values)

            # Normalize: sum / max_sum
            for param_val, summed_val in param_sums.items():
                normalized_value = summed_val / max_sum
                metrics_map[(param_val, entity)] = normalized_value

    return metrics_map


def peak_time(mean_response: torch.Tensor) -> torch.Tensor:
    """Calculate peak time for each sample.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak times for each sample
    """
    return mean_response.argmax(dim=1)


def peak_height(mean_response: torch.Tensor) -> torch.Tensor:
    """Calculate peak height for each sample.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak heights for each sample
    """
    max_values, max_indices = mean_response.max(dim=1)
    return max_values


def peak_ratio(mean_response: torch.Tensor, min_delay: int = 5) -> torch.Tensor:
    """Calculate ratio between the two strongest positive peaks.

    Args:
        mean_response: Tensor of shape ``[batch, timesteps]`` containing temporal
            responses (already averaged across spatial/channel dimensions).
        min_delay: Minimum separation in index space between peaks. If no pair of
            peaks satisfies this separation, the next-best peak is used so long as
            it remains positive.

    Returns:
        Tensor of peak ratios (later_peak / earlier_peak) for each sample. If
        fewer than two positive peaks are found, the corresponding entry is zero.
    """

    if mean_response.ndim != 2:
        raise ValueError(
            f"mean_response must be 2D (batch, timesteps); got shape {mean_response.shape}"
        )

    min_delay = max(int(min_delay), 1)
    batch_size, _ = mean_response.shape
    ratios = torch.zeros(
        batch_size, dtype=mean_response.dtype, device=mean_response.device
    )

    for batch_idx in range(batch_size):
        signal = mean_response[batch_idx].detach().cpu()
        timesteps = signal.numel()
        if timesteps < 2:
            continue

        left = torch.empty_like(signal)
        right = torch.empty_like(signal)
        left[0] = signal[0]
        left[1:] = signal[:-1]
        right[-1] = signal[-1]
        right[:-1] = signal[1:]

        peak_mask = (signal >= left) & (signal >= right)
        peak_mask[0] = timesteps == 1 or signal[0] >= signal[1]
        peak_mask[-1] = timesteps == 1 or signal[-1] >= signal[-2]

        peak_indices = [
            idx
            for idx in torch.nonzero(peak_mask, as_tuple=False).flatten().tolist()
            if signal[idx].item() > 0
        ]

        if len(peak_indices) < 2:
            continue

        peak_indices.sort(key=lambda idx: signal[idx].item(), reverse=True)

        first_idx = peak_indices[0]
        first_val = signal[first_idx].item()
        if first_val <= 0:
            continue

        second_idx = None
        for candidate in peak_indices[1:]:
            if abs(candidate - first_idx) >= min_delay:
                second_idx = candidate
                break
        if second_idx is None:
            second_idx = peak_indices[1]

        second_val = signal[second_idx].item()
        if second_val <= 0:
            continue

        if first_idx < second_idx:
            ratio_value = second_val / first_val if first_val != 0 else 0.0
        else:
            ratio_value = first_val / second_val if second_val != 0 else 0.0

        ratios[batch_idx] = ratio_value

    return ratios


def spatial_variance(response: torch.Tensor) -> torch.Tensor:
    """Calculate variance across spatial dimensions at each timepoint.

    Args:
        response: Layer response tensor [batch_size, n_timesteps, ...spatial_dims]

    Returns:
        Variance tensor of shape [batch_size, n_timesteps]
    """
    # Handle different tensor dimensionalities
    if response.dim() < 3:
        raise ValueError(
            f"Response tensor must have at least 3 dimensions, got {response.dim()}"
        )
    elif response.dim() == 3:
        # [batch, time, channels] - no spatial dimensions
        return torch.zeros_like(response[:, :, 0])  # Return zeros for spatial variance
    elif response.dim() == 4:
        # [batch, time, channels, spatial] - 1D spatial
        spatial_var = response.var(dim=3)  # [batch, time, channels]
        return spatial_var.mean(dim=2)  # [batch, time]
    elif response.dim() == 5:
        # [batch, time, channels, dim_y, dim_x] - 2D spatial
        spatial_var = response.var(dim=(3, 4))  # [batch, time, channels]
        return spatial_var.mean(dim=2)  # [batch, time]
    else:
        # Higher dimensions - treat all beyond channels as spatial
        spatial_dims = list(range(3, response.dim()))
        spatial_var = response.var(dim=spatial_dims)  # [batch, time, channels]
        return spatial_var.mean(dim=2)  # [batch, time]


def feature_variance(response: torch.Tensor) -> torch.Tensor:
    """Calculate variance across channel dimensions at each timepoint.

    Args:
        response: Layer response tensor [batch_size, n_timesteps, n_channels, ...spatial_dims]

    Returns:
        Feature variance tensor of shape [batch_size, n_timesteps]
    """
    # Handle different tensor dimensionalities
    if response.dim() < 3:
        raise ValueError(
            f"Response tensor must have at least 3 dimensions, got {response.dim()}"
        )
    elif response.dim() == 3:
        # [batch, time, channels] - no spatial dimensions
        return response.var(dim=2)  # [batch, time]
    elif response.dim() == 4:
        # [batch, time, channels, spatial] - 1D spatial
        feature_var = response.var(dim=2)  # [batch, time, spatial]
        return feature_var.mean(dim=2)  # [batch, time]
    elif response.dim() == 5:
        # [batch, time, channels, dim_y, dim_x] - 2D spatial
        feature_var = response.var(dim=2)  # [batch, time, dim_y, dim_x]
        return feature_var.mean(dim=(2, 3))  # [batch, time]
    else:
        # Higher dimensions - treat all beyond channels as spatial
        feature_var = response.var(dim=2)  # [batch, time, ...spatial_dims]
        spatial_dims = list(range(2, feature_var.dim()))
        return feature_var.mean(dim=spatial_dims)  # [batch, time]


def calculate_accuracy(df: pd.DataFrame) -> float:
    """Calculate classification accuracy.

    Args:
        df: DataFrame with label_index and guess_index columns

    Returns:
        Classification accuracy as float
    """
    dfi = df[df.label_index >= 0]
    accuracy = (dfi.guess_index == dfi.label_index).mean()
    return accuracy


def load_config_from_args(palette_str=None, naming_str=None, ordering_str=None):
    """Load configuration from JSON strings passed via command line.

    Args:
        palette_str: JSON string for palette
        naming_str: JSON string for naming
        ordering_str: JSON string for ordering

    Returns:
        Dictionary with config sections
    """
    config = {}

    if palette_str:
        try:
            config["palette"] = json.loads(palette_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse palette JSON: {e}")
            config["palette"] = {}
    else:
        config["palette"] = {}

    if naming_str:
        try:
            config["naming"] = json.loads(naming_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse naming JSON: {e}")
            config["naming"] = {}
    else:
        config["naming"] = {}

    if ordering_str:
        try:
            config["ordering"] = json.loads(ordering_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse ordering JSON: {e}")
            config["ordering"] = {}
    else:
        config["ordering"] = {}

    return config


def get_display_name(key: str, config: Dict) -> str:
    """Get display name/symbol for a key from config.

    Args:
        key: The key to look up
        config: Configuration dictionary

    Returns:
        Display name or symbol, fallback to key if not found
    """
    naming = config.get("naming", {})
    return naming.get(key, naming.get(key.lower(), key))


def get_color(key: str, config: Dict) -> Optional[str]:
    """Get color for a key from config.

    Args:
        key: The key to look up
        config: Configuration dictionary

    Returns:
        Color string or None if not found
    """
    palette = config.get("palette", {})
    return palette.get(key)


def get_ordering(category: str, config: Dict) -> Optional[List[str]]:
    """Get ordering for a category from config.

    Args:
        category: The category to look up
        config: Configuration dictionary

    Returns:
        List of ordered values or None if not found
    """
    ordering = config.get("ordering", {})
    return ordering.get(category)


def order_layers(layer_names: List[str], config: Dict) -> List[str]:
    """Order layers according to visual hierarchy.

    Args:
        layer_names: List of layer names
        config: Configuration dictionary

    Returns:
        Ordered list of layer names (IT, V4, V2, V1)
    """
    # Get ordering from config or use default
    hierarchy_order = get_ordering("layers", config) or ["IT", "V4", "V2", "V1"]

    # Get naming mapping from config
    naming = config.get("naming", {})

    # Filter out classifier layers
    filtered_layers = [
        layer for layer in layer_names if "classifier" not in layer.lower()
    ]

    # Sort layers according to hierarchy
    def get_sort_key(layer_name):
        # Map layer name to display name
        mapped_name = naming.get(layer_name, layer_name)
        if mapped_name in hierarchy_order:
            return hierarchy_order.index(mapped_name)
        else:
            return len(hierarchy_order)  # Put unknown layers at the end

    ordered_layers = sorted(filtered_layers, key=get_sort_key)
    return ordered_layers


def calculate_label_indicator(
    df: pd.DataFrame, category: str, y_range: tuple, step_height: float = 0.25
) -> pd.DataFrame:
    """Calculate label indicator (step function) at each time step.

    Args:
        df: DataFrame containing classifier responses
        category: Category column name
        y_range: Tuple of (y_min, y_max) for the current subplot

    Returns:
        DataFrame with times_index and label_indicator (relative to subplot height)
    """
    y_min, y_max = y_range

    if "label_valid" in df.columns:
        indicator_source = (
            df.groupby("times_index", observed=True)["label_valid"].max().reset_index()
        )
        indicator_source["label_indicator"] = indicator_source["label_valid"].apply(
            lambda flag: y_min + ((y_max - y_min) * step_height) if flag else y_min
        )
        return indicator_source[["times_index", "label_indicator"]]

    # Fallback for legacy datasets that still require label_index logic
    # Get first model's data to determine label validity at each time step
    first_model = df[category].iloc[0]
    model_data = df[df[category] == first_model]

    # Calculate step height as 25% of the y-axis range
    if step_height >= 0 and step_height < 1:
        # step_height is a fraction of the y-axis range
        step_height = (y_max - y_min) * step_height
    elif step_height >= 1:
        # step_height is an absolute value, ensure it does not exceed 25% of y-axis range
        max_step = (y_max - y_min) * 0.25
        step_height = min(step_height, max_step)
    else:
        raise ValueError("step_height must be positive")

    indicator_data = []
    for time_step in sorted(model_data.times_index.unique()):
        time_data = model_data[model_data.times_index == time_step]

        # Check if any labels are valid (>= 0) at this time step
        valid_labels = (time_data.label_index >= 0).any()
        indicator_value = y_min + step_height if valid_labels else y_min

        indicator_data.append(
            {"times_index": time_step, "label_indicator": indicator_value}
        )

    return pd.DataFrame(indicator_data)


def get_category_plotting_settings(
    category: str, category_values: List, config: Dict
) -> Tuple[List, Dict[str, str]]:
    """Get plotting settings for different categories.

    Args:
        category: The category name
        category_values: List of unique category values
        config: Configuration dictionary

    Returns:
        Tuple of (order_list, colors_dict)
    """
    # Get ordering from config
    model_order = get_ordering(category, config)

    if model_order:
        # Filter to only include values that exist in the data
        model_order = [val for val in model_order if val in category_values]
        # Add any missing values from data
        for val in category_values:
            if val not in model_order:
                model_order.append(val)
    else:
        # Try to order by float values if no config ordering
        try:
            float_values = [(float(val), val) for val in category_values]
            float_values.sort(key=lambda x: x[0])
            model_order = [val for _, val in float_values]
        except (ValueError, TypeError):
            # Can't convert to float, use alphabetical order
            model_order = sorted(category_values)

    # Get colors from config
    colors = {}
    for val in model_order:
        color = get_color(val, config)
        if color:
            colors[val] = color

    # Fill in missing colors with defaults
    if len(colors) < len(model_order):
        if len(model_order) <= 10:
            default_colors = plt.cm.tab10(np.arange(len(model_order)))
        else:
            default_colors = plt.cm.tab20(np.arange(len(model_order)) % 20)

        for i, val in enumerate(model_order):
            if val not in colors:
                colors[val] = mcolors.rgb2hex(default_colors[i])

    return model_order, colors


def load_responses(
    pt_files: List[Path],
    csv_files: List[Path],
    data_arg_key: str = "contrast",
    measures: List[str] = ["power", "peak_time", "peak_height"],
    category: str = "rctype",
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and process model responses.

    Args:
        pt_files: List of PyTorch response files
        csv_files: List of CSV label files
        data_arg_key: Key for data arguments
        measures: List of measures to compute
        category: Category key for grouping

    Returns:
        Tuple of (DataFrame with responses, list of layer names)
    """
    dfs = []

    for pt_file, csv_file in zip(pt_files, csv_files):
        arg_value = extract_param_from_string(
            pt_file.name, key=data_arg_key, value_type=float
        )

        cat_value = extract_param_from_string(
            pt_file.name, key=category, value_type=None
        )

        if not arg_value == extract_param_from_string(
            csv_file.name, key=data_arg_key, value_type=float
        ):
            raise ValueError(f"{data_arg_key} values do not match!")
        if not cat_value == extract_param_from_string(
            csv_file.name, key=category, value_type=None
        ):
            raise ValueError(f"{category} do not match!")

        df = load_df(csv_file)
        df[data_arg_key] = arg_value
        df[category] = cat_value
        n_classes = len(df.class_index.unique())

        # Remove 'class_index' and 'response' columns if present
        for col in ["class_index", "response"]:
            if col in df.columns:
                df = df.drop(columns=col)

        # Collapse redundant rows by grouping on all remaining columns and aggregating with 'first'
        df = df.groupby(list(df.columns), as_index=False).first()

        responses = torch.load(pt_file, map_location=torch.device("cpu"))

        max_timesteps = max(tensor.shape[1] for tensor in responses.values())

        for layer_name, tensor in responses.items():
            pad_len = max_timesteps - tensor.shape[1]
            if pad_len > 0:
                # Pad at the start along axis 1 (time steps) with zeros
                # For shape [batch, timesteps, n_channels, dim_y, dim_x], pad for axis 1
                # torch.nn.functional.pad expects (dim_x, dim_y, n_channels, timesteps)
                # So pad = (0,0, 0,0, 0,0, pad_len,0)
                pad = (0, 0, 0, 0, 0, 0, pad_len, 0)
                tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0)
                responses[layer_name] = tensor

        layer_names = list(responses.keys())
        n_samples, n_timesteps, *_ = responses[layer_names[0]].shape

        for layer in layer_names:
            try:
                if "response_avg" in measures:
                    response_avg_tensor = (
                        layer_response_avg(responses[layer])
                        .flatten()
                        .repeat_interleave(n_classes)
                    )
                    df[f"{layer}_response_avg"] = tensor_to_numpy(response_avg_tensor)

                if "response_std" in measures:
                    response_std_tensor = (
                        layer_response_std(responses[layer])
                        .flatten()
                        .repeat_interleave(n_classes)
                    )
                    df[f"{layer}_response_std"] = tensor_to_numpy(response_std_tensor)

                if "peak_time" in measures:
                    peak_time_tensor = peak_time(responses[layer]).repeat_interleave(
                        n_classes * n_timesteps
                    )
                    df[f"{layer}_peak_time"] = tensor_to_numpy(peak_time_tensor)

                if "peak_height" in measures:
                    peak_height_tensor = peak_height(
                        responses[layer]
                    ).repeat_interleave(n_classes * n_timesteps)
                    df[f"{layer}_peak_height"] = tensor_to_numpy(peak_height_tensor)

                if "peak_ratio" in measures:
                    peak_ratio_tensor = peak_ratio(responses[layer]).repeat_interleave(
                        n_classes * n_timesteps
                    )
                    df[f"{layer}_peak_ratio"] = tensor_to_numpy(peak_ratio_tensor)
            except Exception as e:
                logger.error(
                    f"Failed to convert tensor to numpy for layer {layer}: {e}"
                )
                raise

        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    return df, layer_names


def chunk_lists(lst1, lst2, chunk_size):
    """Split two lists into chunks of specified size."""
    for i in range(0, len(lst1), chunk_size):
        yield lst1[i : i + chunk_size], lst2[i : i + chunk_size]


def load_responses_in_batches(
    responses_files,
    test_outputs_files,
    data_arg_key,
    measures,
    category,
    batch_size=1,
    filter_first_label_set=False,
):
    """Load responses in batches to manage memory efficiently."""

    if len(responses_files) != len(test_outputs_files):
        raise ValueError(
            "Number of response files must match number of test output files"
        )

    print(f"Processing {len(responses_files)} files in batches of {batch_size}")

    all_dataframes = []
    layer_names = None
    first_label_set = None

    # Process files in batches
    batch_count = 0
    for response_batch, output_batch in chunk_lists(
        responses_files, test_outputs_files, batch_size
    ):
        batch_count += 1
        print(
            f"Processing batch {batch_count}/{(len(responses_files) + batch_size - 1) // batch_size}"
        )
        print(
            f"  Files: {len(response_batch)} response files, {len(output_batch)} output files"
        )

        try:
            # Load this batch
            batch_df, batch_layer_names = load_responses(
                response_batch,
                output_batch,
                data_arg_key=data_arg_key,
                measures=measures,
                category=category,
            )

            # Store layer names from first batch
            if layer_names is None:
                layer_names = batch_layer_names
            else:
                # Verify layer names are consistent across batches
                if set(layer_names) != set(batch_layer_names):
                    print(
                        f"Warning: Layer names differ between batches. "
                        f"First batch: {layer_names}, Current batch: {batch_layer_names}"
                    )

            if filter_first_label_set:
                # Get first label_set from first batch and filter immediately for efficiency
                first_label_set = batch_df.label_set.unique()[0]
                print(f"  Using first label_set: {first_label_set}")

                # Filter for first label_set only to reduce memory usage
                if first_label_set is not None:
                    batch_df = batch_df[
                        batch_df.label_set.str.contains(str(first_label_set))
                    ]
                    print(f"  After filtering for label_set: {len(batch_df)} rows")

                # Remove label_set column to save memory since we no longer need it
                if "label_set" in batch_df.columns:
                    batch_df = batch_df.drop("label_set", axis=1)

            all_dataframes.append(batch_df)
            print(f"  Batch {batch_count} processed: {len(batch_df)} rows")

            # Clear memory
            del batch_df

        except Exception as e:
            print(f"Error processing batch {batch_count}: {e}")
            print(f"  Response files: {response_batch}")
            print(f"  Output files: {output_batch}")
            continue

    if not all_dataframes:
        raise ValueError("No data was successfully loaded from any batch")

    # Combine all dataframes
    print(f"Combining {len(all_dataframes)} batches...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Clear memory
    del all_dataframes

    print(f"Total combined data: {len(combined_df)} rows")
    return combined_df, layer_names


def save_plot(file_path: Path, dpi: int = 300, **kwargs) -> None:
    """Save matplotlib plot with error handling.

    Args:
        file_path: Path to save plot
        dpi: Resolution for saved plot
        **kwargs: Additional arguments for savefig

    Returns:
        None
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        plt.savefig(fname=file_path, dpi=dpi, bbox_inches="tight", **kwargs)

    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        try:  # save empty plot
            plt.subplots()
            plt.savefig(fname=file_path)
        except Exception as e:
            logging.error(f"Failed to save empty plot: {e}")

    finally:
        if "ipykernel" in sys.modules:
            plt.show()
        else:
            plt.close()

    logging.info(f"Plot saved successfully at {file_path}")

    return None
