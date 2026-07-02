"""plot_dynamics.py
Unified plot showing layer dynamics across different parameter values with consistent layout.

This script creates a 4-panel visualization of neural network layer responses:
    - Panel A (upper left): Different layers for a reference parameter value
    - Panel B (lower left): Focus layer across different parameter values
    - Panel C (upper right): Empty panel for custom content
    - Panel D (lower right): Summary plot showing parameter effect on metrics

Example:
    ```bash
    python plot_dynamics.py --data responses.csv --output dynamics_plot.png --parameter tau \
        --category interval --focus-layer V1 --dt 1.0
    ```
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from dynvision.utils.visualization_utils import (  # noqa: F401
    save_plot,
    load_config_from_args,
    get_display_name,
    order_layers,
    peak_ratio,
    peak_time,
    peak_height,
    format_parameter_value,
    DT_CONVERT_PARAMS,
)

# Import functions from plot_responses.py
from dynvision.visualization.plot_responses import (  # noqa: F401
    _plot_response_ridges,
    _add_horizontal_legend,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _filter_data_for_column,
    _add_layer_circle,
    FORMATTING as RESPONSES_FORMATTING,
    LAYOUT as RESPONSES_LAYOUT,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Layout parameters for the 4-panel figure
DYNAMICS_LAYOUT = {
    "figure_width": 15,
    "figure_height": 10,
    "plots_start": 0.05,
    "left_plots_width": 0.6,
    "right_plots_width": 0.3,
    "gap_width": 0.1,
    # Vertical positions
    "upper_top": 0.98,
    "upper_bottom": 0.65,
    "lower_top": 0.62,  # edited
    "lower_bottom": 0.01,
    "legend_height": 0.1,
    "legend_margin": 0.02,
    # Positioning helpers
    "y_label_x_position": -0.01,
    "panel_label_x_offset": -0.04,
    "panel_label_y_offset": 0.02,
}
#
# Style constants
DYNAMICS_STYLE = {
    "fontsize_panel_labels": 16,
    "marker_edge_width": 1.2,
}


def determine_experiment_info(experiment, config=None):
    """Determine experiment type and corresponding labels from experiment name.

    Args:
        experiment: Experiment name (e.g., 'duration', 'contrast', 'interval')
        config: Optional config dict containing 'naming' for display names

    Returns:
        Dictionary with experiment info including type, labels, and metric
    """
    # Get naming dict from config
    naming = config.get("naming", {}) if config else {}

    if "duration" in experiment.lower():
        param_name = naming.get("duration", "Duration")
        experiment_name = naming.get("duration_experiment", param_name)
        return {
            "type": "duration",
            "label": param_name,
            "full_label": f"{experiment_name} ({param_name})",
            "metric": "summed_response",
            "y_label": "Summed Response",
        }
    elif "contrast" in experiment.lower():
        param_name = naming.get("contrast", "Contrast")
        experiment_name = naming.get("contrast_experiment", param_name)
        return {
            "type": "contrast",
            "label": param_name,
            "full_label": f"{experiment_name} ({param_name})",
            "metric": "peak_time",
            "y_label": "Peak Time",
        }
    elif "interval" in experiment.lower():
        param_name = naming.get("interval", "Interval")
        experiment_name = naming.get("interval_experiment", param_name)
        return {
            "type": "interval",
            "label": param_name,
            "full_label": f"{experiment_name} ({param_name})",
            "metric": "peak_ratio",
            "y_label": "Peak Ratio",
        }
    else:
        param_name = naming.get(experiment.lower(), experiment.capitalize())
        experiment_name = naming.get(f"{experiment.lower()}_experiment", param_name)
        return {
            "type": "unknown",
            "label": param_name,
            "full_label": f"{experiment_name} ({param_name})",
            "metric": "peak_height",
            "y_label": "Peak Height",
        }


def add_panel_labels(fig):
    """Add panel labels A), B), C), D) to the figure."""
    panel_x_offset = DYNAMICS_LAYOUT["panel_label_x_offset"]
    panel_y_offset = DYNAMICS_LAYOUT["panel_label_y_offset"]
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT["gap_width"]
    upper_top = DYNAMICS_LAYOUT["upper_top"]
    lower_top = DYNAMICS_LAYOUT["lower_top"]

    # Left side panel labels (A and B)
    fig.text(
        x=plots_start + panel_x_offset,
        y=upper_top + panel_y_offset,
        s="A)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=plots_start + panel_x_offset,
        y=lower_top + panel_y_offset,
        s="B)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )

    # Right side panel labels (C and D)
    right_x = plots_start + left_width + gap_width + 1.5 * panel_x_offset
    fig.text(
        x=right_x,
        y=upper_top + panel_y_offset,
        s="C)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        x=right_x,
        y=lower_top + panel_y_offset,
        s="D)",
        fontsize=DYNAMICS_STYLE["fontsize_panel_labels"],
        fontweight="bold",
        ha="left",
        va="top",
    )


def calculate_metrics(df, focus_layer, parameter, category, metric, dt=None):
    """Calculate metrics for summary plot.

    Each metric is calculated independently for each (category, sample_index, seed)
    combination across all parameter values, producing one metric value per parameter value.

    If sample_index is not available, first_label_index is used as a fallback.
    If seed column exists, metrics are calculated separately for each seed.

    For 'summed_response', normalization (divide by max) and baseline subtraction
    (subtract min) are applied after calculation to each group independently.
    """
    df = df.copy()
    metric_col = f"{focus_layer}_{metric}"

    if metric_col in df.columns:
        return df

    df[metric_col] = np.nan

    # Log dataset statistics
    response_col = f"{focus_layer}_response_avg"
    n_samples = df["sample_index"].nunique() if "sample_index" in df.columns else 1
    n_times = df["times_index"].nunique()
    n_categories = df[category].nunique()
    n_params = df[parameter].nunique()
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
    n_first_label = (
        df["first_label_index"].nunique() if "first_label_index" in df.columns else 1
    )

    logger.info(
        f"Calculating {metric} for {focus_layer}: "
        f"{n_samples} samples (from {n_first_label} classes) × {n_times} timesteps × {n_categories} categories × "
        f"{n_params} parameter values × {n_seeds} seeds"
    )

    # Build groupby columns: category + sample identifier + seed (if available)
    groupby_cols = [category]

    # Use sample_index if available, otherwise fallback to first_label_index
    if "sample_index" in df.columns:
        groupby_cols.append("sample_index")
        entity_col = "sample_index"
    else:
        groupby_cols.append("first_label_index")
        entity_col = "first_label_index"

    # Add seed to group by if available (metrics calculated separately per seed)
    if "seed" in df.columns:
        groupby_cols.append("seed")
        has_seed = True
    else:
        has_seed = False

    for group_key, group in df.groupby(groupby_cols):
        # Unpack group_key based on what columns we grouped by
        if has_seed:
            cat_val, entity_val, seed_val = group_key
            logger.debug(
                f"Processing {metric}: category={cat_val}, {entity_col}={entity_val}, seed={seed_val}"
            )
        else:
            cat_val, entity_val = group_key
            logger.debug(
                f"Processing {metric}: category={cat_val}, {entity_col}={entity_val}"
            )

        # For each parameter value in this group, calculate the metric
        for param_val in group[parameter].unique():
            param_group = group[group[parameter] == param_val].sort_values(
                "times_index"
            )

            if len(param_group) == 0 or response_col not in param_group.columns:
                continue

            try:
                time_series_grouped = param_group.groupby("times_index")[
                    response_col
                ].mean()

                # Get values in sorted order by times_index
                time_series = time_series_grouped.sort_index().values

                if len(time_series) == 0 or np.isnan(time_series).all():
                    continue

                # Convert to tensor (shape: [1, n_timepoints])
                response_tensor = torch.from_numpy(
                    np.array([time_series], dtype=np.float32)
                )

                # Calculate metric
                if metric == "peak_ratio":
                    if param_val == 0:
                        # Skip interval=0 for peak_ratio (ill-defined with single/overlapping pulses)
                        continue
                    metric_value = peak_ratio(response_tensor, min_delay=5)[0].item()
                    # Debug logging for peak_ratio
                    if metric_value == 0:
                        logger.info(
                            f"peak_ratio={metric_value:.3f} for {parameter}={param_val}, {category}={cat_val}, "
                            f"{entity_col}={entity_val}. Response shape: {response_tensor.shape}, "
                            f"range: [{time_series.min():.3f}, {time_series.max():.3f}]"
                        )
                        logger.warning(
                            f"peak_ratio=0 for {parameter}={param_val}, {category}={cat_val}, {entity_col}={entity_val}. "
                            f"Possible causes: (1) <2 positive peaks, or (2) peaks <5 timesteps apart"
                        )
                elif metric == "peak_time":
                    metric_value = peak_time(response_tensor)[0].item()
                    if dt is not None:
                        metric_value *= dt
                elif metric == "peak_height":
                    metric_value = peak_height(response_tensor)[0].item()
                elif metric == "summed_response":
                    # Sum over time (no normalization yet)
                    metric_value = response_tensor.sum().item()
                else:
                    continue

                # Store metric value for all rows in this param_group
                df.loc[param_group.index, metric_col] = metric_value

            except Exception as e:
                logger.debug(
                    f"Error calculating {metric} for param={param_val}, "
                    f"category={cat_val}, entity={entity_val}: {e}"
                )

    # Post-process summed_response: normalize by max and optionally subtract baseline
    if metric == "summed_response":
        logger.info(
            f"Normalizing {metric} for each (category, {entity_col}, seed) combination"
        )

        # Group by category, entity, and seed (if available) to normalize across parameter values
        norm_groupby_cols = [category, entity_col]
        if "seed" in df.columns:
            norm_groupby_cols.append("seed")

        for norm_key, norm_group in df.groupby(norm_groupby_cols):
            # Get all metric values for this combination
            metric_values = norm_group[metric_col].dropna()

            if len(metric_values) == 0:
                continue

            # Subtract minimum (baseline correction)
            min_val = 0  # metric_values.min()
            baseline_corrected = metric_values - min_val

            # Normalize by maximum
            max_val = baseline_corrected.max()

            if max_val > 0:
                normalized = baseline_corrected / max_val
                df.loc[metric_values.index, metric_col] = normalized
            else:
                # All values at or below baseline
                df.loc[metric_values.index, metric_col] = 0.0

    return df


def create_panel_a(
    fig,
    df,
    layer_names,
    parameter,
    category,
    param_value,
    hue_values,
    colors,
    dt,
    config,
    time_offset: float = 0.0,
):
    """Create Panel A: Ridge plot for layers at middle parameter value."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    upper_top = DYNAMICS_LAYOUT["upper_top"]
    upper_bottom = DYNAMICS_LAYOUT["upper_bottom"]

    # Filter data for middle parameter value
    panel_data = _filter_data_for_column(df, parameter, param_value)
    if dt is not None:
        panel_data["time_ms"] = panel_data["times_index"] * dt + time_offset

    # Calculate layout parameters for ridge plots compatible with plot_responses
    panel_height = upper_top - upper_bottom

    # Create ridge plots for layers with proper layout parameters
    ridge_axes = _plot_response_ridges(
        fig=fig,
        column_left=plots_start,
        column_width=left_width,
        data=panel_data,
        subplot_var="layers",
        subplot_key="layers",
        subplot_values=layer_names,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        dt=dt or 1.0,
        time_offset=time_offset,
        show_ylabel=False,  # We'll add custom y-label
        config=config,
        # Override layout parameters to position correctly in Panel A
        title_bot=upper_top,
        title_pad=0.0,  # No title in this panel
        accuracy_height=0.0,  # No accuracy panel
        accuracy_pad=0.0,
        legend_height=0.0,  # No legend in individual panels
        legend_pad=0.0,
        ridge_top=upper_top,
        ridge_height=panel_height,
        ridge_overlap=0.2,
        **RESPONSES_FORMATTING,
    )

    # Remove x-axis margins for all ridge plot axes
    for ax in ridge_axes:
        if hasattr(ax, "get_xlim"):
            xlim = ax.get_xlim()
            # Set xlim to data range with no margins
            ax.set_xlim(left=time_offset, right=xlim[1])
            # ax.margins(x=0)

    # Add custom Y-axis label for the panel
    exp_info = determine_experiment_info(parameter, config)
    param_display = format_parameter_value(param_value, parameter, dt)
    fig.text(
        x=DYNAMICS_LAYOUT["y_label_x_position"],
        y=(upper_top + upper_bottom) / 2,
        s=f"Layer Response ({exp_info['label']} = {param_display})",
        rotation=90,
        verticalalignment="center",
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )

    return ridge_axes


def create_panel_b(
    fig,
    df,
    focus_layer,
    parameter,
    category,
    param_values,
    hue_values,
    colors,
    dt,
    config,
    normalize=False,
    time_offset: float = 0.0,
):
    """Create Panel B: Ridge plot for focus layer across all parameter values."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    lower_top = DYNAMICS_LAYOUT["lower_top"]
    lower_bottom = DYNAMICS_LAYOUT["lower_bottom"]

    # Prepare data with time conversion and ensure we have the focus layer response
    panel_data = df.copy()
    if dt is not None:
        panel_data["time_ms"] = panel_data["times_index"] * dt + time_offset

    # Add focus layer response as the y-variable for each parameter subplot
    focus_response_col = f"{focus_layer}_response_avg"
    if focus_response_col not in panel_data.columns:
        logger.error(
            f"Focus layer response column '{focus_response_col}' not found in data"
        )
        return []

    # Calculate layout parameters
    panel_height = lower_top - lower_bottom

    # For Panel B, we plot parameter values as subplots and category as hue
    # But we need to use a single focus layer response column for all subplots
    ridge_axes = _plot_response_ridges(
        fig=fig,
        column_left=plots_start,
        column_width=left_width,
        data=panel_data,
        subplot_var="parameter",
        subplot_key=parameter,
        subplot_values=param_values,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        dt=dt or 1.0,
        time_offset=time_offset,
        show_ylabel=False,  # We'll add custom y-label
        config=config,
        # Override layout parameters to position correctly in Panel B
        title_bot=lower_top,
        title_pad=0.0,  # No title in this panel
        accuracy_height=0.0,  # No accuracy panel
        accuracy_pad=0.0,
        legend_height=0.0,  # No legend in individual panels
        legend_pad=0.0,
        ridge_top=lower_top,
        ridge_height=panel_height,
        ridge_overlap=0.2,
        # Add focus layer info so the function knows which response to plot
        focus_layer=focus_layer,
        normalize=normalize,
        **RESPONSES_FORMATTING,
    )

    # Remove x-axis margins for all ridge plot axes
    for ax in ridge_axes:
        if hasattr(ax, "get_xlim"):
            xlim = ax.get_xlim()
            # Set xlim to data range with no margins
            ax.set_xlim(left=time_offset, right=xlim[1])

    # Add custom Y-axis label
    layer_display = get_display_name(focus_layer, config) or focus_layer.upper()
    label = (
        f"Normalized Layer Response ({layer_display})"
        if normalize
        else f"Layer Response ({layer_display})"
    )
    fig.text(
        x=DYNAMICS_LAYOUT["y_label_x_position"],
        y=(lower_top + lower_bottom) / 2,
        s=label,
        rotation=90,
        verticalalignment="center",
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )

    return ridge_axes


def create_panel_c(fig):
    """Create Panel C: Empty panel for custom content."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT["gap_width"]
    right_width = DYNAMICS_LAYOUT["right_plots_width"]
    upper_top = DYNAMICS_LAYOUT["upper_top"]
    upper_bottom = DYNAMICS_LAYOUT["upper_bottom"]

    panel_height = upper_top - upper_bottom
    ax = fig.add_axes(
        [
            plots_start + left_width + gap_width,
            upper_bottom,
            right_width,
            panel_height,
        ]
    )
    ax.patch.set_alpha(0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def create_panel_d(
    fig, df, focus_layer, parameter, category, exp_info, hue_values, colors, dt, config
):
    """Create Panel D: Summary plot showing parameter effect on metrics."""
    plots_start = DYNAMICS_LAYOUT["plots_start"]
    left_width = DYNAMICS_LAYOUT["left_plots_width"]
    gap_width = DYNAMICS_LAYOUT["gap_width"]
    right_width = DYNAMICS_LAYOUT["right_plots_width"]
    lower_top = DYNAMICS_LAYOUT["lower_top"]
    lower_bottom = DYNAMICS_LAYOUT["lower_bottom"]

    # Calculate the actual visual height of Panel B ridge plots to match alignment
    # Ridge plots use overlap, so actual bottom is different from panel bottom
    n_param_values = len(df[parameter].unique())
    panel_height = lower_top - lower_bottom
    ridge_overlap = 0.2
    spacing = panel_height / n_param_values * (1 - ridge_overlap)
    plot_height = panel_height / n_param_values * 1.4
    actual_bottom = lower_top - (n_param_values - 1) * spacing - plot_height

    # Use the ridge plots visual area for Panel D
    summary_height = lower_top - actual_bottom
    summary_bottom = actual_bottom
    summary_left = plots_start + left_width + gap_width

    summary_ax = fig.add_axes(
        [summary_left, summary_bottom, right_width, summary_height]
    )
    summary_ax.patch.set_alpha(0)

    # Add title
    _ = get_display_name(focus_layer, config) or focus_layer.upper()
    _add_layer_circle(
        x=0.5, y=1.1, ax=summary_ax, layer_name=focus_layer, config=config
    )

    # Calculate metrics
    metric = exp_info["metric"]
    metric_col = f"{focus_layer}_{metric}"

    if metric_col not in df.columns:
        df = calculate_metrics(
            df=df,
            focus_layer=focus_layer,
            parameter=parameter,
            category=category,
            metric=metric,
            dt=dt,
        )

    df[category] = df[category].astype(str)
    hue_values_str = [str(val) for val in hue_values]

    if df[metric_col].notna().any():
        sns.lineplot(
            data=df,
            x=parameter,
            y=metric_col,
            ax=summary_ax,
            marker=None,
            hue=category,
            hue_order=hue_values_str,
            palette=colors,
            legend=False,
            errorbar="sd",
            err_style="bars",
            linewidth=RESPONSES_FORMATTING["linewidth_main"],
            alpha=RESPONSES_FORMATTING["alpha_line"],
        )

        for line in summary_ax.get_lines():
            line.set_markeredgecolor("white")
            line.set_markeredgewidth(DYNAMICS_STYLE["marker_edge_width"])

        param_values = df[parameter].unique()
        param_values_sorted = sorted(param_values)
        summary_ax.set_xticks(param_values_sorted)

        # Remove x-axis and y-axis margins
        if len(param_values_sorted) > 0:
            summary_ax.set_xlim(min(param_values_sorted), max(param_values_sorted))
        summary_ax.margins(y=0)

        # Convert tick labels to ms if parameter needs dt conversion
        if dt is not None and parameter.lower() in DT_CONVERT_PARAMS:
            summary_ax.set_xticklabels([f"{int(x * dt)}" for x in param_values_sorted])

        summary_ax.grid(True, which="major", linestyle=":", alpha=0.3)

        if exp_info["metric"] == "peak_ratio":
            summary_ax.axhline(
                y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7
            )
        elif exp_info["metric"] == "summed_response":
            # Add linear prediction reference line
            # Linear prediction: normalized value = parameter / max_parameter
            param_values_unique = sorted(param_values)
            max_param = max(param_values_unique)
            linear_prediction = [p / max_param for p in param_values_unique]
            summary_ax.plot(
                param_values_unique,
                linear_prediction,
                "k--",
                linewidth=1.5,
                alpha=0.5,
                label="Linear prediction",
                zorder=1,
            )
    else:
        summary_ax.text(
            x=0.5,
            y=0.5,
            s="No summary data available",
            ha="center",
            va="center",
            fontsize=RESPONSES_FORMATTING["fontsize_label"],
            alpha=0.7,
        )

    # Add unit to x-axis label if parameter was dt-converted
    xlabel = exp_info["full_label"]
    if dt is not None and parameter.lower() in DT_CONVERT_PARAMS:
        xlabel = f"{xlabel} (ms)"
    summary_ax.set_xlabel(xlabel, fontsize=RESPONSES_FORMATTING["fontsize_label"])
    summary_ax.set_ylabel(
        exp_info["y_label"],
        fontsize=RESPONSES_FORMATTING["fontsize_label"],
        fontweight="bold",
    )
    summary_ax.tick_params(
        axis="both", labelsize=RESPONSES_FORMATTING["fontsize_tick"]
    )
    sns.despine(ax=summary_ax, left=True, bottom=True)

    return summary_ax


def plot_unified_dynamics(
    df,
    focus_layer,
    *,
    parameter="tau",
    category="interval",
    experiment=None,
    save_path=None,
    dt=None,
    time_offset: float = 0.0,
    config=None,
):
    """Create unified 4-panel dynamics visualization."""
    logger.info("=" * 60)
    logger.info("Starting unified dynamics plotting")
    logger.info("=" * 60)

    # Validate inputs
    if parameter not in df.columns:
        raise ValueError(f"Parameter column '{parameter}' not found")
    if category not in df.columns:
        raise ValueError(f"Category column '{category}' not found")

    # Initialize config if not provided
    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Determine experiment info
    experiment = experiment or parameter
    exp_info = determine_experiment_info(experiment, config)

    # Extract dimension values using plot_responses functions
    layer_names = _extract_dimension_values(df, "layers", "layers", config)
    hue_values = _extract_dimension_values(df, "category", category, config)
    param_values = _extract_dimension_values(df, "parameter", parameter, config)

    logger.info(f"Layers: {layer_names}")
    logger.info(f"Category values: {hue_values}")
    logger.info(f"Parameter values: {param_values}")

    # Get colors using plot_responses function
    colors = _get_colors_for_dimension(hue_values, category, config)

    # Select middle parameter value
    middle_idx = len(param_values) // 2
    middle_param_value = param_values[middle_idx]
    logger.info(f"Using middle parameter value: {middle_param_value}")

    # Convert time if needed
    if dt is not None:
        df = df.copy()
        df["time_ms"] = df["times_index"] * dt + time_offset

    # Create figure
    fig = plt.figure(
        figsize=(DYNAMICS_LAYOUT["figure_width"], DYNAMICS_LAYOUT["figure_height"])
    )
    sns.set_style("ticks")
    sns.set_context("talk")

    # Add panel labels
    add_panel_labels(fig)

    # Create panels
    logger.info("Creating Panel A: Layers")
    _ = create_panel_a(
        fig,
        df,
        layer_names,
        parameter,
        category,
        middle_param_value,
        hue_values,
        colors,
        dt,
        config,
        time_offset=time_offset,
    )

    logger.info("Creating Panel B: Parameter variation")
    _ = create_panel_b(
        fig,
        df,
        focus_layer,
        parameter,
        category,
        param_values,
        hue_values,
        colors,
        dt,
        config,
        time_offset=time_offset,
    )

    logger.info("Creating Panel C: Empty panel")
    _ = create_panel_c(fig)

    logger.info("Creating Panel D: Summary metrics")
    _ = create_panel_d(
        fig,
        df,
        focus_layer,
        parameter,
        category,
        exp_info,
        hue_values,
        colors,
        dt,
        config,
    )

    # Add horizontal legend (positioned manually for dynamics layout)
    legend_left = DYNAMICS_LAYOUT["plots_start"]
    legend_width = DYNAMICS_LAYOUT[
        "left_plots_width"
    ]  # + DYNAMICS_LAYOUT["right_plots_width"]
    legend_height = DYNAMICS_LAYOUT["legend_height"]
    legend_bottom = DYNAMICS_LAYOUT["lower_top"] + DYNAMICS_LAYOUT["legend_margin"]

    _add_horizontal_legend(
        fig=fig,
        hue_var="category",
        hue_key=category,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt or 1.0,
        legend_bot=legend_bottom,
        # Override legend positioning for dynamics layout
        **(
            RESPONSES_FORMATTING
            | {
                "legend_height": legend_height,
                "left_margin": legend_left,
                "column_width": legend_width,
                "max_elements": 4,
            }
        ),
    )

    # Save figure
    if save_path:
        logger.info(f"Saving figure to {save_path}")
        save_plot(file_path=Path(save_path), dpi=300)

    logger.info("=" * 60)
    logger.info("Dynamics plotting complete")
    logger.info("=" * 60)

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create unified dynamics visualization"
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to response data CSV file(s). Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save output figure"
    )
    parser.add_argument(
        "--parameter", type=str, required=True, help="Parameter column name"
    )
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument(
        "--category", type=str, required=True, help="Category column name"
    )
    parser.add_argument(
        "--focus-layer", type=str, default="V1", help="Layer to focus on"
    )
    parser.add_argument("--dt", type=float, help="Time step in milliseconds")
    parser.add_argument(
        "--idle-timesteps",
        type=int,
        default=0,
        help="Number of idle timesteps before recorded data (shifts time axis)",
    )
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")

    args = parser.parse_args()

    # Load and concatenate data
    if len(args.data) > 1:
        logger.info(f"Loading and concatenating data from {len(args.data)} files")
        dfs = []
        for i, path in enumerate(args.data):
            logger.info(f"  Loading file {i+1}/{len(args.data)}: {path}")
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Concatenated total: {len(df)} rows")
    else:
        logger.info(f"Loading data from: {args.data[0]}")
        df = pd.read_csv(args.data[0])

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Compute time offset from idle timesteps
    time_offset = args.idle_timesteps * (args.dt if args.dt is not None else 1.0)

    # Create plot
    plot_unified_dynamics(
        df=df,
        focus_layer=args.focus_layer,
        parameter=args.parameter,
        category=args.category,
        experiment=args.experiment,
        save_path=args.output,
        dt=args.dt,
        time_offset=time_offset,
        config=config,
    )


if __name__ == "__main__":
    main()
