"""Plot performance traces with flexible dimension mapping.

This module provides a specialized plotting function for visualizing accuracy
and confidence traces across different experimental conditions with three flexible dimensions:
- Horizontal subplots
- Vertical rows
- Hue (color coding)

Each dimension can represent: category, parameter, or experiment.
Supports multiple input files for different experiments.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

# Import functions and configurations from plot_responses
from dynvision.visualization.plot_responses import (  # noqa: F401
    FORMATTING as RESPONSES_FORMATTING,
    _coerce_measure_list,
    _filter_data_for_column,
    _format_legend_label,
    _format_measure_label,
    _get_colors_for_dimension,
    _get_dimension_key,
    _extract_dimension_values,
    _normalize_dimension,
    _standardize_category_value,
    _validate_dimensions as _validate_dimension_choices,
)

from dynvision.utils.visualization_utils import (  # noqa: F401
    calculate_label_indicator,
    get_color,
    get_display_name,
    get_ordering,
    load_config_from_args,
    order_layers,
    resolve_measure_columns,
    save_plot,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Layout configuration for performance plots
PERFORMANCE_LAYOUT = {
    # Figure dimensions
    "subplot_width": 3,  # Width of each subplot in inches
    "subplot_height": 7.0,  # Height of each subplot in inches
    "subplot_spacing_x": 0.3,  # Horizontal spacing between subplots
    "subplot_spacing_y": 0.7,  # Vertical spacing between rows
    "peak_panel_width": 3.0,  # Width of peak height panel
    "peak_panel_spacing": 1,  # Spacing before peak panel
    "peak_time_panel_width": 3.0,  # Width of peak time panel
    "peak_time_panel_spacing": 1,  # Spacing before peak time panel
    # Margins
    "left_margin": 0.1,  # Left margin in inches
    "right_margin": 0.0,  # Right margin in inches
    "top_margin": 0.1,  # Top margin in inches
    "bottom_margin": 0.0,  # Bottom margin in inches
    # Title spacing
    "title_spacing": 0.08,
    # Panel letters
    "panel_letter_offset_x": -0.01,
    "panel_letter_offset_y": 0.03,
}

# Inherit formatting from plot_responses and override specific values
FORMATTING = {
    **RESPONSES_FORMATTING,  # Import all defaults from plot_responses
    "title_fontsize": 16,
    "fontsize_panel_label": 18,
}

_NONE_DIMENSION_TOKENS = {"", "none", "null"}


def _coerce_optional_dimension(value: Optional[str]) -> Optional[str]:
    """Normalize optional dimension strings allowing textual None tokens."""

    if value is None:
        return None

    text = str(value).strip()
    if text.lower() in _NONE_DIMENSION_TOKENS:
        return None
    return text


def _resolve_measure_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    """Resolve a requested metric name to an available dataframe column."""

    resolved, _ = resolve_measure_columns(df.columns, [requested])
    if resolved:
        return resolved[0][1]
    return None


# Errorbar configuration for lineplot
ERRORBAR_CONFIG = {
    "errorbar": None,  # ("se", 1),  # Standard error of mean with shaded area
    "err_style": "band",  # Shaded band style for errorbars
}


def _append_suffix_to_label(label: str, suffix: str) -> str:
    """Append a suffix to an existing legend label, handling parentheses."""

    if not suffix:
        return label

    cleaned_suffix = suffix.strip()
    if not cleaned_suffix:
        return label

    if label.endswith(")"):
        return f"{label[:-1]}, {cleaned_suffix})"
    return f"{label} ({cleaned_suffix})"


def _filter_data_for_cell(
    df: pd.DataFrame,
    row_key: Optional[str],
    row_value: Optional[str],
    subplot_key: str,
    subplot_value: str,
) -> pd.DataFrame:
    """Filter data for a specific cell in the performance grid.

    Args:
        df: Input DataFrame
        row_key: Column name for row dimension (optional)
        row_value: Value for row dimension (optional)
        subplot_key: Column name for subplot dimension
        subplot_value: Value for subplot dimension

    Returns:
        Filtered DataFrame for the specific cell
    """
    # Filter data for this cell
    cell_data = df.copy()

    # Filter by row dimension
    if row_key and row_key in cell_data.columns and row_value is not None:
        cell_data[row_key] = cell_data[row_key].apply(_standardize_category_value)
        row_value_std = _standardize_category_value(str(row_value))
        cell_data = cell_data[cell_data[row_key] == row_value_std]

    # Filter by subplot dimension
    if subplot_key in cell_data.columns:
        cell_data[subplot_key] = cell_data[subplot_key].apply(
            _standardize_category_value
        )
    subplot_value_std = _standardize_category_value(str(subplot_value))
    cell_data = cell_data[cell_data[subplot_key] == subplot_value_std]

    return cell_data


def _plot_accuracy_panel_with_ffonly(
    ax: plt.Axes,
    data: pd.DataFrame,
    hue_var: str,
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    dt: float,
    time_offset: float = 0.0,
    show_ylabel: bool = True,
    show_legend: bool = True,
    accuracy_cols: Optional[List[str]] = None,
    confidence_cols: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """Plot accuracy and confidence over time for both full and feedforward-only models."""

    fmt = {**FORMATTING, **kwargs}
    errorbar_mode = kwargs.get("errorbar", ERRORBAR_CONFIG["errorbar"])
    err_style = kwargs.get("err_style", ERRORBAR_CONFIG["err_style"])
    errorbar_settings = {"errorbar": errorbar_mode}
    if errorbar_mode not in (None, "none") and err_style:
        errorbar_settings["err_style"] = err_style

    requested_accuracy = list(accuracy_cols or [])
    requested_confidence = list(confidence_cols or [])

    if not requested_accuracy and not requested_confidence:
        logger.warning("No accuracy or confidence measures requested; skipping panel")
        return

    ax.patch.set_alpha(0)

    # Create time in ms
    data_plot = data.copy()
    if "times_index" not in data_plot.columns:
        logger.warning("Missing 'times_index' column; cannot plot performance panel")
        return
    data_plot["time_ms"] = data_plot["times_index"] * dt + time_offset

    if hue_key in data_plot.columns:
        data_plot[hue_key] = data_plot[hue_key].apply(_standardize_category_value)

    n_datapoints = len(data_plot)
    logger.info("Plotting accuracy panel with %d datapoints", n_datapoints)

    has_ffonly = (
        "model_type" in data_plot.columns
        and "ffonly" in data_plot["model_type"].values
    )
    has_feedforward = (
        "model_type" in data_plot.columns
        and "feedforward" in data_plot["model_type"].values
    )

    full_data = (
        data_plot[data_plot.get("model_type", "full") == "full"]
        if "model_type" in data_plot.columns
        else data_plot
    )
    ffonly_data = (
        data_plot[data_plot.get("model_type", "full") == "ffonly"]
        if has_ffonly
        else pd.DataFrame()
    )
    feedforward_data = (
        data_plot[data_plot.get("model_type", "full") == "feedforward"]
        if has_feedforward
        else pd.DataFrame()
    )

    accuracy_styles_full = [
        {"linestyle": "-", "marker": "o", "markersize": 3},
        {"linestyle": "-", "marker": "D", "markersize": 3},
        {"linestyle": "-", "marker": "^", "markersize": 3},
    ]
    confidence_styles_full = [
        {"linestyle": ":", "marker": "s", "markersize": 2},
        {"linestyle": ":", "marker": "P", "markersize": 2},
        {"linestyle": ":", "marker": "X", "markersize": 2},
    ]
    # Feedforward models (trained without recurrence) shown as dashed grey lines
    accuracy_styles_feedforward = [
        {"linestyle": "--", "marker": "", "markersize": 0, "color": "grey"},
    ]

    plotted_accuracy_full: List[Tuple[str, str, Dict[str, Union[str, float]]]] = []
    plotted_confidence_full: List[Tuple[str, str, Dict[str, Union[str, float]]]] = []
    plotted_accuracy_feedforward: List[
        Tuple[str, str, Dict[str, Union[str, float]]]
    ] = []
    ff_accuracy_markers: Dict[str, bool] = {}

    def _plot_series(
        dataset: pd.DataFrame,
        requested: List[str],
        styles: List[Dict[str, Union[str, float]]],
        storage: List[Tuple[str, str, Dict[str, Union[str, float]]]],
        dataset_label: str,
        use_errorbars: bool = True,
    ) -> None:
        if len(dataset) == 0:
            return
        for idx, column in enumerate(requested):
            resolved = _resolve_measure_column(dataset, column)
            if not resolved:
                logger.warning("Column '%s' missing in %s data", column, dataset_label)
                continue
            style = styles[idx % len(styles)]
            logger.debug(
                "Plotting %s column '%s' (resolved to '%s')",
                dataset_label,
                column,
                resolved,
            )

            # Only apply errorbar settings if requested (e.g., for accuracy, not confidence)
            plot_errorbar_settings = (
                errorbar_settings if use_errorbars else {"errorbar": None}
            )

            if "seed_id" in dataset.columns:
                for seed_val, seed_data in dataset.groupby("seed_id"):
                    sns.lineplot(
                        data=seed_data,
                        x="time_ms",
                        y=resolved,
                        hue=hue_key if hue_var != "layers" else None,
                        hue_order=hue_values if hue_var != "layers" else None,
                        palette=colors if hue_var != "layers" else None,
                        ax=ax,
                        legend=False,
                        linewidth=fmt["linewidth_main"],
                        alpha=fmt["alpha_line"],
                        **style,
                        **plot_errorbar_settings,
                    )
                    storage.append((column, resolved, style))
            else:
                sns.lineplot(
                    data=dataset,
                    x="time_ms",
                    y=resolved,
                    hue=hue_key if hue_var != "layers" else None,
                    hue_order=hue_values if hue_var != "layers" else None,
                    palette=colors if hue_var != "layers" else None,
                    ax=ax,
                    legend=False,
                    linewidth=fmt["linewidth_main"],
                    alpha=fmt["alpha_line"],
                    **style,
                    **plot_errorbar_settings,
                )
                storage.append((column, resolved, style))

    _plot_series(
        dataset=full_data,
        requested=requested_accuracy,
        styles=accuracy_styles_full,
        storage=plotted_accuracy_full,
        dataset_label="full-model accuracy",
        use_errorbars=True,
    )
    _plot_series(
        dataset=full_data,
        requested=requested_confidence,
        styles=confidence_styles_full,
        storage=plotted_confidence_full,
        dataset_label="full-model confidence",
        use_errorbars=False,  # Don't show error bars for confidence traces
    )

    # Plot feedforward-trained models as full time traces (dashed grey lines)
    if has_feedforward and not feedforward_data.empty:
        _plot_series(
            dataset=feedforward_data,
            requested=requested_accuracy,
            styles=accuracy_styles_feedforward,
            storage=plotted_accuracy_feedforward,
            dataset_label="feedforward-trained accuracy",
            use_errorbars=True,
        )

    def _draw_ffonly_max_markers(
        dataset: pd.DataFrame,
        requested: List[str],
        force_color: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Draw star markers at max accuracy positions.

        Parameters
        ----------
        force_color : str, optional
            If provided, use this color for all markers instead of hue colors.
        """
        markers_drawn: Dict[str, bool] = {}
        if dataset.empty or not requested:
            return markers_drawn

        groups: List[Tuple[Optional[str], pd.DataFrame]]
        if hue_key in dataset.columns and hue_var != "layers":
            groups = [
                (group_val, group_df)
                for group_val, group_df in dataset.groupby(hue_key, observed=True)
            ]
        else:
            groups = [(None, dataset)]

        for column in requested:
            resolved = _resolve_measure_column(dataset, column)
            if not resolved or resolved not in dataset.columns:
                continue

            column_marked = False
            for group_val, group_df in groups:
                valid = group_df.dropna(subset=[resolved])
                if valid.empty:
                    continue
                if "time_ms" not in valid.columns:
                    continue
                trace = valid.groupby("time_ms", observed=True)[resolved].mean()
                if trace.empty:
                    continue
                max_val = trace.max()
                if pd.isna(max_val) or not np.isfinite(max_val):
                    continue

                # Place marker at horizontal center of subplot
                x_min = valid["time_ms"].min()
                x_max = valid["time_ms"].max()
                x_pos = (x_min + x_max) / 2
                if pd.isna(x_pos) or not np.isfinite(x_pos):
                    continue

                # Determine marker color
                if force_color:
                    color = force_color
                elif group_val is None:
                    color_key = None
                    if colors:
                        color = (
                            colors.get(color_key)
                            or colors.get(str(color_key))
                            or colors.get(str(color_key).lower())
                        )
                        if not color:
                            # Fall back to the first palette color for visibility
                            color = next(iter(colors.values()), "black")
                    else:
                        color = "black"
                else:
                    color_key = _standardize_category_value(str(group_val))
                    if colors:
                        color = (
                            colors.get(color_key)
                            or colors.get(str(color_key))
                            or colors.get(str(color_key).lower())
                        )
                        if not color:
                            # Fall back to the first palette color for visibility
                            color = next(iter(colors.values()), "black")
                    else:
                        color = "black"
                ax.scatter(
                    [x_pos],
                    [max_val],
                    color=color,
                    s=fmt.get("markersize_large", 120),  # Increased from 80 to 120
                    marker="^",
                    edgecolors="none",
                    zorder=5,
                    alpha=fmt["alpha_line"],
                )
                logger.debug(
                    "FF-only max marker for '%s' (hue=%s) at %.3f",
                    resolved,
                    group_val,
                    max_val,
                )
                column_marked = True

            if column_marked:
                markers_drawn[column] = True

        return markers_drawn

    if has_ffonly:
        ff_accuracy_markers = _draw_ffonly_max_markers(
            dataset=ffonly_data, requested=requested_accuracy
        )

    # Also draw star markers for feedforward-trained models (grey to match dashed line)
    _: Dict[str, bool] = {}
    if has_feedforward and not feedforward_data.empty:
        _ = _draw_ffonly_max_markers(
            dataset=feedforward_data, requested=requested_accuracy, force_color="grey"
        )

    if not plotted_accuracy_full and not plotted_confidence_full:
        logger.warning(
            "No valid accuracy or confidence columns found after filtering; skipping panel"
        )
        return

    if len(plotted_accuracy_full) > 1:
        base_column, base_resolved, _ = plotted_accuracy_full[0]
        base_values = full_data[base_resolved].to_numpy()
        for column, resolved, _ in plotted_accuracy_full[1:]:
            if np.allclose(
                base_values, full_data[resolved].to_numpy(), equal_nan=True
            ):
                logger.info(
                    "Full-model accuracy measure '%s' matches '%s'; traces will overlap",
                    column,
                    base_column,
                )

    if len(plotted_confidence_full) > 1:
        base_column, base_resolved, _ = plotted_confidence_full[0]
        base_values = full_data[base_resolved].to_numpy()
        for column, resolved, _ in plotted_confidence_full[1:]:
            if np.allclose(
                base_values, full_data[resolved].to_numpy(), equal_nan=True
            ):
                logger.info(
                    "Full-model confidence measure '%s' matches '%s'; traces will overlap",
                    column,
                    base_column,
                )

    if show_ylabel:
        ax.set_ylabel("Accuracy", fontsize=fmt["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")

    ax.set_xlabel("Time (ms)", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    if show_legend:
        legend_elements: List[Line2D] = []
        accuracy_columns_in_plot = [
            column
            for column in requested_accuracy
            if any(entry[0] == column for entry in plotted_accuracy_full)
            or column in ff_accuracy_markers
            or any(entry[0] == column for entry in plotted_accuracy_feedforward)
        ]
        confidence_columns_in_plot = [
            column
            for column in requested_confidence
            if any(entry[0] == column for entry in plotted_confidence_full)
        ]

        single_accuracy = len(accuracy_columns_in_plot) <= 1
        single_confidence = len(confidence_columns_in_plot) <= 1

        for column in accuracy_columns_in_plot:
            style = next(
                (style for col, _, style in plotted_accuracy_full if col == column),
                None,
            )
            label = (
                "Accuracy"
                if single_accuracy and len(accuracy_columns_in_plot) == 1
                else _format_measure_label(column, "Accuracy")
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", "o"),
                    markersize=style.get("markersize", 4),
                    label=label,
                    alpha=fmt["alpha_line"],
                )
            )

            if has_ffonly and column in ff_accuracy_markers:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linewidth=0,
                        linestyle="none",
                        marker="^",
                        markersize=12,
                        label=_append_suffix_to_label(label, "Silenced"),
                        alpha=fmt["alpha_line"],
                    )
                )

            # Add legend entry for feedforward-trained models
            if has_feedforward and any(
                entry[0] == column for entry in plotted_accuracy_feedforward
            ):
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="grey",
                        linewidth=fmt["linewidth_main"],
                        linestyle="--",
                        marker="",
                        label=_append_suffix_to_label(label, "FF-trained"),
                        alpha=fmt["alpha_line"],
                    )
                )

        for column in confidence_columns_in_plot:
            style = next(
                (style for col, _, style in plotted_confidence_full if col == column),
                {},
            )
            label = (
                "Confidence"
                if single_confidence and len(confidence_columns_in_plot) == 1
                else _format_measure_label(column, "Confidence")
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=fmt["linewidth_main"],
                    linestyle=style.get("linestyle", ":"),
                    marker=style.get("marker", "s"),
                    markersize=style.get("markersize", 4),
                    label=label,
                    alpha=fmt["alpha_line"],
                )
            )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                frameon=False,
                fontsize=fmt["fontsize_legend"],
                ncol=1,
            )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True)


def _plot_peak_height_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    row_key: Optional[str],
    row_value: Optional[str],
    subplot_key: str,
    subplot_values: List[str],
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float,
    show_ylabel: bool = False,
    show_legend: bool = True,
    accuracy_cols: Optional[List[str]] = None,
    confidence_cols: Optional[List[str]] = None,
    primary_accuracy: Optional[str] = None,
    primary_confidence: Optional[str] = None,
    ylabel: str = "Max Performance",
    show_seed_errorbars: bool = True,
) -> None:
    """Plot max accuracy and confidence vs subplot value for each hue category.

    Parameters
    ----------
    ylabel : str, default="Max Performance"
        Y-axis label
    show_seed_errorbars : bool, default=True
        If True and multiple seeds present (seed_id or source_file column),
        show error bars indicating std across seeds
    """

    requested_accuracy = list(accuracy_cols or [])
    requested_confidence = list(confidence_cols or [])
    primary_accuracy = primary_accuracy or (
        requested_accuracy[0] if requested_accuracy else None
    )
    primary_confidence = primary_confidence or (
        requested_confidence[0] if requested_confidence else None
    )

    if primary_accuracy is None and primary_confidence is None:
        ax.text(
            0.5,
            0.5,
            "No Performance Measures",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    row_data = data.copy()
    if row_key and row_key in row_data.columns and row_value is not None:
        row_data[row_key] = row_data[row_key].apply(_standardize_category_value)
        row_value_std = _standardize_category_value(str(row_value))
        row_data = row_data[row_data[row_key] == row_value_std]

    if len(row_data) == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return

    accuracy_column = (
        _resolve_measure_column(row_data, primary_accuracy)
        if primary_accuracy is not None
        else None
    )
    if primary_accuracy and not accuracy_column:
        logger.warning(
            "Primary accuracy measure '%s' not found for peak height panel",
            primary_accuracy,
        )

    confidence_column = (
        _resolve_measure_column(row_data, primary_confidence)
        if primary_confidence is not None
        else None
    )
    if primary_confidence and not confidence_column:
        logger.warning(
            "Primary confidence measure '%s' not found for peak height panel",
            primary_confidence,
        )

    has_accuracy = accuracy_column is not None
    has_confidence = confidence_column is not None
    has_ffonly = (
        "model_type" in row_data.columns and "ffonly" in row_data["model_type"].values
    )
    has_feedforward = (
        "model_type" in row_data.columns
        and "feedforward" in row_data["model_type"].values
    )

    if not has_accuracy and not has_confidence:
        ax.text(
            0.5,
            0.5,
            "No Performance Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    logger.debug(
        "Computing max performance metrics using accuracy='%s', confidence='%s'",
        accuracy_column,
        confidence_column,
    )

    if hue_key in row_data.columns:
        row_data[hue_key] = row_data[hue_key].apply(_standardize_category_value)
    if subplot_key in row_data.columns:
        row_data[subplot_key] = row_data[subplot_key].apply(
            _standardize_category_value
        )

    full_data = (
        row_data[row_data.get("model_type", "full") == "full"]
        if "model_type" in row_data.columns
        else row_data
    )
    ffonly_data = (
        row_data[row_data.get("model_type", "full") == "ffonly"]
        if has_ffonly
        else pd.DataFrame()
    )
    feedforward_data = (
        row_data[row_data.get("model_type", "full") == "feedforward"]
        if has_feedforward
        else pd.DataFrame()
    )

    accuracy_plot_data: List[Dict[str, Union[float, str]]] = []
    confidence_plot_data: List[Dict[str, Union[float, str]]] = []
    ff_accuracy_plot_data: List[Dict[str, Union[float, str]]] = []
    feedforward_accuracy_plot_data: List[Dict[str, Union[float, str]]] = []

    # Determine if we have multiple seeds
    seed_col = None
    if "seed_id" in row_data.columns:
        seed_col = "seed_id"
    elif "source_file" in row_data.columns:
        seed_col = "source_file"
    has_seeds = seed_col is not None and show_seed_errorbars

    def _compute_peak_value(subset: pd.DataFrame, column: str) -> Optional[float]:
        if column not in subset.columns:
            return None
        valid_subset = subset.dropna(subset=[column])
        if valid_subset.empty:
            return None
        if "times_index" in valid_subset.columns:
            grouped = valid_subset.groupby("times_index")[column].mean()
            if grouped.empty or grouped.isna().all():
                return None
            peak_val = grouped.max()
        else:
            peak_val = valid_subset[column].mean()
        if peak_val is None or not np.isfinite(peak_val):
            return None
        return float(peak_val)

    def _compute_peak_per_seed(subset: pd.DataFrame, column: str) -> pd.DataFrame:
        """Compute peak value for each seed separately."""
        if column not in subset.columns or seed_col not in subset.columns:
            return pd.DataFrame()

        results = []
        for seed_val, seed_data in subset.groupby(seed_col):
            peak_val = _compute_peak_value(seed_data, column)
            if peak_val is not None:
                results.append({seed_col: seed_val, "peak": peak_val})

        return pd.DataFrame(results)

    accuracy_label = (
        _format_measure_label(primary_accuracy, "Accuracy")
        if primary_accuracy
        else "Accuracy"
    )
    confidence_label = (
        _format_measure_label(primary_confidence, "Confidence")
        if primary_confidence
        else "Confidence"
    )

    for hue_val in hue_values:
        hue_val_std = _standardize_category_value(str(hue_val))

        hue_data_full = (
            full_data[full_data[hue_key] == hue_val_std]
            if hue_key in full_data.columns
            else full_data
        )
        hue_data_ff = (
            ffonly_data[ffonly_data[hue_key] == hue_val_std]
            if hue_key in ffonly_data.columns and not ffonly_data.empty
            else ffonly_data
        )
        hue_data_feedforward = (
            feedforward_data[feedforward_data[hue_key] == hue_val_std]
            if hue_key in feedforward_data.columns and not feedforward_data.empty
            else feedforward_data
        )

        for subplot_val in subplot_values:
            subplot_val_std = _standardize_category_value(str(subplot_val))
            subset_full = (
                hue_data_full[hue_data_full[subplot_key] == subplot_val_std]
                if subplot_key in hue_data_full.columns
                else hue_data_full
            )
            subset_ff = (
                hue_data_ff[hue_data_ff[subplot_key] == subplot_val_std]
                if subplot_key in hue_data_ff.columns and not hue_data_ff.empty
                else hue_data_ff
            )
            subset_feedforward = (
                hue_data_feedforward[
                    hue_data_feedforward[subplot_key] == subplot_val_std
                ]
                if subplot_key in hue_data_feedforward.columns
                and not hue_data_feedforward.empty
                else hue_data_feedforward
            )

            try:
                x_val = float(subplot_val)
            except (ValueError, TypeError):
                x_val = subplot_values.index(subplot_val)

            if len(subset_full) > 0:
                if has_accuracy and accuracy_column:
                    if has_seeds:
                        # Compute peak per seed and add one row per seed
                        seed_peaks = _compute_peak_per_seed(
                            subset_full, accuracy_column
                        )
                        for _, seed_row in seed_peaks.iterrows():
                            accuracy_plot_data.append(
                                {
                                    "x": x_val,
                                    "y": seed_row["peak"],
                                    "hue": hue_val,
                                    "subplot_val": subplot_val,
                                    "metric": accuracy_label,
                                    "model_type": "full",
                                }
                            )
                    else:
                        # Compute single aggregated peak
                        peak_val = _compute_peak_value(subset_full, accuracy_column)
                        if peak_val is not None:
                            accuracy_plot_data.append(
                                {
                                    "x": x_val,
                                    "y": peak_val,
                                    "hue": hue_val,
                                    "subplot_val": subplot_val,
                                    "metric": accuracy_label,
                                    "model_type": "full",
                                }
                            )

                if has_confidence and confidence_column:
                    # Always compute single aggregated peak for confidence (no seed-level error bars)
                    peak_val = _compute_peak_value(subset_full, confidence_column)
                    if peak_val is not None:
                        confidence_plot_data.append(
                            {
                                "x": x_val,
                                "y": peak_val,
                                "hue": hue_val,
                                "subplot_val": subplot_val,
                                "metric": confidence_label,
                                "model_type": "full",
                            }
                        )

            if has_ffonly and len(subset_ff) > 0 and has_accuracy and accuracy_column:
                if has_seeds:
                    # Compute peak per seed and add one row per seed
                    seed_peaks = _compute_peak_per_seed(subset_ff, accuracy_column)
                    for _, seed_row in seed_peaks.iterrows():
                        ff_accuracy_plot_data.append(
                            {
                                "x": x_val,
                                "y": seed_row["peak"],
                                "hue": hue_val,
                                "subplot_val": subplot_val,
                                "metric": accuracy_label,
                                "model_type": "ffonly",
                            }
                        )
                else:
                    # Compute single aggregated peak
                    peak_val = _compute_peak_value(subset_ff, accuracy_column)
                    if peak_val is not None:
                        ff_accuracy_plot_data.append(
                            {
                                "x": x_val,
                                "y": peak_val,
                                "hue": hue_val,
                                "subplot_val": subplot_val,
                                "metric": accuracy_label,
                                "model_type": "ffonly",
                            }
                        )

            # Feedforward-trained model data collection
            if (
                has_feedforward
                and len(subset_feedforward) > 0
                and has_accuracy
                and accuracy_column
            ):
                if has_seeds:
                    # Compute peak per seed and add one row per seed
                    seed_peaks = _compute_peak_per_seed(
                        subset_feedforward, accuracy_column
                    )
                    for _, seed_row in seed_peaks.iterrows():
                        feedforward_accuracy_plot_data.append(
                            {
                                "x": x_val,
                                "y": seed_row["peak"],
                                "hue": hue_val,
                                "subplot_val": subplot_val,
                                "metric": accuracy_label,
                                "model_type": "feedforward",
                            }
                        )
                else:
                    # Compute single aggregated peak
                    peak_val = _compute_peak_value(subset_feedforward, accuracy_column)
                    if peak_val is not None:
                        feedforward_accuracy_plot_data.append(
                            {
                                "x": x_val,
                                "y": peak_val,
                                "hue": hue_val,
                                "subplot_val": subplot_val,
                                "metric": accuracy_label,
                                "model_type": "feedforward",
                            }
                        )

    if (
        not accuracy_plot_data
        and not confidence_plot_data
        and not ff_accuracy_plot_data
        and not feedforward_accuracy_plot_data
    ):
        ax.text(
            0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes
        )
        return

    if accuracy_plot_data:
        accuracy_df = pd.DataFrame(accuracy_plot_data).dropna(subset=["x", "y"])
        if not accuracy_df.empty:
            # Use std error bars when showing seed-level data, otherwise use SEM
            errorbar_config = (
                {"errorbar": ("sd", 1), "err_style": "bars"}
                if has_seeds
                else ERRORBAR_CONFIG
            )
            sns.lineplot(
                data=accuracy_df,
                x="x",
                y="y",
                hue="hue",
                hue_order=hue_values,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=FORMATTING["linewidth_main"],
                marker="o",
                markersize=6,
                alpha=FORMATTING["alpha_line"],
                linestyle="-",
                **errorbar_config,
            )

    if has_ffonly and ff_accuracy_plot_data:
        ff_df = pd.DataFrame(ff_accuracy_plot_data).dropna(subset=["x", "y"])
        if not ff_df.empty:
            # Use std error bars when showing seed-level data, otherwise use SEM
            errorbar_config = (
                {"errorbar": ("sd", 1), "err_style": "bars"}
                if has_seeds
                else ERRORBAR_CONFIG
            )
            sns.lineplot(
                data=ff_df,
                x="x",
                y="y",
                hue="hue",
                hue_order=hue_values,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=FORMATTING["linewidth_main"],
                marker="o",
                markersize=6,
                alpha=FORMATTING["alpha_line"],
                linestyle="--",
                **errorbar_config,
            )

    # Plot feedforward-trained models as grey dashed lines with star markers
    if has_feedforward and feedforward_accuracy_plot_data:
        feedforward_df = pd.DataFrame(feedforward_accuracy_plot_data).dropna(
            subset=["x", "y"]
        )
        if not feedforward_df.empty:
            # Use std error bars when showing seed-level data, otherwise use SEM
            errorbar_config = (
                {"errorbar": ("sd", 1), "err_style": "bars"}
                if has_seeds
                else ERRORBAR_CONFIG
            )
            sns.lineplot(
                data=feedforward_df,
                x="x",
                y="y",
                hue="hue",
                hue_order=hue_values,
                palette={val: "grey" for val in hue_values},  # Grey for all hue values
                ax=ax,
                legend=False,
                linewidth=FORMATTING["linewidth_main"],
                marker="^",
                markersize=10,
                alpha=FORMATTING["alpha_line"],
                linestyle="--",
                **errorbar_config,
            )

    if confidence_plot_data:
        confidence_df = pd.DataFrame(confidence_plot_data).dropna(subset=["x", "y"])
        if not confidence_df.empty:
            sns.lineplot(
                data=confidence_df,
                x="x",
                y="y",
                hue="hue",
                hue_order=hue_values,
                palette=colors,
                ax=ax,
                legend=False,
                linewidth=FORMATTING["linewidth_main"],
                marker="s",
                markersize=4,
                alpha=FORMATTING["alpha_line"],
                linestyle=":",
                errorbar=None,  # No error bars for confidence
            )

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=FORMATTING["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")

    subplot_display = get_display_name(subplot_key, config)
    ax.set_xlabel(subplot_display, fontsize=FORMATTING["fontsize_axis"])
    ax.tick_params(labelsize=FORMATTING["fontsize_tick"])

    try:
        ymin, ymax = ax.get_ylim()
        if not (np.isfinite(ymin) and np.isfinite(ymax)):
            ax.set_ylim(0, 1)
    except Exception:
        ax.set_ylim(0, 1)

    try:
        [float(v) for v in subplot_values]
    except (ValueError, TypeError):
        ax.set_xticks(range(len(subplot_values)))
        ax.set_xticklabels(
            [_format_legend_label(subplot_key, v, config, dt) for v in subplot_values],
            rotation=45,
            ha="right",
        )

    legend_elements: List[Line2D] = []
    if show_legend and accuracy_plot_data:
        legend_text = f"Max {accuracy_label}"
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="o",
                markersize=6,
                linestyle="-",
                label=legend_text,
                alpha=FORMATTING["alpha_line"],
            )
        )
        if has_ffonly and ff_accuracy_plot_data:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=FORMATTING["linewidth_main"],
                    marker="^",
                    markersize=6,
                    linestyle="--",
                    label=_append_suffix_to_label(legend_text, "Feedforward"),
                    alpha=FORMATTING["alpha_line"],
                )
            )

    if show_legend and confidence_plot_data:
        # Only add specifics to label if there are multiple confidence measures
        if len(requested_confidence) > 1:
            conf_legend_label = f"Max {confidence_label}"
        else:
            conf_legend_label = "Max Confidence"

        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="s",
                markersize=4,
                linestyle=":",
                label=conf_legend_label,
                alpha=FORMATTING["alpha_line"],
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0, 1 + PERFORMANCE_LAYOUT["panel_letter_offset_y"]),
            frameon=False,
            fontsize=FORMATTING["fontsize_legend"],
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def _plot_peak_time_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    row_key: Optional[str],
    row_value: Optional[str],
    subplot_key: str,
    subplot_values: List[str],
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float,
    show_ylabel: bool = False,
    show_legend: bool = True,
    accuracy_cols: Optional[List[str]] = None,
    confidence_cols: Optional[List[str]] = None,
    primary_accuracy: Optional[str] = None,
    primary_confidence: Optional[str] = None,
) -> None:
    """Plot time of max accuracy and confidence vs subplot value for each hue category."""

    requested_accuracy = list(accuracy_cols or [])
    requested_confidence = list(confidence_cols or [])
    primary_accuracy = primary_accuracy or (
        requested_accuracy[0] if requested_accuracy else None
    )
    primary_confidence = primary_confidence or (
        requested_confidence[0] if requested_confidence else None
    )

    if primary_accuracy is None and primary_confidence is None:
        ax.text(
            0.5,
            0.5,
            "No Performance Measures",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    row_data = data.copy()
    if row_key and row_key in row_data.columns and row_value is not None:
        row_data[row_key] = row_data[row_key].apply(_standardize_category_value)
        row_value_std = _standardize_category_value(str(row_value))
        row_data = row_data[row_data[row_key] == row_value_std]

    if len(row_data) == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return

    accuracy_column = (
        _resolve_measure_column(row_data, primary_accuracy)
        if primary_accuracy is not None
        else None
    )
    if primary_accuracy and not accuracy_column:
        logger.warning(
            "Primary accuracy measure '%s' not found for peak time panel",
            primary_accuracy,
        )

    confidence_column = (
        _resolve_measure_column(row_data, primary_confidence)
        if primary_confidence is not None
        else None
    )
    if primary_confidence and not confidence_column:
        logger.warning(
            "Primary confidence measure '%s' not found for peak time panel",
            primary_confidence,
        )

    has_accuracy = accuracy_column is not None
    has_confidence = confidence_column is not None

    if not has_accuracy and not has_confidence:
        ax.text(
            0.5,
            0.5,
            "No Performance Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    if hue_key in row_data.columns:
        row_data[hue_key] = row_data[hue_key].apply(_standardize_category_value)
    if subplot_key in row_data.columns:
        row_data[subplot_key] = row_data[subplot_key].apply(
            _standardize_category_value
        )

    accuracy_plot_data: List[Dict[str, Union[float, str]]] = []
    confidence_plot_data: List[Dict[str, Union[float, str]]] = []

    def _compute_peak_time(subset: pd.DataFrame, column: str) -> Optional[float]:
        if column not in subset.columns or "times_index" not in subset.columns:
            return None
        valid_subset = subset.dropna(subset=[column, "times_index"])
        if valid_subset.empty:
            return None
        grouped = valid_subset.groupby("times_index")[column].mean()
        if grouped.empty or grouped.isna().all():
            return None
        peak_index = grouped.idxmax()
        if peak_index is None or not np.isfinite(peak_index):
            return None
        return float(peak_index) * dt

    accuracy_label = (
        _format_measure_label(primary_accuracy, "Accuracy")
        if primary_accuracy
        else "Accuracy"
    )
    confidence_label = (
        _format_measure_label(primary_confidence, "Confidence")
        if primary_confidence
        else "Confidence"
    )

    for hue_val in hue_values:
        hue_val_std = _standardize_category_value(str(hue_val))
        hue_data = (
            row_data[row_data[hue_key] == hue_val_std]
            if hue_key in row_data.columns
            else row_data
        )

        for subplot_val in subplot_values:
            subplot_val_std = _standardize_category_value(str(subplot_val))
            subset = (
                hue_data[hue_data[subplot_key] == subplot_val_std]
                if subplot_key in hue_data.columns
                else hue_data
            )

            if len(subset) == 0:
                continue

            try:
                x_val = float(subplot_val)
            except (ValueError, TypeError):
                x_val = subplot_values.index(subplot_val)

            if has_accuracy and accuracy_column:
                peak_time = _compute_peak_time(subset, accuracy_column)
                if peak_time is not None:
                    accuracy_plot_data.append(
                        {
                            "x": x_val,
                            "y": peak_time,
                            "hue": hue_val,
                            "subplot_val": subplot_val,
                            "metric": accuracy_label,
                        }
                    )

            if has_confidence and confidence_column:
                peak_time = _compute_peak_time(subset, confidence_column)
                if peak_time is not None:
                    confidence_plot_data.append(
                        {
                            "x": x_val,
                            "y": peak_time,
                            "hue": hue_val,
                            "subplot_val": subplot_val,
                            "metric": confidence_label,
                        }
                    )

    if not accuracy_plot_data and not confidence_plot_data:
        ax.text(
            0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes
        )
        return

    if accuracy_plot_data:
        accuracy_df = pd.DataFrame(accuracy_plot_data)
        sns.lineplot(
            data=accuracy_df,
            x="x",
            y="y",
            hue="hue",
            hue_order=hue_values,
            palette=colors,
            ax=ax,
            legend=False,
            linewidth=FORMATTING["linewidth_main"],
            marker="o",
            markersize=6,
            alpha=FORMATTING["alpha_line"],
            linestyle="-",
            **ERRORBAR_CONFIG,
        )

    if confidence_plot_data:
        confidence_df = pd.DataFrame(confidence_plot_data)
        sns.lineplot(
            data=confidence_df,
            x="x",
            y="y",
            hue="hue",
            hue_order=hue_values,
            palette=colors,
            ax=ax,
            legend=False,
            linewidth=FORMATTING["linewidth_main"],
            marker="s",
            markersize=4,
            alpha=FORMATTING["alpha_line"],
            linestyle=":",
            **ERRORBAR_CONFIG,
        )

    if show_ylabel:
        ax.set_ylabel(
            "Peak Time (ms)", fontsize=FORMATTING["fontsize_axis"], fontweight="bold"
        )
    else:
        ax.set_ylabel("")

    subplot_display = get_display_name(subplot_key, config)
    ax.set_xlabel(subplot_display, fontsize=FORMATTING["fontsize_axis"])
    ax.tick_params(labelsize=FORMATTING["fontsize_tick"])

    try:
        [float(v) for v in subplot_values]
    except (ValueError, TypeError):
        ax.set_xticks(range(len(subplot_values)))
        ax.set_xticklabels(
            [_format_legend_label(subplot_key, v, config, dt) for v in subplot_values],
            rotation=45,
            ha="right",
        )

    legend_elements: List[Line2D] = []
    if show_legend and accuracy_plot_data:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="o",
                markersize=6,
                linestyle="-",
                label=f"Peak Time ({accuracy_label})",
                alpha=FORMATTING["alpha_line"],
            )
        )

    if show_legend and confidence_plot_data:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=FORMATTING["linewidth_main"],
                marker="s",
                markersize=4,
                linestyle=":",
                label=f"Peak Time ({confidence_label})",
                alpha=FORMATTING["alpha_line"],
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0, 1 + PERFORMANCE_LAYOUT["panel_letter_offset_y"]),
            frameon=False,
            fontsize=FORMATTING["fontsize_legend"],
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def _add_panel_letters(
    fig: plt.Figure,
    perf_axes: List[List[plt.Axes]],
    peak_axes: List[plt.Axes],
    peak_time_axes: List[plt.Axes],
    layout: Dict,
) -> None:
    """Add panel letters A), B), C) to performance, peak, and peak time panels."""
    fmt = FORMATTING

    # Get positions from the first row of axes
    if not perf_axes or not perf_axes[0]:
        return

    # Panel A) - Above first performance subplot
    first_perf_ax = perf_axes[0][0]
    first_perf_ax.text(
        layout["panel_letter_offset_x"],
        1 + layout["panel_letter_offset_y"],
        "A)",
        fontsize=fmt["fontsize_panel_label"],
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=first_perf_ax.transAxes,
    )

    # Panel B) - Above peak height panel
    if peak_axes:
        peak_ax = peak_axes[0]
        _ = peak_ax.get_position()
        peak_ax.text(
            layout["panel_letter_offset_x"],
            1 + layout["panel_letter_offset_y"],
            "B)",
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=peak_ax.transAxes,
        )

    # Panel C) - Above peak time panel
    if peak_time_axes:
        peak_time_ax = peak_time_axes[0]
        _ = peak_time_ax.get_position()
        peak_time_ax.text(
            layout["panel_letter_offset_x"],
            1 + layout["panel_letter_offset_y"],
            "C)",
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=peak_time_ax.transAxes,
        )


def _add_hue_legend(
    fig: plt.Figure,
    perf_axes: List[List[plt.Axes]],
    hue_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float,
    show_without_recurrence: bool = False,
) -> None:
    """Add hue legend to the right of accuracy/confidence legend.

    Parameters
    ----------
    show_without_recurrence : bool, default=False
        If True, add a "without recurrence" entry showing star marker + dashed line
        in black to indicate ffonly and feedforward-trained models.
    """
    if not hue_values or len(hue_values) <= 1 or not perf_axes or not perf_axes[0]:
        return

    # Place on first subplot of first row
    legend_ax = perf_axes[0][0]

    # Create legend elements
    legend_elements = []
    for hue_val in hue_values:
        color = colors.get(hue_val, "#000000")
        label = _format_legend_label(hue_key, hue_val, config, dt)
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=FORMATTING["linewidth_main"],
                label=label,
                alpha=FORMATTING["alpha_line"],
            )
        )

    # Add "without recurrence" entry if requested
    if show_without_recurrence:
        # Create a combined legend entry showing star marker + dashed line
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="0.1",
                linewidth=FORMATTING["linewidth_main"],
                linestyle="--",
                marker="^",
                markersize=10,
                label="Without Recurrence",
                alpha=FORMATTING["alpha_line"],
            )
        )

    # Create hue legend
    hue_legend = legend_ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=False,
        fontsize=FORMATTING["fontsize_legend"],
        ncol=1,  # Vertical layout
        title=get_display_name(hue_key, config),
        title_fontsize=FORMATTING["fontsize_legend"],
    )

    # Make the legend title bold
    hue_legend.get_title().set_fontweight("bold")

    # Add the hue legend as an artist (preserves the existing accuracy/confidence legend)
    legend_ax.add_artist(hue_legend)


def plot_performance_grid(
    data_paths: List[Path],
    output: Path,
    subplot_var: str,
    row_var: Optional[str],
    hue_var: str,
    category_key: Optional[str] = None,
    parameter_key: Optional[str] = None,
    experiment_names: Optional[List[str]] = None,
    data_ffonly_paths: Optional[List[Path]] = None,
    confidence_measure: Optional[Union[str, List[str]]] = "first_label_confidence",
    accuracy_measure: Optional[Union[str, List[str]]] = "accuracy",
    dt: float = 2.0,
    time_offset: float = 0.0,
    config: Optional[Dict] = None,
    subplot_filter: Optional[List[str]] = None,
    plot_individual_seeds: bool = False,
    **kwargs,
) -> None:
    """Plot performance traces in a grid layout with flexible dimension mapping.

    Parameters
    ----------
    plot_individual_seeds : bool, default=False
        If True and multiple seed files are provided, plot each seed as a separate
        trace with the same color and linestyle instead of averaging over seeds.
    """
    logger.info("=" * 60)
    logger.info("Starting performance grid plotting")
    logger.info("=" * 60)

    # Normalize dimensions and validate uniqueness
    raw_subplot_var = subplot_var
    raw_row_var = row_var
    raw_hue_var = hue_var

    subplot_clean = _coerce_optional_dimension(subplot_var)
    row_clean = _coerce_optional_dimension(row_var)
    hue_clean = _coerce_optional_dimension(hue_var)

    subplot_var, subplot_limit = _normalize_dimension(subplot_clean)
    row_var, row_limit = _normalize_dimension(row_clean)
    hue_var, hue_limit = _normalize_dimension(hue_clean)

    logger.info(
        "Dimension mapping: subplot=%s (base=%s, limit=%s), row=%s (base=%s, limit=%s), hue=%s (base=%s, limit=%s)",
        raw_subplot_var,
        subplot_var,
        subplot_limit,
        raw_row_var,
        row_var,
        row_limit,
        raw_hue_var,
        hue_var,
        hue_limit,
    )

    if subplot_var is None or hue_var is None:
        raise ValueError("subplot and hue dimensions cannot be empty")

    _validate_dimension_choices(
        subplot_var=subplot_var, hue_var=hue_var, column_var=row_var
    )

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Parse accuracy and confidence selections
    accuracy_measures = _coerce_measure_list(accuracy_measure, default="accuracy")
    confidence_measures = _coerce_measure_list(
        confidence_measure, default="first_label_confidence"
    )

    logger.info(
        "Selected accuracy measures: %s",
        accuracy_measures if accuracy_measures else "[none]",
    )
    logger.info(
        "Selected confidence measures: %s",
        confidence_measures if confidence_measures else "[none]",
    )

    primary_accuracy_measure: Optional[str] = (
        accuracy_measures[0] if accuracy_measures else None
    )
    primary_confidence_measure: Optional[str] = (
        confidence_measures[0] if confidence_measures else None
    )

    # Load and combine data from multiple files
    logger.info(f"Loading data from {len(data_paths)} full model files...")
    combined_data = []

    experiment_names = experiment_names or []
    use_experiment_dimension = "experiment" in {
        dim for dim in (subplot_var, row_var, hue_var) if dim is not None
    }
    if not use_experiment_dimension and len(data_paths) > 1:
        logger.info(
            "Dimension selection excludes 'experiment'; combining %d inputs as replicates",
            len(data_paths),
        )

    combined_label = (
        experiment_names[0]
        if experiment_names
        else (data_paths[0].stem if data_paths else "combined")
    )

    for i, data_path in enumerate(data_paths):
        logger.info(f"Loading full model file {i+1}/{len(data_paths)}: {data_path}")
        df = pd.read_csv(data_path)

        # Add experiment identifier
        if use_experiment_dimension:
            if experiment_names and i < len(experiment_names):
                experiment_name = experiment_names[i]
            elif experiment_names:
                experiment_name = experiment_names[0]
            else:
                experiment_name = data_path.stem
        else:
            experiment_name = combined_label

        df["experiment"] = experiment_name
        df["model_type"] = "full"  # Mark as full model
        df["source_file"] = data_path.stem

        # Add seed identifier if plotting individual seeds
        if plot_individual_seeds:
            df["seed_id"] = i

        combined_data.append(df)
        logger.info(
            f"Loaded {len(df)} rows for full model experiment '{experiment_name}'"
        )

    # Load feedforward-only data if provided and available
    if data_ffonly_paths:
        valid_ffonly_paths: List[Tuple[Path, Optional[str]]] = []
        for idx, raw_path in enumerate(data_ffonly_paths):
            if raw_path is None:
                continue
            path_obj = Path(raw_path)
            if path_obj.exists():
                if use_experiment_dimension:
                    if experiment_names and idx < len(experiment_names):
                        exp_name = experiment_names[idx]
                    elif experiment_names:
                        exp_name = experiment_names[0]
                    else:
                        exp_name = path_obj.stem
                else:
                    exp_name = combined_label
                valid_ffonly_paths.append((path_obj, exp_name))
            else:
                logger.warning(
                    "Feedforward-only data path '%s' does not exist; skipping",
                    path_obj,
                )

        if valid_ffonly_paths:
            logger.info(
                "Loading data from %d feedforward-only files...",
                len(valid_ffonly_paths),
            )

            for i, (data_path, experiment_name) in enumerate(valid_ffonly_paths):
                logger.info(
                    "Loading feedforward-only file %d/%d: %s",
                    i + 1,
                    len(valid_ffonly_paths),
                    data_path,
                )
                df = pd.read_csv(data_path)

                df["experiment"] = experiment_name
                df["model_type"] = "ffonly"  # Mark as feedforward-only model
                df["source_file"] = data_path.stem

                # Add seed identifier if plotting individual seeds
                if plot_individual_seeds:
                    df["seed_id"] = i

                combined_data.append(df)
                logger.info(
                    "Loaded %d rows for feedforward-only experiment '%s'",
                    len(df),
                    experiment_name,
                )
        else:
            logger.info(
                "No existing feedforward-only data paths found; continuing without them"
            )

    # Combine all data
    df = pd.concat(combined_data, ignore_index=True)
    logger.info(f"Combined data: {len(df)} rows, {len(df.columns)} columns")

    # Get dimension keys
    subplot_key = _get_dimension_key(
        dimension=subplot_var, category_key=category_key, parameter_key=parameter_key
    )
    row_key = (
        _get_dimension_key(
            dimension=row_var, category_key=category_key, parameter_key=parameter_key
        )
        if row_var is not None
        else None
    )
    hue_key = _get_dimension_key(
        dimension=hue_var, category_key=category_key, parameter_key=parameter_key
    )
    logger.info(
        f"Keys: subplot_key='{subplot_key}', row_key='{row_key}', hue_key='{hue_key}'"
    )

    # Extract dimension values
    subplot_values_all = _extract_dimension_values(
        df=df,
        dimension=subplot_var,
        dimension_key=subplot_key,
        config=config,
        dimension_limit=subplot_limit,
    )
    if row_var is None:
        row_values = [None]
    else:
        row_values = _extract_dimension_values(
            df=df,
            dimension=row_var,
            dimension_key=row_key,
            config=config,
            dimension_limit=row_limit,
        )
    hue_values = _extract_dimension_values(
        df=df,
        dimension=hue_var,
        dimension_key=hue_key,
        config=config,
        dimension_limit=hue_limit,
    )

    # Filter subplot values for display if filter is provided
    if subplot_filter:
        subplot_values_display = [v for v in subplot_values_all if v in subplot_filter]
        logger.info(f"Applied subplot filter: {subplot_filter}")
    else:
        subplot_values_display = subplot_values_all

    if not subplot_values_display or not row_values or not hue_values:
        raise ValueError("No valid dimension values found")

    logger.info(
        f"Display subplot values ({len(subplot_values_display)}): {subplot_values_display}"
    )
    logger.info(
        f"All subplot values ({len(subplot_values_all)}): {subplot_values_all}"
    )
    logger.info(f"Row values ({len(row_values)}): {row_values}")
    logger.info(f"Hue values ({len(hue_values)}): {hue_values}")

    # Get colors for hue dimension
    colors = _get_colors_for_dimension(
        values=hue_values, dimension_key=hue_key, config=config
    )

    # Calculate figure size and layout
    layout = {
        **PERFORMANCE_LAYOUT,
        **{k: v for k, v in kwargs.items() if k in PERFORMANCE_LAYOUT},
    }

    n_subplots_display = len(subplot_values_display)
    n_rows = len(row_values)

    # Calculate total figure width using absolute spacing
    performance_section_width = (
        n_subplots_display * layout["subplot_width"]
        + (n_subplots_display - 1) * layout["subplot_spacing_x"]
    )

    fig_width = (
        layout["left_margin"]
        + performance_section_width
        + layout["peak_panel_spacing"]
        + layout["peak_panel_width"]
        + layout["peak_time_panel_spacing"]
        + layout["peak_time_panel_width"]
        + layout["right_margin"]
    )

    fig_height = (
        layout["top_margin"]
        + n_rows * layout["subplot_height"]
        + (n_rows - 1) * layout["subplot_spacing_y"]
        + layout["bottom_margin"]
    )

    logger.info(
        f'Creating figure: {fig_width:.2f}" x {fig_height:.2f}" ({n_subplots_display} performance + 1 peak + 1 peak time panel × {n_rows})'
    )

    # Create figure and calculate positions manually for better control
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Calculate absolute positions for each panel type
    def calculate_panel_positions():
        """Calculate absolute positions for all panels."""
        positions = {"performance": [], "peak_height": [], "peak_time": []}

        # Performance panels positions
        perf_left_start = layout["left_margin"]
        for row_idx in range(n_rows):
            row_positions = []
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )

            for col_idx in range(n_subplots_display):
                left = perf_left_start + col_idx * (
                    layout["subplot_width"] + layout["subplot_spacing_x"]
                )
                row_positions.append(
                    [
                        left / fig_width,  # Convert to relative coordinates
                        row_bottom / fig_height,
                        layout["subplot_width"] / fig_width,
                        layout["subplot_height"] / fig_height,
                    ]
                )
            positions["performance"].append(row_positions)

        # Peak height panels positions
        peak_left = (
            perf_left_start + performance_section_width + layout["peak_panel_spacing"]
        )

        for row_idx in range(n_rows):
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )
            positions["peak_height"].append(
                [
                    peak_left / fig_width,
                    row_bottom / fig_height,
                    layout["peak_panel_width"] / fig_width,
                    layout["subplot_height"] / fig_height,
                ]
            )

        # Peak time panels positions
        peak_time_left = (
            peak_left + layout["peak_panel_width"] + layout["peak_time_panel_spacing"]
        )

        for row_idx in range(n_rows):
            row_bottom = layout["bottom_margin"] + (n_rows - 1 - row_idx) * (
                layout["subplot_height"] + layout["subplot_spacing_y"]
            )
            positions["peak_time"].append(
                [
                    peak_time_left / fig_width,
                    row_bottom / fig_height,
                    layout["peak_time_panel_width"] / fig_width,
                    layout["subplot_height"] / fig_height,
                ]
            )

        return positions

    # Get all panel positions
    panel_positions = calculate_panel_positions()

    # Create axes using calculated positions
    perf_axes = []
    peak_axes = []
    peak_time_axes = []

    for row_idx in range(n_rows):
        # Performance axes for this row
        perf_row = []
        for col_idx in range(n_subplots_display):
            pos = panel_positions["performance"][row_idx][col_idx]
            ax = fig.add_axes(pos)
            perf_row.append(ax)
        perf_axes.append(perf_row)

        # Peak height axis for this row
        pos = panel_positions["peak_height"][row_idx]
        peak_ax = fig.add_axes(pos)
        peak_axes.append(peak_ax)

        # Peak time axis for this row
        pos = panel_positions["peak_time"][row_idx]
        peak_time_ax = fig.add_axes(pos)
        peak_time_axes.append(peak_time_ax)

    # Plot performance panels (filtered subplot values)
    for row_idx, row_value in enumerate(row_values):
        for col_idx, subplot_value in enumerate(subplot_values_display):
            ax = perf_axes[row_idx][col_idx]

            logger.debug(
                f"Plotting performance [{row_idx}, {col_idx}]: {row_key}={row_value}, {subplot_key}={subplot_value}"
            )

            cell_data = _filter_data_for_cell(
                df=df,
                row_key=row_key,
                row_value=row_value,
                subplot_key=subplot_key,
                subplot_value=subplot_value,
            )

            logger.debug(f"  Cell data: {len(cell_data)} rows")

            if len(cell_data) == 0:
                logger.warning(f"No data for cell [{row_idx}, {col_idx}]")
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Plot accuracy panel with hue coloring (now includes both full and ffonly data)
            _plot_accuracy_panel_with_ffonly(
                ax=ax,
                data=cell_data,
                hue_var=hue_var,
                hue_key=hue_key,
                hue_values=hue_values,
                colors=colors,
                dt=dt,
                time_offset=time_offset,
                show_ylabel=(col_idx == 0),
                show_legend=(
                    row_idx == 0 and col_idx == 0
                ),  # Legend only in first subplot
                accuracy_cols=accuracy_measures,
                confidence_cols=confidence_measures,
                **kwargs,
            )
            if col_idx == 0:
                sns.despine(ax=ax, left=False)

            # Add subplot title (only on top row)
            if row_idx == 0:
                title_text = get_display_name(subplot_key, config)
                title_text += " = " + _format_legend_label(
                    subplot_key, subplot_value, config, dt
                )
                ax.set_title(title_text, fontsize=FORMATTING["fontsize_axis"], pad=10)

        if row_var is not None:
            # Get experiment name from the row value based on row_var
            if row_var == "experiment":
                experiment_name = row_value
            else:
                experiment_name = (
                    experiment_names[0]
                    if experiment_names
                    else get_display_name(row_key or "row", config)
                )

            row_label_text = get_display_name(f"{experiment_name}_experiment", config)
            if not row_label_text or row_label_text == f"{experiment_name}_experiment":
                formatted_name = str(experiment_name) if experiment_name else ""
                row_label_text = formatted_name.replace("_", " ").title()

            perf_axes[row_idx][0].text(
                -0.5,
                0.5,
                row_label_text,
                rotation=90,
                ha="center",
                va="center",
                fontsize=FORMATTING["fontsize_axis"],
                fontweight="bold",
                transform=perf_axes[row_idx][0].transAxes,
            )

    # Plot peak height panels (using all subplot values for analysis)
    for row_idx, row_value in enumerate(row_values):
        ax = peak_axes[row_idx]

        logger.debug(f"Plotting peak height [{row_idx}]: {row_key}={row_value}")

        _plot_peak_height_panel(
            ax=ax,
            data=df,
            row_key=row_key,
            row_value=row_value,
            subplot_key=subplot_key,
            subplot_values=subplot_values_all,  # Use all values for comprehensive analysis
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            config=config,
            dt=dt,
            show_ylabel=(row_idx == 0),  # Only show ylabel on first row
            show_legend=(row_idx == 0),
            accuracy_cols=accuracy_measures,
            confidence_cols=confidence_measures,
            primary_accuracy=primary_accuracy_measure,
            primary_confidence=primary_confidence_measure,
        )

    # Plot peak time panels (using all subplot values for analysis)
    for row_idx, row_value in enumerate(row_values):
        ax = peak_time_axes[row_idx]

        logger.debug(f"Plotting peak time [{row_idx}]: {row_key}={row_value}")

        _plot_peak_time_panel(
            ax=ax,
            data=df,
            row_key=row_key,
            row_value=row_value,
            subplot_key=subplot_key,
            subplot_values=subplot_values_all,  # Use all values for comprehensive analysis
            hue_key=hue_key,
            hue_values=hue_values,
            colors=colors,
            config=config,
            dt=dt,
            show_ylabel=(row_idx == 0),  # Only show ylabel on first row
            show_legend=(row_idx == 0),
            accuracy_cols=accuracy_measures,
            confidence_cols=confidence_measures,
            primary_accuracy=primary_accuracy_measure,
            primary_confidence=primary_confidence_measure,
        )

    # Synchronize y-limits and y-ticks across performance panels (Panel A) and peak height panels (Panel B)
    all_perf_axes = [ax for row in perf_axes for ax in row]
    all_peak_axes = peak_axes.copy()

    # Combine performance and peak height axes to determine global y-limits
    all_performance_related_axes = all_perf_axes + all_peak_axes

    # Get global y-limits from all performance-related panels
    try:
        global_ymin = min(axis.get_ylim()[0] for axis in all_performance_related_axes)
        global_ymax = max(axis.get_ylim()[1] for axis in all_performance_related_axes)
    except ValueError:
        # Fallback if no valid limits found
        global_ymin, global_ymax = 0.0, 1.0

    # Ensure reasonable bounds for performance data
    global_ymin = max(global_ymin, 0.0)  # Performance shouldn't go below 0
    global_ymax = min(global_ymax, 1.0)  # Performance shouldn't go above 1

    global_y_range = global_ymax - global_ymin

    # Determine appropriate tick spacing
    if global_y_range <= 0.2:
        tick_step = 0.05
    elif global_y_range <= 0.5:
        tick_step = 0.1
    else:
        tick_step = 0.2

    # Calculate y-ticks
    yticks = []
    tick = 0.0
    while tick <= 1.0:
        if tick >= global_ymin and tick <= global_ymax:
            yticks.append(tick)
        tick += tick_step

    # Apply consistent y-limits and y-ticks to performance panels and peak height panels
    for ax in all_performance_related_axes:
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_yticks(yticks)

    # Hide y-tick labels for all performance panels except the leftmost column
    for row_idx in range(n_rows):
        for col_idx in range(n_subplots_display):
            ax = perf_axes[row_idx][col_idx]
            if col_idx > 0:  # Hide labels for all columns except the first
                ax.set_yticklabels([])

    # Add label indicators
    for row_idx, row_value in enumerate(row_values):
        for col_idx, subplot_value in enumerate(subplot_values_display):
            try:
                calculate_label_indicator(perf_axes[row_idx][col_idx])
            except Exception as e:
                logger.debug(f"Could not add label indicator: {e}")

    # Add hue legend in second performance subplot (if exists)
    _add_hue_legend(
        fig=fig,
        perf_axes=perf_axes,
        hue_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt,
    )

    # Add panel letters after all plotting is complete but before saving
    _add_panel_letters(
        fig=fig,
        perf_axes=perf_axes,
        peak_axes=peak_axes,
        peak_time_axes=peak_time_axes,
        layout=layout,
    )

    # Save
    logger.info(f"Saving figure to: {output}")
    save_plot(output)
    logger.info("Performance grid plotting complete")

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot performance traces in grid layout"
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to test_data.csv files",
    )
    parser.add_argument(
        "--data-ffonly",
        type=Path,
        nargs="*",
        help="Paths to feedforward-only test_data.csv files",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output figure path"
    )
    parser.add_argument(
        "--subplot",
        type=str,
        required=True,
        choices=["category", "parameter", "experiment"],
        help="Variable for horizontal subplots",
    )
    parser.add_argument(
        "--row",
        type=str,
        required=True,
        help="Variable for vertical rows (use 'none' to collapse all data)",
    )
    parser.add_argument(
        "--hue",
        type=str,
        required=True,
        choices=["category", "parameter", "experiment"],
        help="Variable for color coding",
    )
    parser.add_argument("--category-key", type=str, help="Category column name")
    parser.add_argument("--parameter-key", type=str, help="Parameter column name")
    parser.add_argument(
        "--experiment",
        "--experiment-names",
        dest="experiment_names",
        type=str,
        nargs="*",
        help="Experiment names",
    )
    parser.add_argument(
        "--confidence-measure",
        "--confidence_measure",
        type=str,
        default="first_label_confidence",
        help="Confidence measure column name or list (comma-separated or Python literal list)",
    )
    parser.add_argument(
        "--accuracy-measure",
        "--accuracy_measure",
        type=str,
        default="accuracy",
        help="Accuracy measure column name or list (comma-separated or Python literal list)",
    )
    parser.add_argument("--dt", type=float, default=2.0, help="Time resolution (ms)")
    parser.add_argument(
        "--idle-timesteps",
        type=int,
        default=0,
        help="Number of idle timesteps before recorded data (shifts time axis)",
    )
    parser.add_argument("--palette", type=str, help="JSON color palette")
    parser.add_argument("--naming", type=str, help="JSON naming dict")
    parser.add_argument("--ordering", type=str, help="JSON ordering dict")
    parser.add_argument(
        "--subplot-filter",
        type=str,
        nargs="*",
        help="Filter subplot values to display in performance panels",
    )
    parser.add_argument(
        "--plot-individual-seeds",
        action="store_true",
        help="Plot each seed as a separate trace instead of averaging over seeds",
    )

    args = parser.parse_args()

    # Compute time offset from idle timesteps
    time_offset = args.idle_timesteps * args.dt

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Plot
    plot_performance_grid(
        data_paths=args.data,
        output=args.output,
        subplot_var=args.subplot,
        row_var=args.row,
        hue_var=args.hue,
        category_key=args.category_key,
        parameter_key=args.parameter_key,
        experiment_names=args.experiment_names,
        data_ffonly_paths=args.data_ffonly,
        confidence_measure=args.confidence_measure,
        accuracy_measure=args.accuracy_measure,
        dt=args.dt,
        time_offset=time_offset,
        config=config,
        subplot_filter=args.subplot_filter,
        plot_individual_seeds=args.plot_individual_seeds,
    )


if __name__ == "__main__":
    main()
