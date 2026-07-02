"""plot_response_tripytch.py

Refactored triptych plotting script using plot_responses.py functionality.
Creates three-column response plots with training accuracy, performance, and ridge plots.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dynvision.utils.visualization_utils import (
    save_plot,
    load_config_from_args,
    get_display_name,
)
from dynvision.visualization.plot_responses import (  # noqa: F401
    plot_temporal_ridge_responses,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _format_legend_label,
    _plot_accuracy_panel,
    _add_horizontal_legend,
    LAYOUT as RESPONSES_LAYOUT,
    FORMATTING as RESPONSES_FORMATTING,
)
from dynvision.visualization.plot_training import (
    _parse_accuracy_data,
    _standardize_category_value,
)

# Configure logging #
logger = logging.getLogger(__name__)

# Triptych-specific layout configuration with relative positioning
TRIPTYCH_LAYOUT = {
    # Figure dimensions in inches
    "figure_width": 19,
    "figure_height": 14,
    # Column layout (relative coordinates 0-1)
    "n_columns": 3,
    "column_spacing": 0.04,  # Space between columns
    "left_margin": 0.06,
    "right_margin": 0.02,
    # Vertical layout (relative coordinates 0-1)
    "title_height": 0.03,
    "title_pad": 0.02,
    "training_accuracy_height": 0.13,  # Same height as performance panel
    "training_accuracy_pad": 0.07,
    "performance_height": 0.13,
    "performance_pad": 0.05,
    "legend_height": 0.10,
    "legend_pad": 0.02,
    "ridge_height": 0.5,  # Remaining space for ridge plots
    # Panel letters
    "panel_letter_offset_x": -0.02,
    "panel_letter_offset_y": 0.03,
}

# Formatting configuration
TRIPTYCH_FORMATTING = {
    **RESPONSES_FORMATTING,
    "fontsize_panel_label": 20,
}


def _calculate_column_positions(layout: Dict) -> List[float]:
    """Calculate left positions for each column."""
    available_width = 1 - layout["left_margin"] - layout["right_margin"]
    column_width = (
        available_width - (layout["n_columns"] - 1) * layout["column_spacing"]
    ) / layout["n_columns"]

    positions = []
    for i in range(layout["n_columns"]):
        left = layout["left_margin"] + i * (column_width + layout["column_spacing"])
        positions.append(left)

    return positions, column_width


def _get_title_and_symbol(
    category: str, config: Optional[Dict] = None
) -> Tuple[str, str]:
    """Get display title and mathematical symbol for a category.

    Uses the naming dict from config for symbol translation, with fallback
    to hardcoded mappings for full descriptive names.
    """
    # Get symbol from config naming dict
    symbol = get_display_name(category, config) if config else None

    # Fallback mappings for full descriptive names
    name_mappings = {
        "tau": "Time Constant",
        "trc": "Recurrence Delay",
        "tsk": "Skip Delay",
        "rctarget": "Recurrence Target",
        "rctype": "Recurrence Type",
        "lossrt": "Loss Reaction Time",
        "tsteps": "Training Timesteps",
        "idle": "Initial Idle Time",
        "feedback": "Feedback Connections",
        "skip": "Skip Connections",
        "seed": "Random Seed",
        "pattern": "Presentation Pattern",
    }

    # Fallback symbol mappings (used when config doesn't have the symbol)
    symbol_mappings = {
        "tau": r"$\tau$",
        "trc": r"$\Delta_{RC}$",
        "tsk": r"$\Delta_{SK}$",
        "lossrt": r"$t_{loss}$",
        "tsteps": r"$T$",
        "idle": r"$t_{idle}$",
    }

    # Get descriptive name
    category_name = name_mappings.get(category, category.capitalize())

    # If no symbol from config, use fallback
    if symbol is None or symbol == category:
        symbol = symbol_mappings.get(category, category_name)

    # Build title: only add symbol in parentheses if it's a mathematical symbol
    if symbol and "$" in symbol:
        title = f"Varying {category_name} ({symbol})"
    else:
        title = f"Varying {category_name}"

    return title, symbol


def _add_panel_letters(
    fig: plt.Figure, column_positions: List[float], layout: Dict
) -> None:
    """Add panel letters A), B), C) for columns and i), ii), iii) for plots within columns."""
    fmt = TRIPTYCH_FORMATTING

    # Column letters A), B), C)
    column_letters = ["A)", "B)", "C)"]
    for i, left in enumerate(column_positions):
        fig.text(
            left + layout["panel_letter_offset_x"],
            1,
            column_letters[i],
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="top",
        )

        # Panel letters within each column: i), ii), iii)
        plot_letters = ["i)", "ii)", "iii)"]

        # Calculate y-positions for each subplot
        title_top = 1 - layout["title_height"]
        training_top = title_top - layout["title_pad"]
        performance_top = (
            training_top
            - layout["training_accuracy_height"]
            - layout["training_accuracy_pad"]
        )
        ridge_top = (
            performance_top
            - layout["performance_height"]
            - layout["performance_pad"]
            - layout["legend_height"]
            - layout["legend_pad"]
        )

        plot_y_positions = [
            training_top + layout["panel_letter_offset_y"],  # Above training accuracy
            performance_top + layout["panel_letter_offset_y"],  # Above performance
            ridge_top + layout["panel_letter_offset_y"],  # Above ridge plots
        ]

        for j, (letter, y_pos) in enumerate(zip(plot_letters, plot_y_positions)):
            fig.text(
                left + layout["panel_letter_offset_x"] + 0.01,
                y_pos,
                letter,
                fontsize=fmt["fontsize_panel_label"] - 2,
                fontweight="bold",
                ha="left",
                va="top",
            )


def _extract_numerical_sort_key(value: str) -> Tuple[bool, float, str]:
    """Extract numerical sort key from a value that may have units.

    Args:
        value: Value string, possibly with units (e.g., "0.05 ms", "10", "true")

    Returns:
        Tuple of (is_numeric, numeric_value, original_string)
        Used for sorting: numeric values sort before strings, then by value
    """
    value_str = str(value).strip()

    # Try to extract numerical value (first token)
    try:
        # Split on whitespace to separate value from units
        tokens = value_str.split()
        if tokens:
            numeric_value = float(tokens[0])
            return (True, numeric_value, value_str)
    except (ValueError, TypeError):
        pass

    # Not numeric, return as string for string sorting
    return (False, float("inf"), value_str)


def _get_ordering_sort_key(category_key: str, config: Optional[Dict]) -> callable:
    """Get a sort key function based on the ordering dict from config.

    If the category_key exists in config['ordering'], returns a function
    that sorts by the order defined there. Otherwise returns a function
    that sorts by numerical value (or string if not numeric).
    """
    ordering_list = None
    if config and "ordering" in config:
        # Try exact match first, then lowercase
        ordering_list = config["ordering"].get(
            category_key, config["ordering"].get(category_key.lower())
        )

    if ordering_list:
        # Create a lookup dict for ordering
        order_lookup = {str(v).lower(): i for i, v in enumerate(ordering_list)}

        def sort_by_ordering(item):
            val, label = item
            # Try to match the value (lowercase) in the ordering list
            val_lower = str(val).lower()
            if val_lower in order_lookup:
                return (0, order_lookup[val_lower], label)
            # Not in ordering list, sort after ordered items by label
            return (1,) + _extract_numerical_sort_key(label)

        return sort_by_ordering
    else:
        # Fall back to numerical sorting by label
        return lambda item: _extract_numerical_sort_key(item[1])


def _format_seed_value(value) -> str:
    return str(value).strip().split(".")[0]


def _format_pattern_value(value) -> str:
    return str(value).strip().split(".")[0]


def _filter_first_parameter_value(
    df: pd.DataFrame, parameter_key: str
) -> pd.DataFrame:
    """Filter to use only the first parameter value (numerically sorted)."""
    if parameter_key not in df.columns:
        logger.warning(f"Parameter '{parameter_key}' not found in data")
        return df

    unique_params = df[parameter_key].unique()

    if len(unique_params) > 1:
        # Try to sort numerically
        try:
            sorted_params = sorted(unique_params, key=float)
            first_param = sorted_params[0]
            logger.warning(
                f"Multiple parameter values found: {unique_params}. "
                f"Using first value: {first_param}"
            )
        except (ValueError, TypeError):
            # Fall back to string sorting
            sorted_params = sorted(unique_params, key=str)
            first_param = sorted_params[0]
            logger.warning(
                f"Multiple parameter values found: {unique_params}. "
                f"Using first value (alphabetically): {first_param}"
            )

        # Filter to first parameter value only
        filtered_df = df[df[parameter_key] == first_param].copy()
        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        return filtered_df

    return df


def _plot_training_accuracy_panel(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    accuracy_df: pd.DataFrame,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    show_ylabel: bool = False,
    show_legend: bool = False,
    max_epoch: Optional[float] = None,
) -> None:
    """Plot training and validation accuracy panel.

    Args:
        accuracy_df: DataFrame with columns: epoch, category_value, train_accuracy, val_accuracy
                     (as returned by _parse_accuracy_data from plot_training.py)
        max_epoch: Maximum epoch to display (None for no limit)
    """
    ax = fig.add_axes([column_left, bottom, column_width, height])
    ax.patch.set_alpha(0)

    if accuracy_df is None or accuracy_df.empty:
        ax.text(
            0.5,
            0.5,
            "No training accuracy data",
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.6,
        )
        logger.warning("No training accuracy data to plot")
    else:
        # Filter by max_epoch if specified
        if max_epoch is not None:
            original_len = len(accuracy_df)
            accuracy_df = accuracy_df[accuracy_df["epoch"] <= max_epoch].copy()
            if len(accuracy_df) < original_len:
                logger.info(
                    f"Filtered accuracy data to epochs <= {max_epoch}: {len(accuracy_df)} rows (was {original_len})"
                )

        logger.info(f"Plotting training accuracy data with {len(accuracy_df)} rows")

        try:
            # Note: accuracy_df["category_value"] is already standardized by _parse_accuracy_data
            # Standardize hue_values for comparison
            hue_values_std = [_standardize_category_value(v) for v in hue_values]

            # Plot training accuracy (solid lines)
            for cat_val_orig, cat_val_std in zip(hue_values, hue_values_std):
                cat_data = accuracy_df[accuracy_df["category_value"] == cat_val_std]
                if not cat_data.empty:
                    train_data = cat_data.dropna(subset=["train_accuracy"])
                    if not train_data.empty:
                        ax.plot(
                            train_data["epoch"],
                            train_data["train_accuracy"],
                            color=colors.get(
                                cat_val_orig, TRIPTYCH_FORMATTING["greyscale_color"]
                            ),
                            linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                            linestyle="-",
                            alpha=TRIPTYCH_FORMATTING["alpha_line"],
                        )

            # Plot validation accuracy (dotted lines)
            for cat_val_orig, cat_val_std in zip(hue_values, hue_values_std):
                cat_data = accuracy_df[accuracy_df["category_value"] == cat_val_std]
                if not cat_data.empty:
                    val_data = cat_data.dropna(subset=["val_accuracy"])
                    if not val_data.empty:
                        ax.plot(
                            val_data["epoch"],
                            val_data["val_accuracy"],
                            color=colors.get(
                                cat_val_orig, TRIPTYCH_FORMATTING["greyscale_color"]
                            ),
                            linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                            linestyle=":",
                            alpha=TRIPTYCH_FORMATTING["alpha_line"],
                        )

        except Exception as e:
            logger.error(f"Error plotting training accuracy: {e}")
            ax.text(
                0.5,
                0.5,
                "Error plotting training accuracy",
                ha="center",
                va="center",
                fontsize=12,
                alpha=0.6,
            )

    # Formatting
    if show_ylabel:
        ax.set_ylabel(
            "Accuracy",
            fontsize=TRIPTYCH_FORMATTING["fontsize_axis"],
            fontweight="bold",
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Epoch", fontsize=TRIPTYCH_FORMATTING["fontsize_axis"])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=TRIPTYCH_FORMATTING["fontsize_tick"])

    # Add legend for line styles on first panel
    if show_legend:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                linestyle="-",
                label="Training",
                alpha=TRIPTYCH_FORMATTING["alpha_line"],
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linewidth=TRIPTYCH_FORMATTING["linewidth_main"],
                linestyle=":",
                label="Validation",
                alpha=TRIPTYCH_FORMATTING["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            frameon=False,
            fontsize=TRIPTYCH_FORMATTING["fontsize_legend"] - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def create_triptych_plot(
    data_paths: List[Path],
    category_list: List[str],
    parameter_key: str,
    dt: float,
    accuracy_paths: Optional[List[Path]] = None,
    experiment: Optional[str] = None,
    config: Optional[Dict] = None,
    accuracy_measure: str = "accuracy",
    confidence_measure: str = "first_label_confidence",
    max_epoch: Optional[float] = None,
    default_category_values: Optional[List[str]] = None,
    time_offset: float = 0.0,
) -> plt.Figure:
    """Create a triptych plot using plot_responses.py functionality.

    Args:
        data_paths: List of paths to response data files (one per column)
        category_list: List of category keys (one per column)
        parameter_key: Parameter name for ridge subplots
        dt: Temporal resolution in ms per timestep
        accuracy_paths: Optional list of paths to training accuracy files
        experiment: Optional experiment name for panel labels
        config: Optional configuration dict with palette, naming, ordering
        accuracy_measure: Column name for accuracy metric
        confidence_measure: Column name for confidence metric
        max_epoch: Maximum epoch to display in accuracy panels (None for no limit)
        default_category_values: Default values for each category (matching category_list order).
            Legend entries matching these values will be marked with an asterisk (*).
        time_offset: Time offset in ms to shift the time axis (default: 0.0).
    """

    logger.info("=" * 60)
    logger.info("Creating triptych plot")
    logger.info("=" * 60)

    layout = TRIPTYCH_LAYOUT
    fmt = TRIPTYCH_FORMATTING

    # Calculate column positions
    column_positions, column_width = _calculate_column_positions(layout)

    # Create figure
    fig = plt.figure(figsize=(layout["figure_width"], layout["figure_height"]))
    sns.set_context("talk")

    # Add panel letters
    _add_panel_letters(fig, column_positions, layout)

    # Process each column
    all_ridge_axes = []
    all_accuracy_axes = []
    column_data_list = []

    for col_idx, (data_path, category_key) in enumerate(
        zip(data_paths, category_list)
    ):
        logger.info(f"Processing column {col_idx + 1}: {category_key}")

        column_left = column_positions[col_idx]

        # Load main data - handle both single Path and list of Paths
        df = None
        if data_path:
            try:
                if isinstance(data_path, list):
                    # Load and concatenate multiple files
                    logger.info(f"Loading and concatenating {len(data_path)} files")
                    dfs = []
                    for i, path in enumerate(data_path):
                        if path and path.exists():
                            logger.info(
                                f"  Loading file {i+1}/{len(data_path)}: {path}"
                            )
                            dfs.append(pd.read_csv(path))
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Concatenated total: {len(df)} rows")
                elif data_path.exists():
                    # Single file
                    df = pd.read_csv(data_path)
                    logger.info(f"Loaded {len(df)} rows from {data_path}")

                if df is not None:
                    # Filter to first parameter value only
                    df = _filter_first_parameter_value(df, parameter_key)

                    # Standardize category values in the dataframe to match dimension extraction
                    # Use special standardization for seed to avoid float conversion
                    if category_key in df.columns:
                        if category_key == "seed":
                            df[category_key] = df[category_key].apply(
                                _format_seed_value
                            )
                        elif category_key == "pattern":
                            df[category_key] = df[category_key].apply(
                                _format_pattern_value
                            )
                        else:
                            df[category_key] = df[category_key].apply(
                                _standardize_category_value
                            )

                    if category_key == "rctype":
                        # Remove rows with rctype=None, NaN, null values, or empty strings
                        initial_len = len(df)
                        df = df[
                            (df["rctype"].notna())
                            & (df["rctype"].astype(str).str.strip() != "")
                        ].copy()
                        if len(df) < initial_len:
                            logger.info(
                                f"Filtered out {initial_len - len(df)} rows with null/NaN/empty rctype values"
                            )
                    #
            except Exception as e:
                logger.error(f"Error loading data: {e}")

        # Load training accuracy data using plot_training.py parser
        accuracy_df = None
        if (
            accuracy_paths
            and col_idx < len(accuracy_paths)
            and accuracy_paths[col_idx]
        ):
            try:
                logger.info(
                    f"Parsing training accuracy data from {accuracy_paths[col_idx]}"
                )
                accuracy_df, detected_category_key = _parse_accuracy_data(
                    accuracy_paths[col_idx], category_key
                )
                logger.info(
                    f"Parsed {len(accuracy_df)} accuracy rows for category key: {detected_category_key}"
                )
            except Exception as e:
                logger.error(f"Error loading training accuracy data: {e}")

        # Store data for later y-scaling
        column_data_list.append((df, accuracy_df, category_key))

        # Handle missing data: for "seed" category, reuse data from other columns
        # Filter to default values only to show seed variation for the default model
        if df is None or df.empty:
            if category_key == "seed":
                filtered_dfs = []
                for prev_idx, (prev_df, _, prev_cat) in enumerate(
                    column_data_list[:-1]
                ):
                    if (
                        prev_df is not None
                        and not prev_df.empty
                        and prev_cat in prev_df.columns
                    ):
                        # Get default value for this category (standardized)
                        if default_category_values and prev_idx < len(
                            default_category_values
                        ):
                            default_val = _standardize_category_value(
                                default_category_values[prev_idx]
                            )
                            # Filter to rows matching the default value
                            filtered = prev_df[prev_df[prev_cat] == default_val].copy()

                            if not filtered.empty:
                                logger.info(
                                    f"Column {col_idx + 1}: Using {len(filtered)} rows "
                                    f"from column {prev_idx + 1} where {prev_cat}={default_val}"
                                )
                                filtered_dfs.append(filtered)
                if filtered_dfs:
                    df = pd.concat(filtered_dfs, ignore_index=True)
                    # Standardize seed column values as integers (not floats)
                    if "seed" in df.columns:
                        df["seed"] = df["seed"].apply(_format_seed_value)
                    logger.info(
                        f"Column {col_idx + 1}: Combined {len(df)} rows for seed variation"
                    )
            if df is None or df.empty:
                logger.warning(f"No valid data for column {col_idx + 1}")
                continue

        # Extract dimension values and colors
        try:
            hue_values = _extract_dimension_values(
                df=df,
                dimension="category",
                dimension_key=category_key,
                config=config or {},
            )
            colors = _get_colors_for_dimension(
                values=hue_values, dimension_key=category_key, config=config or {}
            )
            logger.info(f"Found {len(hue_values)} category values: {hue_values}")

        except Exception as e:
            logger.error(f"Error processing dimension values: {e}")
            continue

        # Calculate panel positions
        title_top = 1 - layout["title_height"]
        training_bottom = (
            title_top - layout["title_pad"] - layout["training_accuracy_height"]
        )
        performance_bottom = (
            training_bottom
            - layout["training_accuracy_pad"]
            - layout["performance_height"]
        )
        legend_bottom = (
            performance_bottom - layout["performance_pad"] - layout["legend_height"]
        )
        ridge_bottom = legend_bottom - layout["legend_pad"] - layout["ridge_height"]

        # Add title
        title, symbol = _get_title_and_symbol(category_key, config)
        title_ax = fig.add_axes(
            [column_left, title_top, column_width, layout["title_height"]]
        )
        title_ax.text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            fontsize=fmt["fontsize_title"],
            fontweight="bold",
        )
        title_ax.set_xlim(0, 1)
        title_ax.set_ylim(0, 1)
        title_ax.axis("off")

        # Plot training accuracy panel
        # Note: accuracy_df uses "category_value" column from _parse_accuracy_data
        acc_ax = _plot_training_accuracy_panel(
            fig=fig,
            column_left=column_left,
            column_width=column_width,
            bottom=training_bottom,
            height=layout["training_accuracy_height"],
            accuracy_df=accuracy_df,
            category_key="category_value",  # Column name used by _parse_accuracy_data
            hue_values=hue_values,
            colors=colors,
            show_ylabel=(col_idx == 0),
            show_legend=(col_idx == 0),
            max_epoch=max_epoch,
        )
        all_accuracy_axes.append(acc_ax)

        # Plot performance panel (accuracy & confidence from test data)
        performance_ax = fig.add_axes(
            [
                column_left,
                performance_bottom,
                column_width,
                layout["performance_height"],
            ]
        )
        _plot_accuracy_panel(
            ax=performance_ax,
            data=df,
            hue_var="category",
            hue_key=category_key,
            hue_values=hue_values,
            colors=colors,
            dt=dt,
            show_ylabel=(col_idx == 0),
            show_legend=(col_idx == 0),
            accuracy_cols=[accuracy_measure],
            confidence_cols=[confidence_measure],
            time_offset=time_offset,
        )
        performance_ax.set_ylim(-0.005, 1)

        # Add horizontal legend
        legend_ax = fig.add_axes(
            [column_left, legend_bottom, column_width, layout["legend_height"]]
        )
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis("off")
        legend_ax.patch.set_alpha(0)

        # Create legend elements and labels
        # Build tuples of (value, formatted_label) first, then sort
        # Get default value for this column (if provided)
        # Standardize using the same function used for hue_values for consistent comparison
        default_val = None
        if default_category_values and col_idx < len(default_category_values):
            if category_key == "seed":
                default_val = _format_seed_value(default_category_values[col_idx])
            else:
                default_val = _standardize_category_value(
                    default_category_values[col_idx]
                )

        legend_items = []
        for val in hue_values:
            if val in colors:
                # Format seed values specially to avoid decimals
                if category_key == "seed":
                    formatted_label = _format_seed_value(val)
                else:
                    formatted_label = _format_legend_label(
                        category_key, val, config or {}, dt
                    )
                # Add asterisk if this value matches the default
                # Both val and default_val are standardized, so compare directly
                try:
                    is_default = float(val) == float(default_val)
                except (ValueError, TypeError):
                    is_default = str(val) == str(default_val)
                if is_default:
                    formatted_label = f"{formatted_label} *"
                legend_items.append((val, formatted_label))

        # Sort using ordering dict if available, otherwise by numerical value
        sort_key = _get_ordering_sort_key(category_key, config)
        legend_items_sorted = sorted(legend_items, key=sort_key)

        # Build final legend elements and labels in sorted order
        legend_elements = []
        legend_labels = []
        for val, formatted_label in legend_items_sorted:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=colors[val],
                    linewidth=fmt["linewidth_main"],
                    marker=".",
                    markersize=8,
                    alpha=fmt["alpha_line"],
                )
            )
            legend_labels.append(formatted_label)

        if legend_elements:
            n_cols = min(len(legend_elements), 4 if category_key == "rctype" else 3)
            # Get legend title from naming dict (symbol already comes from _get_title_and_symbol)
            legend_title = get_display_name(category_key, config) or symbol
            legend = legend_ax.legend(
                legend_elements,
                legend_labels,
                loc="center",
                ncol=n_cols,
                frameon=False,
                fontsize=fmt["fontsize_legend"],
                handlelength=1.5,
                handletextpad=0.5,
                columnspacing=1.0,
                title=legend_title,
                title_fontsize=fmt["fontsize_legend"],
            )
            # Make legend title bold
            legend.get_title().set_fontweight("bold")

        # Plot ridge plots using plot_responses functionality
        try:
            # Get layer values
            layer_values = _extract_dimension_values(
                df, "layers", "layers", config or {}
            )
            logger.info(f"Found {len(layer_values)} layers: {layer_values}")

            if layer_values:
                from dynvision.visualization.plot_responses import (
                    _plot_response_ridges,
                )

                # Calculate ridge positioning manually since we're using custom layout
                ridge_top = ridge_bottom + layout["ridge_height"]

                ridge_axes = _plot_response_ridges(
                    fig=fig,
                    column_left=column_left,
                    column_width=column_width,
                    data=df,
                    subplot_var="layers",
                    subplot_key="layers",
                    subplot_values=layer_values,
                    hue_var="category",
                    hue_key=category_key,
                    hue_values=hue_values,
                    colors=colors,
                    dt=dt,
                    show_ylabel=(col_idx == 0),
                    config=config or {},
                    # Override LAYOUT values for our custom positioning
                    ridge_height=layout["ridge_height"],
                    title_bot=ridge_top,  # Use ridge_top as the starting point
                    title_pad=0,  # No title in ridge section
                    accuracy_height=0,  # No accuracy in ridge section
                    accuracy_pad=0,
                    legend_height=0,  # No legend in ridge section
                    legend_pad=0,
                    time_offset=time_offset,
                )

                all_ridge_axes.extend(ridge_axes)
                logger.info(
                    f"Created {len(ridge_axes)} ridge axes for column {col_idx + 1}"
                )

        except Exception as e:
            logger.error(f"Error creating ridge plots for column {col_idx + 1}: {e}")
            import traceback

            traceback.print_exc()

    # Apply consistent y-scaling across all ridge axes
    if all_ridge_axes:
        # Calculate global y-limits
        global_ymin, global_ymax = float("inf"), float("-inf")
        for ax in all_ridge_axes:
            ymin, ymax = ax.get_ylim()
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)

        # Apply consistent limits
        global_ymin = max(global_ymin, fmt["min_global_ymin"])
        global_ymax = min(global_ymax, fmt["max_global_ymax"])

        for ax in all_ridge_axes:
            ax.set_ylim(global_ymin, global_ymax)

        logger.info(
            f"Applied consistent y-scaling: [{global_ymin:.3f}, {global_ymax:.3f}]"
        )

    # Adjust y-scaling for accuracy panels if needed
    if all_accuracy_axes:
        global_acc_ymin = min([ax.get_ylim()[0] for ax in all_accuracy_axes])
        for ax in all_accuracy_axes:
            ax.set_ylim(global_acc_ymin, 1.0)
            # ax.set_xlim(0, 100)

    fig.align_ylabels()

    logger.info("Triptych plot creation complete")
    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create response triptych plots using plot_responses.py functionality"
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        help="Path(s) to first dataset. Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--data2",
        type=Path,
        nargs="+",
        help="Path(s) to second dataset. Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--data3",
        type=Path,
        nargs="+",
        help="Path(s) to third dataset. Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--accuracy1", type=Path, help="Path to first training accuracy CSV file"
    )
    parser.add_argument(
        "--accuracy2", type=Path, help="Path to second training accuracy CSV file"
    )
    parser.add_argument(
        "--accuracy3", type=Path, help="Path to third training accuracy CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output file"
    )
    parser.add_argument(
        "--parameter", type=str, required=True, help="Parameter column name"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Categories to plot (space-separated)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Temporal resolution in ms per timestep",
    )
    parser.add_argument(
        "--idle-timesteps",
        type=int,
        default=0,
        help="Number of idle timesteps before recorded data (shifts time axis)",
    )
    parser.add_argument("--experiment", type=str, help="Experiment name for title")
    parser.add_argument(
        "--palette", type=str, help="JSON formatted dictionary of colors"
    )
    parser.add_argument("--naming", type=str, help="JSON formatted naming dictionary")
    parser.add_argument(
        "--ordering", type=str, help="JSON formatted ordering dictionary"
    )
    parser.add_argument(
        "--accuracy-measure",
        type=str,
        default="accuracy",
        help="Column name for accuracy measure (default: accuracy)",
    )
    parser.add_argument(
        "--confidence-measure",
        type=str,
        default="first_label_confidence",
        help="Column name for confidence measure (default: first_label_confidence)",
    )
    parser.add_argument(
        "--max-epoch",
        type=float,
        default=300,
        help="Maximum epoch to display in training accuracy panels (default: 300)",
    )
    parser.add_argument(
        "--default-category-values",
        type=str,
        default=None,
        help="Default values for each category (space-separated, matching --category order). "
        "Legend entries matching these values will be marked with an asterisk (*).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s"
    )

    # Load configuration
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Parse categories
    category_list = args.category.split()
    if len(category_list) != 3:
        raise ValueError(
            f"Expected 3 categories, got {len(category_list)}: {category_list}"
        )

    # Handle single vs multiple data files - convert lists to single Path or keep as list
    data_paths = [
        (
            args.data
            if args.data and len(args.data) > 1
            else (args.data[0] if args.data else None)
        ),
        (
            args.data2
            if args.data2 and len(args.data2) > 1
            else (args.data2[0] if args.data2 else None)
        ),
        (
            args.data3
            if args.data3 and len(args.data3) > 1
            else (args.data3[0] if args.data3 else None)
        ),
    ]
    accuracy_paths = [args.accuracy1, args.accuracy2, args.accuracy3]

    # Validate that we have some data (check both single Path and lists)
    valid_data_paths = []
    for dp in data_paths:
        if dp is None:
            continue
        if isinstance(dp, list):
            if any(p and p.exists() for p in dp):
                valid_data_paths.append(dp)
        elif dp.exists():
            valid_data_paths.append(dp)

    if not valid_data_paths:
        raise ValueError("No valid data files found")

    # Compute time offset from idle timesteps
    time_offset = args.idle_timesteps * args.dt

    logger.info(f"Processing {len(valid_data_paths)} data file groups")
    logger.info(f"Categories: {category_list}")
    logger.info(f"Parameter: {args.parameter}")
    logger.info(f"Temporal resolution: {args.dt} ms")
    if time_offset != 0.0:
        logger.info(
            f"Time offset: {time_offset} ms ({args.idle_timesteps} idle timesteps)"
        )

    try:
        # Create the triptych plot
        _ = create_triptych_plot(
            data_paths=data_paths,
            category_list=category_list,
            parameter_key=args.parameter,
            dt=args.dt,
            accuracy_paths=accuracy_paths,
            experiment=args.experiment,
            config=config,
            accuracy_measure=args.accuracy_measure,
            confidence_measure=args.confidence_measure,
            max_epoch=args.max_epoch,
            default_category_values=(
                args.default_category_values.split()
                if args.default_category_values
                else None
            ),
            time_offset=time_offset,
        )

        # Save the plot
        save_plot(args.output)
        logger.info(f"Triptych plot saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error creating triptych plot: {e}")
        raise


if __name__ == "__main__":
    main()
