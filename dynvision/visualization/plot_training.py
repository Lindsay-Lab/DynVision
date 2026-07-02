"""Plot training overview with training/validation accuracy, losses, and test performance.

This module creates a comprehensive training visualization with:
- Left column: Training/validation accuracy (top), activity/cross-entropy loss (bottom)
- Right column: Test performance panel (top), layer response ridge plots (bottom)
- Horizontal category legend between left panels
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dynvision.utils.visualization_utils import (
    load_config_from_args,
    get_display_name,
    save_plot,
)
from dynvision.visualization.plot_responses import (
    _plot_accuracy_panel,
    _plot_response_ridges,
    _extract_dimension_values,
    _get_colors_for_dimension,
    _get_dimension_key,
    _normalize_dimension,
    _validate_dimensions,
    _format_legend_label,
    FORMATTING as RESPONSES_FORMATTING,
    ERRORBAR_CONFIG,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Training plot layout configuration
TRAINING_LAYOUT = {
    # Figure dimensions
    "figure_height": 12,
    "figure_width": 8,
    # Column layout (relative coordinates 0-1)
    "left_column_width": 0.35,  # Configurable
    "right_column_width": 0.60,
    "column_spacing": 0.16,
    "left_margin": 0.06,
    "right_margin": 0.02,
    # Left column vertical layout (A:legend:B:C ~ 0.3:0.1:0.3:0.3)
    "top_margin": 0.05,
    "accuracy_height": 0.27,
    "accuracy_bottom_margin": 0.015,
    "legend_height": 0.15,
    "legend_top_margin": 0.023,  # Space above legend
    "legend_bottom_margin": 0.015,  # Space below legend
    "activity_loss_height": 0.27,
    "activity_loss_bottom_margin": 0.015,
    "ce_loss_height": 0.27,
    "ce_loss_bottom_margin": 0.05,
    # Right column - panel D aligned to panel A with same height
    "performance_height": 0.27,
    "performance_pad": 0.08,
    "ridge_legend_height": 0.08,
    "ridge_legend_pad": 0.01,
    "ridge_overlap": 0.0,
    # Panel letters
    "panel_letter_offset_x": -0.09,
    "panel_letter_offset_y": 0.015,
}

# Formatting configuration (inherit from plot_responses)
TRAINING_FORMATTING = {
    **RESPONSES_FORMATTING,
    "fontsize_panel_label": 20,
    "max_global_ymax": 5.5,  # Increased from 1.5 to accommodate higher response values
}


def _auto_detect_category_key(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect category key from grouped W&B CSV column names.

    Looks for columns with pattern: "<key>: <value> - <metric>"

    Args:
        df: DataFrame with W&B export columns

    Returns:
        Detected category key or None
    """
    # Pattern: "category_key: category_value - metric_name"
    pattern = r"^([^:]+):\s*[^-]+-\s*.+$"

    for col in df.columns:
        match = re.match(pattern, col)
        if match:
            category_key = match.group(1).strip()
            logger.info(
                f"Auto-detected category key: '{category_key}' from column '{col}'"
            )
            return category_key

    logger.warning("Could not auto-detect category key from column names")
    return None


def _extract_category_value_from_column(
    column_name: str, category_key: str
) -> Optional[str]:
    """Extract category value from grouped W&B column name.

    Args:
        column_name: Column name like "activity_loss_weight: 0.05 - train_accuracy"
        category_key: Category key like "activity_loss_weight"

    Returns:
        Category value like "0.05" or None if not found
    """
    # Pattern: "category_key: category_value - metric_name"
    pattern = rf"^{re.escape(category_key)}:\s*([^-]+)\s*-\s*.+$"
    match = re.match(pattern, column_name)

    if match:
        return match.group(1).strip()
    return None


def _standardize_category_value(value: str) -> str:
    """Standardize category value formatting consistently.

    Converts numeric strings to float for consistent comparison.
    E.g., "0", "0.0", "0.00" all become 0.0
    """
    value_str = str(value).strip()

    # Handle boolean-like values consistently
    if value_str.lower() in ["true", "false"]:
        return value_str.lower()

    # Try to convert to numeric for consistent comparison
    try:
        return float(value_str)
    except (ValueError, TypeError):
        # Not numeric, return as string
        return value_str


def _parse_accuracy_data(
    accuracy_csv: Path, category_key: str
) -> Tuple[pd.DataFrame, str]:
    """Parse training/validation accuracy CSV from W&B export.

    Args:
        accuracy_csv: Path to accuracy CSV file
        category_key: Category key to look for (may be auto-detected)

    Returns:
        Tuple of (processed DataFrame, actual category_key used)
        DataFrame has columns: epoch, category_value, train_accuracy, val_accuracy
    """
    df = pd.read_csv(accuracy_csv)

    # Auto-detect category key if needed
    actual_category_key = category_key
    if category_key not in df.columns:
        detected_key = _auto_detect_category_key(df)
        if detected_key:
            actual_category_key = detected_key
            logger.info(f"Using auto-detected category key: '{actual_category_key}'")
        else:
            logger.warning(
                f"Category key '{category_key}' not found and auto-detection failed"
            )
            return pd.DataFrame(), category_key

    # Find training and validation accuracy columns
    train_acc_cols = [
        col
        for col in df.columns
        if "train_accuracy" in col.lower()
        and "__MIN" not in col
        and "__MAX" not in col
    ]
    val_acc_cols = [
        col
        for col in df.columns
        if "val_accuracy" in col.lower() and "__MIN" not in col and "__MAX" not in col
    ]

    if not train_acc_cols and not val_acc_cols:
        logger.warning("No train_accuracy or val_accuracy columns found")
        return pd.DataFrame(), actual_category_key

    logger.info(
        f"Found {len(train_acc_cols)} train_accuracy columns, {len(val_acc_cols)} val_accuracy columns"
    )

    # Extract data for each category value
    processed_data = []

    # Get all category values from column names
    category_values = set()
    for col in train_acc_cols + val_acc_cols:
        cat_val = _extract_category_value_from_column(col, actual_category_key)
        if cat_val:
            category_values.add(cat_val)

    for cat_val in category_values:
        # Find columns for this category value
        train_col = next(
            (
                col
                for col in train_acc_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
            ),
            None,
        )
        val_col = next(
            (
                col
                for col in val_acc_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
            ),
            None,
        )

        for idx, row in df.iterrows():
            train_acc = (
                row[train_col] if train_col and not pd.isna(row[train_col]) else None
            )
            val_acc = row[val_col] if val_col and not pd.isna(row[val_col]) else None

            if train_acc is not None or val_acc is not None:
                processed_data.append(
                    {
                        "epoch": row["epoch"],
                        "category_value": _standardize_category_value(cat_val),
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                    }
                )

    result_df = pd.DataFrame(processed_data)
    logger.info(
        f"Parsed {len(result_df)} accuracy data points for {len(category_values)} category values"
    )

    return result_df, actual_category_key


def _parse_loss_data(
    loss_csv: Path, category_key: str, validation_frequency: int = 10
) -> Tuple[pd.DataFrame, str]:
    """Parse loss CSV from W&B export.

    Args:
        loss_csv: Path to loss CSV file
        category_key: Category key to look for (may be auto-detected)
        validation_frequency: Frequency of validation epochs (default: 10).
            Validation epochs are identified as (epoch + 1) % validation_frequency == 0,
            i.e., epochs 9, 19, 29, ... for frequency 10.

    Returns:
        Tuple of (processed DataFrame, actual category_key used)
        DataFrame has columns: epoch, category_value, cross_entropy_loss, activity_loss, is_validation
    """
    df = pd.read_csv(loss_csv)

    # Auto-detect category key if needed
    actual_category_key = category_key
    if category_key not in df.columns:
        detected_key = _auto_detect_category_key(df)
        if detected_key:
            actual_category_key = detected_key
            logger.info(f"Using auto-detected category key: '{actual_category_key}'")
        else:
            logger.warning(
                f"Category key '{category_key}' not found and auto-detection failed"
            )
            return pd.DataFrame(), category_key

    # Find loss columns
    ce_loss_cols = [
        col
        for col in df.columns
        if "CrossEntropyLoss" in col and "__MIN" not in col and "__MAX" not in col
    ]
    activity_loss_cols = [
        col
        for col in df.columns
        if "ActivityLoss" in col and "__MIN" not in col and "__MAX" not in col
    ]

    if not ce_loss_cols and not activity_loss_cols:
        logger.warning("No CrossEntropyLoss or ActivityLoss columns found")
        return pd.DataFrame(), actual_category_key

    logger.info(
        f"Found {len(ce_loss_cols)} CrossEntropyLoss columns, {len(activity_loss_cols)} ActivityLoss columns"
    )

    # Extract data for each category value
    processed_data = []

    # Get all category values from column names
    category_values = set()
    for col in ce_loss_cols + activity_loss_cols:
        cat_val = _extract_category_value_from_column(col, actual_category_key)
        if cat_val:
            category_values.add(cat_val)

    for cat_val in category_values:
        # Find columns for this category value
        ce_col = next(
            (
                col
                for col in ce_loss_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
            ),
            None,
        )
        activity_col = next(
            (
                col
                for col in activity_loss_cols
                if _extract_category_value_from_column(col, actual_category_key)
                == cat_val
            ),
            None,
        )

        for idx, row in df.iterrows():
            ce_loss = row[ce_col] if ce_col and not pd.isna(row[ce_col]) else None
            activity_loss = (
                row[activity_col]
                if activity_col and not pd.isna(row[activity_col])
                else None
            )

            if ce_loss is not None or activity_loss is not None:
                epoch = row["epoch"]
                # Validation epochs: (epoch + 1) % validation_frequency == 0
                # i.e., epochs 9, 19, 29, ... for frequency 10
                is_validation = (epoch + 1) % validation_frequency == 0
                processed_data.append(
                    {
                        "epoch": epoch,
                        "category_value": _standardize_category_value(cat_val),
                        "cross_entropy_loss": ce_loss,
                        "activity_loss": activity_loss,
                        "is_validation": is_validation,
                    }
                )

    result_df = pd.DataFrame(processed_data)
    logger.info(
        f"Parsed {len(result_df)} loss data points for {len(category_values)} category values"
    )

    return result_df, actual_category_key


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
    show_ylabel: bool = True,
    show_legend: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> plt.Axes:
    """Plot training and validation accuracy panel.

    Args:
        fig: Matplotlib figure
        column_left: Left position of panel
        column_width: Width of panel
        bottom: Bottom position of panel
        height: Height of panel
        accuracy_df: DataFrame with train/val accuracy data
        category_key: Column name for category dimension
        hue_values: Ordered list of category values
        colors: Color mapping for category values
        show_ylabel: Whether to show y-axis label
        show_legend: Whether to show legend
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**TRAINING_FORMATTING, **kwargs}

    ax = fig.add_axes([column_left, bottom, column_width, height])
    ax.patch.set_alpha(0)

    if accuracy_df.empty:
        ax.text(
            0.5,
            0.5,
            "No training accuracy data",
            ha="center",
            va="center",
            fontsize=14,
            alpha=0.6,
        )
        return ax

    # Note: accuracy_df["category_value"] is already standardized to numeric/string
    # Standardize hue_values for comparison
    hue_values_std = [_standardize_category_value(v) for v in hue_values]

    palette = {
        v_std: colors.get(v_orig, fmt["greyscale_color"])
        for v_orig, v_std in zip(hue_values, hue_values_std)
    }

    # Plot training accuracy (solid lines)
    train_acc_df = accuracy_df.dropna(subset=["train_accuracy"])
    if not train_acc_df.empty:
        sns.lineplot(
            data=train_acc_df,
            x="epoch",
            y="train_accuracy",
            hue="category_value",
            hue_order=hue_values_std,
            palette=palette,
            ax=ax,
            legend=False,
            linewidth=fmt["linewidth_main"],
            linestyle="-",
            alpha=fmt["alpha_line"],
            **ERRORBAR_CONFIG,
        )

    # Plot validation accuracy (dotted lines)
    val_acc_df = accuracy_df.dropna(subset=["val_accuracy"])
    if not val_acc_df.empty:
        sns.lineplot(
            data=val_acc_df,
            x="epoch",
            y="val_accuracy",
            hue="category_value",
            hue_order=hue_values_std,
            palette=palette,
            ax=ax,
            legend=False,
            linewidth=fmt["linewidth_main"],
            linestyle=":",
            alpha=fmt["alpha_line"],
            **ERRORBAR_CONFIG,
        )

    # Styling
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.margins(x=0)

    # Log axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    logger.info(
        f"Training accuracy panel: x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]"
    )

    if show_ylabel:
        ax.set_ylabel("Accuracy", fontsize=fmt["fontsize_axis"], fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Epoch", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    # Add legend for line styles
    if show_legend:
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle="-",
                label="Training",
                alpha=fmt["alpha_line"],
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle=":",
                label="Validation",
                alpha=fmt["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            frameon=False,
            fontsize=fmt["fontsize_legend"] - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def _plot_activity_loss_panel(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    loss_df: pd.DataFrame,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    show_ylabel: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """Plot activity loss panel (training only).

    activity loss validation values are not properly computed (always 0),
    so only training loss is shown.

    Args:
        fig: Matplotlib figure
        column_left: Left position of panel
        column_width: Width of panel
        bottom: Bottom position of panel
        height: Height of panel
        loss_df: DataFrame with loss data including is_validation column
        category_key: Column name for category dimension
        hue_values: Ordered list of category values
        colors: Color mapping for category values
        show_ylabel: Whether to show y-axis label
        xlim: X-axis limits (min, max)
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**TRAINING_FORMATTING, **kwargs}

    ax = fig.add_axes([column_left, bottom, column_width, height])
    ax.patch.set_alpha(0)

    if loss_df.empty:
        ax.text(
            0.5,
            0.5,
            "No activity loss data",
            ha="center",
            va="center",
            fontsize=14,
            alpha=0.6,
        )
        return

    # Standardize hue_values for comparison
    hue_values_std = [_standardize_category_value(v) for v in hue_values]

    # Filter to training data only (validation activity loss is always 0)
    train_df = loss_df[~loss_df["is_validation"]]

    palette = {
        v_std: colors.get(v_orig, fmt["greyscale_color"])
        for v_orig, v_std in zip(hue_values, hue_values_std)
    }

    # Plot activity loss (training only, solid lines)
    activity_df = train_df.dropna(subset=["activity_loss"])
    if not activity_df.empty:
        sns.lineplot(
            data=activity_df,
            x="epoch",
            y="activity_loss",
            hue="category_value",
            hue_order=hue_values_std,
            palette=palette,
            ax=ax,
            legend=False,
            linewidth=fmt["linewidth_main"],
            linestyle="-",
            alpha=fmt["alpha_line"],
            **ERRORBAR_CONFIG,
        )

    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.margins(x=0)

    # Log axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    logger.info(
        f"activity loss panel: x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]"
    )

    # Styling
    if show_ylabel:
        ax.set_ylabel(
            "Activity Loss", fontsize=fmt["fontsize_axis"], fontweight="bold"
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    # No xlabel - shared x-axis with CE loss panel below
    ax.set_xlabel("")
    ax.tick_params(labelsize=fmt["fontsize_tick"])
    ax.set_xticklabels([])

    # No legend needed (only training is shown)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def _plot_ce_loss_panel(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    loss_df: pd.DataFrame,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    show_ylabel: bool = True,
    show_legend: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """Plot cross-entropy loss panel (training and validation).

    Args:
        fig: Matplotlib figure
        column_left: Left position of panel
        column_width: Width of panel
        bottom: Bottom position of panel
        height: Height of panel
        loss_df: DataFrame with loss data including is_validation column
        category_key: Column name for category dimension
        hue_values: Ordered list of category values
        colors: Color mapping for category values
        show_ylabel: Whether to show y-axis label
        show_legend: Whether to show legend for train/val distinction
        xlim: X-axis limits (min, max)
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**TRAINING_FORMATTING, **kwargs}

    ax = fig.add_axes([column_left, bottom, column_width, height])
    ax.patch.set_alpha(0)

    if loss_df.empty:
        ax.text(
            0.5,
            0.5,
            "No cross-entropy loss data",
            ha="center",
            va="center",
            fontsize=14,
            alpha=0.6,
        )
        return

    # Standardize hue_values for comparison
    hue_values_std = [_standardize_category_value(v) for v in hue_values]

    # Separate training and validation data
    train_df = loss_df[~loss_df["is_validation"]]
    val_df = loss_df[loss_df["is_validation"]]

    palette = {
        v_std: colors.get(v_orig, fmt["greyscale_color"])
        for v_orig, v_std in zip(hue_values, hue_values_std)
    }

    # Plot training cross-entropy loss (solid lines)
    train_ce_df = train_df.dropna(subset=["cross_entropy_loss"])
    if not train_ce_df.empty:
        sns.lineplot(
            data=train_ce_df,
            x="epoch",
            y="cross_entropy_loss",
            hue="category_value",
            hue_order=hue_values_std,
            palette=palette,
            ax=ax,
            legend=False,
            linewidth=fmt["linewidth_main"],
            linestyle="-",
            alpha=fmt["alpha_line"],
            **ERRORBAR_CONFIG,
        )

    # Plot validation cross-entropy loss (dotted lines)
    val_ce_df = val_df.dropna(subset=["cross_entropy_loss"])
    if not val_ce_df.empty:
        sns.lineplot(
            data=val_ce_df,
            x="epoch",
            y="cross_entropy_loss",
            hue="category_value",
            hue_order=hue_values_std,
            palette=palette,
            ax=ax,
            legend=False,
            linewidth=fmt["linewidth_main"],
            linestyle=":",
            alpha=fmt["alpha_line"],
            **ERRORBAR_CONFIG,
        )

    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.margins(x=0)

    # Log axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    logger.info(f"CE loss panel: x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]")

    # Styling
    if show_ylabel:
        ax.set_ylabel(
            "Cross-Entropy Loss", fontsize=fmt["fontsize_axis"], fontweight="bold"
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.set_xlabel("Epoch", fontsize=fmt["fontsize_axis"])
    ax.tick_params(labelsize=fmt["fontsize_tick"])

    # Add legend for line styles (training vs validation)
    if show_legend:
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle="-",
                label="Training",
                alpha=fmt["alpha_line"],
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=fmt["linewidth_main"],
                linestyle=":",
                label="Validation",
                alpha=fmt["alpha_line"],
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            fontsize=fmt["fontsize_legend"] - 2,
        )

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def _add_panel_letters(
    fig: plt.Figure,
    left_col_left: float,
    right_col_left: float,
    accuracy_top: float,
    activity_loss_top: float,
    ce_loss_top: float,
    performance_top: float,
    ridge_top: float,
    layout: Dict,
    fmt: Dict,
) -> None:
    """Add panel letters A-E to identify plot sections.

    Args:
        fig: Matplotlib figure
        left_col_left: Left position of left column
        right_col_left: Left position of right column
        accuracy_top: Top position of accuracy panel
        activity_loss_top: Top position of activity loss panel
        ce_loss_top: Top position of cross-entropy loss panel
        performance_top: Top position of performance panel
        ridge_top: Top position of ridge plots
        layout: Layout configuration dict
        fmt: Formatting configuration dict
    """
    panel_letters = [
        ("A)", left_col_left, accuracy_top),  # Training/validation accuracy
        ("B)", left_col_left, activity_loss_top),  # activity loss panel
        ("C)", left_col_left, ce_loss_top),  # Cross-entropy loss panel
        ("D)", right_col_left, performance_top),  # Performance panel
        ("E)", right_col_left, ridge_top),  # Ridge plots
    ]

    for letter, x_pos, y_pos in panel_letters:
        fig.text(
            x_pos + layout["panel_letter_offset_x"],
            y_pos + layout["panel_letter_offset_y"],
            letter,
            fontsize=fmt["fontsize_panel_label"],
            fontweight="bold",
            ha="center",
            va="top",
        )


def _plot_category_legend(
    fig: plt.Figure,
    column_left: float,
    column_width: float,
    bottom: float,
    height: float,
    category_key: str,
    hue_values: List[str],
    colors: Dict[str, str],
    config: Dict,
    dt: float = 2.0,
    **kwargs,
) -> None:
    """Add horizontal category legend between left panels.

    Args:
        fig: Matplotlib figure
        column_left: Left position of legend
        column_width: Width of legend
        bottom: Bottom position of legend
        height: Height of legend
        category_key: Column name for category dimension
        hue_values: Ordered list of category values
        colors: Color mapping for category values
        config: Configuration dict with naming mappings
        dt: Temporal resolution in ms per timestep
        **kwargs: Override FORMATTING defaults
    """
    fmt = {**TRAINING_FORMATTING, **kwargs}

    legend_ax = fig.add_axes([column_left, bottom, column_width, height])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.patch.set_alpha(0)

    # Create legend elements
    legend_elements = []
    legend_labels = []

    for val in hue_values:
        if val in colors:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=colors[val],
                    linewidth=fmt["linewidth_main"],
                    alpha=fmt["alpha_line"],
                )
            )
            legend_labels.append(_format_legend_label(category_key, val, config, dt))

    if legend_elements:
        # Get symbol for title
        symbol = get_display_name(category_key, config)

        n_cols = min(len(legend_elements), 3)
        logger.debug(
            f"Adding category legend with {len(legend_elements)} elements in {n_cols} columns"
        )

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
            title=symbol,
            title_fontsize=fmt["fontsize_legend"],
        )
        # Make legend title bold
        legend.get_title().set_fontweight("bold")


def _align_ylabels_tight(axes: List[plt.Axes], renderer, gap_px: int = 4) -> None:
    """Align ylabels across a group of add_axes() panels.

    Finds the leftmost tick-label edge across all axes (i.e. the panel with
    the widest tick text), then places every ylabel so its right edge sits
    ``gap_px`` pixels to the left of that edge.  This keeps labels as close
    to the spines as possible while staying aligned.

    Args:
        axes: Axes whose ylabels should be aligned.
        renderer: Active renderer (from ``fig.canvas.get_renderer()``).
        gap_px: Gap in display pixels between ylabel right edge and tick labels.
    """
    if not axes:
        return

    labeled_axes = [ax for ax in axes if ax.yaxis.label.get_text()]
    if not labeled_axes:
        return

    # Leftmost tick-label x0 across all axes (widest tick text sets the reference).
    tick_x0_values = []
    for ax in labeled_axes:
        for tick in ax.yaxis.get_ticklabels():
            if tick.get_visible() and tick.get_text():
                tick_x0_values.append(tick.get_window_extent(renderer).x0)

    if not tick_x0_values:
        return

    min_tick_x0 = min(tick_x0_values)

    for ax in labeled_axes:
        lbl_ext = ax.yaxis.label.get_window_extent(renderer)
        # For a 90°-rotated label the horizontal span equals the font cap-height.
        lbl_half_w = (lbl_ext.x1 - lbl_ext.x0) / 2
        # Place label center so its right edge is gap_px left of the ticks.
        center_x = min_tick_x0 - gap_px - lbl_half_w
        x_ax = ax.transAxes.inverted().transform([center_x, 0])[0]
        ax.yaxis.set_label_coords(x_ax, 0.5)


def plot_training_overview(
    test_data: Union[Path, List[Path]],
    accuracy_csv: Path,
    loss_csv: Path,
    output: Path,
    subplot_var: str = "layers",
    hue_var: str = "category",
    column_var: Optional[str] = None,
    category_key: Optional[str] = None,
    parameter_key: Optional[str] = None,
    confidence_measure: Optional[Union[str, List[str]]] = "first_label_confidence",
    accuracy_measure: Optional[Union[str, List[str]]] = "accuracy",
    dt: float = 2.0,
    time_offset: float = 0.0,
    validation_frequency: int = 10,
    config: Optional[Dict] = None,
    **kwargs,
) -> plt.Figure:
    """Create comprehensive training overview plot.

    Args:
        test_data: Path(s) to test_data.csv with layer responses and performance
        accuracy_csv: Path to training/validation accuracy CSV
        loss_csv: Path to loss CSV
        output: Path to save figure
        subplot_var: Variable for ridge plot subplots (default: "layers")
        hue_var: Variable for color coding (default: "category")
        column_var: Variable for columns (optional)
        category_key: Column name for category dimension
        parameter_key: Column name for parameter dimension
        confidence_measure: Column name(s) for confidence metrics
        accuracy_measure: Column name(s) for accuracy metrics
        dt: Temporal resolution in ms per timestep
        time_offset: Time offset in ms from idle timesteps (shifts time axis)
        validation_frequency: Frequency of validation epochs (default: 10).
            Validation epochs are (epoch + 1) % validation_frequency == 0.
        config: Configuration dict with palette, naming, ordering
        **kwargs: Override LAYOUT and FORMATTING defaults

    Returns:
        Matplotlib figure
    """
    logger.info("=" * 60)
    logger.info("Starting training overview plot")
    logger.info("=" * 60)

    # Normalize dimension configuration
    subplot_var, subplot_limit = _normalize_dimension(subplot_var)
    hue_var, hue_limit = _normalize_dimension(hue_var)
    column_var, column_limit = _normalize_dimension(column_var)

    if subplot_var is None or hue_var is None:
        raise ValueError("subplot and hue dimensions cannot be empty")

    _validate_dimensions(
        subplot_var=subplot_var, hue_var=hue_var, column_var=column_var
    )

    if config is None:
        config = {"palette": {}, "naming": {}, "ordering": {}}

    # Merge layout and formatting defaults with overrides
    layout = {
        **TRAINING_LAYOUT,
        **{k: v for k, v in kwargs.items() if k in TRAINING_LAYOUT},
    }
    fmt = {
        **TRAINING_FORMATTING,
        **{k: v for k, v in kwargs.items() if k in TRAINING_FORMATTING},
    }

    # Load test data
    if isinstance(test_data, list):
        logger.info(f"Loading and concatenating test data from {len(test_data)} files")
        dfs = [pd.read_csv(path) for path in test_data]
        test_df = pd.concat(dfs, ignore_index=True)
    else:
        logger.info(f"Loading test data from: {test_data}")
        test_df = pd.read_csv(test_data)

    logger.info(f"Test data: {len(test_df)} rows, {len(test_df.columns)} columns")

    # Parse training data
    logger.info(f"Parsing accuracy data from: {accuracy_csv}")
    accuracy_df, detected_category_key = _parse_accuracy_data(
        accuracy_csv, category_key
    )

    logger.info(f"Parsing loss data from: {loss_csv}")
    loss_df, _ = _parse_loss_data(
        loss_csv, detected_category_key, validation_frequency
    )

    # Use original category_key for test data lookups (may be alias like "activityloss")
    # Use detected_category_key for W&B CSV parsing (full name like "activity_loss_weight")
    test_data_category_key = category_key
    _ = detected_category_key

    # Extract dimension values from test data
    logger.info("Extracting dimension values from test data...")
    logger.info(f"Using category key '{test_data_category_key}' for test data lookups")

    hue_key = _get_dimension_key(
        dimension=hue_var,
        category_key=test_data_category_key,
        parameter_key=parameter_key,
    )
    subplot_key = _get_dimension_key(
        dimension=subplot_var,
        category_key=test_data_category_key,
        parameter_key=parameter_key,
    )
    _ = (
        _get_dimension_key(
            dimension=column_var,
            category_key=test_data_category_key,
            parameter_key=parameter_key,
        )
        if column_var
        else None
    )

    hue_values = _extract_dimension_values(
        df=test_df,
        dimension=hue_var,
        dimension_key=hue_key,
        config=config,
        dimension_limit=hue_limit,
    )
    subplot_values = _extract_dimension_values(
        df=test_df,
        dimension=subplot_var,
        dimension_key=subplot_key,
        config=config,
        dimension_limit=subplot_limit,
    )

    if not hue_values or not subplot_values:
        raise ValueError("Cannot create plot with empty dimension values")

    # Get colors for hue dimension
    colors = _get_colors_for_dimension(
        values=hue_values, dimension_key=hue_key, config=config
    )

    # Standardize and filter W&B data to only include category values present in test data
    # The W&B CSVs may have more category values than the test data
    # (e.g., test data has [0.0, 0.1, 0.5] but W&B has [0, 0.01, 0.05, 0.1, 0.5, ...])
    logger.info(f"Test data category values: {hue_values}")

    # Standardize hue_values to numeric if possible for consistent comparison
    hue_values_standardized = []
    for v in hue_values:
        try:
            hue_values_standardized.append(float(v))
        except (ValueError, TypeError):
            hue_values_standardized.append(v)

    test_values_set = set(hue_values_standardized)
    logger.info(
        f"Test data standardized values: {sorted(test_values_set, key=lambda x: (isinstance(x, str), x))}"
    )

    # Filter accuracy data to only include test data category values
    if not accuracy_df.empty:
        wandb_acc_values_orig = sorted(accuracy_df["category_value"].unique())
        logger.info(f"W&B accuracy category values (all): {wandb_acc_values_orig}")

        # Keep only rows with category values present in test data
        accuracy_df = accuracy_df[
            accuracy_df["category_value"].isin(test_values_set)
        ].copy()

        remaining_acc_values = sorted(accuracy_df["category_value"].unique())
        logger.info(f"W&B accuracy category values (filtered): {remaining_acc_values}")
        logger.info(f"Filtered accuracy data: {len(accuracy_df)} rows")

    # Filter loss data to only include test data category values
    if not loss_df.empty:
        wandb_loss_values_orig = sorted(loss_df["category_value"].unique())
        logger.info(f"W&B loss category values (all): {wandb_loss_values_orig}")

        # Keep only rows with category values present in test data
        loss_df = loss_df[loss_df["category_value"].isin(test_values_set)].copy()

        remaining_loss_values = sorted(loss_df["category_value"].unique())
        logger.info(f"W&B loss category values (filtered): {remaining_loss_values}")
        logger.info(f"Filtered loss data: {len(loss_df)} rows")

    # Create figure
    fig = plt.figure(figsize=(layout["figure_width"], layout["figure_height"]))
    sns.set_context("talk")

    # Calculate column positions
    left_col_left = layout["left_margin"]
    left_col_width = layout["left_column_width"]
    right_col_left = left_col_left + left_col_width + layout["column_spacing"]
    right_col_width = layout["right_column_width"]

    # Calculate left column x-axis limits (epoch range)
    # Start at 0, round max up to next decimal (e.g., 349 -> 350)
    import math

    if not accuracy_df.empty or not loss_df.empty:
        epoch_max_acc = accuracy_df["epoch"].max() if not accuracy_df.empty else 0
        epoch_max_loss = loss_df["epoch"].max() if not loss_df.empty else 0
        epoch_max = max(epoch_max_acc, epoch_max_loss)
        # Round up to next multiple of 10
        left_xmax = math.ceil(epoch_max / 10) * 10
        left_xmin = 0
        logger.info(
            f"Left column x-axis limits: [0, {left_xmax}] (data max: {epoch_max})"
        )
    else:
        left_xmin, left_xmax = 0, 100

    # Calculate left column panel positions (from top to bottom)
    current_top = 1 - layout["top_margin"]

    # Training accuracy panel
    accuracy_bottom = current_top - layout["accuracy_height"]
    logger.info("Creating training accuracy panel...")
    ax_accuracy = _plot_training_accuracy_panel(
        fig=fig,
        column_left=left_col_left,
        column_width=left_col_width,
        bottom=accuracy_bottom,
        height=layout["accuracy_height"],
        accuracy_df=accuracy_df,
        category_key="category_value",
        hue_values=hue_values,
        colors=colors,
        show_ylabel=True,
        show_legend=True,
        xlim=(left_xmin, left_xmax),
        **fmt,
    )

    # Category legend (moved down slightly)
    current_top = (
        accuracy_bottom
        - layout["accuracy_bottom_margin"]
        - layout["legend_top_margin"]
    )
    legend_bottom = current_top - layout["legend_height"]
    logger.info("Adding category legend...")
    _plot_category_legend(
        fig=fig,
        column_left=left_col_left,
        column_width=left_col_width,
        bottom=legend_bottom,
        height=layout["legend_height"],
        category_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        config=config,
        dt=dt,
        **fmt,
    )

    # activity loss panel (Panel B)
    current_top = legend_bottom - layout["legend_bottom_margin"]
    activity_loss_bottom = current_top - layout["activity_loss_height"]
    logger.info("Creating activity loss panel...")
    ax_e_loss = _plot_activity_loss_panel(
        fig=fig,
        column_left=left_col_left,
        column_width=left_col_width,
        bottom=activity_loss_bottom,
        height=layout["activity_loss_height"],
        loss_df=loss_df,
        category_key="category_value",
        hue_values=hue_values,
        colors=colors,
        show_ylabel=True,
        xlim=(left_xmin, left_xmax),
        **fmt,
    )

    # Cross-entropy loss panel (Panel C)
    current_top = activity_loss_bottom - layout["activity_loss_bottom_margin"]
    ce_loss_bottom = current_top - layout["ce_loss_height"]
    logger.info("Creating cross-entropy loss panel...")
    ax_ce_loss = _plot_ce_loss_panel(
        fig=fig,
        column_left=left_col_left,
        column_width=left_col_width,
        bottom=ce_loss_bottom,
        height=layout["ce_loss_height"],
        loss_df=loss_df,
        category_key="category_value",
        hue_values=hue_values,
        colors=colors,
        show_ylabel=True,
        show_legend=True,
        xlim=(left_xmin, left_xmax),
        **fmt,
    )

    # Right column: performance panel + ridge plots
    # Calculate right column x-axis limits (time in ms)
    # Start at 0, round max up to next multiple of 10
    if "times_index" in test_df.columns:
        time_max = test_df["times_index"].max() * dt + time_offset
        # Round up to next multiple of 10
        right_xmax = math.ceil(time_max / 10) * 10
        right_xmin = time_offset
        logger.info(
            f"Right column x-axis limits: [{right_xmin}, {right_xmax}] (data max: {time_max} ms)"
        )
    else:
        right_xmin, right_xmax = time_offset, 120

    # Calculate positions
    right_top = 1 - layout["top_margin"]
    performance_bottom = right_top - layout["performance_height"]

    logger.info("Creating test performance panel...")
    perf_ax = fig.add_axes(
        [
            right_col_left,
            performance_bottom,
            right_col_width,
            layout["performance_height"],
        ]
    )

    _plot_accuracy_panel(
        ax=perf_ax,
        data=test_df,
        hue_var=hue_var,
        hue_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        dt=dt,
        time_offset=time_offset,
        show_ylabel=True,
        show_legend=True,
        accuracy_cols=(
            [accuracy_measure]
            if isinstance(accuracy_measure, str)
            else accuracy_measure
        ),
        confidence_cols=(
            [confidence_measure]
            if isinstance(confidence_measure, str)
            else confidence_measure
        ),
        **fmt,
    )

    # Set x-axis limits
    perf_ax.set_xlim(right_xmin, right_xmax)

    # Log axis limits
    xmin, xmax = perf_ax.get_xlim()
    ymin, ymax = perf_ax.get_ylim()
    logger.info(f"Performance panel: x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]")

    # Ridge plots - align bottom with left column loss panel
    ridge_top = performance_bottom - layout["performance_pad"]

    # Calculate ridge_height to align bottom with ce_loss_bottom
    # The ridge plot function uses (with overlap=0):
    #   spacing = ridge_height / n_subplots
    #   plot_height = ridge_height / n_subplots * 1.4
    #   bottom = ridge_top - (n_subplots - 1) * spacing - plot_height
    #          = ridge_top - ridge_height * (n_subplots + 0.4) / n_subplots
    # Solving for ridge_height:
    #   ridge_height = (ridge_top - ce_loss_bottom) * n_subplots / (n_subplots + 0.4)
    n_subplots = len(subplot_values)
    if n_subplots > 0:
        ridge_height = (ridge_top - ce_loss_bottom) * n_subplots / (n_subplots + 0.4)
    else:
        ridge_height = ridge_top - ce_loss_bottom

    logger.info("Creating ridge plots...")
    logger.info(f"Subplot values for ridges: {subplot_values}")
    logger.info(
        f"Ridge plot: top={ridge_top}, bottom target={ce_loss_bottom}, height={ridge_height:.4f}, n_subplots={n_subplots}"
    )

    ridge_axes = _plot_response_ridges(
        fig=fig,
        column_left=right_col_left,
        column_width=right_col_width,
        data=test_df,
        subplot_var=subplot_var,
        subplot_key=subplot_key,
        subplot_values=subplot_values,
        hue_var=hue_var,
        hue_key=hue_key,
        hue_values=hue_values,
        colors=colors,
        dt=dt,
        time_offset=time_offset,
        show_ylabel=True,
        config=config,
        ridge_top=ridge_top,
        ridge_height=ridge_height,
        ridge_overlap=layout["ridge_overlap"],
        **fmt,
    )

    # Note: No separate ridge legend - we already have the category legend on the left

    # Set x-axis limits to match performance panel
    for ax in ridge_axes:
        ax.set_xlim(right_xmin, right_xmax)

    # Log ridge plot axis limits and verify bottom alignment
    if ridge_axes:
        # Check top and bottom ridge plots
        top_ax = ridge_axes[0]
        bottom_ax = ridge_axes[-1]

        xmin, xmax = top_ax.get_xlim()
        ymin, ymax = top_ax.get_ylim()
        logger.info(
            f"Ridge plots: global x-axis [{xmin}, {xmax}], y-axis [{ymin}, {ymax}]"
        )

        # Get actual position of bottommost ridge plot
        bottom_pos = bottom_ax.get_position()
        logger.info(
            f"Ridge plot actual bottom: {bottom_pos.y0:.4f}, target: {ce_loss_bottom:.4f}, diff: {(bottom_pos.y0 - ce_loss_bottom):.4f}"
        )

    # Add panel letters
    logger.info("Adding panel letters...")
    _add_panel_letters(
        fig=fig,
        left_col_left=left_col_left,
        right_col_left=right_col_left,
        accuracy_top=1 - layout["top_margin"],
        activity_loss_top=legend_bottom - layout["legend_bottom_margin"],
        ce_loss_top=activity_loss_bottom - layout["activity_loss_bottom_margin"],
        performance_top=1 - layout["top_margin"],
        ridge_top=ridge_top,
        layout=layout,
        fmt=fmt,
    )
    # Align ylabels for both columns in a single pass over the fully-built figure.
    # One canvas.draw() here is enough; a prior draw would be reset by this one.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_col_axes = [
        ax for ax in [ax_accuracy, ax_e_loss, ax_ce_loss] if ax is not None
    ]
    _align_ylabels_tight(left_col_axes, renderer, gap_px=-22)
    right_col_axes_with_labels = [
        ax for ax in ([perf_ax] + ridge_axes) if ax.yaxis.label.get_text()
    ]
    _align_ylabels_tight(right_col_axes_with_labels, renderer, gap_px=-6)

    logger.info(f"Saving figure to: {output}")
    save_plot(output)
    logger.info("=" * 60)
    logger.info("Training overview plot complete")
    logger.info("=" * 60)

    return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot training overview with accuracy, loss, and test performance"
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to test_data.csv file(s). Multiple paths will be concatenated.",
    )
    parser.add_argument(
        "--accuracy_csv",
        type=Path,
        required=True,
        help="Path to training/validation accuracy CSV",
    )
    parser.add_argument(
        "--loss_csv",
        type=Path,
        required=True,
        help="Path to loss CSV with CrossEntropyLoss and ActivityLoss",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure path",
    )
    parser.add_argument(
        "--subplot",
        type=str,
        default="layers",
        help="Variable for ridge plot subplots (default: layers)",
    )
    parser.add_argument(
        "--hue",
        type=str,
        default="category",
        help="Variable for color coding (default: category)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Variable for columns (optional)",
    )
    parser.add_argument(
        "--category-key",
        type=str,
        required=True,
        help="Category column name (will be auto-detected if not found)",
    )
    parser.add_argument(
        "--parameter-key",
        type=str,
        help="Parameter column name",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=2.0,
        help="Temporal resolution in ms per timestep",
    )
    parser.add_argument(
        "--idle-timesteps",
        type=int,
        default=0,
        help="Number of idle timesteps before recorded data (shifts time axis)",
    )
    parser.add_argument(
        "--validation-frequency",
        type=int,
        default=10,
        help="Frequency of validation epochs (default: 10). Validation epochs are (epoch + 1) %% freq == 0.",
    )
    parser.add_argument(
        "--confidence-measure",
        type=str,
        default="first_label_confidence",
        help="Confidence measure column name",
    )
    parser.add_argument(
        "--accuracy-measure",
        type=str,
        default="accuracy",
        help="Accuracy measure column name",
    )
    parser.add_argument(
        "--palette",
        type=str,
        help="JSON color palette",
    )
    parser.add_argument(
        "--naming",
        type=str,
        help="JSON naming dict",
    )
    parser.add_argument(
        "--ordering",
        type=str,
        help="JSON ordering dict",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args, unknown = parser.parse_known_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    # Load config
    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    # Handle single vs multiple test data files
    test_data_input = args.test_data if len(args.test_data) > 1 else args.test_data[0]

    # Compute time offset from idle timesteps
    time_offset = args.idle_timesteps * args.dt

    # Plot
    plot_training_overview(
        test_data=test_data_input,
        accuracy_csv=args.accuracy_csv,
        loss_csv=args.loss_csv,
        output=args.output,
        subplot_var=args.subplot,
        hue_var=args.hue,
        column_var=args.column,
        category_key=args.category_key,
        parameter_key=args.parameter_key,
        confidence_measure=args.confidence_measure,
        accuracy_measure=args.accuracy_measure,
        dt=args.dt,
        time_offset=time_offset,
        validation_frequency=args.validation_frequency,
        config=config,
    )


if __name__ == "__main__":
    main()
