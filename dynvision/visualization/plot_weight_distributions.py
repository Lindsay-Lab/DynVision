"""Plot aggregated weight distributions across checkpoints and categories.

This script extends the legacy weight plotting routine by:
- loading multiple checkpoint files (e.g. different seeds or statuses) and aggregating them
- arranging categorical dimensions across subplot rows and columns
- mapping connection types, categories, statuses, or layers to violin hues
- harmonising colours, naming, and ordering via the shared visualization config

The expected inputs are torch ``.pt`` files that contain a state dictionary or a
standard PyTorch Lightning checkpoint with a ``state_dict`` entry.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from dynvision.utils.string_utils import key_value_pattern
from dynvision.utils.visualization_utils import (
    get_display_name,
    get_ordering,
    load_config_from_args,
    save_plot,
    standardize_category_value,
    tensor_to_numpy,
)

DIMENSION_CHOICES = ("category", "connection_type", "status", "layer", "none")
logger = logging.getLogger(__name__)

DEFAULT_LAYOUT = {
    "subplot_height": 4.0,
    "subplot_width": 2.5,
    "spacing_y": 0.3,
    "spacing_x": 0.1,
    "min_width": 2.0,
    "min_height": 3.0,
}

RECURRENT_TOKENS = ("recurrence", "recurrent", "feedback")
SEED_PATTERNS = (
    re.compile(r"seed(?:=|_)([-\d\.]+)", re.IGNORECASE),
    re.compile(r"_([\d\.]+)$"),
)
LAYER_PATTERN = re.compile(r"(layer\d+|classifier)", re.IGNORECASE)


@dataclass
class DimensionMeta:
    """Metadata describing how a dataframe column maps to a plot dimension."""

    column: str
    display_column: str
    values: List[str]
    display_order: List[str]
    display_map: Dict[str, str]
    palette: Optional[Dict[str, str]]
    label_key: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot weight distributions.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="List of checkpoint files produced by DynVision.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination image path (PNG, PDF, ...).",
    )
    parser.add_argument(
        "--row",
        type=str.lower,
        choices=DIMENSION_CHOICES,
        default="category",
        help="Dimension to map to subplot rows (category, connection_type, status, layer).",
    )
    parser.add_argument(
        "--column",
        type=str.lower,
        choices=DIMENSION_CHOICES,
        default="none",
        help="Dimension to map to subplot columns (category, connection_type, status, layer).",
    )
    parser.add_argument(
        "--hue",
        type=str.lower,
        choices=DIMENSION_CHOICES,
        default="connection_type",
        help="Dimension to map to violin hues (category, connection_type, status, layer).",
    )
    parser.add_argument(
        "--x-axis",
        "--x_axis",
        dest="x_axis",
        type=str.lower,
        choices=DIMENSION_CHOICES,
        default="layer",
        help="Dimension to arrange along the violin x-axis (category, connection_type, status, layer).",
    )
    parser.add_argument(
        "--category-key",
        type=str,
        default=None,
        help="Parameter key used to extract category values from paths (e.g. 'rctype').",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        help="JSON string overriding palette entries.",
    )
    parser.add_argument(
        "--naming",
        type=str,
        default=None,
        help="JSON string overriding naming entries.",
    )
    parser.add_argument(
        "--ordering",
        type=str,
        default=None,
        help="JSON string overriding ordering entries.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution used when saving the figure.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=75000,
        help="Maximum number of weight samples to draw per parameter (0 disables subsampling).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=13,
        help="Random seed for the optional subsampling step.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional CSV file storing aggregated weight statistics.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        help="Display weight magnitudes on a log scale. Positive and negative weights "
        "are shown as separate distributions using symmetric log scaling.",
    )

    return parser.parse_args()


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive IO guard
        logger.error("Failed to load checkpoint %s: %s", path, exc)
        return {}

    if isinstance(checkpoint, dict):
        for candidate in ("state_dict", "model_state_dict", "weights"):
            maybe_state = checkpoint.get(candidate)
            if isinstance(maybe_state, dict):
                return maybe_state
        if all(isinstance(tensor, torch.Tensor) for tensor in checkpoint.values()):
            return checkpoint

    logger.warning("No state_dict found in %s", path)
    return {}


def _infer_seed(path: Path) -> Optional[str]:
    run_dir = path.parent.parent.name if path.parent.parent else ""
    for pattern in SEED_PATTERNS:
        match = pattern.search(run_dir)
        if match:
            return match.group(1)
    return None


def _infer_category_value(path: Path, category_key: Optional[str]) -> Optional[str]:
    if not category_key:
        return None

    path_str = str(path)
    pattern = re.compile(key_value_pattern(category_key) + r"([^/\\+_]+)")
    match = pattern.search(path_str)
    if match:
        return match.group(1)

    logger.debug("Could not extract %s from path %s", category_key, path)
    return None


def _infer_layer_name(parameter: str) -> str:
    match = LAYER_PATTERN.search(parameter)
    if match:
        return match.group(1)
    return parameter.split(".")[0]


def _infer_connection_type(parameter: str) -> str:
    lowered = parameter.lower()
    if any(token in lowered for token in RECURRENT_TOKENS):
        return "recurrence"
    return "feedforward"


def _state_dict_to_frame(
    state_dict: Dict[str, torch.Tensor],
    source: Path,
    category: str,
    seed: Optional[str],
    status: Optional[str],
    max_samples: Optional[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for parameter, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if ".source" in parameter:
            continue
        if "weight" not in parameter.lower():
            continue
        if tensor.numel() == 0:
            continue

        layer = _infer_layer_name(parameter)
        connection_type = _infer_connection_type(parameter)

        values = tensor_to_numpy(tensor.detach()).reshape(-1)
        if max_samples and max_samples > 0 and values.size > max_samples:
            indices = rng.choice(values.size, size=max_samples, replace=False)
            values = values[indices]

        frame = pd.DataFrame(
            {
                "weight": values,
                "layer": layer,
                "connection_type": connection_type,
                "category": category,
                "seed": seed,
                "status": status,
                "parameter": parameter,
                "source": str(source),
            }
        )
        frames.append(frame)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "weight",
            "layer",
            "connection_type",
            "category",
            "seed",
            "status",
            "parameter",
            "source",
        ]
    )


def load_weight_table(
    paths: Sequence[Path],
    category_key: Optional[str],
    max_samples: Optional[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            logger.warning("Skipping missing checkpoint %s", path)
            continue

        state_dict = _load_state_dict(path)
        if not state_dict:
            continue

        category_value = _infer_category_value(path, category_key) or "overall"
        category_value = standardize_category_value(category_value)
        seed_value = _infer_seed(path)
        status_value = standardize_category_value(path.stem)

        frame = _state_dict_to_frame(
            state_dict=state_dict,
            source=path,
            category=category_value,
            seed=seed_value,
            status=status_value,
            max_samples=max_samples,
            rng=rng,
        )

        if frame.empty:
            logger.warning("No eligible weights found in %s", path)
            continue

        logger.info(
            "Loaded %d weight samples from %s (category=%s, status=%s, seed=%s)",
            len(frame),
            path.name,
            category_value,
            status_value or "NA",
            seed_value or "NA",
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "weight",
                "layer",
                "connection_type",
                "category",
                "seed",
                "status",
                "parameter",
                "source",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["connection_type"] = combined["connection_type"].map(
        standardize_category_value
    )
    combined["layer"] = combined["layer"].astype(str)
    combined["category"] = combined["category"].replace("", "overall")
    combined["status"] = combined["status"].replace("", "trained")

    logger.info(
        "Aggregated %d total weight samples across %d checkpoints",
        len(combined),
        len(frames),
    )

    return combined


def _prepare_dimension(
    df: pd.DataFrame,
    dimension: Optional[str],
    config: Dict[str, Dict[str, str]],
    category_key: Optional[str],
) -> Optional[DimensionMeta]:
    if dimension is None or dimension == "none":
        return None
    if dimension not in df.columns:
        raise KeyError(f"Column '{dimension}' not present in dataframe")

    column = dimension
    df[column] = df[column].map(standardize_category_value)
    unique_values = [value for value in df[column].tolist() if value != ""]
    if not unique_values:
        return None

    seen: List[str] = []
    for value in unique_values:
        if value not in seen:
            seen.append(value)

    if dimension == "category" and category_key:
        key_for_order = category_key
    elif dimension == "layer":
        key_for_order = "layers"
    else:
        key_for_order = dimension

    ordering = get_ordering(key_for_order, config) or []
    ordering_normalized = [standardize_category_value(value) for value in ordering]
    if dimension == "layer":
        ordering_normalized.reverse()

    # Build display map first so we can also match ordering by display name
    display_map_all = {value: get_display_name(value, config) for value in seen}

    # Match ordering against both raw values and their display names
    raw_by_display = {}
    for raw_value, display_value in display_map_all.items():
        raw_by_display.setdefault(display_value, raw_value)

    ordered_values: List[str] = []
    for ordering_entry in ordering_normalized:
        if ordering_entry in seen and ordering_entry not in ordered_values:
            ordered_values.append(ordering_entry)
        elif ordering_entry in raw_by_display:
            raw_value = raw_by_display[ordering_entry]
            if raw_value not in ordered_values:
                ordered_values.append(raw_value)
    ordered_values.extend([value for value in seen if value not in ordered_values])

    display_map = {value: get_display_name(value, config) for value in ordered_values}
    display_column = f"{dimension}_display"
    df[display_column] = df[column].map(
        lambda value: display_map.get(value, get_display_name(value, config))
    )

    display_order = [display_map[value] for value in ordered_values]

    palette_config = config.get("palette", {})
    palette: Dict[str, str] = {}
    for value in ordered_values:
        display_value = display_map[value]
        color = palette_config.get(value) or palette_config.get(display_value)
        if color:
            palette[display_value] = color
    if not palette:
        palette = None

    return DimensionMeta(
        column=column,
        display_column=display_column,
        values=ordered_values,
        display_order=display_order,
        display_map=display_map,
        palette=palette,
        label_key=key_for_order,
    )


def _format_dimension_label(
    value: str,
    meta: DimensionMeta,
    config: Dict[str, Dict[str, str]],
) -> str:
    label_key = meta.label_key or meta.column
    dimension_label = get_display_name(label_key, config)
    value_label = meta.display_map.get(value, get_display_name(value, config))

    if (
        dimension_label
        and dimension_label != label_key
        and dimension_label != value_label
    ):
        return f"{dimension_label}: {value_label}"
    return value_label


def _resolve_axis_label(
    key: Optional[str], config: Dict[str, Dict[str, str]], fallback: str
) -> str:
    if key:
        label = get_display_name(key, config)
        if label != key:
            return label
        return key.replace("_", " ").title()
    return fallback


def _report_connection_type_metrics(
    data: pd.DataFrame,
    dimension_order: Sequence[str],
    dimension_map: Dict[str, DimensionMeta],
    config: Dict[str, Dict[str, str]],
) -> None:
    if "connection_type" not in dimension_map:
        return

    connection_meta = dimension_map["connection_type"]
    other_dimensions = [dim for dim in dimension_order if dim != "connection_type"]
    group_columns = [dimension_map[dim].column for dim in other_dimensions]

    grouped = data.groupby(group_columns + [connection_meta.column], dropna=False)[
        "weight"
    ]

    stats = grouped.agg(
        mean="mean",
        variance=lambda series: float(series.var(ddof=0)),
    ).reset_index()

    if stats.empty:
        logger.info("Connection metrics: no data available for requested grouping.")
        return

    metrics: Dict[Tuple[str, ...], Dict[str, Dict[str, float]]] = {}
    for _, row in stats.iterrows():
        key = tuple(standardize_category_value(row[col]) for col in group_columns)
        connection_value = standardize_category_value(row[connection_meta.column])
        if connection_value.lower() not in {"feedforward", "recurrence"}:
            continue

        metrics.setdefault(key, {})[connection_value] = {
            "mean": float(row["mean"]),
            "variance": float(row["variance"]),
        }

    if not metrics:
        logger.info("Connection metrics: feedforward/recurrent data not found.")
        return

    for key, values in metrics.items():
        feedforward_stats = values.get("feedforward")
        recurrence_stats = values.get("recurrence")
        if not feedforward_stats or not recurrence_stats:
            continue

        mean_diff = feedforward_stats["mean"] - recurrence_stats["mean"]
        recurrence_variance = recurrence_stats["variance"]
        if recurrence_variance == 0.0:
            variance_ratio_repr = "undefined (variance recurrence=0)"
        else:
            variance_ratio = feedforward_stats["variance"] / recurrence_variance
            variance_ratio_repr = f"{variance_ratio:.4f}"

        if not other_dimensions:
            group_label = "Overall"
        else:
            parts: List[str] = []
            for idx, dimension in enumerate(other_dimensions):
                meta = dimension_map[dimension]
                raw_value = key[idx]
                display_value = meta.display_map.get(
                    raw_value, get_display_name(raw_value, config)
                )
                label = _resolve_axis_label(
                    meta.label_key or dimension, config, dimension.title()
                )
                parts.append(f"{label}: {display_value}")
            group_label = " | ".join(parts)

        logger.info(
            "Connection metrics [%s] -> mean diff (feedforward - recurrence): %.4f | variance ratio: %s",
            group_label,
            mean_diff,
            variance_ratio_repr,
        )


def plot_weight_distributions(
    df: pd.DataFrame,
    row_dimension: str,
    column_dimension: str,
    hue_dimension: str,
    x_dimension: str,
    config: Dict[str, Dict[str, str]],
    category_key: Optional[str],
    log_scale: bool = False,
) -> plt.Figure:
    data = df.copy()
    if "layer" in data.columns:
        data = data[data["layer"] != "classifier"]

    row_dimension = row_dimension if row_dimension != "none" else None
    column_dimension = column_dimension if column_dimension != "none" else None
    hue_dimension = hue_dimension if hue_dimension != "none" else None

    if row_dimension and hue_dimension and row_dimension == hue_dimension:
        logger.warning(
            "Row and hue dimensions are identical (%s); disabling hue dimension.",
            row_dimension,
        )
        hue_dimension = None

    if column_dimension and row_dimension and column_dimension == row_dimension:
        logger.warning(
            "Row and column dimensions are identical (%s); disabling column dimension.",
            column_dimension,
        )
        column_dimension = None

    if column_dimension and hue_dimension and column_dimension == hue_dimension:
        logger.warning(
            "Column and hue dimensions are identical (%s); disabling hue dimension.",
            column_dimension,
        )
        hue_dimension = None

    x_dimension = x_dimension if x_dimension not in (None, "none") else None

    if x_dimension:
        x_meta = _prepare_dimension(data, x_dimension, config, category_key)
        if not x_meta:
            raise ValueError(
                f"No data available for x-axis dimension '{x_dimension}'."
            )
    else:
        constant_column = "__x_axis__"
        constant_display = "__x_axis_display__"
        data[constant_column] = "overall"
        data[constant_display] = "All"
        x_meta = DimensionMeta(
            column=constant_column,
            display_column=constant_display,
            values=["overall"],
            display_order=["All"],
            display_map={"overall": "All"},
            palette=None,
            label_key=None,
        )

    row_meta = _prepare_dimension(data, row_dimension, config, category_key)
    column_meta = _prepare_dimension(data, column_dimension, config, category_key)
    hue_meta = _prepare_dimension(data, hue_dimension, config, category_key)

    row_values = row_meta.values if row_meta else [None]
    column_values = column_meta.values if column_meta else [None]

    n_rows = len(row_values)
    n_cols = len(column_values)

    if x_meta.display_order:
        x_order = list(x_meta.display_order)
    else:
        x_order = data[x_meta.display_column].dropna().unique().tolist()
    if not x_order:
        raise ValueError("X-axis ordering could not be determined.")

    dimension_entries: List[Tuple[str, Optional[DimensionMeta]]] = []
    if row_dimension:
        dimension_entries.append((row_dimension, row_meta))
    if column_dimension:
        dimension_entries.append((column_dimension, column_meta))
    if hue_dimension:
        dimension_entries.append((hue_dimension, hue_meta))
    if x_dimension:
        dimension_entries.append((x_dimension, x_meta))

    dimension_map: Dict[str, DimensionMeta] = {}
    dimension_order: List[str] = []
    for dimension, meta in dimension_entries:
        if not dimension or meta is None:
            continue
        if dimension not in dimension_map:
            dimension_map[dimension] = meta
            dimension_order.append(dimension)

    if "connection_type" in dimension_map:
        _report_connection_type_metrics(data, dimension_order, dimension_map, config)

    width = max(
        DEFAULT_LAYOUT["min_width"],
        DEFAULT_LAYOUT["subplot_width"] * max(1, n_cols),
    )
    if len(x_order) > 4:
        width = max(width, DEFAULT_LAYOUT["subplot_width"] * len(x_order))
    height = max(
        DEFAULT_LAYOUT["min_height"],
        DEFAULT_LAYOUT["subplot_height"] * max(1, n_rows),
    )

    if log_scale:
        abs_nonzero = data["weight"][data["weight"] != 0].abs()
        if abs_nonzero.empty:
            _symlog_linthresh = 1e-4
        else:
            _symlog_linthresh = float(np.percentile(abs_nonzero, 5))
            _symlog_linthresh = max(_symlog_linthresh, 1e-8)

    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height), sharex=True, sharey=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    show_legend = False

    for row_idx, row_value in enumerate(row_values):
        for col_idx, col_value in enumerate(column_values):
            ax = axes[row_idx, col_idx]

            subset = data
            if row_meta and row_value is not None:
                subset = subset[subset[row_meta.column] == row_value]
            if column_meta and col_value is not None:
                subset = subset[subset[column_meta.column] == col_value]

            if subset.empty:
                ax.set_visible(False)
                continue

            hue_column = hue_meta.display_column if hue_meta else None
            hue_order = hue_meta.display_order if hue_meta else None
            palette = hue_meta.palette if hue_meta else None
            split = (
                hue_column is not None
                and hue_dimension == "connection_type"
                and hue_meta is not None
                and len(hue_meta.values) == 2
            )

            if log_scale:
                ax.set_yscale("symlog", linthresh=_symlog_linthresh)

            sns.violinplot(
                data=subset,
                x=x_meta.display_column,
                y="weight",
                hue=hue_column,
                order=x_order,
                hue_order=hue_order,
                palette=palette,
                ax=ax,
                inner="box",
                cut=0,
                density_norm="width",
                split=split,
                width=0.95,
            )

            ax.axhline(0.0, linestyle="-", linewidth=1.5, color="black", zorder=1)

            ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Weight")
            else:
                ax.set_ylabel("")

            title_parts: List[str] = []
            if row_meta and row_value is not None:
                title_parts.append(
                    _format_dimension_label(row_value, row_meta, config)
                )
            if column_meta and col_value is not None:
                title_parts.append(
                    _format_dimension_label(col_value, column_meta, config)
                )
            ax.set_title(
                " | ".join(title_parts) if title_parts else "", fontweight="bold"
            )

            if hue_meta and ax.legend_:
                handles, labels = ax.get_legend_handles_labels()
                deduped_handles: List = []
                deduped_labels: List[str] = []
                for handle, label in zip(handles, labels):
                    if label not in deduped_labels:
                        deduped_handles.append(handle)
                        deduped_labels.append(label)

                legend_title = get_display_name(hue_meta.label_key, config)
                if legend_title == hue_meta.label_key:
                    legend_title = None

                if show_legend:
                    legend = ax.legend(
                        deduped_handles,
                        deduped_labels,
                        title=legend_title,
                        frameon=True,
                        loc="center right",
                    )
                    if legend and legend.get_title():
                        legend.get_title().set_fontweight("bold")
                else:
                    ax.legend_.remove()
                    show_legend = True

            ax.tick_params(axis="x", rotation=20)

    # Show only every second y-tick label for readability (after loop so
    # shared y-limits are finalized)
    for ax in axes.flat:
        if ax.get_visible():
            y_ticks = ax.get_yticks()
            y_labels = [
                "" if idx % 2 else ax.yaxis.get_major_formatter()(tick)
                for idx, tick in enumerate(y_ticks)
            ]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)

    # x_axis_label = _resolve_axis_label(
    #     x_meta.label_key or x_dimension, config, "Group"
    # )
    # for ax in axes[-1, :]:
    #     if ax.get_visible():
    #         ax.set_xlabel(x_axis_label)

    for ax in axes.flat:
        if ax.get_visible():
            sns.despine(ax=ax, left=True)

    fig.tight_layout()
    return fig


def export_summary(
    df: pd.DataFrame,
    output_path: Path,
    group_columns: Optional[List[str]] = None,
) -> None:
    if df.empty:
        logger.warning("No data available to export summary table")
        return

    if group_columns is None:
        group_columns = ["layer", "connection_type", "category", "status"]

    group_columns = [col for col in group_columns if col in df.columns]
    if not group_columns:
        logger.warning("No valid group columns found for summary export")
        return

    summary = (
        df.groupby(group_columns, dropna=False)["weight"]
        .agg(
            mean="mean",
            median="median",
            min="min",
            max="max",
            std="std",
        )
        .reset_index()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    logger.info("Saved summary statistics to %s", output_path)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )

    config = load_config_from_args(
        palette_str=args.palette,
        naming_str=args.naming,
        ordering_str=args.ordering,
    )

    rng = np.random.default_rng(args.sample_seed)
    max_samples = args.max_samples if args.max_samples > 0 else None

    weight_table = load_weight_table(
        paths=args.input,
        category_key=args.category_key,
        max_samples=max_samples,
        rng=rng,
    )

    if weight_table.empty:
        raise ValueError("No weight tensors found in provided checkpoints.")

    if args.summary_output:
        group_dimensions = [
            dim
            for dim in [args.row, args.column, args.hue, args.x_axis]
            if dim and dim != "none"
        ]
        export_summary(
            weight_table,
            args.summary_output,
            group_columns=group_dimensions or None,
        )

    plot_weight_distributions(
        df=weight_table,
        row_dimension=args.row,
        column_dimension=args.column,
        hue_dimension=args.hue,
        x_dimension=args.x_axis,
        config=config,
        category_key=args.category_key,
        log_scale=args.log_scale,
    )

    save_plot(args.output, dpi=args.dpi)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
