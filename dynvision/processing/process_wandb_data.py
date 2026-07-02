import pandas as pd
import numpy as np
import re
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_param_value(column_name: str, param_key: str) -> Optional[str]:
    """Extract parameter value from column name.

    Supports formats:
    - 'rctype=full' -> 'full'
    - 'activity_loss_weight: 0.05' -> '0.05'
    """
    # Try colon-space format first (W&B default): "param_key: value"
    match = re.search(rf"{param_key}:\s*([^-]+?)(?:\s+-|$)", column_name)
    if match:
        return match.group(1).strip()

    # Fallback to equals format: "param_key=value"
    match = re.search(rf"{param_key}=([^+_\s-]+)", column_name)
    return match.group(1) if match else None


def identify_x_variable(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Identify x-axis variable and its unit type.

    Returns:
        (column_name, unit_type) where unit_type is 'time', 'memory', or None
    """
    first_col = df.columns[0]
    x_candidates = ["epoch", "step", "time", "iteration"]

    if any(cand in first_col.lower() for cand in x_candidates):
        x_var = first_col
    else:
        x_var = first_col

    # Detect if x is time or memory
    x_lower = x_var.lower()

    time_indicators = ["time", "duration", "seconds", "secs", "elapsed"]
    memory_indicators = ["memory", "mem", "bytes", "mb", "gb", "ram"]

    if any(ind in x_lower for ind in time_indicators):
        return x_var, "time"
    elif any(ind in x_lower for ind in memory_indicators):
        return x_var, "memory"
    else:
        return x_var, None


def identify_metric_columns(
    df: pd.DataFrame, exclude_suffixes: List[str] = ["__MIN", "__MAX"]
) -> Dict[str, List[str]]:
    """Group columns by metric type, excluding MIN/MAX variants."""
    metric_groups = {}

    for col in df.columns[1:]:
        if any(suffix in col for suffix in exclude_suffixes):
            continue

        # Try to extract metric name from ' - <metric>' pattern
        parts = col.split(" - ")
        metric_name = parts[-1].strip() if len(parts) >= 2 else col

        if metric_name.startswith("_"):
            continue

        metric_groups.setdefault(metric_name, []).append(col)

    return metric_groups


def extract_category_keys(columns: List[str], max_categories: int = 2) -> List[str]:
    """
    Identify varying parameter keys across columns.
    Fallback: if no parameters found, use column index as category.
    """
    # Try parameter extraction
    all_params = {}
    for col in columns:
        # Try colon-space format first (W&B default): "param_key: value"
        params_colon = re.findall(r"([a-zA-Z_]+):\s*([^-]+?)(?:\s+-)", col)
        for param_key, param_value in params_colon:
            all_params.setdefault(param_key, set()).add(param_value.strip())

        # Also try equals format: "param_key=value"
        params_equals = re.findall(r"([a-zA-Z_]+)=([^+_\s-]+)", col)
        for param_key, param_value in params_equals:
            all_params.setdefault(param_key, set()).add(param_value)

    # Find parameters with multiple values
    varying_params = [key for key, values in all_params.items() if len(values) > 1]

    if not varying_params:
        # Fallback: use column index as category
        logger.warning(
            "No varying parameters found. Using column indices as categories."
        )
        return ["column_id"]

    # Prioritize common category parameters
    priority = ["feedback", "rctype", "recurrence_type", "model", "dataset"]
    varying_params.sort(key=lambda x: (x not in priority, x))

    return varying_params[:max_categories]


def calculate_statistics(
    series: pd.Series, x_series: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a time series.
    Robust to sparse data with NaN values.

    Args:
        series: Y-values (e.g., accuracy, epoch number)
        x_series: X-values (e.g., time, epoch number)

    Returns:
        Dictionary of statistics (slope = dy/dx, inverse_slope = dx/dy)
    """
    # Convert series to numeric, coercing errors to NaN
    series_numeric = pd.to_numeric(series, errors="coerce")

    # Remove NaN values from y series
    clean_series = series_numeric.dropna()

    if len(clean_series) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "initial": np.nan,
            "final": np.nan,
            "slope": np.nan,
            "inverse_slope": np.nan,
            "peak_step": np.nan,
        }

    stats = {
        "mean": clean_series.mean(),
        "median": clean_series.median(),
        "std": clean_series.std(),
        "min": clean_series.min(),
        "max": clean_series.max(),
        "initial": clean_series.iloc[0],
        "final": clean_series.iloc[-1],
    }

    # Calculate slope using actual x-values (not indices)
    if len(clean_series) > 1 and x_series is not None:
        # Convert x_series to numeric as well
        x_series_numeric = pd.to_numeric(x_series, errors="coerce")

        # Align x and y by removing NaN from both
        # Get x values at the same indices as clean y values
        clean_x_series = x_series_numeric.loc[clean_series.index].dropna()
        # Get y values at the same indices as clean x values
        clean_series_aligned = clean_series.loc[clean_x_series.index]

        if len(clean_x_series) > 1 and len(clean_series_aligned) > 1:
            x_values = clean_x_series.values
            y_values = clean_series_aligned.values

            # Remove any remaining NaN values
            mask = ~(np.isnan(x_values) | np.isnan(y_values))
            x_values = x_values[mask]
            y_values = y_values[mask]

            # Check for constant x or y (would make slope infinite or zero)
            if np.std(x_values) > 1e-10 and np.std(y_values) > 1e-10:
                slope, _ = np.polyfit(x_values, y_values, 1)
                stats["slope"] = slope
                stats["inverse_slope"] = 1.0 / slope if abs(slope) > 1e-10 else np.nan
            else:
                stats["slope"] = np.nan
                stats["inverse_slope"] = np.nan
        else:
            stats["slope"] = np.nan
            stats["inverse_slope"] = np.nan
    else:
        stats["slope"] = np.nan
        stats["inverse_slope"] = np.nan

    # Find step at which peak occurred
    if x_series is not None and len(clean_series) > 0:
        peak_idx = clean_series.idxmax()
        stats["peak_step"] = (
            x_series.loc[peak_idx] if peak_idx in x_series.index else np.nan
        )
    else:
        stats["peak_step"] = clean_series.idxmax() if len(clean_series) > 0 else np.nan

    return stats


def add_unit_transformations(
    stats_df: pd.DataFrame, x_unit_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Add transformed columns for memory (bytes→GB) and time (seconds→minutes).

    Transforms based on BOTH x-axis and y-axis units:
    - If y is memory/time: transform basic stats (mean, median, etc.)
    - If x is memory/time: transform slopes (affects rate denominators)
    - If both: transform appropriately

    Args:
        stats_df: Statistics dataframe
        x_unit_type: Type of x-axis ('time', 'memory', or None)

    Returns:
        Enhanced dataframe with additional unit columns
    """
    df = stats_df.copy()

    # Conversion factors
    BYTES_TO_GB = 1 / (1024**3)
    SECONDS_TO_MINUTES = 1 / 60

    # Detect memory and time metrics in y-axis
    memory_indicators = ["memory", "mem", "bytes", "mb", "gb", "ram", "gpu"]
    time_indicators = ["time", "duration", "seconds", "secs", "elapsed", "epoch_time"]

    transformed_metrics = []

    for idx, row in df.iterrows():
        metric = row["metric"].lower() if "metric" in row else ""

        y_is_memory = any(ind in metric for ind in memory_indicators)
        y_is_time = any(ind in metric for ind in time_indicators)

        # Transform basic statistics (only if y is memory/time)
        if y_is_memory:
            if row["metric"] not in transformed_metrics:
                transformed_metrics.append(f"{row['metric']} (bytes→GB)")
            for col in ["mean", "median", "std", "min", "max", "initial", "final"]:
                if col in df.columns:
                    df.loc[idx, f"{col}_gb"] = (
                        row[col] * BYTES_TO_GB if pd.notna(row[col]) else np.nan
                    )

        if y_is_time:
            if row["metric"] not in transformed_metrics:
                transformed_metrics.append(f"{row['metric']} (sec→min)")
            for col in ["mean", "median", "std", "min", "max", "initial", "final"]:
                if col in df.columns:
                    df.loc[idx, f"{col}_min"] = (
                        row[col] * SECONDS_TO_MINUTES if pd.notna(row[col]) else np.nan
                    )

        # Transform slopes - depends on BOTH x and y
        if "slope" in df.columns and pd.notna(row["slope"]):
            slope_val = row["slope"]

            # Y transformation
            if y_is_memory:
                df.loc[idx, "slope_gb"] = slope_val * BYTES_TO_GB
                slope_val_y_transformed = slope_val * BYTES_TO_GB
                y_suffix = "_gb"
            elif y_is_time:
                df.loc[idx, "slope_min"] = slope_val * SECONDS_TO_MINUTES
                slope_val_y_transformed = slope_val * SECONDS_TO_MINUTES
                y_suffix = "_min"
            else:
                slope_val_y_transformed = slope_val
                y_suffix = ""

            # X transformation (if x is time/memory)
            if x_unit_type == "time":
                # slope is y/second → y/minute (divide by 60 because minutes are bigger)
                df.loc[idx, f"slope{y_suffix}_per_min"] = (
                    slope_val_y_transformed / SECONDS_TO_MINUTES
                    if pd.notna(slope_val_y_transformed)
                    else np.nan
                )
            elif x_unit_type == "memory":
                # slope is y/byte → y/GB (divide by 1024³ because GB are bigger)
                df.loc[idx, f"slope{y_suffix}_per_gb"] = (
                    slope_val_y_transformed / BYTES_TO_GB
                    if pd.notna(slope_val_y_transformed)
                    else np.nan
                )

        # Transform inverse slopes - depends on BOTH x and y
        if "inverse_slope" in df.columns and pd.notna(row["inverse_slope"]):
            inv_slope_val = row["inverse_slope"]

            # X transformation (denominator in x/y)
            if x_unit_type == "time":
                # inverse_slope is seconds/y → minutes/y (multiply by 1/60)
                df.loc[idx, "inverse_slope_min"] = inv_slope_val * SECONDS_TO_MINUTES
                inv_slope_x_transformed = inv_slope_val * SECONDS_TO_MINUTES
                x_suffix = "_min"
            elif x_unit_type == "memory":
                # inverse_slope is bytes/y → GB/y (multiply by 1/1024³)
                df.loc[idx, "inverse_slope_gb"] = inv_slope_val * BYTES_TO_GB
                inv_slope_x_transformed = inv_slope_val * BYTES_TO_GB
                x_suffix = "_gb"
            else:
                inv_slope_x_transformed = inv_slope_val
                x_suffix = ""

            # Y transformation (if y is time/memory)
            if y_is_memory:
                # inverse_slope is x/bytes → x/GB (divide by 1024³)
                df.loc[idx, f"inverse_slope{x_suffix}_per_gb"] = (
                    inv_slope_x_transformed / BYTES_TO_GB
                    if pd.notna(inv_slope_x_transformed)
                    else np.nan
                )
            elif y_is_time:
                # inverse_slope is x/seconds → x/minutes (divide by 60)
                df.loc[idx, f"inverse_slope{x_suffix}_per_min"] = (
                    inv_slope_x_transformed / SECONDS_TO_MINUTES
                    if pd.notna(inv_slope_x_transformed)
                    else np.nan
                )

    if transformed_metrics:
        logger.info(f"Transformed y-metrics: {', '.join(transformed_metrics)}")
    if x_unit_type:
        x_transform_msg = "seconds→minutes" if x_unit_type == "time" else "bytes→GB"
        logger.info(f"X-axis transformation: {x_transform_msg}")

    return df


def format_for_display(stats_df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Format numeric columns for display with specified decimal places.

    Args:
        stats_df: Statistics dataframe
        decimals: Number of decimal places

    Returns:
        Formatted dataframe for display
    """
    display_df = stats_df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "NaN"
        )

    return display_df


def extract_category_values(
    column_name: str, category_keys: List[str], column_index: int
) -> Dict[str, str]:
    """
    Extract category values from column name.
    Fallback: use column index if no parameters found.
    """
    if "column_id" in category_keys:
        return {"column_id": f"col_{column_index}"}

    values = {}
    for cat_key in category_keys:
        cat_value = extract_param_value(column_name, cat_key)
        if cat_value:
            values[cat_key] = cat_value

    return values


def process_wandb_csv(
    filepath: str,
    metric_filter: Optional[List[str]] = None,
    category_hints: Optional[List[str]] = None,
    add_transforms: bool = True,
    max_epoch: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process W&B CSV export and calculate statistics.

    Args:
        filepath: Path to CSV file
        metric_filter: Optional list of metrics to include
        category_hints: Optional hints for category parameter keys
        add_transforms: Add unit transformations (bytes→GB, seconds→minutes)
        max_epoch: Maximum epoch value to include (default: 300)

    Returns:
        (processed_dataframe, statistics_summary)
    """
    logger.info(f"Loading CSV: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Identify variables
    x_var, x_unit_type = identify_x_variable(df)
    logger.info(
        f"X variable: '{x_var}'" + (f" ({x_unit_type})" if x_unit_type else "")
    )

    # Filter epoch values by max_epoch (set values > max_epoch to NaN)
    if max_epoch is not None:
        # Find all epoch columns
        epoch_cols = [col for col in df.columns if "epoch" in col.lower()]
        if epoch_cols:
            total_filtered = 0
            for epoch_col in epoch_cols:
                # Count values that will be filtered
                filtered_count = (df[epoch_col] > max_epoch).sum()
                total_filtered += filtered_count
                # Set epoch values > max_epoch to NaN
                df.loc[df[epoch_col] > max_epoch, epoch_col] = np.nan

            if total_filtered > 0:
                logger.info(
                    f"Filtered {total_filtered} epoch values > {max_epoch} across {len(epoch_cols)} epoch column(s)"
                )

            # Remove rows where ALL values are NaN (optional cleanup)
            initial_rows = len(df)
            df = df.dropna(how="all", subset=epoch_cols)
            if len(df) < initial_rows:
                logger.info(
                    f"Removed {initial_rows - len(df)} rows with no valid epoch data"
                )
            logger.info(
                f"Shape after filtering: {df.shape[0]} rows × {df.shape[1]} columns"
            )

    metric_groups = identify_metric_columns(df)
    if metric_filter:
        metric_groups = {k: v for k, v in metric_groups.items() if k in metric_filter}

    logger.info(f"Metrics: {list(metric_groups.keys())}")

    # Identify categories
    all_metric_cols = [col for cols in metric_groups.values() for col in cols]
    category_keys = category_hints or extract_category_keys(all_metric_cols)
    logger.info(f"Categories: {category_keys}")

    # Process to long format
    processed_data = []

    for metric_name, columns in metric_groups.items():
        for col_idx, col in enumerate(columns):
            category_values = extract_category_values(col, category_keys, col_idx)

            if not category_values:
                continue

            for _, row in df.iterrows():
                # Only include rows where both x and y values are not NaN
                if pd.notna(row[col]) and pd.notna(row[x_var]):
                    # Create separate rows for each category
                    for cat_name, cat_value in category_values.items():
                        processed_data.append(
                            {
                                x_var: row[x_var],
                                "metric": metric_name,
                                "value": row[col],
                                "category": cat_name,
                                "category_values": cat_value,
                            }
                        )

    processed_df = pd.DataFrame(processed_data)
    logger.info(f"Processed shape: {processed_df.shape}")

    # Check if processed_df is empty
    if processed_df.empty:
        logger.error("No data was extracted. Possible issues:")
        logger.error("  1. Category parameter not found in column names")
        logger.error("  2. All values are NaN")
        logger.error(f"Sample column names from CSV: {list(df.columns[:5])}")
        logger.error(f"Looking for categories: {category_keys}")
        raise ValueError(
            "No data extracted - check category parameter names match column naming pattern"
        )

    # Warn if any category has very few data points
    group_cols = ["metric", "category", "category_values"]
    for group_key, group_df in processed_df.groupby(group_cols):
        if len(group_df) < 2:
            logger.warning(
                f"Category {group_key} has only {len(group_df)} data point(s) - statistics may be limited"
            )

    # Calculate statistics
    group_cols = ["metric", "category", "category_values"]
    stats_data = []

    for group_key, group_df in processed_df.groupby(group_cols):
        group_key = (group_key,) if not isinstance(group_key, tuple) else group_key

        stats = calculate_statistics(group_df["value"], group_df[x_var])
        stats_row = dict(zip(group_cols, group_key))
        stats_row.update(stats)
        stats_data.append(stats_row)

    stats_df = pd.DataFrame(stats_data).sort_values(group_cols).reset_index(drop=True)

    # Add unit transformations (bytes→GB, seconds→minutes)
    if add_transforms:
        stats_df = add_unit_transformations(stats_df, x_unit_type)

    # Log category values
    if "category" in processed_df.columns:
        for cat_name in processed_df["category"].unique():
            cat_df = processed_df[processed_df["category"] == cat_name]
            unique_vals = cat_df["category_values"].unique()
            logger.info(f"  {cat_name}: {list(unique_vals)}")

    return processed_df, stats_df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process Weights & Biases CSV exports and calculate statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data training_metrics.csv
  %(prog)s --data metrics.csv --output stats.csv --metrics train_accuracy val_accuracy
  %(prog)s --data metrics.csv --categories feedback rctype --verbose
  %(prog)s --data metrics.csv --max-epoch 100         # Only use first 100 epochs
  %(prog)s --data memory_data.csv --output stats.csv  # Auto-converts bytes→GB
  %(prog)s --data timing_data.csv --no-transform      # Disable auto-conversion

Notes:
  - Automatically detects and transforms memory metrics (bytes→GB)
  - Automatically detects and transforms time metrics (seconds→minutes)
  - Detects x-axis units and adjusts slopes accordingly
  - Filters data by --max-epoch if x-axis contains 'epoch' (default: 300)
  - Example: if x="time (seconds)" and y="epoch", slope is epochs/sec
    - slope_per_min gives epochs/minute
  - Example: if x="time (seconds)" and y="memory (bytes)":
    - slope_gb gives GB/sec
    - slope_gb_per_min gives GB/minute
  - Use --no-transform to disable automatic unit conversions
        """,
    )

    parser.add_argument(
        "--data", "-d", required=True, help="Path to W&B CSV export file"
    )
    parser.add_argument(
        "--output", "-o", help="Path to save statistics summary CSV (optional)"
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        help="Filter specific metrics (e.g., train_accuracy val_accuracy)",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="Hint category parameter keys (e.g., feedback rctype)",
    )
    parser.add_argument(
        "--save-processed", help="Optionally save processed long-format data"
    )
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Disable automatic unit transformations (bytes→GB, seconds→minutes)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=300,
        help="Maximum epoch value to include in statistics (default: 300)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress info logging"
    )

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    # Validate input file
    if not Path(args.data).exists():
        logger.error(f"File not found: {args.data}")
        return 1

    # Process CSV
    try:
        processed_df, stats_df = process_wandb_csv(
            args.data,
            metric_filter=args.metrics,
            category_hints=args.categories,
            add_transforms=not args.no_transform,
            max_epoch=args.max_epoch,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=not args.quiet)
        return 1

    # Display summary
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80 + "\n")

    # Check if unit transformations were added
    has_gb = any("_gb" in col for col in stats_df.columns)
    has_min = any("_min" in col for col in stats_df.columns)
    has_per_min = any("_per_min" in col for col in stats_df.columns)
    has_per_gb = any("_per_gb" in col for col in stats_df.columns)

    if has_gb or has_min or has_per_min or has_per_gb:
        transformations = []
        if has_gb:
            transformations.append("Y-axis: bytes→GB")
        if has_min:
            transformations.append("Y-axis: seconds→minutes")
        if has_per_min:
            transformations.append("X-axis: per second→per minute")
        if has_per_gb:
            transformations.append("X-axis: per byte→per GB")
        print("Unit transformations applied:")
        for t in transformations:
            print(f"  • {t}")
        print()

    # Format for display
    stats_display = format_for_display(stats_df)

    for metric in stats_df["metric"].unique():
        print(f"\n{metric.upper()}")
        print("-" * 80)
        metric_stats = stats_display[stats_display["metric"] == metric]
        print(metric_stats.to_string(index=False))

    # Save outputs
    if args.output:
        stats_df.to_csv(args.output, index=False)
        logger.info(f"Statistics saved to: {args.output}")

    if args.save_processed:
        processed_df.to_csv(args.save_processed, index=False)
        logger.info(f"Processed data saved to: {args.save_processed}")

    return 0


if __name__ == "__main__":
    exit(main())
