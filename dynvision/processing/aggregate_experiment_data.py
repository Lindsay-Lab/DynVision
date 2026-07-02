"""Aggregate individual test data into experiment-level dataset.

STAGE 2 of two-stage experiment processing pipeline.

This script:
1. Loads all test_data.csv files from individual tests (Stage 1 outputs)
2. Extracts metadata from corresponding .config.yaml files (parameter, category, additional_parameters)
3. Adds metadata columns to each dataframe
4. Concatenates into single dataframe
5. Optionally applies class-level resolution (sample → class aggregation)
6. Handles missing measures gracefully
7. Saves aggregated CSV

This is lightweight - no heavy computation, just concatenation and metadata extraction.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from dynvision.utils import str_to_bool
from dynvision.utils.data_utils import load_df

# Import metadata extraction and resolution functions from process_test_data
from dynvision.processing.process_test_data import (
    compress_to_presentation_level,
    _extract_metadata,
    FileMetadata,
)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def extract_metadata_for_aggregation(
    config_file: Path,
    test_data_file: Path,
    data_arg_key: str,
    category: str,
    extra_parameters: Sequence[str],
    extract_status: bool = False,
) -> FileMetadata:
    """Extract metadata from config file for a test.

    Wraps _extract_metadata but provides better error handling for aggregation context.

    Args:
        config_file: Path to test_outputs.csv.config.yaml
        test_data_file: Path to test_data.csv (for error messages)
        data_arg_key: Parameter key to extract
        category: Category key to extract
        extra_parameters: Additional parameter names
        extract_status: If True, extract status from file path

    Returns:
        FileMetadata with parameter, category, extra values, and status
    """
    try:
        return _extract_metadata(
            config_file=config_file,
            data_arg_key=data_arg_key,
            category=category,
            extra_parameters=extra_parameters,
            extract_status=extract_status,
        )
    except Exception as exc:
        logger.error(f"Failed to extract metadata for {test_data_file.name}: {exc}")
        raise


def load_test_data_with_metadata(
    test_data_file: Path,
    config_file: Path,
    data_arg_key: str,
    category: str,
    extra_parameters: Sequence[str],
    fail_on_missing: bool = True,
    extract_status: bool = False,
) -> Optional[pd.DataFrame]:
    """Load a single test_data.csv file and add metadata columns.

    Args:
        test_data_file: Path to test_data.csv
        config_file: Path to corresponding test_outputs.csv.config.yaml
        data_arg_key: Parameter key to extract
        category: Category key to extract
        extra_parameters: Additional parameter names
        fail_on_missing: Whether to raise error on missing files
        extract_status: If True, extract status from file path

    Returns:
        DataFrame with metadata columns added, or None if file missing and fail_on_missing=False
    """
    # Check files exist
    if not test_data_file.exists():
        msg = f"Test data file not found: {test_data_file}"
        if fail_on_missing:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return None

    if not config_file.exists():
        msg = f"Config file not found: {config_file}"
        if fail_on_missing:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return None

    # Load test data
    try:
        df = load_df(test_data_file)
    except Exception as exc:
        msg = f"Failed to load {test_data_file}: {exc}"
        if fail_on_missing:
            raise RuntimeError(msg) from exc
        logger.warning(msg)
        return None

    # Extract metadata
    try:
        metadata = extract_metadata_for_aggregation(
            config_file=config_file,
            test_data_file=test_data_file,
            data_arg_key=data_arg_key,
            category=category,
            extra_parameters=extra_parameters,
            extract_status=extract_status,
        )
    except Exception as exc:
        msg = f"Failed to extract metadata for {test_data_file}: {exc}"
        if fail_on_missing:
            raise RuntimeError(msg) from exc
        logger.warning(msg)
        return None

    # Detect collision between parameter and category (e.g., both are "idle")
    param_category_collision = category and data_arg_key == category

    # Determine category column name (prefix with train_ if collision)
    category_col_name = f"train_{category}" if param_category_collision else category

    # Track columns already used to avoid overwrites
    used_columns = {data_arg_key}

    # Add parameter column (test/data value)
    df[data_arg_key] = metadata.parameter_value

    # Add category column
    if category:
        df[category_col_name] = metadata.category_value
        used_columns.add(category_col_name)

    # Add status if present
    if metadata.status_value is not None:
        df["status"] = metadata.status_value
        used_columns.add("status")

    # Add extra parameters with collision avoidance
    for param_name, param_value in metadata.extra_values.items():
        if param_value is None:
            continue

        # Skip if column already used (avoids overwriting parameter/category)
        if param_name in used_columns:
            continue

        # Handle case where extra_param matches original category name but collision occurred
        # The train value is already in train_{category}, data value same as parameter_value
        if param_category_collision and param_name == category:
            continue

        df[param_name] = param_value
        used_columns.add(param_name)

    return df


def aggregate_test_data(
    test_data_files: List[Path],
    config_files: List[Path],
    data_arg_key: str,
    category: str,
    extra_parameters: Sequence[str],
    resolution: str = "sample",
    fail_on_missing: bool = True,
    extract_status: bool = False,
) -> tuple[pd.DataFrame, List[Path]]:
    """Aggregate multiple test_data.csv files into single experiment dataset.

    Args:
        test_data_files: List of paths to test_data.csv files
        config_files: List of paths to corresponding .config.yaml files
        data_arg_key: Parameter key to extract from configs
        category: Category key to extract from configs
        extra_parameters: Additional parameter names to extract
        resolution: 'sample' (keep sample-level) or 'class' (aggregate to class-level)
        fail_on_missing: Whether to raise error on missing files
        extract_status: If True, extract status from file paths

    Returns:
        (aggregated_df, successful_files) tuple
    """
    if len(test_data_files) != len(config_files):
        raise ValueError(
            f"Mismatch: {len(test_data_files)} test_data files vs "
            f"{len(config_files)} config files"
        )

    logger.info(f"Aggregating {len(test_data_files)} test data files...")
    logger.info(f"Resolution: {resolution}")
    logger.info(f"Parameter: {data_arg_key}")
    logger.info(f"Category: {category or '(none - single config aggregation)'}")
    if extra_parameters:
        logger.info(f"Additional parameters: {extra_parameters}")
    logger.info(f"Extract status: {extract_status}")

    # Detect and log parameter-category collision
    param_category_collision = category and data_arg_key == category
    if param_category_collision:
        logger.info(
            f"Collision detected: parameter '{data_arg_key}' == category '{category}'. "
            f"Using 'train_{category}' for model category, '{data_arg_key}' for test parameter."
        )

    dfs = []
    successful_files = []

    for test_data_file, config_file in zip(test_data_files, config_files):
        logger.info(f"  Loading: {test_data_file.name}")

        df = load_test_data_with_metadata(
            test_data_file=test_data_file,
            config_file=config_file,
            data_arg_key=data_arg_key,
            category=category,
            extra_parameters=extra_parameters,
            fail_on_missing=fail_on_missing,
            extract_status=extract_status,
        )

        if df is not None:
            logger.info(f"    Loaded {len(df)} rows, {len(df.columns)} columns")
            dfs.append(df)
            successful_files.append(test_data_file)
        else:
            logger.warning(f"    Skipped {test_data_file.name}")

    if not dfs:
        raise RuntimeError("No test data files loaded successfully")

    logger.info(f"\nSuccessfully loaded {len(dfs)}/{len(test_data_files)} files")

    # Concatenate all dataframes
    logger.info("Concatenating dataframes...")
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined shape before resolution: {combined.shape}")

    # Apply resolution transformation if requested
    if resolution == "class":
        logger.info("Applying class-level aggregation...")
        combined = compress_to_presentation_level(
            combined,
            id_cols=["first_label_index", "times_index"],
            drop_cols=["sample_index", "image_index"],
        )
        logger.info(f"Shape after class-level aggregation: {combined.shape}")
    else:
        logger.info("Keeping sample-level resolution (no aggregation)")

    # Sort by metadata and time
    # Status sorts first if present, then category, parameter, extra parameters, identifiers, time
    sort_cols = []

    # Add status first if extracted
    if extract_status and "status" in combined.columns:
        sort_cols.append("status")

    # Add category (if provided) and parameter
    # Use renamed column when parameter-category collision occurred
    param_category_collision = category and data_arg_key == category
    category_col_name = f"train_{category}" if param_category_collision else category

    if category:
        sort_cols.extend([category_col_name, data_arg_key])
    else:
        sort_cols.append(data_arg_key)

    # Add extra parameters (including auto-extracted epoch from status)
    for param in extra_parameters:
        if param in combined.columns:
            sort_cols.append(param)

    # Add identifier columns based on resolution
    if resolution == "sample" and "sample_index" in combined.columns:
        sort_cols.append("sample_index")
    elif resolution == "class" and "first_label_index" in combined.columns:
        sort_cols.append("first_label_index")

    # Add time index last
    sort_cols.append("times_index")

    logger.info(f"Sorting by: {sort_cols}")
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    return combined, successful_files


def validate_aggregated_data(df: pd.DataFrame, resolution: str) -> None:
    """Validate aggregated dataframe for consistency.

    Args:
        df: Aggregated dataframe
        resolution: Expected resolution ('sample' or 'class')

    Raises:
        ValueError: If validation fails
    """
    logger.info("\nValidating aggregated data...")

    # Check required columns based on resolution
    if resolution == "sample":
        required_cols = ["sample_index", "times_index", "first_label_index"]
    else:  # class
        required_cols = ["first_label_index", "times_index"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"  ✓ Required columns present: {required_cols}")

    # Check for unexpected NaN values in key columns
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"    Column '{col}' has {nan_count} NaN values")

    # Summary statistics
    logger.info(f"  Shape: {df.shape}")
    if resolution == "sample":
        logger.info(f"  Unique samples: {df['sample_index'].nunique()}")
    logger.info(f"  Unique classes: {df['first_label_index'].nunique()}")
    logger.info(f"  Unique times: {df['times_index'].nunique()}")

    logger.info("  ✓ Validation passed")


# Command line interface
parser = argparse.ArgumentParser(
    description="Aggregate test data into experiment dataset (Stage 2 of experiment processing)"
)
parser.add_argument(
    "--test_data",
    nargs="+",
    type=Path,
    required=True,
    help="Paths to test_data.csv files from Stage 1",
)
parser.add_argument(
    "--test_configs",
    nargs="+",
    type=Path,
    required=True,
    help="Paths to corresponding test_outputs.csv.config.yaml files",
)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Path to output aggregated CSV file",
)
parser.add_argument(
    "--parameter",
    type=str,
    required=True,
    help="Parameter key to extract from config files (e.g., 'dsteps', 'stim')",
)
parser.add_argument(
    "--category",
    type=str,
    default="",
    help="Category key to extract from file paths (e.g., 'rctype', 'tsteps'). Empty string for single config aggregation.",
)
parser.add_argument(
    "--additional_parameters",
    nargs="*",
    default=None,
    help="Additional parameter names to extract from config files",
)
parser.add_argument(
    "--sample_resolution",
    type=str,
    default="sample",
    choices=["sample", "class"],
    help="Resolution: 'sample' (keep sample-level) or 'class' (aggregate to class-level)",
)
parser.add_argument(
    "--fail_on_missing_inputs",
    type=str_to_bool,
    default="True",
    help="If True, raise error when input files missing; if False, skip and continue",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Determine if status extraction is requested based on additional_parameters
    extract_status = "status" in (args.additional_parameters or [])

    # Log configuration
    logger.info("=" * 80)
    logger.info("STAGE 2: EXPERIMENT DATA AGGREGATION")
    logger.info("=" * 80)
    logger.info(f"Test data files: {len(args.test_data)}")
    logger.info(f"Config files: {len(args.test_configs)}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Parameter: {args.parameter}")
    logger.info(f"Category: {args.category}")
    if args.additional_parameters:
        logger.info(f"Additional parameters: {args.additional_parameters}")
    logger.info(f"Resolution: {args.sample_resolution}")
    if extract_status:
        logger.info("Extract status: True (detected from additional_parameters)")
    logger.info(f"Fail on missing inputs: {args.fail_on_missing_inputs}")

    try:
        # Validate input counts match
        if len(args.test_data) != len(args.test_configs):
            msg = (
                f"Input count mismatch: {len(args.test_data)} test_data files vs "
                f"{len(args.test_configs)} config files"
            )
            if args.fail_on_missing_inputs:
                raise ValueError(msg)
            logger.warning(msg)

        # Aggregate data
        aggregated_df, successful_files = aggregate_test_data(
            test_data_files=args.test_data,
            config_files=args.test_configs,
            data_arg_key=args.parameter,
            category=args.category,
            extra_parameters=args.additional_parameters or [],
            resolution=args.sample_resolution,
            fail_on_missing=args.fail_on_missing_inputs,
            extract_status=extract_status,
        )

        # Validate
        validate_aggregated_data(aggregated_df, resolution=args.sample_resolution)

        # Save
        logger.info(f"\nSaving aggregated data to: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        aggregated_df.to_csv(args.output, index=False)
        logger.info(f" Successfully saved {len(aggregated_df)} rows")

        # Report on skipped files if any
        skipped_count = len(args.test_data) - len(successful_files)
        if skipped_count > 0:
            logger.warning(f"\n  Skipped {skipped_count} files due to errors")
            logger.warning("See warnings above for details")

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("AGGREGATION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        sys.exit(1)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED DATASET SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Resolution: {args.sample_resolution}")
    logger.info(f"Shape: {aggregated_df.shape}")
    logger.info(f"Columns: {len(aggregated_df.columns)}")

    # Handle collision case where category column is renamed to train_{category}
    param_category_collision = args.category and args.parameter == args.category
    category_col_name = (
        f"train_{args.category}" if param_category_collision else args.category
    )

    if category_col_name and category_col_name in aggregated_df.columns:
        logger.info(
            f"Unique {category_col_name} values: {aggregated_df[category_col_name].unique()}"
        )
    if args.parameter in aggregated_df.columns:
        logger.info(
            f"Unique {args.parameter} values: {aggregated_df[args.parameter].unique()}"
        )
    if "times_index" in aggregated_df.columns:
        logger.info(f"Time steps: {sorted(aggregated_df['times_index'].unique())}")

    if args.sample_resolution == "sample" and "sample_index" in aggregated_df.columns:
        logger.info(f"Samples: {aggregated_df['sample_index'].nunique()}")

    if "first_label_index" in aggregated_df.columns:
        logger.info(
            f"Presentation labels: {sorted(aggregated_df['first_label_index'].unique())}"
        )
