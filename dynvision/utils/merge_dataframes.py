"""DataFrame merging utility.

This module provides functionality to:
- Merge multiple DataFrames from different sources
- Add parameters as columns
- Handle parameter distribution across frames
- Validate and optimize merging operations

Usage:
    python merge_dataframes.py \
        --data results1.csv results2.csv \
        --output merged.csv \
        --param1 value1 value2 \
        --param2 shared_value
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from dynvision.utils import load_df, parse_kwargs, tqdm


logger = logging.getLogger(__name__)


def validate_inputs(
    data_files: List[Path], output_file: Path, params: Dict[str, Any]
) -> None:
    """Validate input parameters.

    Args:
        data_files: List of input CSV files
        output_file: Output file path
        params: Additional parameters

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If input files don't exist
    """
    if not data_files:
        raise ValueError("No input files provided")

    for file in data_files:
        if not file.exists():
            raise FileNotFoundError(f"Input file not found: {file}")

    if output_file.exists():
        logger.warning(f"Output file already exists: {output_file}")

    for k, v in params.items():
        if hasattr(v, "__len__") and len(v) not in [1, len(data_files)]:
            raise ValueError(
                f"Parameter {k} must have length 1 or {len(data_files)}, got {len(v)}"
            )


def merge_dataframes(
    data_files: List[Path], params: Dict[str, Any], show_progress: bool = True
) -> pd.DataFrame:
    """Merge DataFrames with parameters.

    Args:
        data_files: List of input CSV files
        params: Additional parameters
        show_progress: Whether to show progress bar

    Returns:
        Merged DataFrame

    Raises:
        RuntimeError: If merging fails
    """
    iterator = tqdm(enumerate(data_files)) if show_progress else enumerate(data_files)
    full_df = None

    try:
        for i, datafile in iterator:
            df = load_df(datafile)

            # Add per-frame parameters
            for k, v in params.items():
                if hasattr(v, "__len__") and len(v) == len(data_files):
                    df[k] = v[i]

            if full_df is None:
                full_df = df
            else:
                full_df = pd.concat((full_df, df), axis=0, ignore_index=True)

            del df  # Free memory

        # Add shared parameters
        for k, v in params.items():
            if hasattr(v, "__len__") and len(v) == 1:
                full_df[k] = v[0]

        return full_df

    except Exception as e:
        raise RuntimeError(f"Failed to merge DataFrames: {e}")


# Configure argument parser
CLI = argparse.ArgumentParser(
    description="Merge DataFrames with parameters",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
CLI.add_argument("--data", nargs="+", type=Path, help="Input CSV files", required=True)
CLI.add_argument("--output", type=Path, help="Output CSV file", required=True)
CLI.add_argument("--progress", type=bool, default=True, help="Show progress bar")

if __name__ == "__main__":
    args, unknown = CLI.parse_known_args()
    # add optional addition parameters
    params = parse_kwargs(unknown)

    try:

        validate_inputs(args.data, args.output, params)

        full_df = merge_dataframes(args.data, params, args.progress)

        logger.info(f"Saving merged DataFrame to {args.output}")
        full_df.to_csv(args.output, index=False)
        logger.info(f"Successfully merged {len(args.data)} files")

    except Exception as e:
        logger.error(f"Failed to merge DataFrames: {e}")
        raise
