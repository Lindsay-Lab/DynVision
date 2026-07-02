"""Data utility functions for the DynVision toolbox.

This module provides data-related utilities:
- DataFrame loading and manipulation
- Progress tracking
- Basic data operations
"""

from typing import Any, Dict, Union
from pathlib import Path

import pandas as pd
from tqdm import tqdm as progress_bar

from dynvision.project_paths import project_paths
from .type_utils import dtypes


def load_df(path: Union[str, Path], dtypes: Dict[str, Any] = dtypes) -> pd.DataFrame:
    """Load DataFrame with appropriate column types.

    Args:
        path: Path to CSV file
        dtypes: Dictionary mapping column names to types

    Returns:
        DataFrame with properly typed columns
    """
    df = pd.read_csv(path, dtype=dtypes)
    df.drop(
        df.columns[df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    return df


def identity(*args: Any, **kwargs: Any) -> Any:
    """Identity function that returns first argument."""
    return args[0]


def tqdm(*args: Any, **kwargs: Any) -> Any:
    """Progress bar that adapts to environment.

    Uses identity function on cluster, progress bar otherwise.

    This function provides a context-aware progress tracking:
    - On cluster environments: Returns identity function (no progress display)
    - On other environments: Returns tqdm progress bar

    Args:
        *args: Positional arguments passed to tqdm
        **kwargs: Keyword arguments passed to tqdm

    Returns:
        Progress bar or identity function based on environment
    """
    if project_paths.iam_on_cluster():
        return identity(*args, **kwargs)
    else:
        return progress_bar(*args, **kwargs)
