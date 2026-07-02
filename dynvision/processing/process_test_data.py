"""Process test data by combining layer responses and test performance metrics.

MEMORY-OPTIMIZED VERSION: Processes layers sequentially to handle large (30GB+) files.
"""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd
import torch

from dynvision.utils import str_to_bool
from dynvision.utils.visualization_utils import extract_param_from_string
from dynvision.utils.data_utils import load_df
from dynvision.utils.performance_measures import (
    calculate_topk_accuracy,
    calculate_confidence,
)

# Import memory-efficient loading system
from dynvision.utils.memory_efficient_loading import (
    MemoryMonitor,
    process_layer_responses_incremental,  # STEP 2: Incremental extraction
)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class MeasureConfig:
    layer_measures: List[str]
    confidence_measures: List[str]
    accuracy_topk: List[int]
    classifier_topk: List[int]


@dataclass
class FileMetadata:
    parameter_value: str
    category_value: str
    extra_values: Dict[str, Optional[str]]
    status_value: Optional[str] = None


def extract_status_from_path(file_path: Path) -> Optional[str]:
    """Extract status value from file path.

    The status is expected to be in the format: {data_name}:{data_group}_{status}
    where the status appears after the last underscore in a path component.

    This function distinguishes between:
    - Model identifier parts: {model_name}:{args}_{seed} where args contains '=' BEFORE underscore
    - Data identifier parts: {data_name}:{data_group}_{status} where '=' only in status (epoch=N)

    Args:
        file_path: Path to extract status from (typically config file or data file)

    Returns:
        Status string if found, None otherwise

    Example:
        Path: .../imagenet1k:01_init/test_data.csv
        Returns: "init"

        Path: .../imagenet1k:01_trained-epoch=150/test_data.csv
        Returns: "trained-epoch=150"

        Path: .../DyRCNNx8:rctype=full_6000/imagenette:all_trained-best/test_data.csv
        Returns: "trained-best" (correctly skips model identifier part)
    """
    # Look through path components for pattern matching {data}:{group}_{status}
    for part in file_path.parts:
        # Check if this part contains a colon (indicating potential data_name:data_group format)
        if ":" in part:
            # Extract the part after the colon
            after_colon = part.split(":", 1)[1]

            # Must have underscore to extract status
            if "_" not in after_colon:
                continue

            # Split on last underscore: before_underscore contains group, after contains status/seed
            before_underscore, after_underscore = after_colon.rsplit("_", 1)

            # Skip model identifier parts - they have '=' in the part BEFORE the underscore
            # (e.g., "rctype=full_6000" has "rctype=full" before underscore)
            # Data parts don't have '=' before underscore (e.g., "all_trained-best" has "all")
            if "=" in before_underscore:
                continue

            return after_underscore

    logger.debug(f"No status found in path: {file_path}")
    return None


def parse_status_string(status: str) -> Tuple[str, Optional[int]]:
    """Parse status string to extract status type and optional epoch number.

    Status values can be:
    - 'init', 'pretrained', 'trained', 'trained-best', 'trained-last'
    - 'trained-epoch=<epoch>' where epoch is an integer

    Args:
        status: Status string to parse

    Returns:
        Tuple of (status_string, epoch_number)
        - epoch_number is None unless status contains 'epoch=N'

    Examples:
        'init' -> ('init', None)
        'trained-best' -> ('trained-best', None)
        'trained-epoch=150' -> ('trained-epoch=150', 150)
    """
    if not status:
        return status, None

    # Look for epoch pattern in status string
    epoch_match = re.search(r"epoch=(\d+)", status)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        return status, epoch

    return status, None


def build_measure_config(measures: List[str]) -> MeasureConfig:
    """Categorize measure strings into layer, confidence, top-k accuracy, and classifier groups."""

    layer_measure_names = {
        "response_avg",
        "response_std",
        "spatial_variance",
        "feature_variance",
    }
    confidence_measure_names = {
        "guess_confidence",
        "label_confidence",
        "first_label_confidence",
    }

    layer_measures: List[str] = []
    confidence_measures: List[str] = []
    accuracy_topk: List[int] = []
    classifier_topk: List[int] = []

    for measure in measures:
        if measure in layer_measure_names:
            layer_measures.append(measure)
        elif measure in confidence_measure_names:
            confidence_measures.append(measure)
        elif measure.startswith("accuracy_top"):
            value_str = measure.replace("accuracy_top", "", 1)
            if not value_str.isdigit():
                raise ValueError(f"Invalid accuracy measure '{measure}'")
            value = int(value_str)
            if value <= 0:
                raise ValueError(f"Top-k accuracy must be positive, got '{measure}'")
            accuracy_topk.append(value)
        elif measure.startswith("classifier_top"):
            value_str = measure.replace("classifier_top", "", 1)
            if not value_str.isdigit():
                raise ValueError(f"Invalid classifier measure '{measure}'")
            value = int(value_str)
            if value <= 0:
                raise ValueError(f"Classifier top-k must be positive, got '{measure}'")
            classifier_topk.append(value)
        else:
            raise ValueError(f"Unsupported measure '{measure}'")

    # Deduplicate while preserving natural ordering for deterministic output
    layer_measures = list(dict.fromkeys(layer_measures))
    confidence_measures = list(dict.fromkeys(confidence_measures))
    accuracy_topk = sorted(set(accuracy_topk))
    classifier_topk = sorted(set(classifier_topk))

    return MeasureConfig(
        layer_measures=layer_measures,
        confidence_measures=confidence_measures,
        accuracy_topk=accuracy_topk,
        classifier_topk=classifier_topk,
    )


def chunk_lists(lst1, lst2, lst3, chunk_size):
    """Split three lists into chunks of specified size."""
    # Ensure all lists are the same length
    for i in range(0, len(lst1), chunk_size):
        yield lst1[i : i + chunk_size], lst2[i : i + chunk_size], lst3[
            i : i + chunk_size
        ]


def calculate_confidence_and_topk_from_classifier(
    classifier_responses: torch.Tensor,
    label_index: torch.Tensor,
    sample_index: torch.Tensor = None,
    times_index: torch.Tensor = None,
    guess_index: torch.Tensor = None,
    first_label_index: torch.Tensor = None,
    topk_values: Optional[List[int]] = None,
    confidence_measures: List[str] = [
        "guess_confidence",
        "label_confidence",
        "first_label_confidence",
    ],
) -> pd.DataFrame:
    """
    Calculate all confidence measures and top-k accuracies from classifier responses.

    Returns a DataFrame with (sample_index, times_index) as index and confidence/topk columns.

    Args:
        classifier_responses: Tensor of shape (n_samples, n_timesteps, n_classes)
        label_index: Ground truth labels (n_samples, n_timesteps)
        guess_index: Model predictions (n_samples, n_timesteps)
        first_label_index: First valid label per sample (n_samples,)
        topk_values: List of k values for top-k accuracy

    Returns:
        DataFrame with columns: sample_index, times_index, guess_confidence,
        label_confidence, first_label_confidence, accuracy_topK (for each K)
    """
    n_samples, n_timesteps, n_classes = classifier_responses.shape
    if sample_index is None:
        sample_index = np.repeat(np.arange(n_samples), n_timesteps)
    if times_index is None:
        times_index = np.tile(np.arange(n_timesteps), n_samples)

    topk_values = topk_values or []

    # Apply softmax once to get probabilities when needed
    probabilities = (
        torch.softmax(classifier_responses, dim=-1) if topk_values else None
    )

    # ===== CONFIDENCE CALCULATIONS =====
    confidence_results = {}

    if "guess_confidence" in confidence_measures and guess_index is not None:
        confidence_results["guess_confidence"] = (
            calculate_confidence(classifier_responses, guess_index).cpu().numpy()
        )

    if "label_confidence" in confidence_measures:
        confidence_results["label_confidence"] = (
            calculate_confidence(classifier_responses, label_index).cpu().numpy()
        )

    if (
        "first_label_confidence" in confidence_measures
        and first_label_index is not None
    ):
        confidence_results["first_label_confidence"] = (
            calculate_confidence(classifier_responses, first_label_index).cpu().numpy()
        )

    # ===== TOP-K ACCURACY =====
    topk_results = {}
    if topk_values:
        accuracy_label = (
            label_index if first_label_index is None else first_label_index
        )
        topk_accuracy = calculate_topk_accuracy(
            probabilities, accuracy_label, k=topk_values
        )
        topk_results = {
            f"accuracy_top{k}": acc.cpu().numpy()
            for k, acc in zip(topk_values, topk_accuracy)
        }

    # ===== BUILD DATAFRAME =====
    # Create multi-index for sample and time
    data = {
        "sample_index": sample_index,
        "times_index": times_index,
    }
    # Add confidence columns
    for measure, conf_array in confidence_results.items():
        data[measure] = conf_array.ravel()
    # Add top-k accuracy columns
    for k, acc_array in topk_results.items():
        data[k] = acc_array.ravel()

    return pd.DataFrame(data)


def compress_to_presentation_level(
    df: pd.DataFrame,
    id_cols: List[str] = ["first_label_index", "times_index"],
    drop_cols: List[str] = ["sample_index", "image_index"],
) -> pd.DataFrame:
    """
    Compress dataframe from sample-level to presentation-level resolution.

    For each (first_label_index, times_index) group:
    - If all values identical: keep single value
    - If values vary: create <col>_avg and <col>_std

    Args:
        df: Input dataframe at sample-level
        id_cols: Grouping columns (default: first_label_index, times_index)
        drop_cols: Columns to discard

    Returns:
        Compressed dataframe at presentation-level
    """
    logger.info(f"    Compressing from {len(df)} rows to presentation-level...")

    aggregated = {}

    for col in df.columns:
        if col in id_cols:
            continue
        if col in drop_cols:
            continue

        grouped = df.groupby(id_cols)[col]

        # Check if all non-NaN values are identical within each group
        # For each group, check if nunique (excluding NaN) is <= 1
        nunique_per_group = grouped.nunique()
        all_identical = (nunique_per_group <= 1).all()

        if all_identical:
            # Keep single value
            aggregated[col] = grouped.first()
            logger.debug(f"      {col}: identical values, keeping single column")
        else:
            # Compute mean and std
            aggregated[f"{col}_avg"] = grouped.mean()
            aggregated[f"{col}_std"] = grouped.std()
            logger.debug(
                f"      {col}: varying values, creating _avg and _std columns"
            )

    result = pd.DataFrame(aggregated).reset_index()
    logger.info(
        f"    Compressed to {len(result)} rows with {len(result.columns)} columns"
    )

    return result


def process_test_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process test performance CSV and add first_label_index.

    Returns DataFrame at (sample_index, times_index) resolution with:
    - All original CSV columns
    - first_label_index: first valid label for each sample
    - accuracy: boolean (guess == first_label)
    """
    logger.info("  Processing test performance metrics...")

    # Verify required columns
    required_cols = ["sample_index", "times_index", "label_index", "guess_index"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available: {df.columns.tolist()}"
        )

    logger.info(f"    Input shape: {df.shape}")
    logger.info(
        f"    Samples: {df['sample_index'].nunique()}, Times: {df['times_index'].nunique()}"
    )

    # Ensure one row per (sample, time)
    df = df.groupby(["sample_index", "times_index"]).first().reset_index()

    # Add first_label_index for each sample
    first_labels = (
        df[df["label_index"] >= 0].groupby("sample_index")["label_index"].first()
    )
    df["first_label_index"] = (
        df["sample_index"].map(first_labels).fillna(-1).astype(int)
    )

    # Calculate accuracy (1 if guess matches first label, 0 otherwise, NaN if no valid label)
    df["accuracy"] = (df["guess_index"] == df["first_label_index"]).astype(float)
    df.loc[df["first_label_index"] < 0, "accuracy"] = np.nan

    logger.info(f"    Output shape: {df.shape}")
    logger.info(f"    Unique first labels: {df['first_label_index'].unique()}")

    return df


def _extract_metadata(
    config_file: Path,
    data_arg_key: str,
    category: str,
    extra_parameters: Sequence[str],
    extract_status: bool = False,
) -> FileMetadata:
    """Extract and validate metadata from config YAML file and file path.

    The config file is generated by TestingParams.persist_resolved_config() and contains
    all resolved parameters with keys like 'data.param_name' and 'model.param_name'.

    Args:
        config_file: Path to .config.yaml file
        data_arg_key: Parameter key to extract from data.* namespace
        category: Category key to extract from model arguments
        extra_parameters: Additional parameter keys to extract
        extract_status: If True, extract status from file path and auto-extract epoch

    Returns:
        FileMetadata with parameter value, category value, extra values, and status
    """
    import yaml

    # Load config file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameter value from data namespace
    # Format: data.param_name (e.g., data.dsteps, data.intro, data.stim)
    data_key = f"data.{data_arg_key}"
    arg_value = config.get(data_key)
    if arg_value is None:
        raise ValueError(
            f"Parameter '{data_key}' not found in config file {config_file}"
        )

    # Extract category value from model arguments path (only if category provided)
    # The category comes from the model path wildcards, stored in the parent directory structure
    # We need to extract it from the path since it's not a parameter but a wildcard
    if category:
        cat_value = extract_param_from_string(
            str(config_file.parent.parent), key=category, value_type=None
        )
    else:
        cat_value = None

    # Extract extra parameters (try both data.* and model.* namespaces)
    extra_values: Dict[str, Optional[str]] = {}
    for param_name in extra_parameters:
        # Try data namespace first
        value = config.get(f"data.{param_name}")
        if value is None:
            # Try model namespace
            value = config.get(f"model.{param_name}")
        if value is None:
            # Try unscoped
            value = config.get(param_name)

        extra_values[param_name] = str(value) if value is not None else None

        if value is None:
            logger.debug(
                "    Parameter '%s' not found in config file '%s'",
                param_name,
                config_file,
            )

    # Extract status from file path if requested
    status_value = None
    if extract_status:
        status_value = extract_status_from_path(config_file)
        if status_value:
            logger.debug(f"    Extracted status: {status_value}")
            # Parse status string to check for epoch
            _, epoch = parse_status_string(status_value)
            if epoch is not None:
                # Auto-add epoch to extra_values
                extra_values["epoch"] = str(epoch)
                logger.debug(f"    Auto-extracted epoch from status: {epoch}")
        else:
            logger.warning(
                f"    extract_status=True but no status found in path: {config_file}"
            )

    return FileMetadata(
        parameter_value=str(arg_value),
        category_value=cat_value,
        extra_values=extra_values,
        status_value=status_value,
    )


def _load_responses(
    pt_file: Path,
) -> Tuple[Optional[Dict[str, torch.Tensor]], bool]:
    """Load response tensors from disk, returning (responses, has_responses)."""

    try:
        responses = torch.load(pt_file, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to load responses: {exc}")
        return None, False

    if not responses or len(responses) == 0:
        logger.warning("  Empty response file, skipping response metrics")
        return None, False

    return responses, True


def _append_classifier_metrics(
    sample_df: pd.DataFrame,
    responses: Optional[Dict[str, torch.Tensor]],
    confidence_measures: Sequence[str],
    accuracy_topk_values: Sequence[int],
    classifier_topk_values: Sequence[int],
) -> pd.DataFrame:
    """Augment the sample-level dataframe with classifier-derived metrics."""

    if (
        responses is None
        or "classifier" not in responses
        or not (confidence_measures or accuracy_topk_values or classifier_topk_values)
    ):
        return sample_df

    logger.info("  Calculating classifier metrics...")

    classifier_tensor = responses["classifier"]
    n_samples_csv = sample_df["sample_index"].nunique()
    n_timesteps = sample_df["times_index"].nunique()
    n_samples_pt = classifier_tensor.shape[0]
    n_units = classifier_tensor.shape[-1]

    if n_samples_pt > n_samples_csv:
        raise ValueError(
            f"PT file has MORE samples ({n_samples_pt}) than CSV ({n_samples_csv}). "
            "This should not happen!"
        )

    if n_samples_pt < n_samples_csv:
        logger.warning("⚠️  SAMPLE SIZE MISMATCH detected!")
        logger.warning(f"    CSV file: {n_samples_csv} samples")
        logger.warning(f"    PT file:  {n_samples_pt} samples")
        logger.warning(
            "    Confidence metrics will be NaN for %d samples (sample_index >= %d).",
            n_samples_csv - n_samples_pt,
            n_samples_pt,
        )
        logger.warning(
            "    Top-k accuracies will be calculated from available samples only."
        )

    confidence_measures_needed = [
        m for m in confidence_measures if m not in sample_df.columns
    ]

    valid_accuracy_topk = [k for k in accuracy_topk_values if k <= n_units]
    invalid_accuracy_topk = sorted(
        set(accuracy_topk_values) - set(valid_accuracy_topk)
    )
    if invalid_accuracy_topk:
        logger.warning(
            "    Requested accuracy_topk values %s exceed available classifier units (%d)",
            invalid_accuracy_topk,
            n_units,
        )

    valid_classifier_topk = [k for k in classifier_topk_values if k <= n_units]
    invalid_classifier_topk = sorted(
        set(classifier_topk_values) - set(valid_classifier_topk)
    )
    if invalid_classifier_topk:
        logger.warning(
            "    Requested classifier_topk values %s exceed available classifier units (%d)",
            invalid_classifier_topk,
            n_units,
        )

    if n_samples_pt <= 0:
        logger.warning("    PT file has no samples, skipping classifier metrics")
        return sample_df

    sorted_df = sample_df.sort_values(["sample_index", "times_index"])
    sorted_df_available = sorted_df[sorted_df["sample_index"] < n_samples_pt]

    if len(sorted_df_available) == 0:
        logger.warning("    No valid samples to process for classifier metrics")
        return sample_df

    guess_tensor = torch.tensor(
        sorted_df_available["guess_index"].values.reshape(n_samples_pt, n_timesteps),
        dtype=torch.int,
    )
    label_tensor = torch.tensor(
        sorted_df_available["label_index"].values.reshape(n_samples_pt, n_timesteps),
        dtype=torch.int,
    )
    first_label_tensor = torch.tensor(
        sorted_df_available["first_label_index"].values.reshape(
            n_samples_pt, n_timesteps
        ),
        dtype=torch.int,
    )

    classifier_df = calculate_confidence_and_topk_from_classifier(
        classifier_tensor,
        label_index=label_tensor,
        guess_index=guess_tensor,
        first_label_index=first_label_tensor,
        topk_values=valid_accuracy_topk,
        confidence_measures=confidence_measures_needed,
    )

    sample_df = sample_df.merge(
        classifier_df,
        on=["sample_index", "times_index"],
        how="left",
    )

    logger.info(f"    Added {len(classifier_df.columns) - 2} classifier metrics")

    if valid_classifier_topk:
        max_classifier_top = max(valid_classifier_topk)
        overall_activity = classifier_tensor.sum(dim=(0, 1))
        ranked_units = torch.argsort(overall_activity, descending=True)[
            :max_classifier_top
        ]

        if ranked_units.numel() < max_classifier_top:
            logger.warning(
                "    Fewer classifier units (%d) than requested top-k (%d)",
                ranked_units.numel(),
                max_classifier_top,
            )

        reordered_responses = classifier_tensor[:, :, ranked_units]
        flattened_responses = reordered_responses.reshape(-1, ranked_units.numel())

        if flattened_responses.shape[0] != len(sorted_df_available):
            logger.warning(
                "    Classifier activations shape mismatch (expected %d rows, got %d)",
                len(sorted_df_available),
                flattened_responses.shape[0],
            )
        else:
            classifier_top_data: Dict[str, np.ndarray] = {
                "sample_index": sorted_df_available["sample_index"].values,
                "times_index": sorted_df_available["times_index"].values,
            }
            np_responses = flattened_responses.cpu().numpy()
            for idx in range(ranked_units.numel()):
                classifier_top_label = f"classifier_top{idx + 1}"
                unit_index = int(ranked_units[idx].item())
                classifier_top_data[classifier_top_label] = np_responses[:, idx]
                classifier_top_data[f"{classifier_top_label}_id"] = np.full(
                    np_responses.shape[0], unit_index, dtype=np.int64
                )

            classifier_top_df = pd.DataFrame(classifier_top_data)
            sample_df = sample_df.merge(
                classifier_top_df,
                on=["sample_index", "times_index"],
                how="left",
            )

            logger.info(
                "    Added classifier top-k activations up to k=%d",
                max_classifier_top,
            )

    return sample_df


def _apply_resolution(sample_df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Convert sample-level dataframe to requested resolution."""

    if resolution == "class":
        logger.info("  Compressing sample-level data to presentation-level...")
        return compress_to_presentation_level(
            sample_df,
            id_cols=["first_label_index", "times_index"],
            drop_cols=["sample_index", "image_index"],
        )

    logger.info("  Keeping data at sample-level (no compression)...")
    return sample_df.copy()


def _append_layer_metrics(
    result_df: pd.DataFrame,
    pt_file: Path,
    has_responses: bool,
    layer_measures: Sequence[str],
    memory_monitor: MemoryMonitor,
    test_df_with_first_label: pd.DataFrame,
    resolution: str,
) -> pd.DataFrame:
    """Attach layer-level metrics to the result dataframe when available."""

    if not (has_responses and layer_measures):
        return result_df

    logger.info(f"  Calculating layer metrics at '{resolution}' resolution...")

    try:
        sample_to_presentation = (
            test_df_with_first_label.groupby("sample_index")["first_label_index"]
            .first()
            .to_dict()
        )
        presented_classes = sorted(result_df["first_label_index"].unique())
        unique_times = sorted(result_df["times_index"].unique())

        n_samples_expected = None
        if resolution == "sample":
            n_samples_expected = len(sample_to_presentation)
            logger.info(
                f"    Expected samples for layer metrics: {n_samples_expected}"
            )

        layer_metrics = process_layer_responses_incremental(
            pt_file=pt_file,
            measures=layer_measures,
            sample_to_presentation=sample_to_presentation,
            presented_classes=presented_classes,
            unique_times=unique_times,
            memory_monitor=memory_monitor,
            max_retries=3,
            resolution=resolution,
            n_samples_expected=n_samples_expected,
        )

    except Exception as exc:
        logger.error(f"Failed to calculate layer metrics: {exc}")
        raise

    if resolution == "class":
        layer_data = {
            "first_label_index": np.repeat(presented_classes, len(unique_times)),
            "times_index": np.tile(unique_times, len(presented_classes)),
        }
        layer_data.update(layer_metrics)
        layer_df = pd.DataFrame(layer_data)
        merged_df = result_df.merge(
            layer_df,
            on=["first_label_index", "times_index"],
            how="left",
        )
    else:
        n_samples = result_df["sample_index"].nunique()
        n_times = len(unique_times)
        layer_data = {
            "sample_index": np.repeat(np.arange(n_samples), n_times),
            "times_index": np.tile(unique_times, n_samples),
        }
        layer_data.update(layer_metrics)
        layer_df = pd.DataFrame(layer_data)
        merged_df = result_df.merge(
            layer_df,
            on=["sample_index", "times_index"],
            how="left",
        )

    logger.info(f"    Added {len(layer_metrics)} layer metrics")
    return merged_df


def process_single_batch_optimized(
    response_files: List[Path],
    test_output_files: List[Path],
    test_config_files: List[Path],
    data_arg_key: str,
    measure_config: MeasureConfig,
    category: str,
    memory_monitor: MemoryMonitor,
    resolution: str = "sample",
    extra_parameters: Optional[List[str]] = None,
    extract_status: bool = False,
) -> pd.DataFrame:
    """
    Process a single batch of files.

    Workflow:
    1. Load and process test CSV (adds first_label_index, accuracy)
    2. Load response PT file and calculate confidence/topk metrics from classifier
    3. Calculate layer response metrics
    4. Merge all metrics into single DataFrame at (sample_index, times_index) resolution

    Args:
        response_files: List of response PT files
        test_output_files: List of test output CSV files
        test_config_files: List of test config YAML files
        data_arg_key: Parameter key to extract
        measure_config: Configuration for measures to compute
        category: Category key to extract
        memory_monitor: Memory monitoring instance
        resolution: Resolution level ('sample' or 'class')
        extra_parameters: Additional parameters to extract
        extract_status: If True, extract status from file paths

    Returns:
        DataFrame with all metrics at (sample_index, times_index, first_label_index) resolution
    """
    batch_dfs = []

    layer_measures = measure_config.layer_measures
    confidence_measures = measure_config.confidence_measures
    accuracy_topk_values = measure_config.accuracy_topk
    classifier_topk_values = measure_config.classifier_topk
    extra_parameters = extra_parameters or []

    for pt_file, csv_file, config_file in zip(
        response_files, test_output_files, test_config_files
    ):
        logger.info(
            f"  Processing: {pt_file.name} + {csv_file.name} + {config_file.name}"
        )

        try:
            metadata = _extract_metadata(
                config_file,
                data_arg_key=data_arg_key,
                category=category,
                extra_parameters=extra_parameters,
                extract_status=extract_status,
            )

            test_df = load_df(csv_file)
            test_df_with_first_label = process_test_performance(test_df)

            responses, has_responses = _load_responses(pt_file)

            sample_level_df = test_df_with_first_label.copy()
            sample_level_df = _append_classifier_metrics(
                sample_level_df,
                responses=responses,
                confidence_measures=confidence_measures,
                accuracy_topk_values=accuracy_topk_values,
                classifier_topk_values=classifier_topk_values,
            )

            result_df = _apply_resolution(sample_level_df, resolution=resolution)
            result_df = _append_layer_metrics(
                result_df,
                pt_file=pt_file,
                has_responses=has_responses,
                layer_measures=layer_measures,
                memory_monitor=memory_monitor,
                test_df_with_first_label=test_df_with_first_label,
                resolution=resolution,
            )

            result_df[data_arg_key] = metadata.parameter_value
            result_df[category] = metadata.category_value
            if metadata.status_value is not None:
                result_df["status"] = metadata.status_value
            for param_name, param_value in metadata.extra_values.items():
                if param_value is not None:
                    result_df[param_name] = param_value

            batch_dfs.append(result_df)
            logger.info(
                f"    Final shape: {result_df.shape} with {len(result_df.columns)} columns"
            )

            # Cleanup
            if responses is not None:
                del responses
            memory_monitor.cleanup()

        except Exception as e:
            logger.error(
                f"ERROR: {pt_file.name} + {csv_file.name} + {config_file.name}"
            )
            logger.error(f"  {type(e).__name__}: {str(e)}")
            raise

    # Combine all file results
    if not batch_dfs:
        raise RuntimeError("No dataframes created")

    combined = pd.concat(batch_dfs, ignore_index=True)
    logger.info(f"  Batch combined: {combined.shape}")

    return combined


# Command line interface
parser = argparse.ArgumentParser(
    description="Process test data with memory-optimized layer loading"
)
parser.add_argument(
    "--responses",
    nargs="+",
    type=Path,
    required=True,
    help="Path to response .pt files",
)
parser.add_argument(
    "--test_outputs",
    nargs="+",
    type=Path,
    required=True,
    help="Path to test output .csv files",
)
parser.add_argument(
    "--test_configs",
    nargs="+",
    type=Path,
    required=True,
    help="Path to test config .yaml files",
)
parser.add_argument(
    "--output", type=Path, required=True, help="Path to output CSV file"
)
parser.add_argument(
    "--parameter",
    type=str,
    required=True,
    help="Parameter to extract from filenames",
)
parser.add_argument(
    "--category",
    type=str,
    required=True,
    help="Category to extract from filenames",
)
parser.add_argument(
    "--additional_parameters",
    nargs="*",
    default=None,
    help="Optional additional parameter names to extract from response directory names",
)
parser.add_argument(
    "--measures",
    nargs="+",
    type=str,
    default=["response_avg", "response_std"],
    help=(
        "Measures to compute. Supported values include layer metrics (response_avg, "
        "response_std, spatial_variance, feature_variance), confidence measures "
        "(guess_confidence, label_confidence, first_label_confidence), "
        "top-k accuracies (accuracy_topK), and classifier activations "
        "(classifier_topK)."
    ),
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for processing files (default: 1)",
)
parser.add_argument(
    "--memory_limit_gb",
    type=float,
    default=68.0,
    help="Soft memory limit in GB (triggers warnings, default: 20.0)",
)
parser.add_argument(
    "--remove_input_responses",
    type=str_to_bool,
    default="False",
    help="If 'True', remove input response files after SUCCESSFUL processing",
)
parser.add_argument(
    "--sample_resolution",
    type=str,
    default="sample",
    choices=["sample", "class"],
    help="Resolution for output data: 'sample' for (sample_index, times_index), "
    "'class' for (first_label_index, times_index) with aggregation",
)
parser.add_argument(
    "--fail_on_missing_inputs",
    type=str_to_bool,
    default="True",
    help="If 'True', raise an error when input files are missing or a batch fails; "
    "if 'False', skip problematic inputs and continue processing",
)


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize measure configuration and memory monitor
    measure_config = build_measure_config(args.measures)
    memory_monitor = MemoryMonitor(memory_limit_gb=args.memory_limit_gb)
    additional_parameters = args.additional_parameters or []

    # Determine if status extraction is requested based on additional_parameters
    extract_status = "status" in additional_parameters

    # Log configuration
    logger.info("=" * 80)
    logger.info("MEMORY-OPTIMIZED TEST DATA PROCESSING")
    logger.info("=" * 80)
    logger.info(f"Resolution: {args.sample_resolution}")
    logger.info(f"Measures: {args.measures}")
    if measure_config.accuracy_topk:
        logger.info(f"Accuracy top-k measures: {measure_config.accuracy_topk}")
    if measure_config.classifier_topk:
        logger.info(f"Classifier top-k measures: {measure_config.classifier_topk}")
    logger.info(f"Memory limit: {args.memory_limit_gb}GB")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Files to process: {len(args.responses)}")
    logger.info(f"Fail on missing inputs: {args.fail_on_missing_inputs}")
    if additional_parameters:
        logger.info(f"Additional parameters: {additional_parameters}")
    if extract_status:
        logger.info(
            "Extract status from paths: True (detected from additional_parameters)"
        )
    if args.sample_resolution == "sample":
        logger.info("NOTE: Output at sample-level (no compression)")
    else:
        logger.info("NOTE: Output at class-level (with compression and aggregation)")
    logger.info("NOTE: New CSV format - classifier responses no longer in CSV")
    logger.info("NOTE: Empty response files will be handled gracefully")

    processing_successful = False
    df = None

    try:
        # Process data
        if len(args.responses) != len(args.test_outputs) or len(args.responses) != len(
            args.test_configs
        ):
            mismatch_msg = (
                "Response file count (%d) does not match test output file count (%d) or config file count (%d)."
                % (len(args.responses), len(args.test_outputs), len(args.test_configs))
            )
            if args.fail_on_missing_inputs:
                logger.error(mismatch_msg)
                sys.exit(1)
            logger.warning(mismatch_msg + " Extra files will be ignored.")

        paired_inputs: List[Tuple[Path, Path, Path]] = []
        max_pairs = min(
            len(args.responses), len(args.test_outputs), len(args.test_configs)
        )
        for idx in range(max_pairs):
            response_path = args.responses[idx]
            output_path = args.test_outputs[idx]
            config_path = args.test_configs[idx]
            missing_paths = [
                str(path)
                for path in (response_path, output_path, config_path)
                if not path.exists()
            ]
            if missing_paths:
                missing_msg = f"Missing input file(s) for set {idx + 1}: {', '.join(missing_paths)}"
                if args.fail_on_missing_inputs:
                    logger.error(missing_msg)
                    sys.exit(1)
                logger.warning(missing_msg + " Skipping set.")
                continue
            paired_inputs.append((response_path, output_path, config_path))

        if not paired_inputs:
            logger_msg = "No valid response/test output pairs to process."
            if args.fail_on_missing_inputs:
                logger.error(logger_msg)
                sys.exit(1)
            logger.warning(logger_msg)
            df = pd.DataFrame()
            logger.info(f"\nSaving processed data to: {args.output}")
            df.to_csv(args.output, index=False)
            logger.info(f"✅ Successfully saved {len(df)} rows")
        else:
            valid_responses = [pair[0] for pair in paired_inputs]
            valid_test_outputs = [pair[1] for pair in paired_inputs]
            valid_test_configs = [pair[2] for pair in paired_inputs]

            logger.info(f"\nProcessing {len(paired_inputs)} file sets...")
            memory_monitor.log_memory("start")

            # Process in batches
            all_dataframes = []
            batch_count = 0

            for response_batch, output_batch, config_batch in chunk_lists(
                valid_responses,
                valid_test_outputs,
                valid_test_configs,
                args.batch_size,
            ):
                batch_count += 1
                logger.info(
                    f"\nBatch {batch_count}/{(len(valid_responses) + args.batch_size - 1) // args.batch_size}"
                )

                try:
                    # STEP 2: Use incremental extraction
                    batch_df = process_single_batch_optimized(
                        response_batch,
                        output_batch,
                        config_batch,
                        data_arg_key=args.parameter,
                        measure_config=measure_config,
                        category=args.category,
                        memory_monitor=memory_monitor,
                        resolution=args.sample_resolution,
                        extra_parameters=additional_parameters,
                        extract_status=extract_status,
                    )

                    if not batch_df.empty:
                        all_dataframes.append(batch_df)
                        logger.info(
                            f"  Batch {batch_count} processed: {len(batch_df)} rows"
                        )
                    else:
                        raise RuntimeError(
                            f"Batch {batch_count} produced empty dataframe"
                        )

                    # Cleanup between batches
                    del batch_df
                    memory_monitor.cleanup()

                except Exception as e:
                    logger.error(f"Failed to process batch {batch_count}")
                    logger.error(
                        f"  Response files: {[f.name for f in response_batch]}"
                    )
                    logger.error(f"  Output files: {[f.name for f in output_batch]}")
                    memory_monitor.cleanup()
                    if args.fail_on_missing_inputs:
                        raise RuntimeError(f"Batch {batch_count} failed: {e}") from e
                    logger.warning(
                        "Batch %s skipped due to error and fail_on_missing_inputs=False",
                        batch_count,
                    )
                    continue

            if not all_dataframes:
                message = "No data was successfully processed from any batch"
                if args.fail_on_missing_inputs:
                    raise RuntimeError(message)
                logger.warning(message)
                df = pd.DataFrame()
                logger.info(f"\nSaving processed data to: {args.output}")
                df.to_csv(args.output, index=False)
                logger.info(f"✅ Successfully saved {len(df)} rows")
            else:
                # Combine all dataframes
                logger.info(f"\nCombining {len(all_dataframes)} batches...")
                df = pd.concat(all_dataframes, ignore_index=True)
                del all_dataframes
                memory_monitor.cleanup()

                logger.info(f"Total combined data: {len(df)} rows")
                memory_monitor.log_memory("after combining")

                # Save processed data
                logger.info(f"\nSaving processed data to: {args.output}")
                # Ensure output directory exists (redundant safety check before save)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(args.output, index=False)
                logger.info(f"✅ Successfully saved {len(df)} rows")

                processing_successful = len(df) > 0

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("PROCESSING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        logger.error("\nDue to errors, input files will NOT be removed")
        sys.exit(1)

    finally:
        # Memory summary
        logger.info("\n" + "=" * 80)
        logger.info("MEMORY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Peak memory usage: {memory_monitor.peak_memory_gb:.2f}GB")
        logger.info(f"Memory limit: {args.memory_limit_gb}GB")

        if memory_monitor.peak_memory_gb > args.memory_limit_gb:
            logger.warning(
                f"⚠️  Peak memory ({memory_monitor.peak_memory_gb:.2f}GB) "
                f"exceeded limit ({args.memory_limit_gb}GB)"
            )

    # Only remove input files if processing was successful
    if processing_successful and args.remove_input_responses:
        logger.info("\n" + "=" * 80)
        logger.info("REMOVING INPUT FILES")
        logger.info("=" * 80)
        logger.info(
            "Processing completed successfully, removing input response files..."
        )

        for response_file in args.responses:
            if response_file.exists():
                try:
                    response_file.unlink()
                    logger.info(f"  ✓ Removed: {response_file.name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to remove {response_file.name}: {e}")

    # Print summary statistics
    if df is not None:
        logger.info("\n" + "=" * 80)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Resolution: {args.sample_resolution}")
        logger.info(f"Shape: {df.shape}")
        if df.empty:
            logger.info("Dataset is empty; skipping detailed summary.")
        else:
            if args.category in df.columns:
                logger.info(
                    f"Unique {args.category} values: {df[args.category].unique()}"
                )
            if args.parameter in df.columns:
                logger.info(
                    f"Unique {args.parameter} values: {df[args.parameter].unique()}"
                )
            if "times_index" in df.columns:
                logger.info(f"Time steps: {sorted(df['times_index'].unique())}")

            if args.sample_resolution == "sample" and "sample_index" in df.columns:
                logger.info(f"Samples: {df['sample_index'].nunique()}")

            if "first_label_index" in df.columns:
                logger.info(
                    f"Presentation labels: {sorted(df['first_label_index'].unique())}"
                )

            # Layer metrics summary
            layer_cols = [
                col
                for col in df.columns
                if any(pattern in col for pattern in measure_config.layer_measures)
            ]
            if measure_config.classifier_topk:
                layer_cols.extend(
                    [col for col in df.columns if col.startswith("classifier_top")]
                )
            layer_cols = sorted(set(layer_cols))
            logger.info(f"Layer metric columns: {len(layer_cols)}")

            # Performance metrics summary
            if "accuracy" in df.columns and not df["accuracy"].empty:
                logger.info(
                    f"Accuracy range: {df['accuracy'].min():.3f} to {df['accuracy'].max():.3f}"
                )

            accuracy_columns = []
            for col in df.columns:
                if col.startswith("accuracy_top"):
                    suffix = col.replace("accuracy_top", "", 1)
                    if suffix.isdigit():
                        accuracy_columns.append((int(suffix), col))
            accuracy_columns.sort()
            for k, topk_col in accuracy_columns:
                min_val = df[topk_col].min(skipna=True)
                max_val = df[topk_col].max(skipna=True)
                logger.info(f"Top-{k} accuracy range: {min_val:.3f} to {max_val:.3f}")

            classifier_columns = []
            for col in df.columns:
                if col.startswith("classifier_top"):
                    suffix = col.replace("classifier_top", "", 1)
                    if suffix.isdigit():
                        classifier_columns.append((int(suffix), col))
            classifier_columns.sort()
            for rank, classifier_col in classifier_columns:
                min_val = df[classifier_col].min(skipna=True)
                max_val = df[classifier_col].max(skipna=True)
                logger.info(
                    f"Classifier top{rank} activation range: {min_val:.3f} to {max_val:.3f}"
                )

            logger.info("\n✅ Processing completed successfully!")
