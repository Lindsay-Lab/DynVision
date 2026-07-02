"""Process individual test output to extract metrics.

STAGE 1 of two-stage experiment processing pipeline.

This script processes a single test's outputs (test_responses.pt, test_outputs.csv)
to compute all metrics and save a processed test_data.csv file at sample-level resolution.

NO metadata extraction - that happens in Stage 2 (aggregate_experiment_data.py).

Memory-optimized: Processes layers incrementally to handle large (30GB+) files.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from dynvision.utils import str_to_bool
from dynvision.utils.data_utils import load_df

# Import memory-efficient loading system
from dynvision.utils.memory_efficient_loading import (
    MemoryMonitor,
    process_layer_responses_incremental,
)

# Import shared processing functions from process_test_data
from dynvision.processing.process_test_data import (
    MeasureConfig,
    build_measure_config,
    process_test_performance,
    calculate_confidence_and_topk_from_classifier,
)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_responses(
    pt_file: Path,
) -> tuple[Optional[Dict[str, torch.Tensor]], bool, str]:
    """Load response tensors from disk.

    Returns:
        (responses_dict, has_responses, response_resolution) tuple.
        response_resolution is "unit" or "layer".
    """
    try:
        responses = torch.load(pt_file, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning(f"Failed to load responses from {pt_file}: {exc}")
        return None, False, "unit"

    if not responses or len(responses) == 0:
        logger.warning(f"Empty response file: {pt_file}")
        return None, False, "unit"

    # Extract metadata if present
    metadata = responses.pop("_metadata", None)
    if metadata is not None:
        response_resolution = metadata.get("response_resolution", "unit")
        logger.info(
            f"Response metadata: resolution={response_resolution}, version={metadata.get('version')}"
        )
    else:
        # Auto-detect: if non-classifier keys end with _response_avg/_response_std, it's layer-wise
        non_classifier_keys = [k for k in responses if k != "classifier"]
        has_avg_keys = any(k.endswith("_response_avg") for k in non_classifier_keys)
        response_resolution = "layer" if has_avg_keys else "unit"
        if response_resolution == "layer":
            logger.info("Auto-detected layer-wise response format (no metadata)")

    return responses, True, response_resolution


def append_classifier_metrics(
    sample_df: pd.DataFrame,
    responses: Optional[Dict[str, torch.Tensor]],
    confidence_measures: List[str],
    accuracy_topk_values: List[int],
    classifier_topk_values: List[int],
) -> pd.DataFrame:
    """Augment sample-level dataframe with classifier-derived metrics.

    Args:
        sample_df: DataFrame at (sample_index, times_index) resolution
        responses: Dict of layer responses including 'classifier'
        confidence_measures: List of confidence measure names to compute
        accuracy_topk_values: List of k values for top-k accuracy
        classifier_topk_values: List of k values for classifier activations

    Returns:
        DataFrame with classifier metrics added
    """
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

    # Validate sample counts
    if n_samples_pt > n_samples_csv:
        raise ValueError(
            f"PT file has MORE samples ({n_samples_pt}) than CSV ({n_samples_csv})"
        )

    if n_samples_pt < n_samples_csv:
        logger.warning("⚠️  SAMPLE SIZE MISMATCH detected!")
        logger.warning(f"    CSV file: {n_samples_csv} samples")
        logger.warning(f"    PT file:  {n_samples_pt} samples")
        logger.warning(
            f"    Confidence metrics will be NaN for {n_samples_csv - n_samples_pt} samples"
        )

    # Filter confidence measures to those not already in dataframe
    confidence_measures_needed = [
        m for m in confidence_measures if m not in sample_df.columns
    ]

    # Validate top-k values against available units
    valid_accuracy_topk = [k for k in accuracy_topk_values if k <= n_units]
    invalid_accuracy_topk = sorted(
        set(accuracy_topk_values) - set(valid_accuracy_topk)
    )
    if invalid_accuracy_topk:
        logger.warning(
            f"    Requested accuracy_topk values {invalid_accuracy_topk} exceed "
            f"available classifier units ({n_units})"
        )

    valid_classifier_topk = [k for k in classifier_topk_values if k <= n_units]
    invalid_classifier_topk = sorted(
        set(classifier_topk_values) - set(valid_classifier_topk)
    )
    if invalid_classifier_topk:
        logger.warning(
            f"    Requested classifier_topk values {invalid_classifier_topk} exceed "
            f"available classifier units ({n_units})"
        )

    if n_samples_pt <= 0:
        logger.warning("    PT file has no samples, skipping classifier metrics")
        return sample_df

    # Prepare sorted data for tensor operations
    sorted_df = sample_df.sort_values(["sample_index", "times_index"])
    sorted_df_available = sorted_df[sorted_df["sample_index"] < n_samples_pt]

    if len(sorted_df_available) == 0:
        logger.warning("    No valid samples to process for classifier metrics")
        return sample_df

    # Create tensors from CSV data
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

    # Calculate confidence and top-k metrics
    classifier_df = calculate_confidence_and_topk_from_classifier(
        classifier_tensor,
        label_index=label_tensor,
        guess_index=guess_tensor,
        first_label_index=first_label_tensor,
        topk_values=valid_accuracy_topk,
        confidence_measures=confidence_measures_needed,
    )

    # Merge with original dataframe
    sample_df = sample_df.merge(
        classifier_df,
        on=["sample_index", "times_index"],
        how="left",
    )

    logger.info(f"    Added {len(classifier_df.columns) - 2} classifier metrics")

    # Add classifier top-k activations if requested
    if valid_classifier_topk:
        max_classifier_top = max(valid_classifier_topk)
        overall_activity = classifier_tensor.sum(dim=(0, 1))
        ranked_units = torch.argsort(overall_activity, descending=True)[
            :max_classifier_top
        ]

        if ranked_units.numel() < max_classifier_top:
            logger.warning(
                f"    Fewer classifier units ({ranked_units.numel()}) than "
                f"requested top-k ({max_classifier_top})"
            )

        reordered_responses = classifier_tensor[:, :, ranked_units]
        flattened_responses = reordered_responses.reshape(-1, ranked_units.numel())

        if flattened_responses.shape[0] != len(sorted_df_available):
            logger.warning(
                f"    Classifier activations shape mismatch (expected "
                f"{len(sorted_df_available)} rows, got {flattened_responses.shape[0]})"
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
                f"    Added classifier top-k activations up to k={max_classifier_top}"
            )

    return sample_df


def append_layer_metrics(
    result_df: pd.DataFrame,
    pt_file: Path,
    has_responses: bool,
    layer_measures: List[str],
    memory_monitor: MemoryMonitor,
    test_df_with_first_label: pd.DataFrame,
    response_resolution: str = "unit",
) -> pd.DataFrame:
    """Attach layer-level metrics to result dataframe.

    Always computes at sample-level resolution (resolution applied in Stage 2).

    Args:
        result_df: DataFrame to augment with layer metrics
        pt_file: Path to test_responses.pt file
        has_responses: Whether responses were loaded successfully
        layer_measures: List of layer measure names to compute
        memory_monitor: Memory monitoring instance
        test_df_with_first_label: Original test dataframe with first_label_index
        response_resolution: "unit" or "layer" — when "layer", uses fast path
            for pre-averaged data

    Returns:
        DataFrame with layer metrics added
    """
    if not (has_responses and layer_measures):
        return result_df

    # Fast path for pre-averaged (layer-wise) data
    if response_resolution == "layer":
        return _append_preaveraged_layer_metrics(
            result_df=result_df,
            pt_file=pt_file,
            layer_measures=layer_measures,
        )

    logger.info("  Calculating layer metrics at sample-level resolution...")

    try:
        # Map samples to presentation classes
        sample_to_presentation = (
            test_df_with_first_label.groupby("sample_index")["first_label_index"]
            .first()
            .to_dict()
        )
        presented_classes = sorted(result_df["first_label_index"].unique())
        unique_times = sorted(result_df["times_index"].unique())

        n_samples_expected = len(sample_to_presentation)
        logger.info(f"    Expected samples for layer metrics: {n_samples_expected}")

        # Process layer responses incrementally (always at sample resolution)
        layer_metrics = process_layer_responses_incremental(
            pt_file=pt_file,
            measures=layer_measures,
            sample_to_presentation=sample_to_presentation,
            presented_classes=presented_classes,
            unique_times=unique_times,
            memory_monitor=memory_monitor,
            max_retries=3,
            resolution="sample",  # Always sample-level in Stage 1
            n_samples_expected=n_samples_expected,
        )

    except Exception as exc:
        logger.error(f"Failed to calculate layer metrics: {exc}")
        raise

    # Build dataframe with layer metrics (sample-level)
    n_samples = result_df["sample_index"].nunique()
    n_times = len(unique_times)
    layer_data = {
        "sample_index": np.repeat(np.arange(n_samples), n_times),
        "times_index": np.tile(unique_times, n_samples),
    }
    layer_data.update(layer_metrics)
    layer_df = pd.DataFrame(layer_data)

    # Merge with result dataframe
    merged_df = result_df.merge(
        layer_df,
        on=["sample_index", "times_index"],
        how="left",
    )

    logger.info(f"    Added {len(layer_metrics)} layer metrics")
    return merged_df


def _append_preaveraged_layer_metrics(
    result_df: pd.DataFrame,
    pt_file: Path,
    layer_measures: List[str],
) -> pd.DataFrame:
    """Fast path for layer-wise (pre-averaged) response data.

    When responses were stored with response_resolution="layer", the .pt file
    already contains {layer}_response_avg and {layer}_response_std tensors of
    shape (n_samples, n_timesteps). This maps them directly to DataFrame columns,
    bypassing the memory-heavy incremental loading pipeline.

    Args:
        result_df: DataFrame to augment with layer metrics
        pt_file: Path to test_responses.pt file
        layer_measures: List of layer measure names to compute

    Returns:
        DataFrame with layer metrics added
    """
    logger.info("  Using fast path for pre-averaged layer responses...")

    responses = torch.load(pt_file, map_location="cpu", weights_only=True)
    responses.pop("_metadata", None)

    # Check for measures that require full 5D tensors
    spatial_measures = {"spatial_variance", "feature_variance"}
    unavailable = spatial_measures & set(layer_measures)
    if unavailable:
        logger.warning(
            f"  Skipping measures {unavailable} — requires full unit-wise tensors "
            f"(not available in layer-wise mode)"
        )

    n_samples = result_df["sample_index"].nunique()
    unique_times = sorted(result_df["times_index"].unique())
    n_times = len(unique_times)

    layer_data = {
        "sample_index": np.repeat(np.arange(n_samples), n_times),
        "times_index": np.tile(unique_times, n_samples),
    }

    metrics_added = 0
    for key, tensor in responses.items():
        # Skip classifier — handled separately
        if key == "classifier":
            continue

        # Match keys like layer0_response_avg, layer0_response_std
        for suffix in ("response_avg", "response_std"):
            if key.endswith(f"_{suffix}"):
                _ = suffix.split("_", 1)[1] if "_" in suffix else suffix
                # Only include if the metric is in the requested measures
                if suffix not in layer_measures:
                    continue
                # Flatten (n_samples, n_timesteps) -> (n_samples * n_timesteps,)
                values = tensor.float().numpy()
                n_samples_pt = values.shape[0]
                if n_samples_pt < n_samples:
                    logger.warning(
                        f"    {key}: PT has {n_samples_pt} samples, CSV has {n_samples}. "
                        f"Padding with NaN."
                    )
                    padded = np.full((n_samples, n_times), np.nan)
                    padded[:n_samples_pt, :n_times] = values[:, :n_times]
                    values = padded
                else:
                    values = values[:n_samples, :n_times]
                layer_data[key] = values.flatten()
                metrics_added += 1
                break

    layer_df = pd.DataFrame(layer_data)

    merged_df = result_df.merge(
        layer_df,
        on=["sample_index", "times_index"],
        how="left",
    )

    logger.info(f"    Added {metrics_added} pre-averaged layer metrics")
    return merged_df


def process_single_test(
    response_file: Path,
    test_output_file: Path,
    measure_config: MeasureConfig,
    memory_monitor: MemoryMonitor,
) -> pd.DataFrame:
    """Process a single test's outputs to compute all metrics.

    Args:
        response_file: Path to test_responses.pt
        test_output_file: Path to test_outputs.csv
        measure_config: Configuration for which measures to compute
        memory_monitor: Memory monitoring instance

    Returns:
        DataFrame at sample-level with all computed metrics (no metadata)
    """
    logger.info(f"Processing: {response_file.name} + {test_output_file.name}")

    # Load test outputs CSV
    test_df = load_df(test_output_file)
    test_df_with_first_label = process_test_performance(test_df)

    # Load response tensors
    responses, has_responses, response_resolution = load_responses(response_file)

    # Start with CSV data at sample level
    sample_level_df = test_df_with_first_label.copy()

    # Add classifier metrics
    sample_level_df = append_classifier_metrics(
        sample_level_df,
        responses=responses,
        confidence_measures=measure_config.confidence_measures,
        accuracy_topk_values=measure_config.accuracy_topk,
        classifier_topk_values=measure_config.classifier_topk,
    )

    # Add layer metrics
    sample_level_df = append_layer_metrics(
        sample_level_df,
        pt_file=response_file,
        has_responses=has_responses,
        layer_measures=measure_config.layer_measures,
        memory_monitor=memory_monitor,
        test_df_with_first_label=test_df_with_first_label,
        response_resolution=response_resolution,
    )

    logger.info(
        f"  Final shape: {sample_level_df.shape} with {len(sample_level_df.columns)} columns"
    )

    # Cleanup
    if responses is not None:
        del responses
    memory_monitor.cleanup()

    return sample_level_df


# Command line interface
parser = argparse.ArgumentParser(
    description="Process single test output (Stage 1 of experiment processing)"
)
parser.add_argument(
    "--response",
    type=Path,
    required=True,
    help="Path to test_responses.pt file",
)
parser.add_argument(
    "--test_output",
    type=Path,
    required=True,
    help="Path to test_outputs.csv file",
)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Path to output test_data.csv file",
)
parser.add_argument(
    "--measures",
    nargs="+",
    type=str,
    default=["response_avg", "response_std"],
    help=(
        "Measures to compute. Supported: layer metrics (response_avg, "
        "response_std, spatial_variance, feature_variance), confidence measures "
        "(guess_confidence, label_confidence, first_label_confidence), "
        "top-k accuracies (accuracy_topK), classifier activations (classifier_topK)."
    ),
)
parser.add_argument(
    "--memory_limit_gb",
    type=float,
    default=68.0,
    help="Soft memory limit in GB (triggers warnings, default: 68.0)",
)
parser.add_argument(
    "--remove_input_responses",
    type=str_to_bool,
    default="False",
    help="If True, delete test_responses.pt after successful processing",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize configuration
    measure_config = build_measure_config(args.measures)
    memory_monitor = MemoryMonitor(memory_limit_gb=args.memory_limit_gb)

    # Log configuration
    logger.info("=" * 80)
    logger.info("STAGE 1: SINGLE TEST PROCESSING")
    logger.info("=" * 80)
    logger.info(f"Response file: {args.response}")
    logger.info(f"Test output file: {args.test_output}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Measures: {args.measures}")
    if measure_config.accuracy_topk:
        logger.info(f"Accuracy top-k: {measure_config.accuracy_topk}")
    if measure_config.classifier_topk:
        logger.info(f"Classifier top-k: {measure_config.classifier_topk}")
    logger.info(f"Memory limit: {args.memory_limit_gb}GB")
    logger.info(f"Remove responses after processing: {args.remove_input_responses}")
    logger.info("NOTE: Output at sample-level (no metadata, no resolution applied)")

    processing_successful = False
    df = None

    try:
        # Check input files exist
        if not args.response.exists():
            raise FileNotFoundError(f"Response file not found: {args.response}")
        if not args.test_output.exists():
            raise FileNotFoundError(f"Test output file not found: {args.test_output}")

        memory_monitor.log_memory("start")

        # Process single test
        df = process_single_test(
            response_file=args.response,
            test_output_file=args.test_output,
            measure_config=measure_config,
            memory_monitor=memory_monitor,
        )

        # Save processed data
        logger.info(f"\nSaving processed data to: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Successfully saved {len(df)} rows")

        processing_successful = True
        memory_monitor.log_memory("after processing")

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

    # Remove input response file if successful and requested
    if processing_successful and args.remove_input_responses:
        logger.info("\n" + "=" * 80)
        logger.info("REMOVING INPUT RESPONSE FILE")
        logger.info("=" * 80)
        logger.info("Processing completed successfully, removing response file...")

        if args.response.exists():
            try:
                response_size_gb = args.response.stat().st_size / (1024**3)
                args.response.unlink()
                logger.info(
                    f"  ✓ Removed: {args.response.name} ({response_size_gb:.2f}GB freed)"
                )
            except Exception as e:
                logger.error(f"  ✗ Failed to remove {args.response.name}: {e}")

    # Print summary
    if df is not None:
        logger.info("\n" + "=" * 80)
        logger.info("OUTPUT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Samples: {df['sample_index'].nunique()}")
        logger.info(f"Time steps: {sorted(df['times_index'].unique())}")
        logger.info(f"Unique first labels: {sorted(df['first_label_index'].unique())}")

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
        if "accuracy" in df.columns:
            logger.info(
                f"Accuracy range: {df['accuracy'].min():.3f} to {df['accuracy'].max():.3f}"
            )

        logger.info("\n✅ Stage 1 processing completed successfully!")
