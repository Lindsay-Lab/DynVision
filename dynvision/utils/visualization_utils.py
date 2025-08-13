"""Visualization utility functions for the DynVision toolbox.

This module provides visualization-related utilities:
- Layer response analysis
- Peak detection and analysis
- Accuracy calculation
- Response data loading
- Plot saving
"""

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data_utils import load_df
from .string_utils import extract_param_from_string


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array with proper dtype handling.

    Args:
        tensor: PyTorch tensor in any precision format

    Returns:
        NumPy array in float32 format
    """
    return tensor.cpu().float().numpy()


logger = logging.getLogger(__name__)


def layer_power(response: torch.Tensor) -> torch.Tensor:
    """Calculate mean power of layer responses.

    Args:
        response: Layer response tensor

    Returns:
        Mean absolute power tensor of shape [n_features, n_timesteps]
    """
    return response.abs().mean(dim=list(range(2, response.dim())))


def peak_time(response: torch.Tensor) -> torch.Tensor:
    """Calculate peak time for each feature.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak times for each feature
    """
    mean_power = layer_power(response)
    return mean_power.argmax(dim=1)


def peak_height(response: torch.Tensor) -> torch.Tensor:
    """Calculate peak height for each feature.

    Args:
        response: Layer response tensor

    Returns:
        Tensor of peak heights for each feature
    """
    mean_power = layer_power(response)
    max_values, max_indices = mean_power.max(dim=1)
    return max_values


def peak_ratio(response: torch.Tensor, min_delay: int = 3) -> torch.Tensor:
    """Calculate ratio between first and second peaks.

    Args:
        response: Layer response tensor
        min_delay: Minimum separation between peaks

    Returns:
        Tensor of peak ratios for each feature
    """
    mean_power = layer_power(response)
    peak1_index = mean_power.argmax(dim=1)
    peak1_value = torch.tensor(
        [
            deepcopy(mean_power[channel, i].item())
            for channel, i in enumerate(peak1_index)
        ]
    )

    for channel, i in enumerate(peak1_index):
        mean_power[channel, i - min_delay : i + min_delay] = float("-inf")

    peak2_index = mean_power.argmax(dim=1)
    peak2_value = [mean_power[channel, i] for channel, i in enumerate(peak2_index)]

    ratio = torch.Tensor(
        [
            p1 / p2 if i1 < i2 else p2 / p1
            for i1, p1, i2, p2 in zip(
                peak1_index, peak1_value, peak2_index, peak2_value
            )
        ]
    )
    return ratio


def calculate_accuracy(df: pd.DataFrame) -> float:
    """Calculate classification accuracy.

    Args:
        df: DataFrame with label_index and guess_index columns

    Returns:
        Classification accuracy as float
    """
    dfi = df[df.label_index >= 0]
    accuracy = (dfi.guess_index == dfi.label_index).mean()
    return accuracy


def load_responses(
    pt_files: List[Path],
    csv_files: List[Path],
    data_arg_key: str = "contrast",
    measures: List[str] = ["power", "peak_time", "peak_height"],
    category: str = "rctype",
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and process model responses.

    Args:
        pt_files: List of PyTorch response files
        csv_files: List of CSV label files
        data_arg_key: Key for data arguments
        measures: List of measures to compute
        category: Category key for grouping

    Returns:
        Tuple of (DataFrame with responses, list of layer names)
    """
    dfs = []

    for pt_file, csv_file in zip(pt_files, csv_files):
        arg_value = extract_param_from_string(
            pt_file.stem, key=data_arg_key, value_type=float
        )

        cat_value = extract_param_from_string(
            pt_file.stem, key=category, value_type=None
        )

        if not arg_value == extract_param_from_string(
            csv_file.stem, key=data_arg_key, value_type=float
        ):
            raise ValueError(f"{data_arg_key} values do not match!")
        if not cat_value == extract_param_from_string(
            csv_file.stem, key=category, value_type=None
        ):
            raise ValueError(f"{category} do not match!")

        df = load_df(csv_file)
        df[data_arg_key] = arg_value
        df[category] = cat_value

        n_classes = len(df.class_index.unique())

        responses = torch.load(pt_file, map_location=torch.device("cpu"))

        max_timesteps = max(tensor.shape[1] for tensor in responses.values())

        for layer_name, tensor in responses.items():
            pad_len = max_timesteps - tensor.shape[1]
            if pad_len > 0:
                # Pad at the start along axis 1 (time steps) with zeros
                # For shape [batch, timesteps, n_channels, dim_y, dim_x], pad for axis 1
                # torch.nn.functional.pad expects (dim_x, dim_y, n_channels, timesteps)
                # So pad = (0,0, 0,0, 0,0, pad_len,0)
                pad = (0, 0, 0, 0, 0, 0, pad_len, 0)
                tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0)
                responses[layer_name] = tensor

        layer_names = list(responses.keys())
        n_samples, n_timesteps, *_ = responses[layer_names[0]].shape

        for layer in layer_names:
            try:
                if "power" in measures:
                    power_tensor = (
                        layer_power(responses[layer])
                        .flatten()
                        .repeat_interleave(n_classes)
                    )
                    df[f"{layer}_power"] = tensor_to_numpy(power_tensor)

                if "peak_time" in measures:
                    peak_time_tensor = peak_time(responses[layer]).repeat_interleave(
                        n_classes * n_timesteps
                    )
                    df[f"{layer}_peak_time"] = tensor_to_numpy(peak_time_tensor)

                if "peak_height" in measures:
                    peak_height_tensor = peak_height(
                        responses[layer]
                    ).repeat_interleave(n_classes * n_timesteps)
                    df[f"{layer}_peak_height"] = tensor_to_numpy(peak_height_tensor)

                if "peak_ratio" in measures:
                    peak_ratio_tensor = peak_ratio(responses[layer]).repeat_interleave(
                        n_classes * n_timesteps
                    )
                    df[f"{layer}_peak_ratio"] = tensor_to_numpy(peak_ratio_tensor)
            except Exception as e:
                logger.error(
                    f"Failed to convert tensor to numpy for layer {layer}: {e}"
                )
                raise

        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    return df, layer_names


def chunk_lists(lst1, lst2, chunk_size):
    """Split two lists into chunks of specified size."""
    for i in range(0, len(lst1), chunk_size):
        yield lst1[i : i + chunk_size], lst2[i : i + chunk_size]


def load_responses_in_batches(
    responses_files,
    test_outputs_files,
    data_arg_key,
    measures,
    category,
    batch_size=1,
    filter_first_label_set=False,
):
    """Load responses in batches to manage memory efficiently."""

    if len(responses_files) != len(test_outputs_files):
        raise ValueError(
            "Number of response files must match number of test output files"
        )

    print(f"Processing {len(responses_files)} files in batches of {batch_size}")

    all_dataframes = []
    layer_names = None
    first_label_set = None

    # Process files in batches
    batch_count = 0
    for response_batch, output_batch in chunk_lists(
        responses_files, test_outputs_files, batch_size
    ):
        batch_count += 1
        print(
            f"Processing batch {batch_count}/{(len(responses_files) + batch_size - 1) // batch_size}"
        )
        print(
            f"  Files: {len(response_batch)} response files, {len(output_batch)} output files"
        )

        try:
            # Load this batch
            batch_df, batch_layer_names = load_responses(
                response_batch,
                output_batch,
                data_arg_key=data_arg_key,
                measures=measures,
                category=category,
            )

            # Store layer names from first batch
            if layer_names is None:
                layer_names = batch_layer_names
            else:
                # Verify layer names are consistent across batches
                if set(layer_names) != set(batch_layer_names):
                    print(
                        f"Warning: Layer names differ between batches. "
                        f"First batch: {layer_names}, Current batch: {batch_layer_names}"
                    )

            if filter_first_label_set:
                # Get first label_set from first batch and filter immediately for efficiency
                first_label_set = batch_df.label_set.unique()[0]
                print(f"  Using first label_set: {first_label_set}")

                # Filter for first label_set only to reduce memory usage
                if first_label_set is not None:
                    batch_df = batch_df[
                        batch_df.label_set.str.contains(str(first_label_set))
                    ]
                    print(f"  After filtering for label_set: {len(batch_df)} rows")

                # Remove label_set column to save memory since we no longer need it
                if "label_set" in batch_df.columns:
                    batch_df = batch_df.drop("label_set", axis=1)

            all_dataframes.append(batch_df)
            print(f"  Batch {batch_count} processed: {len(batch_df)} rows")

            # Clear memory
            del batch_df

        except Exception as e:
            print(f"Error processing batch {batch_count}: {e}")
            print(f"  Response files: {response_batch}")
            print(f"  Output files: {output_batch}")
            continue

    if not all_dataframes:
        raise ValueError("No data was successfully loaded from any batch")

    # Combine all dataframes
    print(f"Combining {len(all_dataframes)} batches...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Clear memory
    del all_dataframes

    print(f"Total combined data: {len(combined_df)} rows")
    return combined_df, layer_names


def save_plot(file_path: Path, dpi: int = 300, **kwargs) -> None:
    """Save matplotlib plot with error handling.

    Args:
        file_path: Path to save plot
        dpi: Resolution for saved plot
        **kwargs: Additional arguments for savefig

    Returns:
        None
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        plt.savefig(fname=file_path, dpi=dpi, bbox_inches="tight", **kwargs)

    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        try:  # save empty plot
            plt.subplots()
            plt.savefig(fname=file_path)
        except Exception as e:
            logging.error(f"Failed to save empty plot: {e}")

    finally:
        if "ipykernel" in sys.modules:
            plt.show()
        else:
            plt.close()

    logging.info(f"Plot saved successfully at {file_path}")

    return None
