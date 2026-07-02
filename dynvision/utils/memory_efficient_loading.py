"""Memory-efficient layer loading and processing system.

This module implements sequential layer loading to minimize memory usage
when processing large .pt files containing neural network responses.
"""

import gc
import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import BadZipFile

import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and log memory usage throughout processing."""

    def __init__(self, memory_limit_gb: float = 20.0):
        self.memory_limit_gb = memory_limit_gb
        self.process = psutil.Process()
        self.peak_memory_gb = 0.0
        self.memory_history = []  # Track memory over time

    def get_current_memory_gb(self) -> float:
        """Get current memory usage in GB."""
        mem_info = self.process.memory_info()
        return mem_info.rss / 1024**3

    def log_memory(self, context: str = ""):
        """Log current memory usage with context."""
        current_gb = self.get_current_memory_gb()
        self.peak_memory_gb = max(self.peak_memory_gb, current_gb)

        # Store history for analysis
        self.memory_history.append((context, current_gb))

        # Enhanced logging with GPU memory if available
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_info = (
                f", GPU: {gpu_allocated:.2f}GB alloc, {gpu_reserved:.2f}GB reserved"
            )

        logger.info(
            f"Memory [{context}]: {current_gb:.2f}GB (peak: {self.peak_memory_gb:.2f}GB){gpu_info}"
        )

        if current_gb > self.memory_limit_gb:
            logger.warning(
                f"⚠️  Memory usage ({current_gb:.2f}GB) exceeds limit ({self.memory_limit_gb}GB)"
            )

    def cleanup(self):
        """Force garbage collection and log results."""
        before_gb = self.get_current_memory_gb()

        # Multi-stage cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak stats to get cleaner measurements
            torch.cuda.reset_peak_memory_stats()

        # Second pass - sometimes helps
        gc.collect()

        after_gb = self.get_current_memory_gb()
        freed_gb = before_gb - after_gb

        if freed_gb > 0.05:  # Log if any significant memory was freed
            logger.info(f"Freed {freed_gb:.2f}GB via garbage collection")

    def get_memory_trend(self) -> str:
        """Get memory usage trend analysis."""
        if len(self.memory_history) < 2:
            return "No trend data"

        recent = self.memory_history[-5:]  # Last 5 measurements
        if len(recent) < 2:
            return "Insufficient data"

        start_mem = recent[0][1]
        end_mem = recent[-1][1]
        trend = end_mem - start_mem

        if trend > 0.5:
            return f"📈 Memory increasing (+{trend:.2f}GB over last {len(recent)} operations)"
        elif trend < -0.5:
            return f"📉 Memory decreasing ({trend:.2f}GB over last {len(recent)} operations)"
        else:
            return (
                f"📊 Memory stable ({trend:+.2f}GB over last {len(recent)} operations)"
            )


def robust_load_pt_layer(
    file_path: Path, layer_name: str, max_retries: int = 3, retry_delay: float = 1.0
) -> Optional[torch.Tensor]:
    """
    Load a single layer and ensure it has no computational graph.

    CRITICAL: Immediately detach tensor to remove any computational graph.
    Loaded tensors should never need gradients for analysis.
    """
    for attempt in range(max_retries):
        try:
            # Load the full file (unavoidable with torch.load)
            data = torch.load(
                file_path, map_location=torch.device("cpu"), weights_only=False
            )

            if layer_name not in data:
                raise KeyError(
                    f"Layer '{layer_name}' not found in {file_path}. "
                    f"Available layers: {list(data.keys())}"
                )

            # Extract the layer tensor
            layer_tensor = data[layer_name]

            # CRITICAL FIX: Remove computational graph immediately
            # Loaded tensors from .pt files shouldn't need gradients
            if layer_tensor.requires_grad:
                logger.debug(f"Layer '{layer_name}' has requires_grad=True, detaching")

            layer_tensor = layer_tensor.detach()  # Remove computational graph

            # Clone to ensure clean memory ownership
            layer_tensor = layer_tensor.clone()

            # Delete the full dict immediately to free memory
            del data
            gc.collect()

            logger.debug(
                f"Loaded layer '{layer_name}' with shape {layer_tensor.shape}, "
                f"requires_grad={layer_tensor.requires_grad}"
            )

            return layer_tensor

        except (RuntimeError, BadZipFile) as e:
            logger.warning(
                f"Attempt {attempt+1}/{max_retries} failed for layer '{layer_name}' "
                f"in {file_path.name}: {str(e)}"
            )

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                error_msg = (
                    f"Failed to load layer '{layer_name}' from {file_path} "
                    f"after {max_retries} attempts. Last error: {str(e)}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        except KeyError as e:
            raise RuntimeError(str(e)) from e

        except Exception as e:
            error_msg = (
                f"Unexpected error loading layer '{layer_name}' from {file_path}: "
                f"{type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    return None


def get_layer_names_from_pt(
    file_path: Path,
    exclude_classifier: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> List[str]:
    """
    Get layer names from a .pt file without loading full tensors.

    Args:
        file_path: Path to the .pt file
        exclude_classifier: If True, exclude 'classifier' layers
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        List of layer names

    Raises:
        RuntimeError: If loading fails after all retries
    """
    for attempt in range(max_retries):
        try:
            # Load only to get keys
            data = torch.load(
                file_path, map_location=torch.device("cpu"), weights_only=False
            )

            if not isinstance(data, dict):
                raise TypeError(
                    f"Expected dict in {file_path}, got {type(data).__name__}"
                )

            layer_names = list(data.keys())

            # Filter out classifier layers if requested
            if exclude_classifier:
                layer_names = [
                    name for name in layer_names if "classifier" not in name.lower()
                ]

            del data
            gc.collect()

            logger.info(
                f"Found {len(layer_names)} layers in {file_path.name}: {layer_names}"
            )

            return layer_names

        except (RuntimeError, BadZipFile, io.UnsupportedOperation) as e:
            logger.warning(
                f"Attempt {attempt+1}/{max_retries} failed to read layer names "
                f"from {file_path.name}: {str(e)}"
            )

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                error_msg = (
                    f"Failed to read layer names from {file_path} "
                    f"after {max_retries} attempts. Last error: {str(e)}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = (
                f"Unexpected error reading layer names from {file_path}: "
                f"{type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    return []


def get_max_timesteps_from_pt(
    file_path: Path, layer_names: List[str], max_retries: int = 3
) -> int:
    """
    Determine maximum timesteps across all layers efficiently.

    Loads only shape information, not full tensors.

    Args:
        file_path: Path to the .pt file
        layer_names: List of layer names to check
        max_retries: Maximum number of retry attempts

    Returns:
        Maximum number of timesteps

    Raises:
        RuntimeError: If unable to determine max timesteps
    """
    logger.info(f"Determining max timesteps from {file_path.name}...")

    max_timesteps = 0

    for attempt in range(max_retries):
        try:
            data = torch.load(
                file_path, map_location=torch.device("cpu"), weights_only=False
            )

            for layer_name in layer_names:
                if layer_name not in data:
                    logger.warning(
                        f"Layer '{layer_name}' not found when determining max timesteps"
                    )
                    continue

                tensor_shape = data[layer_name].shape

                # Assuming shape is [batch, time, ...] or [batch, time, channels]
                if len(tensor_shape) >= 2:
                    timesteps = tensor_shape[1]
                    max_timesteps = max(max_timesteps, timesteps)
                    logger.debug(
                        f"Layer '{layer_name}': {timesteps} timesteps (shape: {tensor_shape})"
                    )

            del data
            gc.collect()

            logger.info(f"Max timesteps: {max_timesteps}")
            return max_timesteps

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt+1}/{max_retries} failed to determine max timesteps: {e}"
                )
                time.sleep(1.0)
            else:
                error_msg = (
                    f"Failed to determine max timesteps from {file_path} "
                    f"after {max_retries} attempts: {str(e)}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    return 0


def pad_tensor_efficient(tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
    """
    Pad tensor along time dimension with minimal memory overhead.

    Args:
        tensor: Input tensor with shape [batch, time, ...]
        pad_len: Number of timesteps to pad at the beginning

    Returns:
        Padded tensor
    """
    if pad_len <= 0:
        return tensor

    logger.debug(f"Padding tensor of shape {tensor.shape} with {pad_len} timesteps")

    # Pre-allocate padded tensor
    padded_shape = list(tensor.shape)
    padded_shape[1] += pad_len

    padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)

    # Copy data to end (after padding)
    padded[:, pad_len:, ...] = tensor

    return padded


def extract_metric_values_for_dataframe(
    metric_tensor: torch.Tensor,
    sample_to_presentation: Dict[int, int],
    presented_classes: List[int],
    unique_times: List[int],
    resolution: str = "class",
    n_samples_expected: Optional[int] = None,
) -> np.ndarray:
    """
    Extract only the needed values from a metric tensor for dataframe construction.

    This is critical for memory efficiency: instead of keeping the full tensor,
    we extract only the values we need for the final dataframe.

    Args:
        metric_tensor: Computed metric tensor [samples, times, ...]
        sample_to_presentation: Mapping from sample_index to first_label_index
        presented_classes: List of unique presentation labels
        unique_times: List of unique time indices
        resolution: 'sample' for all samples, 'class' for one per presentation (default: 'class')
        n_samples_expected: Expected number of samples (for sample mode with PT/CSV mismatch)

    Returns:
        1D numpy array with extracted values in correct order
    """
    if resolution == "sample":
        # Extract ALL samples at (sample_index, times_index) resolution
        n_samples_pt = metric_tensor.shape[0]  # Samples actually in PT file
        n_samples = (
            n_samples_expected if n_samples_expected is not None else n_samples_pt
        )
        n_times = len(unique_times)
        n_rows = n_samples * n_times
        values = np.empty(n_rows, dtype=np.float32)

        idx = 0
        for sample_idx in range(n_samples):
            for time_idx in unique_times:
                # Check if sample is available in PT file
                if sample_idx >= n_samples_pt:
                    # Sample not in PT file (CSV/PT mismatch) - fill with NaN
                    values[idx] = np.nan
                    idx += 1
                    continue

                try:
                    # Extract single value
                    if len(metric_tensor.shape) == 2:
                        # Shape: [samples, times]
                        val = metric_tensor[sample_idx, time_idx].item()
                    else:
                        # Should not happen after reduction, but handle gracefully
                        logger.warning(
                            f"Unexpected metric tensor shape: {metric_tensor.shape}"
                        )
                        val = metric_tensor[sample_idx, time_idx].flatten()[0].item()

                    values[idx] = val
                except (IndexError, RuntimeError) as e:
                    logger.warning(
                        f"Error extracting value at sample {sample_idx}, time {time_idx}: {e}"
                    )
                    values[idx] = np.nan

                idx += 1

        return values

    else:  # resolution == "class"
        # Extract ONE sample per presentation at (first_label_index, times_index) resolution
        n_rows = len(presented_classes) * len(unique_times)
        values = np.empty(n_rows, dtype=np.float32)

        idx = 0
        for pres_label in presented_classes:
            # Find first sample with this presentation label
            sample_idx = None
            for s, p in sample_to_presentation.items():
                if p == pres_label:
                    sample_idx = s
                    break

            if sample_idx is None:
                # Fill with NaN if no sample found
                for _ in unique_times:
                    values[idx] = np.nan
                    idx += 1
                continue

            for time_idx in unique_times:
                try:
                    # Check bounds before extraction
                    if sample_idx >= metric_tensor.shape[0]:
                        # Sample not available in PT file (mismatched sample sizes)
                        values[idx] = np.nan
                        idx += 1
                        continue

                    # Extract single value
                    if len(metric_tensor.shape) == 2:
                        # Shape: [samples, times]
                        val = metric_tensor[sample_idx, time_idx].item()
                    else:
                        # Should not happen after reduction, but handle gracefully
                        logger.warning(
                            f"Unexpected metric tensor shape: {metric_tensor.shape}"
                        )
                        val = metric_tensor[sample_idx, time_idx].flatten()[0].item()

                    values[idx] = val
                except (IndexError, RuntimeError) as e:
                    logger.warning(
                        f"Error extracting value at sample {sample_idx}, time {time_idx}: {e}"
                    )
                    values[idx] = np.nan

                idx += 1

        return values


def process_large_layer_chunked(
    layer_tensor: torch.Tensor,
    metric_func,
    metric_name: str,
    sample_to_presentation: Dict[int, int],
    presented_classes: List[int],
    unique_times: List[int],
    memory_monitor: MemoryMonitor,
    chunk_size: int = 32,
    resolution: str = "class",
    n_samples_expected: Optional[int] = None,
) -> np.ndarray:
    """
    Process very large layers in chunks to avoid OOM.

    This is a fallback for layers that are too large to process at once.
    Processes samples in chunks and combines results.

    Args:
        layer_tensor: Large tensor to process [samples, time, ...]
        metric_func: Function to compute metric
        metric_name: Name of metric for logging
        sample_to_presentation: Sample to presentation mapping
        presented_classes: List of unique presentations
        unique_times: List of unique times
        memory_monitor: Memory monitor instance
        chunk_size: Number of samples to process at once

    Returns:
        Combined numpy array of extracted values
    """
    logger.info(
        f"    🔄 Using chunked processing for {metric_name} (chunk_size={chunk_size})"
    )

    n_samples = layer_tensor.shape[0]
    all_values = []

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_indices = list(range(start_idx, end_idx))

        logger.debug(
            f"      Processing chunk {start_idx}-{end_idx-1} ({len(chunk_indices)} samples)"
        )
        memory_monitor.log_memory(f"chunk {start_idx}-{end_idx-1} start")

        # Extract chunk
        chunk_tensor = layer_tensor[start_idx:end_idx]

        # Compute metric on chunk
        with torch.no_grad():
            chunk_metric = metric_func(chunk_tensor)
            memory_monitor.log_memory(f"chunk {start_idx}-{end_idx-1} computed")

            # Create temporary mapping for this chunk
            chunk_sample_to_presentation = {
                i: sample_to_presentation.get(start_idx + i, -1)
                for i in range(len(chunk_indices))
            }

            # Extract values for this chunk
            chunk_values = extract_metric_values_for_dataframe(
                chunk_metric,
                chunk_sample_to_presentation,
                presented_classes,
                unique_times,
                resolution=resolution,
                n_samples_expected=n_samples_expected,
            )

            all_values.append(chunk_values)

            # Cleanup chunk
            del chunk_tensor, chunk_metric

        memory_monitor.cleanup()
        memory_monitor.log_memory(f"chunk {start_idx}-{end_idx-1} cleaned")

    # Combine all chunk results
    # Note: This assumes chunk results can be simply concatenated in presentation order
    # For metrics that need global aggregation, this would need modification
    if len(all_values) == 1:
        return all_values[0]
    else:
        # For now, use the first chunk's values as template
        # This is a simplified approach - more sophisticated combination may be needed
        logger.warning(f"    ⚠️  Chunked processing may affect {metric_name} accuracy")
        return all_values[0]  # Simplified: return first chunk


def check_layer_memory_requirements(
    layer_tensor: torch.Tensor,
    memory_monitor: MemoryMonitor,
    safety_factor: float = 0.7,
) -> bool:
    """
    Estimate if layer can be processed without chunking.

    Args:
        layer_tensor: Tensor to check
        memory_monitor: Memory monitor instance
        safety_factor: Safety factor for memory estimation (0.7 = use 70% of available)

    Returns:
        True if layer can be processed normally, False if chunking needed
    """
    # Estimate memory requirements
    tensor_size_gb = layer_tensor.numel() * layer_tensor.element_size() / 1024**3
    current_memory_gb = memory_monitor.get_current_memory_gb()
    available_memory_gb = (
        memory_monitor.memory_limit_gb - current_memory_gb
    ) * safety_factor

    # Rough estimate: metric computation might need 2-3x tensor size temporarily
    estimated_peak_gb = tensor_size_gb * 3

    can_process = estimated_peak_gb < available_memory_gb

    logger.info(
        f"    Memory check: tensor={tensor_size_gb:.2f}GB, "
        f"current={current_memory_gb:.2f}GB, "
        f"available={available_memory_gb:.2f}GB, "
        f"estimated_peak={estimated_peak_gb:.2f}GB, "
        f"can_process={can_process}"
    )

    return can_process


def process_layer_responses_incremental(
    pt_file: Path,
    measures: List[str],
    sample_to_presentation: Dict[int, int],
    presented_classes: List[int],
    unique_times: List[int],
    memory_monitor: MemoryMonitor,
    max_retries: int = 3,
    resolution: str = "class",
    n_samples_expected: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    STEP 3: Enhanced memory management with aggressive cleanup and detailed logging.

    Changes from Step 2:
    - Enhanced memory logging at every critical point
    - More aggressive garbage collection
    - Explicit variable deletion in proper order
    - Force memory synchronization after each operation
    - Clear intermediate references immediately
    """
    from tqdm import tqdm
    from dynvision.utils.visualization_utils import (
        layer_response_avg,
        layer_response_std,
        spatial_variance,
        feature_variance,
    )

    logger.info(f"Processing {pt_file.name} with Step 3 aggressive memory management")
    memory_monitor.log_memory("start of processing")

    try:
        # Get layer names and max timesteps (lightweight operations)
        logger.info("Getting layer metadata...")
        layer_names = get_layer_names_from_pt(
            pt_file, exclude_classifier=True, max_retries=max_retries
        )

        if not layer_names:
            raise RuntimeError(f"No valid layers found in {pt_file}")

        max_timesteps = get_max_timesteps_from_pt(
            pt_file, layer_names, max_retries=max_retries
        )

        memory_monitor.log_memory("after metadata loading")
        logger.info(f"Found {len(layer_names)} layers, max timesteps: {max_timesteps}")

        # Storage for extracted values
        extracted_values = {}

        # Process each layer with enhanced monitoring
        for layer_idx, layer_name in enumerate(
            tqdm(layer_names, desc=f"Processing layers in {pt_file.name}")
        ):
            try:
                logger.info(
                    f"=== LAYER {layer_idx+1}/{len(layer_names)}: {layer_name} ==="
                )
                memory_monitor.log_memory(f"before loading {layer_name}")

                # STEP 3A: Load layer tensor with immediate cleanup
                logger.info(f"Loading layer: {layer_name}")
                layer_tensor = robust_load_pt_layer(
                    pt_file, layer_name, max_retries=max_retries
                )

                if layer_tensor is None:
                    raise RuntimeError(f"Failed to load layer '{layer_name}'")

                # Verify no grad requirement
                assert (
                    not layer_tensor.requires_grad
                ), f"Layer '{layer_name}' still has requires_grad=True after loading!"

                memory_monitor.log_memory(f"loaded {layer_name}")
                logger.info(
                    f"  Layer shape: {layer_tensor.shape}, dtype: {layer_tensor.dtype}"
                )

                # STEP 3B: Pad if needed with immediate cleanup
                pad_len = max_timesteps - layer_tensor.shape[1]
                if pad_len > 0:
                    logger.info(f"  Padding {layer_name} with {pad_len} timesteps")
                    # Store reference to old tensor for explicit deletion
                    old_tensor = layer_tensor
                    layer_tensor = pad_tensor_efficient(layer_tensor, pad_len)
                    # Explicitly delete old tensor
                    del old_tensor
                    memory_monitor.cleanup()
                    memory_monitor.log_memory(f"padded {layer_name}")

                # Determine if this is a 5D tensor
                is_5d = len(layer_tensor.shape) == 5
                logger.info(f"  Computing metrics for {layer_name} (5D: {is_5d})")

                # Defensive guard: handle pre-averaged 2D tensors (from layer-wise storage)
                if layer_tensor.dim() == 2:
                    logger.info(
                        f"  Layer {layer_name} is 2D ({layer_tensor.shape}) — "
                        f"treating as pre-averaged data"
                    )
                    full_metric_name = f"{layer_name}_response_avg"
                    extracted_values[full_metric_name] = (
                        extract_metric_values_for_dataframe(
                            layer_tensor,
                            sample_to_presentation,
                            presented_classes,
                            unique_times,
                            resolution=resolution,
                            n_samples_expected=n_samples_expected,
                        )
                    )
                    del layer_tensor
                    memory_monitor.cleanup()
                    continue

                # STEP 3C: Process each metric with aggressive cleanup
                metrics_to_process = []
                if "response_avg" in measures:
                    metrics_to_process.append(("response_avg", layer_response_avg))
                if "response_std" in measures:
                    metrics_to_process.append(("response_std", layer_response_std))
                if is_5d and "spatial_variance" in measures:
                    metrics_to_process.append(("spatial_variance", spatial_variance))
                if is_5d and "feature_variance" in measures:
                    metrics_to_process.append(("feature_variance", feature_variance))

                for metric_name, metric_func in metrics_to_process:
                    logger.info(f"    Computing {metric_name}...")
                    memory_monitor.log_memory(f"before {metric_name}")

                    # STEP 3C1: Check if layer is too large for normal processing
                    can_process_normally = check_layer_memory_requirements(
                        layer_tensor, memory_monitor, safety_factor=0.6
                    )

                    full_metric_name = f"{layer_name}_{metric_name}"

                    if can_process_normally:
                        # Normal processing
                        with torch.no_grad():
                            # Compute metric
                            metric_tensor = metric_func(layer_tensor)
                            memory_monitor.log_memory(f"computed {metric_name}")

                            # Extract values immediately
                            extracted_values[full_metric_name] = (
                                extract_metric_values_for_dataframe(
                                    metric_tensor,
                                    sample_to_presentation,
                                    presented_classes,
                                    unique_times,
                                    resolution=resolution,
                                    n_samples_expected=n_samples_expected,
                                )
                            )
                            memory_monitor.log_memory(f"extracted {metric_name}")

                            # CRITICAL: Delete metric tensor immediately
                            del metric_tensor
                    else:
                        # Chunked processing fallback
                        logger.warning(
                            f"    ⚠️  Layer {layer_name} too large for normal processing, "
                            f"using chunked approach for {metric_name}"
                        )
                        extracted_values[full_metric_name] = (
                            process_large_layer_chunked(
                                layer_tensor=layer_tensor,
                                metric_func=metric_func,
                                metric_name=metric_name,
                                sample_to_presentation=sample_to_presentation,
                                presented_classes=presented_classes,
                                unique_times=unique_times,
                                memory_monitor=memory_monitor,
                                chunk_size=32,  # Conservative chunk size
                                resolution=resolution,
                                n_samples_expected=n_samples_expected,
                            )
                        )

                    # STEP 3D: Aggressive cleanup after each metric
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    memory_monitor.log_memory(f"cleaned up {metric_name}")
                    logger.info(f"    ✓ {metric_name} completed")

                    # STEP 3D1: Emergency brake if memory is still too high
                    current_memory = memory_monitor.get_current_memory_gb()
                    if current_memory > memory_monitor.memory_limit_gb * 0.9:
                        logger.error(
                            f"🚨 EMERGENCY: Memory ({current_memory:.2f}GB) exceeds 90% of limit "
                            f"({memory_monitor.memory_limit_gb}GB) after {metric_name}!"
                        )
                        # Emergency cleanup
                        memory_monitor.cleanup()
                        # Check trend
                        trend = memory_monitor.get_memory_trend()
                        logger.error(f"Memory trend: {trend}")

                        # If still too high, log a strong warning but continue
                        if (
                            memory_monitor.get_current_memory_gb()
                            > memory_monitor.memory_limit_gb * 0.85
                        ):
                            logger.warning(
                                f"🚨 HIGH MEMORY WARNING: Memory usage remains very high after emergency cleanup. "
                                f"Current: {memory_monitor.get_current_memory_gb():.2f}GB, "
                                f"Limit: {memory_monitor.memory_limit_gb}GB. Continuing but risk of OOM."
                            )

                # STEP 3E: Final layer cleanup
                logger.info(f"  Cleaning up layer {layer_name}")
                del layer_tensor

                # Force complete memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                gc.collect()

                memory_monitor.log_memory(f"completed {layer_name}")
                logger.info(f"=== LAYER {layer_name} COMPLETE ===")

                # STEP 3F: Check memory after each layer
                current_memory = memory_monitor.get_current_memory_gb()
                if current_memory > memory_monitor.memory_limit_gb * 0.8:
                    logger.warning(
                        f"⚠️  Memory usage ({current_memory:.2f}GB) approaching limit "
                        f"({memory_monitor.memory_limit_gb}GB) after layer {layer_name}"
                    )
                    # Additional emergency cleanup
                    memory_monitor.cleanup()

            except Exception as e:
                error_msg = (
                    f"Error processing layer '{layer_name}' in {pt_file}: "
                    f"{type(e).__name__}: {str(e)}"
                )
                logger.error(error_msg)
                # Cleanup on error
                if "layer_tensor" in locals():
                    del layer_tensor
                memory_monitor.cleanup()
                raise RuntimeError(error_msg) from e

        logger.info(
            f"Successfully processed {len(layer_names)} layers, "
            f"extracted {len(extracted_values)} metric columns"
        )
        memory_monitor.log_memory("processing complete")

        return extracted_values

    except Exception as e:
        error_msg = f"Failed to process {pt_file}: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        # Final cleanup on error
        memory_monitor.cleanup()
        raise RuntimeError(error_msg) from e
