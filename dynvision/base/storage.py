"""
Refined data buffer for DynVision PyTorch Lightning workflows.

Provides efficient storage for neural network responses and records with
clear indexing behavior, thread safety, and efficient tensor operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import threading
import gc
from pytorch_lightning import LightningModule

import torch
import pandas as pd
import numpy as np
import time

logger = logging.getLogger(__name__)


class SamplingStrategy:
    """Base class for sampling strategies."""

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        """Determine if we should store the current sample."""
        raise NotImplementedError

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        """Get the index where to store the sample, or None if shouldn't store."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset strategy state."""
        pass


class CyclicStrategy(SamplingStrategy):
    """
    Circular buffer strategy - maintains last N samples.

    Indexing behavior:
    - get(0): oldest item in buffer (when full)
    - get(-1): newest item in buffer
    - get(-x): x-th newest item

    Note: When buffer is not full, get(0) is the first item inserted.
    """

    def __init__(self):
        self.head = 0  # Points to next insertion slot

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        return True  # Always store in cyclic mode

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        if buffer_size < max_size:
            return buffer_size  # Still filling up
        else:
            idx = self.head
            self.head = (self.head + 1) % max_size
            return idx

    def get_logical_index(
        self, requested_index: int, buffer_size: int, max_size: int
    ) -> int:
        """Convert logical index to physical storage index."""
        if buffer_size < max_size:
            # Buffer not full - simple indexing
            if requested_index < 0:
                requested_index = buffer_size + requested_index
            return max(0, min(requested_index, buffer_size - 1))
        else:
            # Buffer full - circular indexing
            # Negative indexing: -1 is newest, -2 is second newest, etc.
            # Positive indexing: 0 is oldest, 1 is second oldest, etc.
            physical_index = (self.head + requested_index) % max_size
            return physical_index

    def reset(self) -> None:
        self.head = 0


class FixedStrategy(SamplingStrategy):
    """
    Fixed size strategy - stores first N samples then stops.

    Indexing behavior:
    - get(0): first item stored
    - get(-1): last item stored
    - get(-x): x-th from last item stored

    Simple, predictable indexing like a regular list.
    """

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        return buffer_size < max_size

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        return (
            buffer_size
            if self.should_store(buffer_size, total_seen, max_size)
            else None
        )

    def get_logical_index(
        self, requested_index: int, buffer_size: int, max_size: int
    ) -> int:
        """Convert logical index to physical storage index."""
        if requested_index < 0:
            requested_index = buffer_size + requested_index
        return max(0, min(requested_index, buffer_size - 1))

    def reset(self) -> None:
        pass


class ReservoirStrategy(SamplingStrategy):
    """
    Reservoir sampling strategy - probabilistic representative sampling.

    Indexing behavior:
    - get(0): some stored item (order not guaranteed)
    - get(-1): some stored item (order not guaranteed)
    - get(-x): some stored item (order not guaranteed)

    Warning: Order is not meaningful with reservoir sampling!
    Use only when you don't care about temporal order.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        if buffer_size < max_size:
            return True  # Still filling reservoir

        # Probability = max_size / (total_seen + 1)
        return self.rng.random() < (max_size / (total_seen + 1))

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        if not self.should_store(buffer_size, total_seen, max_size):
            return None

        if buffer_size < max_size:
            return buffer_size  # Still filling
        else:
            return self.rng.randint(0, max_size)  # Random replacement

    def get_logical_index(
        self, requested_index: int, buffer_size: int, max_size: int
    ) -> int:
        """Convert logical index to physical storage index."""
        # For reservoir sampling, order doesn't matter - just clamp to valid range
        if requested_index < 0:
            requested_index = buffer_size + requested_index
        return max(0, min(requested_index, buffer_size - 1))

    def reset(self) -> None:
        pass


class DataBuffer:
    """
    High-performance buffer for neural network data.

    Thread Safety:
    - self._lock: Ensures thread-safe operations for multi-GPU distributed training
    - When multiple processes access the buffer simultaneously, the lock prevents
      data corruption and ensures consistent state updates

    Indexing Behavior:
    - Depends on the sampling strategy used
    - CyclicStrategy: Maintains temporal order (newest/oldest semantics)
    - FixedStrategy: Simple list-like indexing (predictable order)
    - ReservoirStrategy: Order not guaranteed (for statistical sampling)

    Memory Management:
    - Explicit cleanup to prevent memory leaks
    - Efficient tensor operations with minimal copying
    """

    __slots__ = [
        "max_size",
        "strategy_name",
        "cpu_offload",
        "detach_tensors",
        "thread_safe",
        "_storage",
        "_strategy",
        "_size",
        "_total_seen",
        "_lock",
        "name",
    ]

    def __init__(
        self,
        max_size: int,
        strategy: str = "cyclic",
        cpu_offload: bool = True,
        detach_tensors: bool = True,
        thread_safe: bool = True,
        name: str = "DataBuffer",
    ):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum number of samples to store
            strategy: Sampling strategy ("cyclic", "fixed", "reservoir")
            cpu_offload: Move data to CPU to save GPU memory
            detach_tensors: Detach tensors from computation graph
            thread_safe: Use thread-safe operations (important for distributed training)
            name: Buffer name for debugging
        """
        self.max_size = max_size
        self.strategy_name = strategy
        self.cpu_offload = cpu_offload
        self.detach_tensors = detach_tensors
        self.thread_safe = thread_safe
        self.name = name

        # Pre-allocate storage for efficiency
        self._storage: List[Any] = [None] * max_size if max_size > 0 else []
        self._strategy = self._create_strategy(strategy)
        self._size = 0
        self._total_seen = 0

        # Thread safety for distributed training
        # This lock prevents race conditions when multiple GPU processes
        # access the buffer simultaneously (e.g., in distributed validation)
        self._lock = threading.RLock() if thread_safe else None

        logger.debug(
            f"Created {self.name}: max_size={max_size}, strategy={strategy}, "
            f"thread_safe={thread_safe}"
        )

    def _create_strategy(self, strategy: str) -> SamplingStrategy:
        """Create sampling strategy with validation."""
        strategies = {
            "cyclic": CyclicStrategy,
            "fixed": FixedStrategy,
            "reservoir": ReservoirStrategy,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {list(strategies.keys())}"
            )

        return strategies[strategy]()

    def _preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess tensor for storage."""
        # Detach from computation graph to prevent memory leaks
        if self.detach_tensors and tensor.requires_grad:
            tensor = tensor.detach()

        # Move to CPU if requested (saves GPU memory)
        if self.cpu_offload and tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Ensure contiguous memory layout for efficiency
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    def _preprocess_data(self, data: Any) -> Any:
        """Recursively preprocess data."""
        if isinstance(data, torch.Tensor):
            return self._preprocess_tensor(data)
        elif isinstance(data, dict):
            return {k: self._preprocess_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._preprocess_data(item) for item in data)
        else:
            return data

    def should_store(self) -> bool:
        """Determine if the current data should be stored based on the strategy."""
        return self._strategy.should_store(self._size, self._total_seen, self.max_size)

    def append(self, data: Any) -> bool:
        """
        Append data to buffer.

        Args:
            data: Data to store (tensor, dict, list, etc.)

        Returns:
            bool: True if data was stored, False if rejected by strategy or max_size=0
        """
        # Quick check for zero-size buffer
        if self.max_size == 0:
            return False

        if self._lock:
            with self._lock:
                return self._append_impl(data)
        else:
            return self._append_impl(data)

    def _append_impl(self, data: Any) -> bool:
        """Internal append implementation."""
        # Check if strategy wants to store this sample
        if not self.should_store():
            self._total_seen += 1
            return False

        # Get storage index from strategy
        storage_idx = self._strategy.get_storage_index(
            self._size, self._total_seen, self.max_size
        )
        if storage_idx is None:
            self._total_seen += 1
            return False

        # Preprocess data
        try:
            processed_data = self._preprocess_data(data)
        except Exception as e:
            logger.warning(f"Data preprocessing failed: {e}")
            self._total_seen += 1
            return False

        # Store data with explicit cleanup of old data
        if storage_idx < self._size and self._storage[storage_idx] is not None:
            # Clear old reference to prevent memory leaks
            self._storage[storage_idx] = None

        self._storage[storage_idx] = processed_data

        # Update size only if we're expanding
        if storage_idx >= self._size:
            self._size = storage_idx + 1

        self._total_seen += 1
        return True

    def get(self, index: int) -> Any:
        """
        Get item at logical index.

        Indexing behavior depends on strategy:
        - Cyclic: get(0)=oldest, get(-1)=newest
        - Fixed: get(0)=first stored, get(-1)=last stored
        - Reservoir: order not meaningful

        Args:
            index: Logical index (supports negative indexing)

        Returns:
            Stored item at the logical index
        """
        if self.max_size == 0:
            raise IndexError("Buffer has zero capacity")

        if self._lock:
            with self._lock:
                return self._get_impl(index)
        else:
            return self._get_impl(index)

    def _get_impl(self, index: int) -> Any:
        """Internal get implementation."""
        if self._size == 0:
            raise IndexError("Buffer is empty")

        # Convert logical index to physical storage index
        physical_index = self._strategy.get_logical_index(
            index, self._size, self.max_size
        )

        if physical_index < 0 or physical_index >= self._size:
            raise IndexError(
                f"Index {index} out of range for buffer size {self._size}"
            )

        return self._storage[physical_index]

    def get_all(self) -> List[Any]:
        """
        Get all stored data in logical order.

        For cyclic strategy: returns oldest to newest
        For fixed strategy: returns first stored to last stored
        For reservoir strategy: arbitrary order
        """
        if self.max_size == 0:
            return []

        if self._lock:
            with self._lock:
                return self._get_all_impl()
        else:
            return self._get_all_impl()

    def _get_all_impl(self) -> List[Any]:
        """Internal get all implementation."""
        if self._size == 0:
            return []

        result = []
        for i in range(self._size):
            try:
                physical_index = self._strategy.get_logical_index(
                    i, self._size, self.max_size
                )
                if self._storage[physical_index] is not None:
                    result.append(self._storage[physical_index])
            except IndexError:
                break

        return result

    def clear(self) -> None:
        """Clear buffer with explicit memory cleanup."""
        if self._lock:
            with self._lock:
                self._clear_impl()
        else:
            self._clear_impl()

    def _clear_impl(self) -> None:
        """Internal clear implementation."""
        # Explicitly clear all storage slots to prevent memory leaks
        for i in range(len(self._storage)):
            self._storage[i] = None

        # Reset state
        self._size = 0
        self._total_seen = 0
        self._strategy.reset()

        # clear GPU memory if offloading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

    def to_tensor(self, dim: int = 0) -> torch.Tensor:
        """
        Concatenate all tensor data efficiently.

        Args:
            dim: Dimension along which to concatenate

        Returns:
            Concatenated tensor
        """
        if self.max_size == 0:
            raise ValueError("Buffer has zero capacity")

        data = self.get_all()
        if not data:
            raise ValueError("Buffer is empty")

        if not isinstance(data[0], torch.Tensor):
            raise ValueError("Buffer does not contain tensor data")

        valid_tensors = [item for item in data if isinstance(item, torch.Tensor)]
        if not valid_tensors:
            raise ValueError("No valid tensors in buffer")

        return torch.cat(valid_tensors, dim=dim)

    def to_dict(self, dim: int = 0) -> Dict[str, torch.Tensor]:
        """
        Convert dict data to concatenated tensors by key.

        Args:
            dim: Dimension along which to concatenate

        Returns:
            Dict with concatenated tensors for each key
        """
        if self.max_size == 0:
            return {}

        data = self.get_all()
        if not data or not isinstance(data[0], dict):
            return {}

        # Collect all unique keys across all samples
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        # Concatenate tensors for each key
        result = {}
        for key in all_keys:
            tensors = []
            for item in data:
                if (
                    isinstance(item, dict)
                    and key in item
                    and item[key] is not None
                    and isinstance(item[key], torch.Tensor)
                ):
                    tensors.append(item[key])

            if tensors:
                try:
                    # This can be expensive for large dicts with many keys
                    result[key] = torch.cat(tensors, dim=dim)
                except RuntimeError as e:
                    logger.warning(
                        f"Failed to concatenate tensors for key '{key}': {e}"
                    )

        return result

    def get_storage_size(self, unit="GB") -> float:
        """Return the current size of the storage in specified unit (GB, MB, or bytes)."""
        total_bytes = 0
        for item in self._storage:
            if isinstance(item, torch.Tensor):
                total_bytes += item.element_size() * item.nelement()
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, torch.Tensor):
                        total_bytes += v.element_size() * v.nelement()
            elif hasattr(item, "__sizeof__"):
                total_bytes += item.__sizeof__()

        unit = unit.upper()
        if unit == "GB":
            return total_bytes / (1024**3)
        elif unit == "MB":
            return total_bytes / (1024**2)
        elif unit == "BYTES":
            return float(total_bytes)
        else:
            raise ValueError(f"Unknown unit '{unit}'. Use 'GB', 'MB', or 'BYTES'.")

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __repr__(self) -> str:
        return (
            f"DataBuffer(name='{self.name}', size={self._size}/{self.max_size}, "
            f"strategy={self.strategy_name})"
        )


@dataclass
class Record:
    """
    Combined storage for model records.

    Contains data that varies per batch sample and timestep:
    - guess_indices: Model predictions (batch_size, n_timesteps)
    - label_indices: Ground truth labels (batch_size, n_timesteps)
    - image_indices: Unique identifiers for input images (batch_size, n_timesteps)

    Metadata like sample indices and times indices are generated when needed.
    """

    guess_indices: torch.Tensor
    label_indices: torch.Tensor
    image_indices: torch.Tensor

    def to_cpu(self) -> "Record":
        """Move all tensors to CPU."""
        return Record(
            guess_indices=self.guess_indices.cpu(),
            label_indices=self.label_indices.cpu(),
            image_indices=self.image_indices.cpu(),
        )

    def detach(self) -> "Record":
        """Detach all tensors from computation graph."""
        return Record(
            guess_indices=self.guess_indices.detach(),
            label_indices=self.label_indices.detach(),
            image_indices=self.image_indices.detach(),
        )


class StorageBuffer:
    """
    High-level buffer management for DynVision.

    Combines response and record storage with efficient operations.
    Designed to be created in Lightning hooks and cleared when done.

    Usage:
        # In on_validation_start():
        self.storage = StorageBuffer(max_responses=1000, max_records=500)

        # In validation steps:
        self.storage.store_responses(response_dict)
        self.storage.store_records(guess_indices, label_indices, image_indices)

        # In on_validation_end():
        df = self.storage.get_dataframe()
        self.storage.clear_all()
    """

    def __init__(
        self,
        max_responses: int = 100,
        max_records: int = 100,
        response_strategy: str = "cyclic",
        record_strategy: str = "fixed",
        cpu_offload: bool = True,
        thread_safe: bool = True,
    ):
        # Response buffer for layer activations
        self.responses = DataBuffer(
            max_size=max_responses,
            strategy=response_strategy,
            cpu_offload=cpu_offload,
            thread_safe=thread_safe,
            name="ResponseBuffer",
        )

        # Record buffer for model predictions and metadata
        self.records = DataBuffer(
            max_size=max_records,
            strategy=record_strategy,
            cpu_offload=cpu_offload,
            thread_safe=thread_safe,
            name="RecordBuffer",
        )

        logger.debug(
            f"Storage buffers created: responses={max_responses}({response_strategy}), "
            f"records={max_records}({record_strategy})"
        )

    def store_responses(self, response_dict: Dict[str, torch.Tensor]) -> bool:
        """Store neural network layer responses."""
        return self.responses.append(response_dict)

    def store_records(
        self,
        guess_indices: torch.Tensor,
        label_indices: torch.Tensor,
        image_indices: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Store model records.

        Args:
            guess_indices: Model predictions (batch_size, n_timesteps)
            label_indices: Ground truth labels (batch_size, n_timesteps)
            image_indices: Unique image identifiers (batch_size, n_timesteps)
        """
        if not self.records.should_store():
            self.records._total_seen += 1
            return False

        if image_indices is None:
            batch_size, n_timesteps = label_indices.shape
            image_indices = (
                torch.arange(batch_size).unsqueeze(1).expand(-1, n_timesteps)
            )

        record = Record(
            guess_indices=guess_indices,
            label_indices=label_indices,
            image_indices=image_indices,
        )

        # Apply preprocessing
        if self.records.cpu_offload:
            record = record.to_cpu()
        if self.records.detach_tensors:
            record = record.detach()

        return self.records.append(record)

    def get_dataframe(self, layer_name: str = "classifier") -> pd.DataFrame:
        """
        Generate classifier DataFrame efficiently.

        IMPORTANT: This function requires both response and record buffers to use
        the same sampling strategy to ensure proper alignment between responses and records.

        Generates sample_indices and times_indices on-the-fly to save memory.
        """

        try:
            # Quick check for zero-capacity buffers
            if self.responses.max_size == 0 or self.records.max_size == 0:
                return pd.DataFrame()

            # Critical alignment check: ensure both buffers use same strategy
            if self.responses.strategy_name != self.records.strategy_name:
                logger.error(
                    f"Cannot combine responses (strategy='{self.responses.strategy_name}') "
                    f"with records (strategy='{self.records.strategy_name}'). "
                    f"Different strategies lead to misaligned data. Both buffers must use the same strategy."
                )
                return pd.DataFrame()

            # Get raw data from buffers
            response_data = self.responses.get_all()
            record_data = self.records.get_all()

            if not response_data or not record_data:
                logger.warning("No data stored in buffers")
                return pd.DataFrame()

            # Ensure matching lengths between responses and records
            min_length = min(len(response_data), len(record_data))
            if min_length == 0:
                return pd.DataFrame()

            # Take matching portion from both buffers to ensure alignment
            # Since both use the same strategy, index i corresponds to the same logical sample
            response_data = response_data[:min_length]
            record_data = record_data[:min_length]

            # Check if the layer exists in responses
            if layer_name not in response_data[0]:
                available_layers = (
                    list(response_data[0].keys()) if response_data else []
                )
                logger.warning(
                    f"Layer '{layer_name}' not found. Available: {available_layers}"
                )
                return pd.DataFrame()

            # Extract response tensors and pad them to the same time step length
            max_timesteps = max(item[layer_name].shape[1] for item in response_data)
            response_tensors = []
            for item in response_data:
                tensor = item[layer_name]
                pad_len = max_timesteps - tensor.shape[1]
                if pad_len > 0:
                    # Pad at the start along axis 1 (time steps) with zeros
                    # For shape [batch, timesteps, n_channels, dim_y, dim_x], pad for axis 1
                    # torch.nn.functional.pad expects (dim_x, dim_y, n_channels, timesteps)
                    # So pad = (0,0, 0,0, 0,0, pad_len,0)
                    pad = (0, 0, 0, 0, 0, 0, pad_len, 0)
                    tensor = torch.nn.functional.pad(
                        tensor, pad, mode="constant", value=0
                    )
                response_tensors.append(tensor)

            layer_responses = torch.cat(response_tensors, dim=0)

            # Extract and concatenate records
            guess_tensors = [record.guess_indices for record in record_data]
            guess_data = torch.cat(guess_tensors, dim=0)

            label_tensors = [record.label_indices for record in record_data]
            label_data = torch.cat(label_tensors, dim=0)

            image_tensors = [record.image_indices for record in record_data]
            image_data = torch.cat(image_tensors, dim=0)

            # Ensure all data has the same length (number of samples)
            valid_data_length = min(
                len(layer_responses),
                len(guess_data),
                len(label_data),
                len(image_data),
            )

            layer_responses = layer_responses[:valid_data_length]
            guess_data = guess_data[:valid_data_length]
            label_data = label_data[:valid_data_length]
            image_data = image_data[:valid_data_length]

            # Convert to CPU and numpy
            response = layer_responses.cpu().float().numpy()
            label_indices = label_data.cpu().numpy()
            guess_indices = guess_data.cpu().numpy()
            image_indices = image_data.cpu().numpy()

            # Get dimensions
            n_samples, n_timesteps, n_classes = response.shape

            # Create indices for DataFrame structure
            sample_indices, times_indices, class_indices = np.meshgrid(
                np.arange(n_samples),
                np.arange(n_timesteps),
                np.arange(n_classes),
                indexing="ij",
            )
            label_sets = np.array(["".join(row.astype(str)) for row in label_indices])
            label_sets = (
                label_sets[:, None, None]
                .repeat(n_timesteps, axis=-2)
                .repeat(n_classes, axis=-1)
            )
            label_indices = label_indices[..., None].repeat(n_classes, axis=-1)
            guess_indices = guess_indices[..., None].repeat(n_classes, axis=-1)
            image_indices = image_indices[..., None].repeat(n_classes, axis=-1)

            df = pd.DataFrame(
                {
                    "sample_index": sample_indices.ravel(),
                    "times_index": times_indices.ravel(),
                    "class_index": class_indices.ravel(),
                    "response": response.ravel(),
                    "label_index": label_indices.ravel(),
                    "guess_index": guess_indices.ravel(),
                    "image_index": image_indices.ravel(),
                    "label_set": label_sets.ravel(),
                }
            )

            # Clean up memory
            del (
                response,
                label_indices,
                guess_indices,
                image_indices,
                sample_indices,
                times_indices,
                class_indices,
            )

            return df

        except Exception as e:
            logger.error(f"Error generating classifier DataFrame: {e}")
            return pd.DataFrame()

    def clear_all(self) -> None:
        """Clear all buffers."""
        self.responses.clear()
        self.records.clear()

    def get_storage_size(self, unit="GB") -> None:
        """Get the size of the storage."""
        return self.responses.get_storage_size(unit) + self.records.get_storage_size(
            unit
        )

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        return {
            "responses_size": len(self.responses),
            "records_size": len(self.records),
            "total_items": len(self.responses) + len(self.records),
        }


class StorageBufferMixin(LightningModule):
    """
    Mixin class for automatic StorageBuffer lifecycle management in PyTorch Lightning.

    IMPORTANT: For get_dataframe() to work correctly, both response and record
    buffers must use the same sampling strategy to ensure proper data alignment.
    """

    # Default storage configurations - both use same strategy for alignment
    training_storage_config: Dict[str, Any] = {
        "max_responses": 0,  # Disabled by default
        "max_records": 0,  # Disabled by default
        "response_strategy": "fixed",  # Same strategy for alignment
        "record_strategy": "fixed",  # Same strategy for alignment
        "cpu_offload": True,
        "thread_safe": True,
    }

    validation_storage_config: Dict[str, Any] = {
        "max_responses": 1,  # Enabled by default
        "max_records": 1,  # Enabled by default
        "response_strategy": "fixed",  # Same strategy for alignment
        "record_strategy": "fixed",  # Same strategy for alignment
        "cpu_offload": True,
        "thread_safe": True,
    }

    testing_storage_config: Dict[str, Any] = {
        "max_responses": 10,  # Enabled by default for analysis
        "max_records": 10,  # Enabled by default for analysis
        "response_strategy": "fixed",  # Same strategy for alignment
        "record_strategy": "fixed",  # Same strategy for alignment
        "cpu_offload": True,
        "thread_safe": True,
    }

    def __init__(
        self,
        store_train_responses: int = 0,
        store_val_responses: int = 1,
        store_test_responses: int = 10,
        **kwargs,
    ):
        """Initialize with empty storage buffer."""
        super().__init__(**kwargs)

        self.training_storage_config["max_responses"] = store_train_responses
        self.training_storage_config["max_records"] = store_train_responses
        self.validation_storage_config["max_responses"] = store_val_responses
        self.validation_storage_config["max_records"] = store_val_responses
        self.testing_storage_config["max_responses"] = store_test_responses
        self.testing_storage_config["max_records"] = store_test_responses

        # Always create a storage instance (with zero capacity initially)
        self.storage = StorageBuffer(
            max_responses=0,
            max_records=0,
            cpu_offload=True,
            thread_safe=True,
        )

    def get_dataframe(self, **kwargs) -> pd.DataFrame:
        return self.storage.get_dataframe(**kwargs)

    def on_train_epoch_start(self) -> None:
        """Initialize storage buffer at start of training epoch."""
        try:
            super().on_train_epoch_start()
        except AttributeError:
            pass

        logger.info(f"Storage: {self.storage.get_storage_size('GB')} GB")
        # Create storage with training configuration
        self.storage.clear_all()
        self.storage = StorageBuffer(**self.training_storage_config)
        logger.debug(f"Training storage: {self.training_storage_config}")

    def on_train_epoch_end(self) -> None:
        """Clear storage buffer at end of training epoch."""
        try:
            super().on_train_epoch_end()
        except AttributeError:
            pass

        # Clear storage and reset to empty
        self.storage.clear_all()
        self.storage = StorageBuffer(max_responses=0, max_records=0)
        logger.debug("Training storage cleared")

    def on_validation_start(self) -> None:
        """Initialize storage buffer at start of validation epoch."""
        try:
            super().on_validation_start()
        except AttributeError:
            pass

        self.storage.clear_all()

    def on_validation_epoch_start(self) -> None:
        """Initialize storage buffer at start of validation epoch."""
        try:
            super().on_validation_epoch_start()
        except AttributeError:
            pass

        logger.info(f"Storage: {self.storage.get_storage_size('GB')} GB")
        self.storage.clear_all()
        # Create storage with validation configuration
        self.storage = StorageBuffer(**self.validation_storage_config)
        logger.debug(f"Validation storage: {self.validation_storage_config}")

    def on_validation_epoch_end(self) -> None:
        """Clear storage buffer at end of validation."""
        try:
            super().on_validation_epoch_end()
        except AttributeError:
            pass

        # Clear storage and reset to empty
        self.storage.clear_all()
        self.storage = StorageBuffer(max_responses=0, max_records=0)
        logger.debug("Validation storage cleared")

    def on_test_start(self) -> None:
        """Initialize storage buffer at start of testing."""
        try:
            super().on_test_start()
        except AttributeError:
            pass

        # Create storage with testing configuration
        self.storage.clear_all()
        self.storage = StorageBuffer(**self.testing_storage_config)
        logger.debug(f"Testing storage: {self.testing_storage_config}")

    def on_test_end(self) -> None:
        """Clear storage buffer at end of testing."""
        try:
            super().on_test_end()
        except AttributeError:
            pass


if __name__ == "__main__":
    print("Testing refined DataBuffer implementation...")

    # Test zero-capacity buffer behavior
    print("\n=== Testing Zero-Capacity Buffer ===")
    zero_buffer = DataBuffer(max_size=0, strategy="fixed")
    print(f"Zero buffer append result: {zero_buffer.append(torch.tensor([1]))}")
    print(f"Zero buffer get_all: {zero_buffer.get_all()}")

    storage_zero = StorageBuffer(max_responses=0, max_records=0)
    print(f"Zero storage DataFrame: {storage_zero.get_dataframe().shape}")

    # Test StorageBufferMixin
    print("\n=== Testing StorageBufferMixin ===")

    class TestModel(StorageBufferMixin):
        """Test model with mixin."""

        # Override storage configs
        training_storage_config = {
            "max_responses": 10,  # No training storage
            "max_records": 10,
            "response_strategy": "fixed",
            "record_strategy": "fixed",
        }

        validation_storage_config = {
            "max_responses": 100,  # Validation storage enabled
            "max_records": 100,
            "response_strategy": "cyclic",
            "record_strategy": "cyclic",
        }

        def __init__(self):
            super().__init__()

    # Test lifecycle
    model = TestModel()
    print(
        f"Initial storage: responses={model.storage.responses.max_size}, records={model.storage.records.max_size}"
    )

    # Start validation
    model.on_validation_epoch_start()
    print(
        f"Validation storage: responses={model.storage.responses.max_size}, records={model.storage.records.max_size}"
    )

    # Store some data
    for i in range(3):
        responses = {"classifier": torch.randn(2, 5, 4)}  # (batch, timesteps, classes)
        records_stored = model.storage.store_responses(responses)

        records_stored = model.storage.store_records(
            guess_indices=torch.randint(0, 4, (2, 5)),
            label_indices=torch.randint(0, 4, (2, 5)),
            image_indices=torch.arange(2).unsqueeze(1).expand(2, 5),
        )
        print(
            f"  Batch {i}: stored={records_stored}, storage_size=({len(model.storage.responses)}, {len(model.storage.records)})"
        )

    # Generate DataFrame
    df = model.storage.get_dataframe()
    print(f"Generated DataFrame: {df.shape}")

    # End validation
    model.on_validation_epoch_end()
    print(
        f"After clearing: responses={model.storage.responses.max_size}, records={model.storage.records.max_size}"
    )

    print("\nIntegrated implementation tests completed!")
