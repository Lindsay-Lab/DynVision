"""Dataset implementations for DynVision.

Performance Features:
- LRU caching for frequently accessed data
- Memory pinning for faster GPU transfer
- Background loading with ThreadPoolExecutor
- Efficient memory management
- GPU memory optimization

Usage:
    from dynvision.data.datasets import get_dataset

    dataset = get_dataset(
        data_path=Path('data/images'),
        data_transform='imagenet_train',
        cache_size=1000
    )
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torchvision as tv
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from dynvision.utils import filter_kwargs

from dynvision.data.transforms import (
    get_data_transform,
    get_target_transform,
)

logger = logging.getLogger(__name__)

# File extensions
TENSOR_EXTENSIONS = (".pt", ".pth")
EXTENSIONS = IMG_EXTENSIONS + TENSOR_EXTENSIONS

# Configure thread pool for background loading
MAX_WORKERS = 2
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


@lru_cache(maxsize=1000)
def load_raw_data(path: str) -> Any:
    """Load raw data with caching.

    Args:
        path: Path to data file

    Returns:
        Loaded data

    Raises:
        ValueError: If file extension not supported
    """
    if Path(path).suffix.lower() in IMG_EXTENSIONS:
        return default_loader(path)
    elif Path(path).suffix.lower() in TENSOR_EXTENSIONS:
        return torch.load(path).float()
    else:
        raise ValueError(f"Unsupported file extension: {Path(path).suffix}")


class PathFolder(datasets.DatasetFolder):
    """Optimized dataset folder implementation.

    Features:
    - Memory pinning for faster GPU transfer
    - Background loading with ThreadPoolExecutor
    - LRU caching for frequently accessed data
    - Efficient memory management
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = load_raw_data,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        cache_size: int = 1000,
        pin_memory: bool = True,
        return_path: bool = False,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Root directory
            transform: Data transform
            target_transform: Target transform
            loader: Data loader function
            is_valid_file: File validation function
            cache_size: Size of LRU cache
            pin_memory: Whether to pin memory
        """

        super().__init__(
            root=root,
            loader=loader,
            extensions=EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        self.cache_size = cache_size
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self._prefetch_indices = set()
        self.return_path = return_path

        # Initialize cache
        self._sample_cache = {}

        # Start prefetching first batch
        self._prefetch_next_batch(0, 32)

    def _prefetch_next_batch(self, start_idx: int, batch_size: int) -> None:
        """Prefetch next batch of samples with error handling."""
        end_idx = min(start_idx + batch_size, len(self))
        indices = set(range(start_idx, end_idx))
        new_indices = indices - self._prefetch_indices

        if new_indices:
            futures = []
            for idx in new_indices:
                try:
                    future = executor.submit(self._load_sample, idx)
                    futures.append((idx, future))
                except Exception as e:
                    logger.error(
                        f"Failed to submit prefetch task for index {idx}: {e}"
                    )
            self._prefetch_indices.update(new_indices)

            # Wait for prefetching to complete and handle errors
            for idx, future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error prefetching sample at index {idx}: {e}")

    def _load_sample(self, index: int) -> Tuple[torch.Tensor, Any]:
        """Load a single sample with caching.

        Args:
            index: Sample index

        Returns:
            Tuple of (sample, target)
        """
        try:
            path, target = self.samples[index]

            # Try cache first
            if path in self._sample_cache:
                sample = self._sample_cache[path]
            else:
                sample = self.loader(path)
                if len(self._sample_cache) < self.cache_size:
                    self._sample_cache[path] = sample

            if self.transform is not None:
                sample = self.transform(sample)
            else:
                sample = transforms.ToTensor()(sample)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # Pin memory if requested and GPU is available
            if (
                self.pin_memory
                and isinstance(sample, torch.Tensor)
                and sample.device.type == "cpu"
                and not sample.is_pinned()
            ):
                sample = sample.pin_memory()

            return sample, target
        except Exception as e:
            logger.error(f"Error loading sample at index {index}: {e}")
            raise

    def target_str_to_list(self, target: str) -> List[int]:
        """Convert target string to list of integers.

        Args:
            target: Target string

        Returns:
            List of integer targets
        """
        return [int(t) if t.isdigit() else -1 for t in target]

    def __getitem__(self, index: int) -> Tuple[Any, Any, str]:
        """Get a sample from the dataset.

        Args:
            index: Sample index

        Returns:
            Tuple of (sample, target, path)

        Raises:
            ValueError: If sample shape is invalid
        """
        # Trigger prefetching of next batch
        self._prefetch_next_batch(index + 1, 32)

        # Load sample
        sample, target = self._load_sample(index)
        if self.return_path:
            path = self.samples[index][0]
            return sample, target, path
        else:
            return sample, target


def get_dataset(
    data_path: Path,
    data_name: Optional[str] = None,
    # Transform interface
    transform_backend: str = "torch",
    transform_context: str = "train",
    transform_preset: Optional[str] = None,
    # Target transform interface
    target_data_name: Optional[str] = None,
    target_data_group: str = "all",
    # Other parameters
    dataset_class: Callable[..., Dataset] = PathFolder,
    normalize: Optional[Tuple[List[float], List[float]]] = None,
    dtype: Optional[torch.dtype] = None,
    pixel_range: str = "0-1",
    cache_size: int = 1000,
    pin_memory: bool = False,
    pil_to_tensor: bool = True,
    **kwargs: Any,
) -> Dataset:
    """Get dataset with optimized loading.

    Args:
        data_path: Path to dataset
        data_name: Name of the dataset
        data_transform: Data transform name or callable
        target_transform: Target transform name or callable
        dataset_class: Dataset class to use
        normalize: Normalization parameters (mean, std)
        dtype: Target dtype for tensors
        pixel_range: Pixel value range - "0-1" (scaled) or "0-255" (raw)
        cache_size: Size of LRU cache
        pin_memory: Whether to pin memory
        pil_to_tensor: Whether to convert PIL images to tensors
        **kwargs: Additional arguments

    Returns:
        Dataset instance

    Raises:
        ValueError: If data path is invalid
    """
    # set data_path to the parent directory if it is a file
    if data_path.is_file():
        if data_path.with_suffix("").exists():
            data_path = data_path.with_suffix("")
        else:
            data_path = data_path.parent

    contains_folders = any(f.is_dir() for f in data_path.iterdir())
    if not contains_folders:
        warnings.warn(
            f"{data_path} contains no category folders! Changing to parent directory."
        )
        data_path = data_path.parent

    # image transforms
    transform = []

    # Get data transforms
    additional_transforms = get_data_transform(
        backend=transform_backend,
        context=transform_context,
        dataset_or_preset=transform_preset,
    )

    if additional_transforms is not None:
        transform.extend(additional_transforms)

    # image -> tensor
    if pil_to_tensor:
        transform.append(tv.transforms.PILToTensor())

    # dtype transform with pixel range control
    if dtype is not None:
        if pixel_range == "0-255":
            # Convert dtype WITHOUT scaling (keep [0, 255] range)
            transform.append(tv.transforms.Lambda(lambda x: x.to(dtype)))
        else:
            # Default: convert and scale to [0, 1]
            transform.append(tv.transforms.ConvertImageDtype(dtype))

    # tensor transforms
    if normalize:
        transform.append(
            tv.transforms.Normalize(
                mean=torch.tensor(normalize[0]),
                std=torch.tensor(normalize[1]),
            )
        )

    logger.info(f"Transform sequence: {transform}")

    # target transforms
    target_name = target_data_name or data_name or "unknown"
    target_transform = get_target_transform(
        data_name=target_name,
        data_group=target_data_group,
    )

    if isinstance(transform, list):
        transform = transforms.Compose(transform)
    if isinstance(target_transform, list):
        target_transform = transforms.Compose(target_transform)

    dataset_kwargs = dict(
        root=data_path,
        transform=transform,
        target_transform=target_transform,
        cache_size=cache_size,
        pin_memory=pin_memory,
        **kwargs,
    )
    return dataset_class(**filter_kwargs(dataset_class, dataset_kwargs)[0])
