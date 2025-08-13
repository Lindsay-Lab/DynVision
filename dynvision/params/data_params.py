from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
    ConfigDict,
)
import logging
import torch
from ffcv.loader import OrderOption
import json
import os
from dynvision.params.base_params import BaseParams
from dynvision.utils import get_effective_dtype_from_precision

logger = logging.getLogger(__name__)


class DataParams(BaseParams):
    """
    Data loading and processing parameters with automatic optimization for large datasets.

    This class handles dataset configuration and dataloader parameters, with built-in
    optimization logic for memory-intensive datasets. Path resolution is handled
    externally by Snakemake workflows.

    Supports additional unknown arguments that will be passed to dataloader functions.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,  # Validate fields when assigned after creation
        use_enum_values=True,  # Use enum values in serialization
        validate_by_name=True,  # Allow validation using field names and aliases
        arbitrary_types_allowed=True,  # Allow PyTorch types like torch.dtype
    )

    # === Core Dataset Configuration ===
    data_name: str = Field(..., description="Name of the dataset to use")

    data_group: str = Field(
        default="all",
        description="Data group to use (e.g., 'all', 'invertebrates', specific group name)",
    )

    train: bool = Field(default=True, description="Is in training mode?")

    # === Loader Configuration ===
    use_ffcv: bool = Field(
        default=True, description="Whether to use FFCV for optimized data loading"
    )

    batch_size: int = Field(
        default=32, ge=1, description="Base batch size for data loading"
    )

    num_workers: int = Field(
        default=1,
        ge=0,
        description="Number of worker processes for data loading",
    )

    persistent_workers: bool = Field(
        default=True,
        description="Don't reload dataloader workers",
    )

    drop_last: bool = Field(
        default=True, description="Whether to drop the last incomplete batch"
    )

    # === Data Processing ===

    train_ratio: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Ratio of training data to total data",
    )

    data_transform: Union[str, List[str]] = Field(
        default=None,
        description="Transform specification (e.g., 'ffcv_train', 'ffcv_test')",
    )

    target_transform: str = Field(
        default=None, description="Target transform specification"
    )

    resolution: Optional[int] = Field(
        default=None,
        ge=1,
        description="Image resolution",
    )

    normalize: Optional[Tuple[List[float], List[float]]] = Field(
        default=None,
        description="Custom normalization (mean, std) as JSON string or tuple",
    )

    # === Temporal Parameters ===
    data_timesteps: int = Field(
        default=1, ge=1, description="number of timesteps to load"
    )

    non_input_value: int = Field(default=-3, description="null input value")

    non_label_index: int = Field(default=-1, description="label index to ignore")

    # === Advanced Loader Parameters ===
    use_distributed: bool = Field(
        default=False,
        description="Whether to use distributed data loading",
    )

    encoding: str = Field(default="image", description="FFCV encoding type")

    writer_mode: Literal["jpg", "raw", "smart", "proportion"] = Field(
        default="jpg", description="FFCV writer type"
    )

    max_resolution: int = Field(default=224, description="Max resolution for images")

    compress_probability: float = Field(
        default=1.0, description="Probability of compression applied to images"
    )

    jpeg_quality: int = Field(
        default=100, description="Quality of JPEG compression (1-100)"
    )

    chunksize: int = Field(
        default=100, description="Size of the chunks for data processing"
    )

    page_size: Optional[int] = Field(
        default=4 * 1024 * 1024, description="Size of the page for data processing"
    )

    dtype: Optional[Union[str, torch.dtype]] = Field(
        default=None,
        description="Data type for tensors - if None`, derived from precision",
    )

    precision: Optional[str] = Field(
        default="32", description="Training precision (PyTorch Lightning format)"
    )

    batches_ahead: int = Field(
        default=3, ge=1, description="Number of batches to prefetch with ffcv"
    )

    prefetch_factor: Optional[int] = Field(
        default=None, ge=1, description="Number of batches to prefetch per worker"
    )

    order: OrderOption | Literal["RANDOM", "QUASI_RANDOM", "SEQUENTIAL"] = Field(
        default=OrderOption.QUASI_RANDOM, description="Data traversal order"
    )

    os_cache: Optional[bool] = Field(
        default=None,
        description="Whether to use OS caching (None for auto-optimization)",
    )

    cache_size: Optional[int] = Field(
        default=1000,
        ge=0,
        description="Size of the cache in bytes",
    )

    pin_memory: bool = Field(
        default=False,
        description="Whether to pin memory for faster data transfer to GPU",
    )

    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the dataset",
    )

    # === Custom Dataloader Arguments ===
    dataloader_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom arguments for dataloader functions",
    )

    # === Computed Properties ===

    @computed_field
    @property
    def effective_dtype(self) -> Optional[torch.dtype]:
        """Get effective dtype without optimization (to avoid circular dependency)."""
        # Convert string dtype to torch dtype if specified
        if self.dtype is not None:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
            }
            return dtype_map.get(self.dtype, torch.float32)

        return None

    # === Aliases ===

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names."""
        aliases = super().get_aliases()
        aliases.update(
            {
                "use_distributed_mode": "use_distributed",
                "dsteps": "data_timesteps",
            }
        )
        return aliases

    # === Validators ===

    @field_validator("data_name")
    @classmethod
    def validate_data_name(cls, v: str) -> str:
        """Validate that data_name is not empty."""
        if not v or not v.strip():
            raise ValueError("data_name cannot be empty")
        return v.strip().lower()

    @field_validator("pin_memory", mode="after")
    @classmethod
    def validate_pin_memory(cls, v: bool) -> bool:
        """Ensure pin_memory is only True if CUDA is available."""
        if v and not torch.cuda.is_available():
            logger.warning(
                "pin_memory=True but CUDA is not available. Setting pin_memory=False."
            )
            return False
        return v

    @field_validator("num_workers", mode="after")
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        """validate num_worker is not larger as available"""
        max_workers = len(os.sched_getaffinity(0))
        if max_workers > v:
            logger.info(
                f"num_worker = {v} is more than available ({max_workers}). Scaling down."
            )

        return min(max_workers, v)

    @field_validator("prefetch_factor", mode="after")
    @classmethod
    def validate_prefetch_factor(cls, v, info):
        """
        Ensure prefetch_factor is only set when num_workers > 0.
        """
        # info.data contains the current field values
        num_workers = info.data.get("num_workers", 1)
        if v is not None and num_workers == 0:
            logger.warning(
                "prefetch_factor option could only be specified in multiprocessing. "
                "Set num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."
            )
            return None
        return v

    @field_validator("normalize", mode="before")
    @classmethod
    def parse_normalize(cls, v):
        """Parse JSON string or return existing tuple."""
        if v is None:
            return None

        # If string, parse as JSON
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list) and len(parsed) == 2:
                    return tuple(parsed)
                else:
                    raise ValueError("JSON must be list of two sublists")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format for normalize: {v}")

        # If already correct format
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return tuple(v)

        raise ValueError(f"Invalid normalize format: {v}")

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: Optional[str]) -> Optional[str]:
        """Validate precision matches PyTorch Lightning format."""
        if v is None:
            return "32"  # Default fallback

        # Convert to string for consistent handling
        v_str = str(v).lower()

        # Valid PyTorch Lightning precision values
        valid_precisions = {
            "16",
            "16-mixed",
            "bf16",
            "bf16-mixed",
            "bfloat16",
            "32",
            "64",
        }

        if v_str not in valid_precisions:
            raise ValueError(f"precision must be one of {valid_precisions}, got {v}")

        return v_str

    @field_validator("dtype")
    @classmethod
    def validate_dtype(
        cls, v: Optional[Union[str, torch.dtype]]
    ) -> Optional[torch.dtype]:
        """Validate dtype specification and convert to torch.dtype."""
        if v is None:
            return None  # Will be derived in model_validator

        if isinstance(v, str):
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "bfloat16": torch.bfloat16,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
            }

            v = v.replace("torch.", "")
            if v not in dtype_map:
                raise ValueError(
                    f"dtype must be one of {list(dtype_map.keys())}, got {v}"
                )

            return dtype_map[v]
        elif isinstance(v, torch.dtype):
            return v
        else:
            raise ValueError(
                f"dtype must be a string or torch.dtype, got {type(v).__name__}"
            )

    @field_validator("order")
    @classmethod
    def validate_order(cls, v: Union[OrderOption, str]) -> OrderOption:
        """Validate and convert order specification."""
        if isinstance(v, str):
            return getattr(OrderOption, v)
        return v

    @model_validator(mode="after")
    def validate_persistent_workers(self) -> "DataParams":
        if self.num_workers == 0 and self.persistent_workers:
            logger.info(
                "persistent_workers option needs num_workers > 0. "
                "Setting persistent_workers = False."
            )
            self.update_field(
                "persistent_workers", False, verbose=True, validate=False
            )
        return self

    @model_validator(mode="after")
    def validate_transforms(self) -> "DataParams":
        if self.data_transform is None:
            if self.use_ffcv:
                if self.train:
                    self.update_field(
                        "data_transform", f"ffcv_train_{self.data_name}", verbose=True
                    )
                else:
                    self.update_field(
                        "data_transform", f"ffcv_test_{self.data_name}", verbose=True
                    )
            else:
                if self.train:
                    self.update_field(
                        "data_transform", f"train_{self.data_name}", verbose=True
                    )
                else:
                    self.update_field(
                        "data_transform", f"test_{self.data_name}", verbose=True
                    )
        else:
            if self.use_ffcv:
                if not "ffcv" in self.data_transform:
                    self.update_field(
                        "data_transform", f"ffcv_{self.data_transform}", verbose=True
                    )
            else:
                if "ffcv" in self.data_transform:
                    self.update_field(
                        "data_transform",
                        self.data_transform.replace("ffcv_", ""),
                        verbose=True,
                    )
        if self.target_transform is None:
            if self.train:
                self.update_field(
                    "target_transform", f"{self.data_name}_all", verbose=True
                )
            else:
                self.update_field(
                    "target_transform",
                    f"{self.data_name}_{self.data_group}",
                    verbose=True,
                )
        return self

    @model_validator(mode="after")
    def resolve_dtype_precision_compatibility(self):
        """
        Resolve dtype/precision compatibility using shared logic with TrainerParams.
        """

        # Derive dtype if not explicitly set
        if self.dtype is None:
            derived_dtype_str = get_effective_dtype_from_precision(self.precision)

            # Convert string to torch.dtype
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "bfloat16": torch.bfloat16,
            }

            self.dtype = dtype_map[derived_dtype_str]
            logging.info(
                f"Derived data dtype '{derived_dtype_str}' from precision '{self.precision}'"
            )

        return self

    # === Loader Factory Interface ===
    def get_dataset_kwargs(self) -> Dict[str, Any]:
        """Get dataset configuration optimized for testing."""
        kwargs = self.get_dataloader_kwargs()
        kwargs.update(
            {
                "data_name": self.data_name,
                "cache_size": self.cache_size,
                "pin_memory": self.pin_memory,
            }
        )
        return kwargs

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """
        Get standardized dataloader kwargs with automatic optimization.

        This method provides a clean interface for DataLoaderFactory to get
        all necessary parameters with automatic large dataset optimizations applied.
        Includes any additional custom arguments provided.
        """

        # Add extra fields if any
        kwargs = {}
        if hasattr(self, "__pydantic_extra__"):
            kwargs.update(self.__pydantic_extra__)

        # Base dataloader arguments
        kwargs.update(
            {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "persistent_workers": self.persistent_workers,
                "encoding": self.encoding,
                "resolution": self.resolution,
                "normalize": self.normalize,
                "data_transform": self.data_transform,
                "target_transform": self.target_transform,
                "drop_last": self.drop_last,
                "dtype": self.dtype,
                "batches_ahead": self.batches_ahead,
                "order": self.order,
                "os_cache": self.os_cache,
                "data_timesteps": self.data_timesteps,
                "distributed": self.use_distributed,
                "train": self.train,
                "shuffle": self.shuffle,
                "non_label_index": self.non_label_index,
                "non_input_value": self.non_input_value,
            }
        )

        # Add custom dataloader kwargs
        kwargs.update(self.dataloader_kwargs)

        return kwargs


# === Usage Examples ===

if __name__ == "__main__":
    # Example 1: Basic usage with automatic optimization
    data_params = DataParams(
        data_name="cifar100",
        data_group="invertebrates",
        batch_size=32,
        use_ffcv=True,
        data_timesteps=20,
    )

    print(f"Resolution: {data_params.resolution}")
    print(f"Normalization: {data_params.normalize}")
    print(f"Group classes: {data_params.group_class_indices}")
    print(f"Is large dataset: {data_params.is_large_dataset}")

    # Example 3: Custom dataloader arguments
    custom_data_params = DataParams(
        data_name="cifar10",
        batch_size=64,
        # Custom arguments that will be passed to dataloader
        dataloader_kwargs={"shuffle": True, "pin_memory": True, "timeout": 30},
        # Additional unknown arguments via extra fields
        seed=42,
        use_distributed=False,
        verbose=True,
    )

    # Get all kwargs (includes custom arguments)
    all_kwargs = custom_data_params.get_dataloader_kwargs()
    print(f"All dataloader kwargs: {list(all_kwargs.keys())}")

    # Example 5: Using specific dataloader kwargs methods
    ffcv_kwargs = custom_data_params.get_ffcv_dataloader_kwargs()
    pytorch_kwargs = custom_data_params.get_pytorch_dataloader_kwargs()
    print(f"FFCV kwargs count: {len(ffcv_kwargs)}")
    print(f"PyTorch kwargs count: {len(pytorch_kwargs)}")

    # Example 8: Advanced custom arguments usage
    advanced_params = DataParams(
        data_name="tinyimagenet",
        batch_size=32,
        # Mix of explicit dataloader_kwargs and extra fields
        dataloader_kwargs={"persistent_workers": True},
        # These will be captured as extra fields
        custom_augmentation=True,
        experimental_feature="enabled",
        debug_mode=False,
    )

    print(
        f"\nAdvanced params extra fields: {getattr(advanced_params, '__pydantic_extra__', {})}"
    )
    advanced_kwargs = advanced_params.get_dataloader_kwargs()
    print(
        f"Advanced kwargs include custom args: {'custom_augmentation' in advanced_kwargs}"
    )
