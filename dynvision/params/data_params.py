from typing import Dict, List, Optional, Tuple, Union, Any, Literal, Sequence, ClassVar
from pathlib import Path
from collections import OrderedDict
from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
    ConfigDict,
)
import logging
import torch
import json
import os
from dynvision.params.base_params import BaseParams
from dynvision.utils import (
    SummaryItem,
    log_section,
    format_value,
    resolve_signature_defaults,
)

# OrderOption is re-exported (with a non-ffcv stand-in) from the BatchSource
# seam so DataParams never has to import ffcv directly and there's a single
# definition of the stand-in enum. See dynvision.data.batch_source and
# docs/development/planning/ffcv-batch-source-seam.md (issue #16).
from dynvision.data.batch_source import OrderOption

logger = logging.getLogger(__name__)


class DataParams(BaseParams):
    """
    Data loading and processing parameters with automatic optimization for large datasets.

    This class handles dataset configuration and dataloader parameters, with built-in
    optimization logic for memory-intensive datasets. Path resolution is handled
    externally by Snakemake workflows.

    Supports additional unknown arguments that will be passed to dataloader functions.
    """

    # ===== COMMON PARAMETERS =====
    seed: int = Field(description="Random seed for reproducibility")
    log_level: str = Field(description="Logging level")

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,  # Validate fields when assigned after creation
        use_enum_values=True,  # Use enum values in serialization
        validate_by_name=True,  # Allow validation using field names and aliases
        arbitrary_types_allowed=True,  # Allow PyTorch types like torch.dtype
    )

    summary_sections: ClassVar[Dict[str, Sequence[SummaryItem]]] = {
        "Dataset": (
            SummaryItem("data_name", always=True),
            SummaryItem("data_group", always=True),
            SummaryItem("train", always=True),
            SummaryItem("data_loader", always=True),
            SummaryItem("use_ffcv"),
            SummaryItem("sampler"),
        ),
        "Batch": (
            SummaryItem("batch_size", always=True),
            SummaryItem("num_workers", always=True),
            SummaryItem("persistent_workers"),
            SummaryItem("shuffle"),
            SummaryItem("drop_last"),
        ),
        "Processing": (
            SummaryItem("data_timesteps", always=True),
            SummaryItem("resolution"),
            SummaryItem("transform_backend", always=True),
            SummaryItem("transform_context", always=True),
            SummaryItem("transform_preset"),
            SummaryItem("target_data_name"),
            SummaryItem("target_data_group"),
            SummaryItem("normalize"),
            SummaryItem("pixel_range", always=True),
        ),
        "Precision": (
            SummaryItem("dtype"),
            SummaryItem("use_distributed"),
            SummaryItem("pin_memory"),
            SummaryItem("prefetch_factor"),
            SummaryItem("cache_size"),
        ),
    }

    # === Core Dataset Configuration ===
    data_name: str = Field(..., description="Name of the dataset to use")
    data_group: str = Field(
        ...,
        description="Data group to use (e.g., 'all', 'invertebrates', specific group name)",
    )
    train: bool = Field(..., description="Is in training mode?")

    # === Loader Configuration ===
    use_ffcv: bool = Field(
        ..., description="Whether to use FFCV for optimized data loading"
    )
    batch_size: int = Field(..., ge=1, description="Base batch size for data loading")
    num_workers: int = Field(
        ..., ge=0, description="Number of worker processes for data loading"
    )
    persistent_workers: bool = Field(
        ..., description="Don't reload dataloader workers"
    )
    drop_last: bool = Field(
        ..., description="Whether to drop the last incomplete batch"
    )

    # === Data Processing ===
    train_ratio: float = Field(
        ..., ge=0, le=1, description="Ratio of training data to total data"
    )
    # Transform backend/context/preset
    transform_backend: Optional[Literal["torch", "ffcv"]] = Field(
        default=None,
        description="Transform backend - derived from use_ffcv if not specified",
    )
    transform_context: Optional[Literal["train", "test"]] = Field(
        default=None,
        description="Transform context - derived from train if not specified",
    )
    transform_preset: Optional[str] = Field(
        default=None,
        description="Transform preset name (e.g., 'imagenette', 'mnist', 'base') - derived from data_name if not specified",
    )
    # Target transform parameters
    target_data_name: Optional[str] = Field(
        default=None,
        description="Dataset name for target transforms - derived from data_name if not specified",
    )
    target_data_group: Optional[str] = Field(
        default=None,
        description="Data group for target transforms - derived from data_group/train if not specified",
    )
    resolution: Optional[int] = Field(
        default=None, ge=1, description="Image resolution"
    )
    normalize: Optional[Tuple[List[float], List[float]]] = Field(
        default=None,
        description="Custom normalization (mean, std) as JSON string or tuple",
    )
    pixel_range: Literal["0-1", "0-255"] = Field(
        ..., description="Pixel value range: '0-1' (normalized) or '0-255' (raw)"
    )

    # === Temporal Parameters ===
    data_timesteps: int = Field(..., ge=1, description="number of timesteps to load")
    non_input_value: int = Field(..., description="null input value")
    non_label_index: int = Field(..., description="label index to ignore")

    # === Advanced Loader Parameters ===
    use_distributed: bool = Field(
        ..., description="Whether to use distributed data loading"
    )
    encoding: str = Field(..., description="FFCV encoding type")
    writer_mode: Literal["jpg", "raw", "smart", "proportion"] = Field(
        ..., description="FFCV writer type"
    )
    max_resolution: int = Field(..., description="Max resolution for images")
    compress_probability: float = Field(
        ..., description="Probability of compression applied to images"
    )
    jpeg_quality: int = Field(..., description="Quality of JPEG compression (1-100)")
    chunksize: int = Field(..., description="Size of the chunks for data processing")
    page_size: int = Field(..., description="Size of the page for data processing")
    dtype: Optional[Union[str, torch.dtype]] = Field(
        default=None,
        description="Data type for tensors - if None, derived from trainer precision",
    )
    batches_ahead: int = Field(
        ..., ge=1, description="Number of batches to prefetch with ffcv"
    )
    prefetch_factor: Optional[int] = Field(
        default=None, ge=1, description="Number of batches to prefetch per worker"
    )
    order: OrderOption | Literal["RANDOM", "QUASI_RANDOM", "SEQUENTIAL"] = Field(
        ..., description="Data traversal order"
    )
    os_cache: Optional[bool] = Field(
        default=None,
        description="Whether to use OS caching (None for auto-optimization)",
    )
    cache_size: Optional[int] = Field(
        default=None, ge=0, description="Size of the cache in bytes"
    )
    pin_memory: bool = Field(
        ..., description="Whether to pin memory for faster data transfer to GPU"
    )
    shuffle: bool = Field(..., description="Whether to shuffle the dataset")
    sampler: Optional[str] = Field(
        default=None, description="Name of custom sampler for data loading"
    )

    # === Custom Dataloader Arguments ===
    dataloader_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional custom arguments for dataloader functions",
    )

    # === Computed Properties ===

    @computed_field
    @property
    def effective_dtype(self) -> Optional[torch.dtype]:
        """Get effective dtype without optimization (to avoid circular dependency)."""
        # validate_dtype already normalizes self.dtype to a torch.dtype (or
        # None), so no further string-keyed lookup is needed here.
        return self.dtype

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

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log_level is valid."""
        if isinstance(v, str):
            v = v.upper()
            if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(f"Invalid log level: {v}")
        return v

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
        """Parse JSON string or return existing tuple.

        Accepts:
        - None or "null" (JSON): No normalization
        - JSON string with 2-element list: [[mean], [std]]
        - Tuple/list with 2 elements: (mean, std)
        """
        if v is None:
            return None

        # If string, parse as JSON
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                # Handle JSON null explicitly
                if parsed is None:
                    return None
                # Handle list format
                if isinstance(parsed, list) and len(parsed) == 2:
                    return tuple(parsed)
                else:
                    raise ValueError("JSON must be null or a list of two sublists")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format for normalize: {v}")

        # If already correct format
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return tuple(v)

        raise ValueError(f"Invalid normalize format: {v}")

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
            logger.debug(
                "persistent_workers requires num_workers > 0; disabling option"
            )
            self.update_field(
                "persistent_workers",
                False,
                verbose=False,
                validate=False,
                mutation_tag="derived",
            )
        return self

    @model_validator(mode="after")
    def validate_transforms(self) -> "DataParams":
        """Derive transform parameters from other configuration fields.

        Derives:
        - transform_backend from use_ffcv
        - transform_context from train
        - transform_preset from data_name
        - target_data_name from data_name
        - target_data_group based on train mode (all for training, specific group for testing)
        """

        def _derive(field: str, value: Any) -> None:
            self.update_field(
                field,
                value,
                verbose=False,
                mutation_tag="derived",
            )
            logger.debug("Derived %s=%s", field, value)

        # Derive transform_backend if not set
        if self.transform_backend is None:
            backend = "ffcv" if self.use_ffcv else "torch"
            _derive("transform_backend", backend)

        # Derive transform_context if not set
        if self.transform_context is None:
            context = "train" if self.train else "test"
            _derive("transform_context", context)

        # Derive transform_preset if not set
        if self.transform_preset is None:
            # Use data_name as the default preset
            _derive("transform_preset", self.data_name)

        # Derive target_data_name if not set
        if self.target_data_name is None:
            _derive("target_data_name", self.data_name)

        # Derive target_data_group if not set
        if self.target_data_group is None:
            # Use "all" for training, specific group for testing
            target_group = "all" if self.train else self.data_group
            _derive("target_data_group", target_group)

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
        # Filter out None values to allow dataset class defaults
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return kwargs

    def get_dataloader_kwargs(self, dataloader_class=None) -> Dict[str, Any]:
        """
        Get standardized dataloader kwargs with automatic optimization.

        This method provides a clean interface for DataLoaderFactory to get
        all necessary parameters with automatic large dataset optimizations applied.
        Includes any additional custom arguments provided.

        Args:
            dataloader_class: Optional dataloader class to filter kwargs against.
                If provided, only kwargs accepted by that class will be returned.
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
                "sampler": self.sampler,
                "persistent_workers": self.persistent_workers,
                "encoding": self.encoding,
                "resolution": self.resolution,
                "normalize": self.normalize,
                "pixel_range": self.pixel_range,
                # Transform interface
                "transform_backend": self.transform_backend,
                "transform_context": self.transform_context,
                "transform_preset": self.transform_preset,
                # Target transform interface
                "target_data_name": self.target_data_name,
                "target_data_group": self.target_data_group,
                # Other parameters
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
        if self.dataloader_kwargs:
            kwargs.update(self.dataloader_kwargs)

        # Guard against user-injected prefetch_factor when workers are disabled
        if kwargs.get("num_workers", 0) == 0 and "prefetch_factor" in kwargs:
            removed_value = kwargs.pop("prefetch_factor")
            logging.info(
                "Dropping prefetch_factor=%s because num_workers=0 (prefetch requires multiprocessing)",
                removed_value,
            )

        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Filter kwargs if dataloader_class is provided
        if dataloader_class is not None:
            try:
                from dynvision.utils import filter_kwargs

                known, unknown = filter_kwargs(dataloader_class, kwargs)

                if known:
                    logging.debug(
                        "Filtered dataloader kwargs for %s: %s",
                        dataloader_class.__name__,
                        list(known.keys()),
                    )

                return known

            except ImportError:
                logging.warning("filter_kwargs not available, returning all kwargs")
                return kwargs

        return kwargs

    def get_preview_dataloader_kwargs(self) -> Dict[str, Any]:
        """Return dataloader kwargs specialized for lightweight previews."""

        preview_kwargs = self.get_dataloader_kwargs().copy()
        preview_kwargs["distributed"] = False
        preview_kwargs["shuffle"] = False
        preview_kwargs["train"] = False

        num_workers = preview_kwargs.get("num_workers", 4)
        preview_kwargs["num_workers"] = min(num_workers, 1)

        if self.use_ffcv:
            preview_kwargs["train"] = False
            preview_kwargs["distributed"] = False

        return preview_kwargs

    def get_validation_dataloader_kwargs(
        self,
        base_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Derive validation dataloader kwargs without mutating runtime state."""

        kwargs = dict(base_kwargs or self.get_dataloader_kwargs())

        kwargs["train"] = False
        kwargs.setdefault("shuffle", False)

        if self.use_ffcv:
            num_workers = kwargs.get("num_workers", 0)
            kwargs["num_workers"] = max(num_workers // 4, 1) if num_workers else 1

            batch_size = kwargs.get("batch_size")
            if batch_size is not None:
                kwargs["batch_size"] = max(batch_size // 4, 32)

        return kwargs

    def _build_callable_entries(
        self,
        callable_obj: Any,
        provided_kwargs: Dict[str, Any],
        *,
        previous_kwargs: Optional[Dict[str, Any]] = None,
        suppress_defaults: bool = False,
        always_show: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, str, Optional[str]]]:
        always_show = set(always_show or ())
        try:
            resolved_kwargs, default_flags = resolve_signature_defaults(
                callable_obj, provided_kwargs
            )
        except (ValueError, TypeError) as exc:
            logging.debug(
                "Falling back to provided kwargs for %s due to %s",
                callable_obj,
                exc,
            )
            resolved_kwargs = provided_kwargs
            default_flags = {}

        previous_resolved: Optional["OrderedDict[str, Any]"] = None
        if previous_kwargs is not None:
            try:
                previous_resolved, _ = resolve_signature_defaults(
                    callable_obj, previous_kwargs
                )
            except (ValueError, TypeError) as exc:
                logging.debug(
                    "Falling back to provided previous kwargs for %s due to %s",
                    callable_obj,
                    exc,
                )
                previous_resolved = previous_kwargs

        entries: List[Tuple[str, str, Optional[str]]] = []
        for name, value in resolved_kwargs.items():
            marker_parts: List[str] = []
            is_default_value = default_flags.get(name, False)
            if (
                suppress_defaults
                and previous_resolved is None
                and is_default_value
                and name not in always_show
            ):
                continue

            if is_default_value:
                marker_parts.append("default")

            previous_value = None
            if previous_resolved is not None and name in previous_resolved:
                previous_value = previous_resolved[name]
                if previous_value == value:
                    if not marker_parts or marker_parts == ["default"]:
                        # Hide unchanged entries when diffing (even if default repeated)
                        continue
                else:
                    marker_parts.append("changed")
            elif previous_resolved is not None:
                marker_parts.append("new")

            formatted_value = format_value(value)
            if (
                previous_value is not None
                and previous_resolved is not None
                and previous_value != value
            ):
                formatted_value = (
                    f"{formatted_value} (was {format_value(previous_value)})"
                )

            marker = ", ".join(marker_parts) if marker_parts else None
            entries.append((name, formatted_value, marker))

        if previous_resolved is not None:
            for name, value in previous_resolved.items():
                if name not in resolved_kwargs:
                    entries.append((name, format_value(value), "removed"))

        return entries

    def log_dataloader_creation(
        self,
        *,
        dataloader_class: Any,
        dataloader_kwargs: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        context: str = "active",
        previous_kwargs: Optional[Dict[str, Any]] = None,
        level: int = logging.INFO,
    ) -> None:
        """Log dataloader kwargs including defaults and optional diffs."""

        run_logger = logger or logging.getLogger(__name__)

        entries = self._build_callable_entries(
            dataloader_class,
            dataloader_kwargs,
            previous_kwargs=previous_kwargs,
            suppress_defaults=True,
            always_show=("path",),
        )
        title = (
            f"creating_{getattr(dataloader_class, '__name__', 'dataloader').lower()}"
        )
        if context and context != "active":
            title = f"{title} ({context})"
        log_section(run_logger, title, entries, level=level)

    def log_dataset_creation(
        self,
        *,
        dataset_path: Path,
        dataset_kwargs: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        context: str = "active",
        previous_kwargs: Optional[Dict[str, Any]] = None,
        previous_dataset_path: Optional[Path] = None,
        level: int = logging.INFO,
    ) -> None:
        """Log dataset construction arguments with default and diff markers."""

        from dynvision.data.datasets import get_dataset

        run_logger = logger or logging.getLogger(__name__)
        call_kwargs = {"data_path": dataset_path}
        call_kwargs.update(dataset_kwargs)

        previous_call_kwargs = None
        if previous_kwargs is not None:
            previous_call_kwargs = {
                "data_path": previous_dataset_path or dataset_path,
                **previous_kwargs,
            }

        entries = self._build_callable_entries(
            get_dataset,
            call_kwargs,
            previous_kwargs=previous_call_kwargs,
            suppress_defaults=True,
            always_show=("data_path", "data_name"),
        )
        title = "creating_dataset"
        if context and context != "active":
            title = f"{title} ({context})"
        log_section(run_logger, title, entries, level=level)

    def log_configuration(
        self, dataloader: Optional[Any] = None, dataloader_name: str = "DataLoader"
    ) -> None:
        """Log dataloader configuration parameters in a concise, structured format."""

        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.log_summary(
            logger=logger,
            title=f"{dataloader_name} configuration",
            include_defaults=False,
        )

        if dataloader is None:
            return

        comparisons = (
            ("batch_size", getattr(dataloader, "batch_size", None)),
            ("num_workers", getattr(dataloader, "num_workers", None)),
            ("shuffle", getattr(dataloader, "shuffle", None)),
            ("drop_last", getattr(dataloader, "drop_last", None)),
            ("pin_memory", getattr(dataloader, "pin_memory", None)),
            ("persistent_workers", getattr(dataloader, "persistent_workers", None)),
            ("prefetch_factor", getattr(dataloader, "prefetch_factor", None)),
        )

        mismatches = []
        for field, actual in comparisons:
            expected = getattr(self, field, None)
            if expected is not None and actual is not None and expected != actual:
                mismatches.append((field, expected, actual))

        if mismatches:
            entries = [
                (
                    field.replace("_", " ").capitalize(),
                    f"{format_value(actual)} (expected {format_value(expected)})",
                    "adjusted",
                )
                for field, expected, actual in mismatches
            ]
            log_section(
                logger,
                "Configuration differences",
                entries,
                level=logging.WARNING,
            )

        if logger.isEnabledFor(logging.DEBUG):
            extras = []
            if self.dataloader_kwargs:
                for key, value in self.dataloader_kwargs.items():
                    extras.append((key, format_value(value), None))
            extra_attrs = getattr(self, "__pydantic_extra__", {}) or {}
            for key, value in extra_attrs.items():
                extras.append((key, format_value(value), None))

            if extras:
                log_section(
                    logger,
                    "Custom dataloader parameters",
                    extras,
                    level=logging.DEBUG,
                )


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
