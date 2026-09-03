"""
Trainer parameter handling for DynVision using Pydantic.

This module provides type-safe parameter management for PyTorch Lightning Trainer
configuration, including training behavior, system performance, distributed training,
and callback settings.
"""

from pydantic import Field, field_validator, model_validator, ConfigDict
from typing import Dict, Any, Optional, Union, ClassVar, Sequence
import logging
import pytorch_lightning as pl
from datetime import timedelta
import torch
import os
from dynvision.params.base_params import BaseParams
from dynvision.utils import (
    SummaryItem,
    get_effective_dtype_from_precision,
    log_section,
    format_value,
    resolve_signature_defaults,
)


class TrainerParams(BaseParams):
    """
    Training parameters for PyTorch Lightning trainer configuration.

    Handles PyTorch Lightning Trainer-specific parameters including training behavior,
    system performance, distributed training, callbacks, and advanced training options.

    Distributed training is automatically enabled when world_size > 1 (devices x num_nodes).
    """

    # ===== COMMON PARAMETERS =====
    seed: int = Field(description="Random seed for reproducibility")
    log_level: str = Field(description="Logging level")

    # Core Training Parameters
    epochs: int = Field(..., ge=1, description="Number of training epochs")
    check_val_every_n_epoch: int = Field(
        ..., ge=1, description="Validation frequency (epochs)"
    )
    log_every_n_steps: int = Field(..., ge=1, description="Logging frequency (steps)")
    num_sanity_val_steps: int = Field(
        ..., ge=0, description="Number of sanity steps before training"
    )

    # Training Behavior
    accumulate_grad_batches: int = Field(
        ..., ge=1, description="Number of batches to accumulate gradients"
    )
    precision: Union[int, str] = Field(
        ...,
        description="Training precision (16, 32, 64, '16', '32', '64', 'bf16', '16-mixed', 'bf16-mixed') - Lightning 2.0+",
    )
    deterministic: Union[bool, str] = Field(
        ..., description="Enable deterministic training (True, False, 'warn')"
    )

    # Distributed Training Parameters
    strategy: Optional[str] = Field(
        default=None,
        description="Distributed strategy (None=auto-detect, ddp, ddp_spawn, fsdp, deepspeed, etc.)",
    )
    devices: Union[int, str, None] = Field(
        ...,
        description="Number of devices (GPUs) per node or device specification (None=auto-detect)",
    )
    num_nodes: int = Field(..., ge=1, description="Number of compute nodes")
    accelerator: str = Field(..., description="Accelerator type (gpu, cpu, tpu, auto)")
    sync_batchnorm: Optional[bool] = Field(
        default=None,
        description="Synchronize batch normalization across devices (None=auto-set based on strategy)",
    )

    # Strategy-Specific Configuration
    strategy_kwargs: Optional[Dict[str, Any]] = Field(
        default=None, description="Strategy-specific keyword arguments"
    )

    # System Performance
    enable_progress_bar: bool = Field(..., description="Enable training progress bar")
    profiler: Optional[str] = Field(
        default=None, description="Profiler type (simple, advanced, pytorch, None)"
    )
    benchmark: bool = Field(
        ..., description="Enable PyTorch cudnn benchmark for performance"
    )

    # Advanced Training Options
    gradient_clip_val: Optional[float] = Field(
        default=None, description="Gradient clipping value (None = disabled)"
    )
    gradient_clip_algorithm: str = Field(
        ..., description="Gradient clipping algorithm (norm, value)"
    )
    limit_val_batches: float = Field(
        ..., description="Fraction of validation batches to use"
    )
    reload_dataloaders_every_n_epochs: int = Field(
        ..., description="Number of epochs to reload dataloaders"
    )

    # Early Stopping
    early_stopping_patience: Optional[int] = Field(
        default=None, description="Early stopping patience (None = disabled)"
    )
    early_stopping_min_delta: float = Field(
        ..., description="Minimum change for early stopping"
    )
    early_stopping_monitor: str = Field(
        ..., description="Metric to monitor for early stopping"
    )
    early_stopping_mode: str = Field(..., description="Early stopping mode (min, max)")

    # Checkpoint Configuration
    save_top_k: int = Field(..., description="Number of best checkpoints to save")
    monitor_checkpoint: str = Field(
        ..., description="Metric to monitor for checkpoint saving"
    )
    checkpoint_mode: str = Field(..., description="Checkpoint mode (min, max)")
    save_last: bool = Field(..., description="Save the last checkpoint")
    every_n_epochs: int = Field(..., description="Save a checkpoint every n epochs")

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        use_enum_values=True,
        validate_by_name=True,
    )

    summary_sections: ClassVar[Dict[str, Sequence[SummaryItem]]] = {
        "Schedule": (
            SummaryItem("epochs", always=True),
            SummaryItem("check_val_every_n_epoch"),
            SummaryItem("log_every_n_steps"),
            SummaryItem("accumulate_grad_batches"),
            SummaryItem("num_sanity_val_steps"),
            SummaryItem("reload_dataloaders_every_n_epochs"),
        ),
        "Precision": (
            SummaryItem("precision", always=True),
            SummaryItem("deterministic"),
            SummaryItem("benchmark"),
            SummaryItem("profiler"),
        ),
        "Distributed": (
            SummaryItem("strategy"),
            SummaryItem("devices"),
            SummaryItem("num_nodes"),
            SummaryItem("accelerator"),
            SummaryItem("sync_batchnorm"),
            SummaryItem("strategy_kwargs"),
        ),
        "Controls": (
            SummaryItem("enable_progress_bar"),
            SummaryItem("limit_val_batches"),
        ),
        "Gradient": (
            SummaryItem("gradient_clip_val"),
            SummaryItem("gradient_clip_algorithm"),
        ),
        "Checkpointing": (
            SummaryItem("monitor_checkpoint"),
            SummaryItem("save_top_k"),
            SummaryItem("save_last"),
            SummaryItem("every_n_epochs"),
        ),
        "Early stopping": (
            SummaryItem("early_stopping_monitor"),
            SummaryItem("early_stopping_mode"),
            SummaryItem("early_stopping_min_delta"),
            SummaryItem("early_stopping_patience"),
        ),
    }

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names."""
        aliases = super().get_aliases()
        aliases.update(
            {
                "ep": "epochs",
                "val_freq": "check_val_every_n_epoch",
                "log_freq": "log_every_n_steps",
                "accum": "accumulate_grad_batches",
                "prec": "precision",
                "prog": "enable_progress_bar",
                "prof": "profiler",
                "clip": "gradient_clip_val",
                "patience": "early_stopping_patience",
                "topk": "save_top_k",
                "determ": "deterministic",
                "strat": "strategy",
                "dev": "devices",
                "nodes": "num_nodes",
                "accel": "accelerator",
            }
        )
        return aliases

    def log_trainer_creation(
        self,
        *,
        trainer_kwargs: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        run_logger = logger or logging.getLogger(__name__)
        resolved_kwargs, default_flags = resolve_signature_defaults(
            pl.Trainer, trainer_kwargs
        )

        entries = []
        for name, value in resolved_kwargs.items():
            marker = "default" if default_flags.get(name, False) else None
            entries.append((name, format_value(value), marker))

        log_section(run_logger, "creating_trainer", entries)

    @property
    def world_size(self) -> int:
        """Calculate total world size (devices x num_nodes)."""
        if self.devices is None:
            return 1

        if isinstance(self.devices, int):
            devices_count = max(1, self.devices)
        elif isinstance(self.devices, str):
            if self.devices.lower() == "auto":
                devices_count = 1  # Conservative estimate
                # devices_count = torch.cuda.device_count()
            elif "," in self.devices:
                # Handle device lists like "0,1,2,3"
                devices_count = len(
                    [d.strip() for d in self.devices.split(",") if d.strip()]
                )
            else:
                devices_count = 1  # Conservative for other string specs
        else:
            devices_count = 1

        return devices_count * self.num_nodes

    @property
    def is_distributed(self) -> bool:
        """Check if distributed training is enabled based on world size."""
        return self.world_size > 1

    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Ensure log_level is valid."""
        if isinstance(v, str):
            v = v.upper()
            if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(f"Invalid log level: {v}")
        return v

    @field_validator("strategy")
    def validate_strategy(cls, v):
        """Validate distributed strategy."""
        if v is not None:
            valid_strategies = [
                "auto",
                "ddp",
                "ddp_spawn",
                "ddp_sharded",
                "ddp_sharded_spawn",
                "fsdp",
                "deepspeed",
                "ddp_find_unused_parameters_false",
                "ddp_find_unused_parameters_true",
                "horovod",
            ]
            if v not in valid_strategies:
                logging.warning(
                    f"Unknown strategy: {v}. Valid options: {valid_strategies}"
                )
        return v

    @field_validator("accelerator")
    def validate_accelerator(cls, v):
        """Validate accelerator type."""
        valid_accelerators = ["cpu", "gpu", "tpu", "auto"]
        if v not in valid_accelerators:
            raise ValueError(
                f"Invalid accelerator: {v}. Valid options: {valid_accelerators}"
            )
        return v

    @field_validator("devices")
    def validate_devices(cls, v):
        """Validate device specification."""
        if isinstance(v, int) and v <= 0:
            raise ValueError("devices must be positive when specified as integer")
        elif isinstance(v, str):
            # Allow string specifications like "0,1,2,3" or "auto"
            if v.lower() not in ["auto"] and not all(
                c.isdigit() or c in ",[]" for c in v
            ):
                raise ValueError("Invalid device string specification")
        return v

    @field_validator("precision")
    def validate_precision(cls, v):
        """Validate precision setting.

        Note: This validator must match PyTorch Lightning's accepted precision values.
        For Lightning 2.0+, the valid values are:
        - String: '64', '32', '16', 'bf16', '16-mixed', 'bf16-mixed'
        - Integer: 64, 32, 16
        """
        valid_precisions = [
            # Integer precisions
            16,
            32,
            64,
            # String precisions (Lightning 2.0+)
            "16",
            "32",
            "64",
            "bf16",
            "16-mixed",
            "bf16-mixed",
        ]
        if v not in valid_precisions:
            raise ValueError(
                f"Precision '{v}' is invalid. Allowed precision values: {tuple(valid_precisions)}"
            )
        return v

    @field_validator("deterministic")
    def validate_deterministic(cls, v):
        """Validate deterministic setting."""
        if isinstance(v, str) and v not in ["warn"]:
            raise ValueError(
                f"Invalid deterministic string: {v}. Valid options: 'warn'"
            )
        return v

    @field_validator("profiler")
    def validate_profiler(cls, v):
        """Validate profiler setting."""
        if v is not None:
            valid_profilers = ["simple", "advanced", "pytorch", "xla"]
            if v not in valid_profilers:
                raise ValueError(
                    f"Invalid profiler: {v}. Valid options: {valid_profilers}"
                )
        return v

    @field_validator("gradient_clip_algorithm")
    def validate_gradient_clip_algorithm(cls, v):
        """Validate gradient clipping algorithm."""
        valid_algorithms = ["norm", "value"]
        if v not in valid_algorithms:
            raise ValueError(
                f"Invalid gradient clip algorithm: {v}. Valid options: {valid_algorithms}"
            )
        return v

    @field_validator("early_stopping_mode", "checkpoint_mode")
    def validate_mode(cls, v):
        """Validate mode settings."""
        valid_modes = ["min", "max"]
        if v not in valid_modes:
            raise ValueError(f"Invalid mode: {v}. Valid options: {valid_modes}")
        return v

    @model_validator(mode="after")
    def ensure_dtype_precision_consistency(self):
        """
        Ensure trainer precision aligns with expected dtypes.
        """
        # Resolve the stored-parameter dtype for this precision (shared resolver).
        expected_dtype = get_effective_dtype_from_precision(self.precision)

        # Log what dtype this precision will actually use
        logging.debug(
            f"Trainer precision '{self.precision}' will use dtype '{expected_dtype}'"
        )

        # Store the effective dtype for coordination with other components
        self._effective_dtype = expected_dtype

        return self

    @model_validator(mode="after")
    def validate_distributed_consistency(self):
        """Validate distributed training configuration consistency."""
        logger = logging.getLogger(__name__)

        # Check distributed training consistency
        if self.is_distributed:
            # Distributed training is active - validate required parameters
            if self.strategy is None:
                logger.warning(
                    f"Distributed training detected (world_size={self.world_size}) "
                    f"but no strategy specified. Consider setting --strategy ddp"
                )
                # Auto-set strategy for convenience
                self.update_field(
                    "strategy",
                    "ddp",
                    verbose=True,
                    validate=False,
                    mutation_tag="derived",
                )
                logger.info("Auto-set strategy to 'ddp'")

            # Auto-set sync_batchnorm if not specified
            if self.sync_batchnorm is None:
                if self.strategy in ["ddp", "ddp_spawn", "fsdp"]:
                    self.update_field(
                        "sync_batchnorm",
                        True,
                        validate=False,
                        mutation_tag="derived",
                    )
                    logger.debug(
                        "Auto-enabled sync_batchnorm for distributed training"
                    )
                else:
                    self.update_field(
                        "sync_batchnorm",
                        False,
                        validate=False,
                        mutation_tag="derived",
                    )

            # Strategy-specific validations
            self._validate_strategy_specific_config()

        else:
            # Single device training - warn about unused distributed settings
            distributed_params_set = []
            if self.strategy is not None:
                distributed_params_set.append(f"strategy={self.strategy}")
            if self.sync_batchnorm is not None:
                distributed_params_set.append(f"sync_batchnorm={self.sync_batchnorm}")
            if self.strategy_kwargs:
                distributed_params_set.append(
                    f"strategy_kwargs={self.strategy_kwargs}"
                )

            if distributed_params_set:
                logger.warning(
                    f"Distributed parameters set but world_size={self.world_size} <= 1. "
                    f"Unused parameters: {', '.join(distributed_params_set)}"
                )

        # General training parameter validation
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")

        if self.save_top_k < -1:
            raise ValueError("save_top_k must be >= -1 (-1 saves all)")

        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive")

        self._enforce_device_limits()
        return self

    def _validate_strategy_specific_config(self):
        """Validate strategy-specific configuration."""
        if not self.strategy:
            return

        strategy = self.strategy.lower()

        if strategy.startswith("ddp"):
            self._validate_ddp_kwargs()
        elif strategy == "fsdp":
            self._validate_fsdp_kwargs()
            if self.world_size < 2:
                logging.warning(
                    "FSDP typically requires multiple devices for benefits"
                )
        elif strategy == "deepspeed":
            self._validate_deepspeed_kwargs()

    def _validate_ddp_kwargs(self):
        """Validate DDP-specific kwargs."""
        if not self.strategy_kwargs:
            return

        valid_ddp_kwargs = [
            "find_unused_parameters",
            "gradient_as_bucket_view",
            "process_group_backend",
            "bucket_cap_mb",
            "static_graph",
        ]

        for key in self.strategy_kwargs:
            if key not in valid_ddp_kwargs:
                logging.warning(
                    f"Unknown DDP kwarg: {key}. Valid options: {valid_ddp_kwargs}"
                )

        # Validate process_group_backend
        if "process_group_backend" in self.strategy_kwargs:
            backend = self.strategy_kwargs["process_group_backend"]
            valid_backends = ["nccl", "gloo", "mpi"]
            if backend not in valid_backends:
                raise ValueError(
                    f"Invalid process_group_backend: {backend}. Valid options: {valid_backends}"
                )

    def _validate_fsdp_kwargs(self):
        """Validate FSDP-specific kwargs."""
        if not self.strategy_kwargs:
            return

        valid_fsdp_kwargs = [
            "state_dict_type",
            "sharding_strategy",
            "cpu_offload",
            "mixed_precision",
            "auto_wrap_policy",
            "backward_prefetch",
            "forward_prefetch",
        ]

        for key in self.strategy_kwargs:
            if key not in valid_fsdp_kwargs:
                logging.warning(
                    f"Unknown FSDP kwarg: {key}. Valid options: {valid_fsdp_kwargs}"
                )

    def _validate_deepspeed_kwargs(self):
        """Validate DeepSpeed-specific kwargs."""
        if not self.strategy_kwargs:
            return

        valid_deepspeed_kwargs = [
            "config",
            "stage",
            "offload_optimizer",
            "offload_parameters",
            "cpu_checkpointing",
            "contiguous_gradients",
            "overlap_comm",
        ]

        for key in self.strategy_kwargs:
            if key not in valid_deepspeed_kwargs:
                logging.warning(
                    f"Unknown DeepSpeed kwarg: {key}. Valid options: {valid_deepspeed_kwargs}"
                )

    def create_strategy(self):
        """
        Create PyTorch Lightning strategy object based on configuration.

        Returns:
            PyTorch Lightning strategy instance or string for auto-detection
        """
        if not self.is_distributed or not self.strategy:
            return "auto"  # Let Lightning auto-detect

        try:
            import pytorch_lightning as pl
        except ImportError:
            raise ImportError("PyTorch Lightning is required for strategy creation")

        strategy_name = self.strategy.lower()
        strategy_kwargs = (self.strategy_kwargs or {}).copy()

        if strategy_name in ["ddp", "ddp_spawn"]:
            if strategy_name == "ddp":
                return pl.strategies.DDPStrategy(
                    timeout=timedelta(minutes=10), **strategy_kwargs
                )
            else:
                return pl.strategies.DDPSpawnStrategy(**strategy_kwargs)

        elif strategy_name == "fsdp":
            return pl.strategies.FSDPStrategy(**strategy_kwargs)

        elif strategy_name == "deepspeed":
            return pl.strategies.DeepSpeedStrategy(**strategy_kwargs)

        elif strategy_name in ["ddp_sharded", "ddp_sharded_spawn"]:
            # For FairScale sharded strategies (if available)
            try:
                if strategy_name == "ddp_sharded":
                    return pl.strategies.DDPShardedStrategy(**strategy_kwargs)
                else:
                    return pl.strategies.DDPSpawnShardedStrategy(**strategy_kwargs)
            except AttributeError:
                logging.warning(
                    f"Strategy {strategy_name} not available, falling back to DDP"
                )
                return pl.strategies.DDPStrategy(**strategy_kwargs)

        else:
            # For string-based strategies or unknown strategies
            logging.warning(f"Using string-based strategy: {self.strategy}")
            return self.strategy

    def get_trainer_kwargs(self) -> Dict[str, Any]:
        """
        Convert trainer parameters to PyTorch Lightning Trainer kwargs.

        Returns:
            Dictionary of kwargs for pl.Trainer initialization
        """
        trainer_kwargs = {
            "max_epochs": self.epochs,
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
            "log_every_n_steps": self.log_every_n_steps,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "num_sanity_val_steps": self.num_sanity_val_steps,
            "precision": self.precision,
            "enable_progress_bar": self.enable_progress_bar,
            "profiler": self.profiler,
            "benchmark": self.benchmark,
            "deterministic": self.deterministic,
            "limit_val_batches": self.limit_val_batches,
            "reload_dataloaders_every_n_epochs": self.reload_dataloaders_every_n_epochs,
        }

        # Add gradient clipping if specified
        if self.gradient_clip_val is not None:
            trainer_kwargs["gradient_clip_val"] = self.gradient_clip_val
            trainer_kwargs["gradient_clip_algorithm"] = self.gradient_clip_algorithm

        # Add distributed training configuration
        if self.is_distributed:
            trainer_kwargs.update(
                {
                    "strategy": self.create_strategy(),
                    "devices": self.devices,
                    "num_nodes": self.num_nodes,
                    "accelerator": self.accelerator,
                    "sync_batchnorm": self.sync_batchnorm,
                }
            )
        else:
            # For single device training
            if self.accelerator == "auto":
                # Auto-detect: use GPU if available, otherwise let Lightning default to CPU
                if self._detect_available_gpu_count() > 0:
                    trainer_kwargs["accelerator"] = "gpu"
                    # Use configured devices or default to 1 for single-device training
                    trainer_kwargs["devices"] = (
                        self.devices if self.devices is not None else 1
                    )
                # If no GPU available, let Lightning default to CPU (don't set accelerator)
            else:
                # User explicitly specified accelerator
                trainer_kwargs["accelerator"] = self.accelerator
                if self.devices is not None:
                    trainer_kwargs["devices"] = self.devices

        # Filter out None values to allow Trainer class defaults
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

        return trainer_kwargs

    @staticmethod
    def _detect_available_gpu_count() -> int:
        """Return the number of GPUs visible to the current process."""

        if not torch.cuda.is_available():
            return 0

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
            if tokens:
                return len(tokens)

        try:
            return torch.cuda.device_count()
        except Exception:  # pragma: no cover - defensive guard
            return 0

    def _enforce_device_limits(self) -> None:
        """Clamp requested devices so Lightning never asks for more GPUs than available."""

        if self.devices is None:
            return

        # Only enforce when we're aiming for GPU acceleration (explicitly or implicitly)
        if str(self.accelerator).lower() not in {"gpu", "auto"}:
            return

        available = self._detect_available_gpu_count()
        if available <= 0:
            return

        logger = logging.getLogger(__name__)

        if isinstance(self.devices, int):
            if self.devices > available:
                logger.warning(
                    "Requested %s GPUs but only %s visible; clamping to hardware limits",
                    self.devices,
                    available,
                )
                self.update_field(
                    "devices",
                    available,
                    verbose=True,
                    validate=False,
                    mutation_tag="derived",
                )
        elif isinstance(self.devices, str) and self.devices.lower() not in {"auto"}:
            requested = [tok.strip() for tok in self.devices.split(",") if tok.strip()]
            if len(requested) > available:
                logger.warning(
                    "Requested GPU list '%s' exceeds %s visible devices; trimming to first %s entries",
                    self.devices,
                    available,
                    available,
                )
                trimmed = ",".join(requested[:available])
                self.update_field(
                    "devices",
                    trimmed,
                    verbose=True,
                    validate=False,
                    mutation_tag="derived",
                )

    def get_effective_dtype(self) -> torch.dtype:
        """Get the actual torch.dtype that this trainer configuration will use."""
        if hasattr(self, "_effective_dtype"):
            return self._effective_dtype
        else:
            # Fallback to derivation
            return get_effective_dtype_from_precision(self.precision)

    def get_early_stopping_callback_kwargs(self) -> Optional[Dict[str, Any]]:
        """Get early stopping callback configuration."""
        if self.early_stopping_patience is None:
            return None

        kwargs = {
            "monitor": self.early_stopping_monitor,
            "patience": self.early_stopping_patience,
            "min_delta": self.early_stopping_min_delta,
            "mode": self.early_stopping_mode,
            "verbose": True,
        }
        # Filter out None values to allow callback class defaults
        return {k: v for k, v in kwargs.items() if v is not None}

    def get_checkpoint_callback_kwargs(self) -> Dict[str, Any]:
        """Get model checkpoint callback configuration."""
        kwargs = {
            "monitor": self.monitor_checkpoint,
            "save_top_k": self.save_top_k,
            "mode": self.checkpoint_mode,
            "save_last": self.save_last,
            "auto_insert_metric_name": False,
            "save_on_train_epoch_end": True,
            "every_n_epochs": self.every_n_epochs,
        }
        # Filter out None values to allow callback class defaults
        return {k: v for k, v in kwargs.items() if v is not None}


# Example usage and testing
if __name__ == "__main__":
    # Test basic single-device training
    single_device = TrainerParams()
    print(
        f"Single device config: world_size={single_device.world_size}, is_distributed={single_device.is_distributed}"
    )

    # Test distributed training
    distributed = TrainerParams(devices=4, num_nodes=2, strategy="ddp")
    print(
        f"Distributed config: world_size={distributed.world_size}, is_distributed={distributed.is_distributed}"
    )

    # Test CLI parsing
    test_args = [
        "--devices",
        "4",
        "--strategy",
        "fsdp",
        "--precision",
        "bf16-mixed",
        "--epochs",
        "200",
    ]

    try:
        cli_params = TrainerParams.from_cli_and_config(args=test_args)
        print(
            f"CLI params: world_size={cli_params.world_size}, strategy={cli_params.strategy}"
        )

        # Test PyTorch Lightning integration
        trainer_kwargs = cli_params.get_trainer_kwargs()
        print(f"Trainer kwargs: {trainer_kwargs}")

    except Exception as e:
        print(f"Error: {e}")
