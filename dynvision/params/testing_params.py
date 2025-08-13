"""
Testing parameters for DynVision using Pydantic.

This module provides type-safe parameter management for model testing and evaluation,
including response storage configuration and memory-optimized settings.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from pydantic import (
    Field,
    computed_field,
    model_validator,
    ConfigDict,
    field_validator,
)

from dynvision.params.base_params import BaseParams, DynVisionValidationError
from dynvision.params.model_params import ModelParams
from dynvision.params.trainer_params import TrainerParams
from dynvision.params.data_params import DataParams

logger = logging.getLogger(__name__)


class TestingParams(BaseParams):
    """
    Composite configuration for model testing with comprehensive validation.

    Combines ModelParams, TrainerParams, and DataParams with testing-specific configuration
    for model evaluation, response storage, and result analysis.
    """

    # === CORE COMPONENT COMPOSITION ===
    model: ModelParams = Field(
        description="Model architecture and evaluation parameters"
    )
    trainer: TrainerParams = Field(
        description="Trainer configuration (minimal for testing)"
    )
    data: DataParams = Field(description="Data loading and processing parameters")

    # === TESTING-SPECIFIC PARAMETERS ===
    input_model_state: Path = Field(
        description="Path to trained model state for evaluation"
    )
    dataset: Path = Field(description="Path to test dataset")
    output_results: Path = Field(description="Path to save test results (CSV)")
    output_responses: Path = Field(
        description="Path to save model responses (PT tensors)"
    )

    # Testing behavior configuration
    verbose: bool = Field(
        default=False, description="Enable verbose logging and error reporting"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional CLI arguments
        validate_assignment=True,
        use_enum_values=True,
        validate_by_name=True,
    )

    @computed_field
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size for testing (optimized for memory)."""
        base_size = self.data.batch_size

        # Memory optimization based on model complexity
        if hasattr(self.model, "input_dims") and len(self.model.input_dims) >= 3:
            height, width = self.model.input_dims[-2:]
            if height * width > 224 * 224:  # High resolution
                base_size = min(base_size, 16)

        # Further reduce for temporal models with many timesteps
        if self.model.n_timesteps > 10:
            base_size = min(base_size, 8)

        return max(1, base_size)

    @computed_field
    @property
    def data_group(self) -> str:
        """Extract data group from data configuration."""
        return getattr(self.data, "data_group", "all")

    @computed_field
    @property
    def store_responses(self) -> str:
        """Extract store responses from model configuration."""
        return getattr(self.model, "store_responses", "-1")

    @model_validator(mode="after")
    def validate_testing_configuration(self) -> "TestingParams":
        """Comprehensive validation and configuration for testing context."""
        # Validate required paths exist
        self._validate_required_paths()

        # Validate and optimize memory usage
        self._optimize_memory_usage()

        return self

    def _validate_required_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.input_model_state.exists():
            raise DynVisionValidationError(
                f"Input model state not found: {self.input_model_state}"
            )

        if not self.dataset.exists():
            raise DynVisionValidationError(f"Test dataset not found: {self.dataset}")

        # Ensure output directories exist
        self.output_results.parent.mkdir(parents=True, exist_ok=True)
        self.output_responses.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_data_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply testing-specific optimizations to the data configuration."""
        updates = {
            "train": False,
            "shuffle": False,
            "use_distributed": False,
            "use_ffcv": False,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "max_workers": min(4, config.get("num_workers", 0)),
        }
        config = cls.update_kwargs(config, updates, verbose=True)
        return config

    @classmethod
    def validate_trainer_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply testing-specific optimizations to the trainer configuration."""
        updates = {
            "devices": 1,
            "num_nodes": 1,
            "strategy": "auto",
            "accelerator": "auto",
            "logger": None,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
        }
        config = cls.update_kwargs(config, updates, verbose=True)
        return config

    @classmethod
    def validate_model_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply testing-specific optimizations to the model configuration."""
        updates = {
            "store_responses_on_cpu": True,
        }
        config = cls.update_kwargs(config, updates, verbose=True)
        return config

    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage based on model and data characteristics."""

        # Check for memory-intensive configurations
        if self.model.store_responses > 10000:
            logger.warning(
                f"Large store_responses ({self.model.store_responses}) may cause memory issues. "
                "Consider reducing or using store_responses=0 for no response storage."
            )
        elif self.model.store_responses == -1:
            logger.warning(
                "store_responses=-1 (all responses) may cause memory issues with large datasets. "
                "Consider using a specific number or store_responses=0 for no storage."
            )

        # Optimize batch size for memory safety
        memory_safe_batch = self.effective_batch_size
        if self.data.batch_size > memory_safe_batch * 2:
            logger.warning(
                f"Reducing batch size from {self.data.batch_size} to {memory_safe_batch} "
                "for memory safety during testing"
            )
            self.data.update_field(
                "batch_size", memory_safe_batch, verbose=True, validate=False
            )

    def update_model_parameters_from_data(
        self,
        input_dims: Tuple[int, ...],
        n_classes: Optional[int] = None,
        dataset_size: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Update model parameters based on actual data characteristics.

        Args:
            input_dims: Actual input dimensions from data
            n_classes: Number of classes (if None, keeps current value)
            dataset_size: Size of test dataset for response storage optimization
            verbose: Whether to log warnings for mismatches
        """
        # Update input dimensions
        if self.model.input_dims != input_dims:
            self.model.update_field("input_dims", input_dims, verbose=verbose)

        # Update n_timesteps from input_dims
        if len(input_dims) >= 2:
            n_timesteps = input_dims[0]
            if n_timesteps > 1 and self.model.n_timesteps != n_timesteps:
                self.model.update_field("n_timesteps", n_timesteps, verbose=verbose)

        # Update n_classes if provided
        if n_classes is not None and self.model.n_classes != n_classes:
            self.model.update_field("n_classes", n_classes, verbose=verbose)

        # Optimize response storage based on dataset size
        if dataset_size is not None:
            if self.model.store_responses == -1:
                # If storing all responses, update to actual dataset size
                self.model.store_responses = dataset_size
                logger.info(
                    f"Updated store_responses from -1 (all) to dataset size: {dataset_size}"
                )
            elif (
                self.model.store_responses > dataset_size
                and self.model.store_responses > 0
            ):
                # If storing more than available, cap at dataset size
                self.model.store_responses = dataset_size
                logger.info(f"Capped store_responses to dataset size: {dataset_size}")
            elif self.model.store_responses == 0:
                logger.info("store_responses=0: No responses will be stored")

            # Further optimize batch size based on dataset size
            if self.data.batch_size > dataset_size // 4:
                suggested_batch_size = max(1, dataset_size // 8)
                logger.info(
                    f"Optimizing batch size from {self.data.batch_size} to {suggested_batch_size} "
                    f"for dataset size {dataset_size}"
                )
                self.data.update_field(
                    "batch_size", suggested_batch_size, verbose=True, validate=False
                )

    def get_model_kwargs(self, model_class=None) -> Dict[str, Any]:
        return self.model.get_model_kwargs(model_class)

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        return self.data.get_dataloader_kwargs()

    def get_trainer_kwargs(self) -> Dict[str, Any]:
        return self.trainer.get_trainer_kwargs()

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names for all components."""
        aliases = super().get_aliases()

        # Add testing-specific aliases
        aliases.update(
            {
                # Model aliases (routed to model component)
                "model_name": "model.model_name",
                "classes": "model.n_classes",
                "tsteps": "model.n_timesteps",
                "rctype": "model.recurrence_type",
                # Trainer aliases (routed to trainer component)
                "precision": "trainer.precision",
                "benchmark": "trainer.benchmark",
                "devices": "trainer.devices",
                # Data aliases (routed to data component)
                "data_name": "data.data_name",
                "dsteps": "data.data_timesteps",
                "batch_size": "data.batch_size",
                "resolution": "data.resolution",
                "data_loader": "data.data_loader",
                "data_group": "data.data_group",
                # Testing-specific aliases
                "responses": "store_responses",
                "cache": "cache_size",
            }
        )

        return aliases

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[Path] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> "TestingParams":
        """
        Create TestingParams instance from CLI and config with proper component separation.
        """
        # Get raw parameters using BaseParams method
        params = cls.get_params_from_cli_and_config(
            config_path=config_path,
            override_kwargs=override_kwargs,
            args=args,
        )
        # Separate into component configurations
        separated_params = cls._separate_component_configs(params)

        # Create the TestingParams instance
        try:
            return cls(**separated_params)
        except Exception as e:
            raise DynVisionValidationError(f"TestingParams creation failed: {e}")

    @classmethod
    def _separate_component_configs(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Separate flat parameter dict into component configurations."""
        model_params = {}
        trainer_params = {}
        data_params = {}
        base_params = {}

        # Get field names for each component
        model_fields = set(ModelParams.model_fields.keys())
        trainer_fields = set(TrainerParams.model_fields.keys())
        data_fields = set(DataParams.model_fields.keys())
        base_fields = set(cls.model_fields.keys()) - {"model", "trainer", "data"}

        for key, value in params.items():
            # Handle dotted notation (e.g., "model.learning_rate")
            if "." in key:
                component, field = key.split(".", 1)
                if component == "model":
                    model_params[field] = value
                elif component == "trainer":
                    trainer_params[field] = value
                elif component == "data":
                    data_params[field] = value
                else:
                    base_params[key] = value
            else:
                # Use mutually exclusive assignment logic
                if key in model_fields:
                    model_params[key] = value
                elif key in data_fields:
                    data_params[key] = value
                elif key in trainer_fields:
                    trainer_params[key] = value
                elif key in base_fields:
                    base_params[key] = value
                else:
                    # Unknown parameters go to both model and data params for flexibility
                    model_params[key] = value
                    data_params[key] = value
                    logger.debug(
                        f"Assigning unknown parameter '{key}' to both model_params and data_params"
                    )

        # Validate component configurations
        cls.validate_data_config(data_params)
        cls.validate_model_config(model_params)
        cls.validate_trainer_config(trainer_params)

        try:
            components = {
                "model": ModelParams(**model_params),
                "trainer": TrainerParams(**trainer_params),
                "data": DataParams(**data_params),
                **base_params,
            }
            return components
        except Exception as e:
            raise DynVisionValidationError(f"Component configuration failed: {e}")
