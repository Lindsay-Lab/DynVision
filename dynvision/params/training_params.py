import logging
import yaml
import math
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch

# Import the Pydantic parameter classes
from dynvision.params import (
    BaseParams,
    DynVisionValidationError,
    DynVisionConfigError,
    ModelParams,
    TrainerParams,
    DataParams,
)

from pydantic import (
    Field,
    computed_field,
    model_validator,
    ConfigDict,
    field_validator,
)

logger = logging.getLogger(__name__)

# Configuration constants
REFERENCE_BATCH_SIZE = 64


class TrainingParams(BaseParams):
    """
    Composite configuration for model training with comprehensive validation.

    Combines ModelParams, TrainerParams, and DataParams with advanced computed
    properties for training optimization, consistency checking, and automatic
    parameter scaling based on effective batch size and distributed setup.
    """

    # === CORE COMPONENT COMPOSITION ===
    model: ModelParams = Field(
        description="Model architecture and training parameters"
    )
    trainer: TrainerParams = Field(
        description="Training behavior and system configuration"
    )
    data: DataParams = Field(description="Data loading and processing parameters")

    # === SCRIPT-SPECIFIC PARAMETERS ===
    input_model_state: Path = Field(description="Path to initial model state")
    output_model_state: Path = Field(description="Path to save trained model")
    dataset_link: Optional[Path] = Field(
        default=None, description="Path to training dataset"
    )
    dataset_train: Optional[Path] = Field(
        default=None, description="Path to training ffcv dataset"
    )
    dataset_val: Optional[Path] = Field(
        default=None, description="Path to validation ffcv dataset"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional CLI arguments
        validate_assignment=True,
        use_enum_values=True,
        validate_by_name=True,
    )

    def update_model_parameters_from_data(
        self,
        input_dims: Tuple[int, ...],
        n_classes: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Update model parameters based on actual data characteristics.

        Args:
            input_dims: Actual input dimensions from data (n_timesteps, channels, height, width)
            n_classes: Number of classes (if None, keeps current value)
            batch_size: Actual batch size from dataloader (if None, keeps current value)
            verbose: Whether to log warnings for mismatches
        """

        # Update input dimensions
        if self.model.input_dims != input_dims:
            self.model.update_field("input_dims", input_dims, verbose=verbose)

        # Update n_timesteps from input_dims
        n_timesteps = input_dims[0]
        if n_timesteps > 1 and self.model.n_timesteps != n_timesteps:
            self.model.update_field("n_timesteps", n_timesteps, verbose=verbose)

        # Update n_classes if provided
        if n_classes is not None and self.model.n_classes != n_classes:
            self.model.update_field("n_classes", n_classes, verbose=verbose)

        # Update batch size if provided
        # if batch_size is not None and self.data.batch_size != batch_size:
        #     self.data.update_field("batch_size", batch_size, verbose=verbose)

    # === COMPUTED PROPERTIES ===

    # @computed_field
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size considering gradient accumulation."""
        return self.global_batch_size * self.trainer.accumulate_grad_batches

    # @computed_field
    @property
    def local_batch_size(self) -> int:
        """Calculate local batch size per gpu."""
        return self.data.batch_size // self.trainer.world_size

    # @computed_field
    @property
    def global_batch_size(self) -> int:
        """Calculate global batch size across all devices."""
        return self.data.batch_size

    # @computed_field
    @property
    def effective_learning_rate(self) -> float:
        """Scale learning rate based on effective batch size."""
        base_lr = self.model.learning_rate

        # Calculate effective batch size without triggering recursion
        global_batch_size = self.data.batch_size
        effective_batch_size = global_batch_size * self.trainer.accumulate_grad_batches

        # Linear scaling for small batches, square root scaling for large batches
        if effective_batch_size <= 128:
            scaling_factor = effective_batch_size / REFERENCE_BATCH_SIZE
        else:
            scaling_factor = math.sqrt(effective_batch_size / REFERENCE_BATCH_SIZE)

        return base_lr * scaling_factor

    # === VALIDATION METHODS ===

    @field_validator("dataset_link", "dataset_train", "dataset_val", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v is None or v == "":
            return None
        return v

    @model_validator(mode="after")
    def validate_training_configuration(self) -> "TrainingParams":
        """Comprehensive validation for training context."""
        # Validate required paths exist
        self._validate_required_paths()

        # Resolve parameter consistency issues
        self._resolve_parameter_conflicts()
        return self

    @model_validator(mode="after")
    def coordinate_component_dtypes(self):
        """
        Ensure all components (trainer, data, model) use consistent dtypes.
        """
        # Get the effective dtype from trainer precision
        trainer_dtype = self.trainer.get_effective_dtype()

        # Ensure data uses the same dtype
        if self.data.dtype != trainer_dtype:
            logging.warning(
                f"Data dtype ({self.data.dtype}) differs from trainer dtype ({trainer_dtype}). "
                f"Aligning data dtype to trainer."
            )
            self.data.dtype = trainer_dtype

        # Store for model initialization
        self._coordinated_dtype = trainer_dtype

        logging.info(f"Coordinated dtype across all components: {trainer_dtype}")

        return self

    def get_coordinated_dtype(self) -> torch.dtype:
        """Get the dtype that all components should use."""
        if hasattr(self, "_coordinated_dtype"):
            return self._coordinated_dtype
        else:
            return self.trainer.get_effective_dtype()

    def _validate_data_selection(self) -> None:
        if self.data.data_group != "all":
            logger.warning(
                f"Training data group ({self.data.data_group}) "
                "was not set to 'all'! Updating config to train on full "
                "dataset. Build a separate dataset if you want to train on a subset."
            )
            self.data.update_field("data_group", "all", verbose=True, validate=False)

    def apply_parameter_scaling(self) -> None:
        """
        Apply parameter scaling after initialization to avoid recursion.
        Call this method explicitly after creating the TrainingParams instance.
        """
        eps = 1e-6

        # Log scaling information without modifying the actual parameters
        if abs(self.local_batch_size - self.data.batch_size) > eps:
            logger.info(
                f"Distributed training detected: "
                f"batch_size ({self.data.batch_size}) scaled to "
                f"local batch_size ({self.local_batch_size}) "
                f"according to world size ({self.trainer.world_size})"
            )
            self.data.update_field("batch_size", self.local_batch_size)

        if abs(self.model.learning_rate - self.effective_learning_rate) > eps:
            logger.info(
                f"Distributed training detected: "
                f"learning_rate ({self.model.learning_rate}) "
                f"with reference batch_size ({REFERENCE_BATCH_SIZE}) scaled to "
                f"effective_learning_rate ({self.effective_learning_rate}) "
                f"according to effective_effective_size ({self.effective_batch_size})"
            )
            self.model.update_field("learning_rate", self.effective_learning_rate)

    def _validate_required_paths(self) -> None:
        """Validate that required paths exist."""
        # Check input paths exist
        if not self.input_model_state.exists():
            raise DynVisionValidationError(
                f"Input model state not found: {self.input_model_state}"
            )

        print(
            f"use ffcv {self.data.use_ffcv}, {self.dataset_train}, {self.dataset_val}, {self.dataset_link}"
        )  # debugging

        if not self.dataset_train.exists():
            raise DynVisionValidationError(
                f"Training dataset not found: {self.dataset_train}"
            )

        if not self.dataset_val.exists():
            raise DynVisionValidationError(
                f"Validation dataset not found: {self.dataset_val}"
            )

        if not self.dataset_link.exists():
            raise DynVisionValidationError(
                f"dataset folder link not found: {self.dataset_link}"
            )
        # Ensure output directory exists
        self.output_model_state.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_parameter_conflicts(self) -> None:
        """Resolve conflicts between component parameters."""
        # Resolve n_timesteps consistency
        if (
            self.data.data_timesteps > 1
            and self.model.n_timesteps != self.data.data_timesteps
        ):
            resolved_timesteps = self.data.data_timesteps
            logger.info(
                f"Resolving n_timesteps conflict: model={self.model.n_timesteps}, "
                f"data={self.data.data_timesteps}, resolved_to={resolved_timesteps}"
            )
            # Update both to the larger value
            self.model.update_field("n_timesteps", resolved_timesteps)
            self.data.update_field("data_timesteps", resolved_timesteps)

    # === CONFIGURATION EXPORT ===

    def get_full_config(self, flat=True) -> Dict[str, Any]:
        config_dict = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model.model_dump(),
            "trainer": self.trainer.model_dump(),
            "data": self.data.model_dump(),
            "computed": {
                "effective_batch_size": self.effective_batch_size,
                "global_batch_size": self.global_batch_size,
                "effective_learning_rate": self.effective_learning_rate,
            },
            "system": {
                "torch_version": torch.__version__,
                "lightning_version": pl.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "world_size": self.trainer.world_size,  # Fixed: use property
            },
        }
        if flat:
            # Flatten the configuration for easier CLI parsing
            flat_config = {}
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_config[f"{subkey}"] = subvalue
                else:
                    flat_config[key] = value
            config_dict = flat_config

        return config_dict

    def export_full_config(self, path: Path, flat=True) -> None:
        """Export complete configuration for reproducibility."""
        config_dict = self.get_full_config(flat=flat)

        # Clean config for YAML serialization
        def clean_for_yaml(obj):
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (Path,)):
                return str(obj)
            elif isinstance(obj, dict):
                return {
                    str(k): clean_for_yaml(v)
                    for k, v in obj.items()
                    if clean_for_yaml(v) is not None
                }
            elif isinstance(obj, (list, tuple)):
                return [
                    clean_for_yaml(item)
                    for item in obj
                    if clean_for_yaml(item) is not None
                ]
            else:
                # Skip non-serializable objects
                return None

        cleaned_config = clean_for_yaml(config_dict)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(cleaned_config, f, indent=4, default_flow_style=False)

        logger.info(f"Configuration exported to {path}")

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names for all components."""
        aliases = super().get_aliases()

        # Add common aliases for backward compatibility
        aliases.update(
            {
                # Model aliases (will be routed to model component)
                "lr": "model.learning_rate",
                "opt": "model.optimizer",
                "rctype": "model.recurrence_type",
                "tsteps": "model.n_timesteps",
                "classes": "model.n_classes",
                "model_name": "model.model_name",
                # Trainer aliases (will be routed to trainer component)
                "epochs": "trainer.epochs",
                "prec": "trainer.precision",
                "patience": "trainer.early_stopping_patience",
                "devices": "trainer.devices",
                "strategy": "trainer.strategy",
                # Data aliases (will be routed to data component)
                "batch_size": "data.batch_size",
                "resolution": "data.resolution",
                "use_ffcv": "data.use_ffcv",
                "data_name": "data.data_name",
            }
        )

        return aliases

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> "TrainingParams":
        """
        Create TrainingParams instance from CLI and config with proper component separation.
        """
        # Get raw parameters using BaseParams method
        params = cls.get_params_from_cli_and_config(
            config_path=config_path,
            override_kwargs=override_kwargs,
            args=args,
        )

        # Separate into component configurations
        separated_params = cls._separate_component_configs(params)

        print("Training Params:\n", separated_params)

        # Create the TrainingParams instance
        try:
            instance = cls(**separated_params)
            # Apply scaling after successful creation to avoid recursion
            instance.apply_parameter_scaling()
            return instance
        except Exception as e:
            raise DynVisionValidationError(f"TrainingParams creation failed: {e}")

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
                # Assign to ALL component classes that have this field
                assigned_to = []

                if key in model_fields:
                    model_params[key] = value
                    assigned_to.append("model")
                if key in trainer_fields:
                    trainer_params[key] = value
                    assigned_to.append("trainer")
                if key in data_fields:
                    data_params[key] = value
                    assigned_to.append("data")
                if key in base_fields:
                    base_params[key] = value
                    assigned_to.append("base")

                if assigned_to:
                    # Log when parameters are shared across components
                    if len(assigned_to) > 1:
                        logger.debug(
                            f"Parameter '{key}' assigned to multiple components: {assigned_to}"
                        )
                else:
                    # Unknown parameters go to model_params as fallback
                    model_params[key] = value
                    logger.debug(
                        f"Assigning unknown parameter '{key}' to model_params"
                    )
        # Create component instances
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
