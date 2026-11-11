"""
Model initialization parameters for DynVision using Pydantic.

This module provides type-safe parameter management for model initialization,
including dataset dimension inference and pretrained model configuration.
"""

from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, Tuple
import logging

from pydantic import Field, model_validator, ConfigDict

from dynvision.params.base_params import DynVisionValidationError
from dynvision.params.composite_params import CompositeParams
from dynvision.params.model_params import ModelParams
from dynvision.params.data_params import DataParams
from dynvision.utils import log_section, format_value
from dynvision.utils.timing import coordinate_component_timing

logger = logging.getLogger(__name__)


class InitParams(CompositeParams):
    """
    Model initialization parameters with automatic dimension inference.

    Combines ModelParams and DataParams with initialization-specific configuration
    for creating and saving initialized models.
    """

    mode_name: ClassVar[str] = "init"
    component_classes: ClassVar[Dict[str, type]] = {
        "model": ModelParams,
        "data": DataParams,
    }

    # ===== COMMON PARAMETERS =====
    seed: int = Field(description="Random seed for reproducibility")
    log_level: str = Field(description="Logging level")

    # === CORE COMPONENT COMPOSITION ===
    model: ModelParams = Field(description="Model architecture parameters")
    data: DataParams = Field(
        description="Data loading parameters for dimension inference"
    )

    # === INITIALIZATION-SPECIFIC PARAMETERS ===
    dataset_path: Optional[Path] = Field(
        default=None, description="Path to dataset for dimension inference"
    )
    output: Path = Field(description="Path to save initialized model")

    model_config = ConfigDict(
        extra="allow",  # Allow additional CLI arguments
        validate_assignment=True,
        use_enum_values=True,
        validate_by_name=True,
    )

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names for all components."""
        aliases = super().get_aliases()

        # Add initialization-specific aliases
        aliases.update(
            {
                # Initialization aliases
                "dataset": "dataset_path",  # CLI convenience alias
                # Model aliases (routed to model component)
                "model_name": "model.model_name",
                "classes": "model.n_classes",
                "tsteps": "model.n_timesteps",
                "rctype": "model.recurrence_type",
                # Data aliases (routed to data component)
                "data_name": "data.data_name",
                "resolution": "data.resolution",
            }
        )

        return cls.merge_component_aliases(aliases)

    @model_validator(mode="after")
    def coordinate_component_timing(self) -> "InitParams":
        """Convert the data component's ms windows with the model's dt.

        Defined before the other validators so that they see step counts.
        """
        try:
            coordinate_component_timing(self)
        except ValueError as error:
            raise DynVisionValidationError(str(error)) from None
        return self

    @model_validator(mode="after")
    def validate_initialization_config(self) -> "InitParams":
        """Validate initialization-specific configuration."""
        # Ensure output directory exists
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def get_component_assignment_order(cls) -> Iterable[str]:
        return ("data", "model")

    @classmethod
    def get_component_preprocessors(
        cls,
    ) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        return {
            "data": cls.validate_data_config,
            "model": cls.validate_model_config,
        }

    @classmethod
    def _handle_unscoped_param(
        cls,
        key: str,
        value: Any,
        component_data: Dict[str, Dict[str, Any]],
        base_params: Dict[str, Any],
    ) -> None:
        component_data.setdefault("model", {})[key] = value
        logger.debug("Assigning unscoped parameter '%s' to model component", key)

    @classmethod
    def validate_data_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate initialization data config (warnings only)."""
        # No forced updates - values come from config files under init.data.*
        # Optional: Add warnings for unusual configurations
        if config.get("use_ffcv", False):
            logger.info("Initialization with use_ffcv=True (FFCV dataset required)")
        return config

    @classmethod
    def validate_model_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate initialization model config (warnings only)."""
        # No forced updates - values come from config files under init.model.*
        return config

    def log_initialization_overview(
        self,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Log a structured overview of the initialization run."""

        run_logger = logger or logging.getLogger(__name__)
        log_section(
            run_logger,
            "initialization_run",
            [
                ("seed", format_value(self.seed), None),
                ("output", format_value(self.output), None),
                ("dataset_path", format_value(self.dataset_path), None),
            ],
        )

        self.log_overview(logger=run_logger, include_defaults=False)

    def update_model_parameters_from_dataset(
        self,
        input_dims: Tuple[int, ...],
        n_classes: int,
        verbose: bool = True,
    ) -> None:
        """
        Update model parameters based on dataset characteristics.

        Args:
            input_dims: Input dimensions from dataset (channels, height, width) or (timesteps, channels, height, width)
            n_classes: Number of classes in dataset
            verbose: Whether to log warnings for parameter changes
        """
        # Update input dimensions
        if self.model.input_dims != input_dims:
            self.model.update_field("input_dims", input_dims, verbose=verbose)

        # Update n_timesteps if provided in input_dims
        if len(input_dims) == 4:  # (timesteps, channels, height, width)
            n_timesteps = input_dims[0]
            if n_timesteps > 1 and self.model.n_timesteps != n_timesteps:
                self.model.update_field("n_timesteps", n_timesteps, verbose=verbose)

        # Update n_classes
        if self.model.n_classes != n_classes:
            self.model.update_field("n_classes", n_classes, verbose=verbose)

    def get_model_kwargs(self, model_class=None) -> Dict[str, Any]:
        return self.model.get_model_kwargs(model_class)

    def get_dataset_kwargs(self) -> Dict[str, Any]:
        return self.data.get_dataset_kwargs()

    def get_dataloader_kwargs(self, dataloader_class=None) -> Dict[str, Any]:
        return self.data.get_dataloader_kwargs(dataloader_class=dataloader_class)

    def log_model_creation(
        self,
        *,
        model_class,
        model_kwargs: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Delegate model creation logging to the model params."""

        run_logger = logger or logging.getLogger(__name__)
        self.model.log_model_creation(
            model_class=model_class,
            model_kwargs=model_kwargs,
            logger=run_logger,
        )

    def log_dataloader_creation(
        self,
        *,
        dataloader_class,
        dataloader_kwargs: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        context: str = "active",
        previous_kwargs: Optional[Dict[str, Any]] = None,
        level: int = logging.INFO,
    ) -> None:
        run_logger = logger or logging.getLogger(__name__)
        self.data.log_dataloader_creation(
            dataloader_class=dataloader_class,
            dataloader_kwargs=dataloader_kwargs,
            logger=run_logger,
            context=context,
            previous_kwargs=previous_kwargs,
            level=level,
        )
