"""
Testing parameters for DynVision using Pydantic.

This module provides type-safe parameter management for model testing and evaluation,
including response storage configuration and memory-optimized settings.
"""

import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Sequence,
)

from pydantic import (
    Field,
    computed_field,
    model_validator,
    ConfigDict,
)

from dynvision.params.base_params import DynVisionValidationError
from dynvision.params.composite_params import CompositeParams
from dynvision.params.model_params import ModelParams
from dynvision.params.trainer_params import TrainerParams
from dynvision.params.data_params import DataParams
from dynvision.utils import SummaryItem, log_section, format_value

logger = logging.getLogger(__name__)


class TestingParams(CompositeParams):
    """
    Composite configuration for model testing with comprehensive validation.

    Combines ModelParams, TrainerParams, and DataParams with testing-specific configuration
    for model evaluation, response storage, and result analysis.
    """

    mode_name: ClassVar[str] = "test"
    component_classes: ClassVar[Dict[str, type]] = {
        "model": ModelParams,
        "trainer": TrainerParams,
        "data": DataParams,
    }

    summary_sections: ClassVar[Dict[str, Sequence[SummaryItem]]] = {
        "Run": (
            SummaryItem("mode_name", always=True),
            SummaryItem("seed", always=True),
            SummaryItem("log_level", always=True),
            SummaryItem("verbose", always=True),
        ),
        "Paths": (
            SummaryItem("dataset_path", always=True),
            SummaryItem("input_model_state", always=True),
            SummaryItem("output_results", always=True),
            SummaryItem("output_responses", always=True),
        ),
    }

    # ===== COMMON PARAMETERS =====
    seed: int = Field(description="Random seed for reproducibility")
    log_level: str = Field(description="Logging level")

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
    dataset_path: Path = Field(description="Path to test dataset")
    output_results: Path = Field(description="Path to save test results (CSV)")
    output_responses: Path = Field(
        description="Path to save model responses (PT tensors)"
    )

    # Testing behavior configuration
    verbose: bool = Field(
        ..., description="Enable verbose logging and error reporting"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional CLI arguments
        validate_assignment=True,
        use_enum_values=True,
        validate_by_name=True,
    )

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

    @classmethod
    def get_component_assignment_order(cls) -> Iterable[str]:
        return ("model", "data", "trainer")

    @classmethod
    def get_component_preprocessors(
        cls,
    ) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        return {
            "data": cls.validate_data_config,
            "trainer": cls.validate_trainer_config,
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
        component_data.setdefault("data", {})[key] = value
        logger.debug(
            "Assigning unscoped parameter '%s' to model and data components", key
        )

    def _validate_required_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.input_model_state.exists():
            raise DynVisionValidationError(
                f"Input model state not found: {self.input_model_state}"
            )

        if not self.dataset_path.exists():
            raise DynVisionValidationError(
                f"Test dataset not found: {self.dataset_path}"
            )

        # Ensure output directories exist
        self.output_results.parent.mkdir(parents=True, exist_ok=True)
        self.output_responses.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_data_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate testing data config (warnings only)."""
        # Check for unusual configurations
        if config.get("train", False):
            logger.warning(
                "Testing with train=True is unusual - data augmentation may be active"
            )
        if config.get("shuffle", False):
            logger.warning("Testing with shuffle=True may affect reproducibility")

        # Dynamic value: max_workers based on num_workers
        # This is kept as dynamic logic since it depends on another parameter
        if "max_workers" not in config and "num_workers" in config:
            config["max_workers"] = min(4, config["num_workers"])

        return config

    @classmethod
    def validate_trainer_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate testing trainer config (warnings only)."""
        # Check for unusual configurations
        if config.get("devices", 1) > 1:
            logger.warning("Testing with multiple devices may affect determinism")
        return config

    @classmethod
    def validate_model_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate testing model config (warnings only)."""
        # No forced updates - values come from config files under test.model.*
        return config

    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage based on model and data characteristics."""

        # Check for memory-intensive configurations
        store_responses = getattr(self.model, "store_responses", None)

        # Nothing to validate when store_responses is unset/None
        if store_responses is None:
            return

        if store_responses > 10000:
            logger.warning(
                f"Large store_responses ({store_responses}) may cause memory issues. "
                "Consider reducing or using store_responses=0 for no response storage."
            )
        elif store_responses == -1:
            logger.warning(
                "store_responses=-1 (all responses) may cause memory issues with large datasets. "
                "Consider using a specific number or store_responses=0 for no storage."
            )

    def update_model_parameters_from_data(
        self,
        input_dims: Tuple[int, ...],
        n_classes: Optional[int] = None,  # DEPRECATED
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
            logger.warning("Number of classes mismatches between model and data!")
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

    def get_dataloader_kwargs(self, dataloader_class=None) -> Dict[str, Any]:
        return self.data.get_dataloader_kwargs(dataloader_class=dataloader_class)

    def get_trainer_kwargs(self) -> Dict[str, Any]:
        return self.trainer.get_trainer_kwargs()

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
        self.data.log_dataloader_creation(
            dataloader_class=dataloader_class,
            dataloader_kwargs=dataloader_kwargs,
            logger=logger,
            context=context,
            previous_kwargs=previous_kwargs,
            level=level,
        )

    def log_trainer_creation(
        self,
        *,
        trainer_kwargs: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.trainer.log_trainer_creation(
            trainer_kwargs=trainer_kwargs,
            logger=logger,
        )

    def log_testing_overview(
        self,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Log a structured overview of the testing run."""

        run_logger = logger or logging.getLogger(__name__)
        entries = [
            ("seed", format_value(self.seed), None),
            ("input_model_state", format_value(self.input_model_state), None),
            ("dataset_path", format_value(self.dataset_path), None),
            ("output_results", format_value(self.output_results), None),
            ("output_responses", format_value(self.output_responses), None),
            ("data_name", format_value(self.data.data_name), None),
            ("batch_size", format_value(self.data.batch_size), None),
            ("precision", format_value(self.trainer.precision), None),
            ("verbose", format_value(self.verbose), None),
        ]

        log_section(run_logger, "testing_run", entries)
        self.log_overview(
            logger=run_logger.getChild("params"),
            include_components=True,
            include_defaults=False,
        )

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names for all components."""
        aliases = super().get_aliases()

        # Add testing-specific aliases
        aliases.update(
            {
                # Testing-specific aliases
                "dataset": "dataset_path",  # CLI convenience alias
                # Model aliases (routed to model component)
                "model_name": "model.model_name",
                "classes": "model.n_classes",
                "tsteps": "model.n_timesteps",
                "rctype": "model.recurrence_type",
                "rctarget": "model.recurrence_target",
                # Model timing aliases (needed for biological time overrides in test identifiers)
                "tff": "model.t_feedforward",
                "tfb": "model.t_feedback",
                "tsk": "model.t_skip",
                "trc": "model.t_recurrence",
                "dt": "model.dt",
                "tau": "model.tau",
                "lossrt": "model.loss_reaction_time",
                "pattern": "model.data_presentation_pattern",
                "solver": "model.dynamics_solver",
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
                # Storage aliases
                "responses": "store_responses",
                "cache": "cache_size",
            }
        )

        return aliases
