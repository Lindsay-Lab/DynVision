"""
Model parameter handling for DynVision using Pydantic.

This module provides type-safe parameter management for neural network models,
including biological parameters, architecture settings, optimizer/scheduler
configuration, and loss functions.
"""

from collections import OrderedDict

from pydantic import Field, field_validator, model_validator, ConfigDict
from typing import Dict, Any, Optional, List, Union, Tuple, Literal, Sequence, ClassVar
import logging

from dynvision.params.base_params import BaseParams, DynVisionValidationError
from dynvision.utils import (
    SummaryItem,
    log_section,
    format_value,
    filter_kwargs,
    resolve_signature_defaults,
)


class ModelParams(BaseParams):
    """
    Model parameters for neural network architecture and training configuration.

    Handles biological parameters, architecture settings, optimizer/scheduler
    configuration, loss functions, and model-specific parameters.
    """

    # ===== COMMON PARAMETERS =====
    seed: int = Field(description="Random seed for reproducibility")
    log_level: str = Field(description="Logging level")

    # ===== CORE ARCHITECTURE =====
    # All defaults moved to config_defaults.yaml
    # None = "not set" → filtered out → model class default used
    model_name: Optional[str] = Field(
        default=None, description="Name of the model architecture"
    )
    n_classes: Optional[int] = Field(
        default=None, description="Number of output classes"
    )
    input_dims: Optional[Tuple[int, ...]] = Field(
        default=None,
        description="Input dimensions (timesteps, channels, height, width)",
    )
    n_timesteps: Optional[int] = Field(
        default=None, description="Number of timesteps for model processing"
    )
    data_presentation_pattern: Optional[List[int]] = Field(
        default=None,
        description="Pattern for data presentation across timesteps (1 = present, 0 = absent)",
    )
    input_adaption_weight: Optional[float] = Field(
        default=None,
        description="weight to multiply the input for each consecutive timestep",
    )

    # ===== BIOLOGICAL PARAMETERS =====
    dt: Optional[float] = Field(default=None, description="Integration time step (ms)")
    tau: Optional[float] = Field(default=None, description="Neural time constant (ms)")
    t_feedforward: Optional[float] = Field(
        default=None, description="Feedforward delay (ms)"
    )
    t_recurrence: Optional[float] = Field(
        default=None, description="Recurrent delay (ms)"
    )
    t_feedback: Optional[float] = Field(
        default=None, description="Feedback delay (ms)"
    )
    t_skip: Optional[float] = Field(default=None, description="Skip delay (ms)")
    dynamics_solver: Optional[Literal["euler", "rk4"]] = Field(
        default=None, description="Dynamical systems solver"
    )
    idle_timesteps: Optional[int] = Field(
        default=None,
        description="Number of idle timesteps for spontaneous activity to converge",
    )
    feedforward_only: Optional[bool] = Field(
        default=None, description="Use only feedforward connections"
    )
    truncated_bptt_timesteps: Optional[int] = Field(
        default=None,
        description="Truncated backpropagation through time - detach hidden states every N timesteps (0 or None = disabled)",
        ge=0,
    )
    # ===== RECURRENT ARCHITECTURE =====
    recurrence_type: Optional[
        Literal[
            "full",
            "self",
            "depthwise",
            "pointwise",
            "depthpointwise",
            "pointdepthwise",
            "local",
            "localdepthwise",
            "none",
        ]
    ] = Field(default=None, description="Type of recurrent connections")

    # ===== CONNECTIVITY =====
    skip: Optional[bool] = Field(
        default=None, description="Enable skip connections between layers"
    )
    feedback: Optional[Union[bool, str]] = Field(
        default=None,
        description="Enable feedback connections from higher to lower layers",
    )

    # ===== NONLINEARITIES =====
    supralinearity: Optional[float] = Field(
        default=None, description="Supralinearity exponent"
    )

    # ===== RESPONSE STORAGE (DEPRECATED - use storage configuration below) =====
    store_responses: Optional[int] = Field(
        default=None,
        description="DEPRECATED: Use store_test_responses instead. Number of responses to store during evaluation (0 = disabled)",
    )
    store_responses_on_cpu: Optional[bool] = Field(
        default=None, description="Store responses on CPU to save GPU memory"
    )
    classifier_name: Optional[str] = Field(
        default=None, description="Name of the classifier layer"
    )

    # ===== STORAGE BUFFER CONFIGURATION =====
    store_train_responses: Optional[int] = Field(
        default=None,
        description="Number of responses to store during training (None = use default, 0 = disabled, -1 = unlimited)",
        ge=-1,
    )
    store_val_responses: Optional[int] = Field(
        default=None,
        description="Number of responses to store during validation (None = use default, 0 = disabled, -1 = unlimited)",
        ge=-1,
    )
    store_test_responses: Optional[int] = Field(
        default=None,
        description="Number of responses to store during testing (None = use default, 0 = disabled, -1 = unlimited)",
        ge=-1,
    )
    store_train_records: Optional[int] = Field(
        default=None,
        description="Number of records to store during training (None = use default, 0 = disabled, -1 = unlimited)",
        ge=-1,
    )
    store_val_records: Optional[int] = Field(
        default=None,
        description="Number of records to store during validation (None = use default, 0 = disabled, -1 = unlimited)",
        ge=-1,
    )
    store_test_records: Optional[int] = Field(
        default=None,
        description="Number of records to store during testing (None = use default, 0 = disabled, -1 = unlimited)",
        ge=-1,
    )
    early_test_stop: Optional[bool] = Field(
        default=None,
        description="Stop testing early when buffer is filled (only for fixed strategy)",
    )
    response_resolution: Optional[Literal["unit", "layer"]] = Field(
        default=None,
        description="Resolution for stored responses: 'unit' (full tensors) or 'layer' (spatially averaged)",
    )

    # ===== LOSS CONFIGURATION =====
    loss: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Loss function name or list of loss functions",
    )
    loss_configs: Optional[Dict[str, Dict]] = Field(
        default=None, description="Configurations for the loss function"
    )
    loss_reaction_time: Optional[float] = Field(
        default=None, description="Reaction time for loss calculation (ms)"
    )
    # ===== OPTIMIZER CONFIGURATION =====
    learning_rate: Optional[float] = Field(
        default=None, description="Learning rate for optimizer"
    )
    optimizer: Optional[str] = Field(
        default=None, description="Optimizer type (Adam, SGD, AdamW, etc.)"
    )
    optimizer_kwargs: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional optimizer arguments"
    )
    optimizer_configs: Optional[Dict[str, Any]] = Field(
        default=None, description="Optimizer configuration dictionary"
    )

    # ===== LEARNING RATE PARAMETER GROUPS =====
    lr_parameter_groups: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Learning rate factors for different parameter groups",
    )

    # ===== SCHEDULER CONFIGURATION =====
    scheduler: Optional[str] = Field(
        default=None, description="Learning rate scheduler type"
    )
    scheduler_kwargs: Optional[Dict[str, Any]] = Field(
        default=None, description="Scheduler arguments"
    )
    scheduler_configs: Optional[Dict[str, Any]] = Field(
        default=None, description="Scheduler configuration"
    )

    # ===== TRAINING BEHAVIOR =====
    retain_graph: Optional[bool] = Field(
        default=None,
        description="Whether to retain computation graph for backward pass",
    )
    non_label_index: Optional[int] = Field(
        default=None, description="Index for non-label timesteps"
    )

    # ===== CUSTOM MODEL PARAMETERS =====
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific arguments"
    )

    cached_model_kwargs: Dict[Any, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Cached filtered model kwargs for different model classes",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,  # Validate fields when assigned after creation
        use_enum_values=True,  # Use enum values in serialization
        validate_by_name=True,  # Allow validation using field names and aliases
    )

    summary_sections: ClassVar[Dict[str, Sequence[SummaryItem]]] = {
        "Architecture": (
            SummaryItem("model_name", always=True),
            SummaryItem("input_dims"),
            SummaryItem("n_classes"),
            SummaryItem("n_timesteps"),
            SummaryItem("data_presentation_pattern"),
        ),
        "Temporal": (
            SummaryItem("dt"),
            SummaryItem("tau"),
            SummaryItem("dynamics_solver"),
            SummaryItem("t_feedforward"),
            SummaryItem("t_recurrence"),
            SummaryItem("t_feedback"),
            SummaryItem("t_skip"),
            SummaryItem("idle_timesteps"),
            SummaryItem("truncated_bptt_timesteps"),
        ),
        "Connectivity": (
            SummaryItem("recurrence_type"),
            SummaryItem("recurrence_target"),
            SummaryItem("skip"),
            SummaryItem("feedback"),
            SummaryItem("feedforward_only"),
            SummaryItem("supralinearity"),
        ),
        "Training": (
            SummaryItem("learning_rate"),
            SummaryItem("optimizer"),
            SummaryItem("scheduler"),
            SummaryItem("loss_reaction_time"),
        ),
        "Storage": (
            SummaryItem("store_train_responses"),
            SummaryItem("store_val_responses"),
            SummaryItem("store_test_responses"),
            SummaryItem("store_responses_on_cpu"),
            SummaryItem("early_test_stop"),
            SummaryItem("response_resolution"),
        ),
    }

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Return mapping of aliases to full parameter names."""
        aliases = super().get_aliases()
        aliases.update(
            {
                "lr": "learning_rate",
                "opt": "optimizer",
                "sched": "scheduler",
                "rctype": "recurrence_type",
                "rctarget": "recurrence_target",
                "trc": "t_recurrence",
                "tff": "t_feedforward",
                "tfb": "t_feedback",
                "tsk": "t_skip",
                "pattern": "data_presentation_pattern",
                "solver": "dynamics_solver",
                "lossrt": "loss_reaction_time",
                "supralin": "supralinearity",
                "classes": "n_classes",
                "steps": "n_timesteps",
                "tsteps": "n_timesteps",
                "idle": "idle_timesteps",
                "ffonly": "feedforward_only",
                "inadapt": "input_adaption_weight",
                "tbptt": "truncated_bptt_timesteps",
                # Storage aliases
                "store_train_resp": "store_train_responses",
                "store_val_resp": "store_val_responses",
                "store_test_resp": "store_test_responses",
                "store_train_rec": "store_train_records",
                "store_val_rec": "store_val_records",
                "store_test_rec": "store_test_records",
                "early_stop": "early_test_stop",
                "resp_res": "response_resolution",
            }
        )
        return aliases

    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Ensure log_level is valid."""
        if isinstance(v, str):
            v = v.upper()
            if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(f"Invalid log level: {v}")
        return v

    @field_validator("store_responses")
    def convert_store_responses(cls, v) -> Optional[int]:
        """Convert store_responses to int, handling string inputs."""
        if v is None:
            return None
        if isinstance(v, str):
            if v.lower() == "all":
                return -1  # Use -1 to indicate "all responses"
            return int(v)
        return int(v)

    @field_validator("recurrence_type")
    def validate_recurrence_type(cls, v):
        # None means "not set" → will be filtered out → model decides
        # This is different from the old default "none" which disabled recurrence
        if v is None:
            return None
        return v

    @field_validator("loss")
    def validate_loss(cls, v) -> Optional[list]:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        else:
            return [v]

    @field_validator("optimizer")
    def validate_optimizer(cls, v):
        if v is None:
            return None
        valid_optimizers = [
            "Adam",
            "AdamW",
            "SGD",
            "RMSprop",
            "Adagrad",
            "Adadelta",
            "Adamax",
            "ASGD",
            "LBFGS",
            "NAdam",
            "RAdam",
            "Rprop",
            "SparseAdam",
        ]
        if v not in valid_optimizers:
            logging.warning(
                f"Unknown optimizer: {v}. Valid options: {valid_optimizers}"
            )
        return v

    @field_validator("scheduler")
    def validate_scheduler(cls, v):
        """Validate scheduler name."""
        if v is None:
            return None
        valid_schedulers = [
            "StepLR",
            "MultiStepLR",
            "ExponentialLR",
            "CosineAnnealingLR",
            "ReduceLROnPlateau",
            "CyclicLR",
            "OneCycleLR",
            "CosineAnnealingWarmRestarts",
            "LinearLR",
            "PolynomialLR",
            "LambdaLR",
            "MultiplicativeLR",
        ]
        if v not in valid_schedulers:
            logging.warning(
                f"Unknown scheduler: {v}. Valid options: {valid_schedulers}"
            )
        return v

    @field_validator("model_name")
    def validate_model_name(cls, v):
        """Validate model name."""
        if v is None:
            return None
        valid_models = [
            "DyRCNNx2",
            "DyRCNNx4",
            "DyRCNNx8",
            "CorNetRT",
            "CordsNet",
            "AlexNet",
            "ResNet18",
            "ResNet34",
            "ResNet50",
        ]
        if v not in valid_models:
            logging.warning(
                f"Unknown model: {v}. Consider adding to valid_models list"
            )
        return v

    @field_validator("data_presentation_pattern", mode="before")
    @classmethod
    def validate_data_presentation_pattern(cls, v):
        """Convert string pattern specifications to list of ints.

        Handles:
        - String numbers: '1' → [1]
        - Comma/space separated: '1,0,1' or '1 0 1' → [1, 0, 1]
        - Already a list: [1, 0, 1] → [1, 0, 1]
        - None: None → None
        """
        if v is None:
            return None

        if isinstance(v, str):
            # Parse string pattern
            import re

            v_stripped = v.strip()
            # If no separators, treat each character as a digit
            if re.search(r"[,\s]", v_stripped):
                parts = re.split(r"[,\s]+", v_stripped)
                parts = [p for p in parts if p]  # Remove empty strings
            else:
                parts = list(v_stripped)
            try:
                v = [int(p) for p in parts]
            except ValueError as e:
                raise ValueError(
                    f"data_presentation_pattern contains non-integer values: {v}"
                ) from e

        if isinstance(v, list):
            # Validate list contents
            if not all(isinstance(x, int) for x in v):
                raise ValueError(
                    f"data_presentation_pattern must contain only integers, got {v}"
                )
            return v

        raise ValueError(
            f"data_presentation_pattern must be a string or list, got {type(v).__name__}"
        )

    @field_validator("input_dims")
    def validate_input_dims(cls, v):
        """Validate input dimensions."""
        if v is None:
            return None
        if len(v) not in [3, 4]:
            raise ValueError(
                "input_dims must be (channels, height, width) or (timesteps, channels, height, width)"
            )

        # Ensure all dimensions are positive
        if any(dim <= 0 for dim in v):
            raise ValueError("All input dimensions must be positive")

        return v

    @model_validator(mode="after")
    def setup_defaults_and_validate_constraints(self):
        """Set up defaults and validate all constraints.

        Note: Many parameters can now be None (not set), which means they'll
        be filtered out in get_model_kwargs() and model class defaults will be used.
        Only validate parameters that are explicitly set (not None).
        """
        # Set up loss_configs default only if loss is set but configs aren't
        if self.loss is not None and not self.loss_configs:
            self.update_field(
                "loss_configs",
                {
                    "CrossEntropyLoss": {"weight": 1.0, "ignore_index": -1},
                    "ActivityLoss": {"weight": 100},
                },
                mutation_tag="derived",
            )

        # Validate loss configs if both loss and loss_configs are set
        if self.loss is not None and self.loss_configs is not None:
            for loss in self.loss:
                if loss not in self.loss_configs:
                    logging.warning(f"Loss configuration for '{loss}' not found.")

        # Provide default lr_parameter_groups if empty (but not if None)
        if self.lr_parameter_groups is not None and not self.lr_parameter_groups:
            self.update_field(
                "lr_parameter_groups",
                {
                    "regular": {"lr_factor": 1.0},
                    "recurrent": {"lr_factor": 1.0},
                    "feedback": {"lr_factor": 1.0},
                },
                mutation_tag="derived",
            )
            logging.info("Using default lr_parameter_groups")

        # Validate lr_parameter_groups if set
        if self.lr_parameter_groups is not None:
            for group_name, group_config in self.lr_parameter_groups.items():
                if "lr_factor" not in group_config:
                    group_config["lr_factor"] = 1.0
                    logging.info(
                        f"Added missing lr_factor=1.0 to group '{group_name}'"
                    )
                    self.mark_field_mutation("lr_parameter_groups", tag="derived")

        # Provide default scheduler configs if scheduler is set but configs aren't
        if self.scheduler is not None and not self.scheduler_configs:
            default_configs = {
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "learning_rate",
            }
            self.update_field(
                "scheduler_configs",
                default_configs,
                mutation_tag="derived",
            )
            logging.info("Using default scheduler_configs")

        # Provide default scheduler kwargs based on scheduler type
        if self.scheduler is not None and not self.scheduler_kwargs:
            if self.scheduler == "StepLR":
                self.update_field(
                    "scheduler_kwargs",
                    {"step_size": 30, "gamma": 0.1},
                    mutation_tag="derived",
                )
                logging.info("Using default StepLR scheduler_kwargs")
            elif self.scheduler == "CosineAnnealingLR":
                self.update_field(
                    "scheduler_kwargs",
                    {"T_max": 100},
                    mutation_tag="derived",
                )
                logging.info("Using default CosineAnnealingLR scheduler_kwargs")

        # Scheduler configuration consistency
        if (
            self.scheduler == "ReduceLROnPlateau"
            and self.scheduler_configs is not None
        ):
            if "monitor" not in self.scheduler_configs:
                self.scheduler_configs["monitor"] = "val_loss"
                self.mark_field_mutation("scheduler_configs", tag="derived")

        # Biological timing constraints (only validate if both dt and tau are set)
        if self.dt is not None and self.tau is not None:
            if self.dt >= self.tau:
                logging.warning(
                    f"dt ({self.dt}) should be much smaller than tau ({self.tau}) "
                    "for numerical stability (typically dt < tau/10)"
                )

        # Delay constraints (only validate if set)
        if self.t_feedforward is not None and self.t_feedforward < 0:
            raise DynVisionValidationError("t_feedforward must be non-negative")

        if self.t_recurrence is not None and self.t_recurrence < 1:
            raise DynVisionValidationError("t_recurrence must be at least 1")

        # Check if delays are exact multiples of dt (informational, only if both set)
        if self.dt is not None and self.t_feedforward is not None:
            if self.t_feedforward % self.dt != 0:
                steps = int(self.t_feedforward / self.dt)
                logging.info(
                    f"t_feedforward ({self.t_feedforward}ms) rounds to {steps} timesteps "
                    f"(dt={self.dt}ms)"
                )

        if self.dt is not None and self.t_recurrence is not None:
            if self.t_recurrence % self.dt != 0:
                steps = int(self.t_recurrence / self.dt)
                logging.info(
                    f"t_recurrence ({self.t_recurrence}ms) rounds to {steps} timesteps "
                    f"(dt={self.dt}ms)"
                )

        # Architecture consistency checks (only if both set)
        if (
            self.supralinearity is not None
            and self.recurrence_type is not None
            and self.supralinearity != 1
            and self.recurrence_type == "none"
        ):
            logging.warning(
                "Supralinearity enabled but no recurrence - may cause instability"
            )

        # Input dimensions consistency (only if both set)
        if self.input_dims is not None and self.n_timesteps is not None:
            if len(self.input_dims) == 4:
                timesteps_from_dims = self.input_dims[0]
                if timesteps_from_dims > 1 and timesteps_from_dims != self.n_timesteps:
                    logging.warning(
                        f"n_timesteps ({self.n_timesteps}) doesn't match "
                        f"input_dims timesteps ({timesteps_from_dims})"
                    )

        # Input/output consistency (only if set)
        if self.n_classes is not None and self.n_classes < 2:
            logging.warning(
                f"n_classes ({self.n_classes}) should be >= 2 for meaningful classification"
            )

        # Handle deprecated store_responses parameter (only if set)
        if (
            self.store_responses is not None
            and self.store_responses != 0
            and self.store_test_responses is None
        ):
            logging.warning(
                f"'store_responses' is deprecated. Setting 'store_test_responses={self.store_responses}' instead. "
                "Please use store_test_responses, store_val_responses, store_train_responses explicitly."
            )
            self.update_field(
                "store_test_responses",
                self.store_responses,
                mutation_tag="derived",
            )

        # Validate storage configuration consistency
        storage_params = [
            self.store_train_responses,
            self.store_val_responses,
            self.store_test_responses,
            self.store_train_records,
            self.store_val_records,
            self.store_test_records,
        ]

        # Check for negative values (unlimited) and warn
        for param_name in [
            "store_train_responses",
            "store_val_responses",
            "store_test_responses",
            "store_train_records",
            "store_val_records",
            "store_test_records",
        ]:
            param_value = getattr(self, param_name)
            if param_value is not None and param_value < 0 and param_value != -1:
                raise DynVisionValidationError(
                    f"{param_name} must be None, 0 (disabled), positive integer, or -1 (unlimited). Got: {param_value}"
                )

        # Info about storage configuration
        if any(p is not None and p != 0 for p in storage_params):
            storage_info = []
            if self.store_train_responses not in [None, 0]:
                storage_info.append(f"train_responses={self.store_train_responses}")
            if self.store_val_responses not in [None, 0]:
                storage_info.append(f"val_responses={self.store_val_responses}")
            if self.store_test_responses not in [None, 0]:
                storage_info.append(f"test_responses={self.store_test_responses}")
            if self.store_train_records not in [None, 0]:
                storage_info.append(f"train_records={self.store_train_records}")
            if self.store_val_records not in [None, 0]:
                storage_info.append(f"val_records={self.store_val_records}")
            if self.store_test_records not in [None, 0]:
                storage_info.append(f"test_records={self.store_test_records}")

            if storage_info:
                logging.info(f"Storage configuration: {', '.join(storage_info)}")

        return self

    # ===== COMPUTED PROPERTIES (DERIVED PARAMETERS) =====

    @property
    def delay_feedforward(self) -> Optional[int]:
        """Number of timesteps for feedforward delay."""
        if self.t_feedforward is None or self.dt is None:
            return None
        return int(self.t_feedforward / self.dt)

    @property
    def delay_recurrence(self) -> Optional[int]:
        """Number of timesteps for recurrence delay."""
        if self.t_recurrence is None or self.dt is None:
            return None
        return int(self.t_recurrence / self.dt)

    @property
    def stability_ratio(self) -> Optional[float]:
        """Ratio dt/tau for numerical stability assessment."""
        if self.dt is None or self.tau is None:
            return None
        return self.dt / self.tau

    @property
    def criterion_params(self) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
        """Get the parameters for the criterion used in the model."""
        if self.loss is None or self.loss_configs is None:
            return None
        if not isinstance(self.loss, list):
            self.loss = [self.loss]
        return [(layer, self.loss_configs[layer]) for layer in self.loss]

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get comprehensive timing parameter summary."""
        return {
            "base_parameters": {
                "dt": self.dt,
                "tau": self.tau,
                "t_feedforward": self.t_feedforward,
                "t_recurrence": self.t_recurrence,
            },
            "derived_parameters": {
                "delay_feedforward": self.delay_feedforward,
                "delay_recurrence": self.delay_recurrence,
                "effective_tau_steps": self.effective_tau_steps,
                "total_processing_time": self.total_processing_time,
                "total_processing_steps": self.total_processing_steps,
                "stability_ratio": self.stability_ratio,
            },
            "assessment": {
                "is_stable": self.stability_ratio < 0.2,
                "delays_are_exact_multiples": (
                    self.t_feedforward % self.dt == 0
                    and self.t_recurrence % self.dt == 0
                ),
            },
        }

    def get_model_kwargs(self, model_class=None) -> Dict[str, Any]:
        """
        Get filtered kwargs appropriate for model creation.

        None values are filtered out to allow model class defaults to be used.
        This implements the sentinel pattern: None = "not set" → use model default.

        Args:
            model_class: The model class to filter kwargs for

        Returns:
            Dictionary of kwargs suitable for model instantiation
        """
        if (
            hasattr(self, "cached_model_kwargs")
            and model_class in self.cached_model_kwargs
        ):
            return self.cached_model_kwargs[model_class]

        # Base model parameters
        model_kwargs = {
            "n_classes": self.n_classes,
            "input_dims": self.input_dims,
            "n_timesteps": self.n_timesteps,
            "data_presentation_pattern": self.data_presentation_pattern,
            "dt": self.dt,
            "tau": self.tau,
            "t_feedforward": self.t_feedforward,
            "t_recurrence": self.t_recurrence,
            "t_feedback": self.t_feedback,
            "t_skip": self.t_skip,
            "dynamics_solver": self.dynamics_solver,
            "idle_timesteps": self.idle_timesteps,
            "feedforward_only": self.feedforward_only,
            "truncated_bptt_timesteps": self.truncated_bptt_timesteps,
            "recurrence_type": self.recurrence_type,
            "skip": self.skip,
            "feedback": self.feedback,
            "supralinearity": self.supralinearity,
            "store_responses": self.store_responses,
            "store_responses_on_cpu": self.store_responses_on_cpu,
            "classifier_name": self.classifier_name,
            # Storage buffer configuration
            "store_train_responses": self.store_train_responses,
            "store_val_responses": self.store_val_responses,
            "store_test_responses": self.store_test_responses,
            "store_train_records": self.store_train_records,
            "store_val_records": self.store_val_records,
            "store_test_records": self.store_test_records,
            "early_test_stop": self.early_test_stop,
            "response_resolution": self.response_resolution,
            # Loss and training configuration
            "loss_reaction_time": self.loss_reaction_time,
            "criterion_params": self.criterion_params,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "optimizer_configs": self.optimizer_configs,
            "lr_parameter_groups": self.lr_parameter_groups,
            "scheduler": self.scheduler,
            "scheduler_kwargs": self.scheduler_kwargs,
            "scheduler_configs": self.scheduler_configs,
            "retain_graph": self.retain_graph,
            "non_label_index": self.non_label_index,
        }

        # Add custom model kwargs
        if self.model_kwargs:
            model_kwargs.update(self.model_kwargs)

        # Add extra fields if any
        if hasattr(self, "__pydantic_extra__"):
            model_kwargs.update(self.__pydantic_extra__)

        # CRITICAL: Filter out None values to allow model class defaults
        # None = "not set" → don't pass to model → model uses its own default
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # Filter kwargs if model_class is provided
        if model_class is not None:
            known, unknown = filter_kwargs(model_class, model_kwargs)

            if known:
                logging.info(
                    f"Filtered known model kwargs for {model_class.__name__}: {list(known.keys())}"
                )

            if unknown:
                logging.debug(
                    f"Unknown kwargs for {model_class.__name__}: {list(unknown.keys())}"
                )

            return known

        self.cached_model_kwargs[model_class] = model_kwargs

        return model_kwargs

    def get_model_kwargs_with_defaults(
        self, model_class
    ) -> Tuple[Dict[str, Any], OrderedDict[str, Any], Dict[str, bool]]:
        """Return provided kwargs alongside resolved defaults for logging."""

        provided_kwargs = self.get_model_kwargs(model_class)
        resolved_kwargs, default_flags = resolve_signature_defaults(
            model_class, provided_kwargs
        )
        return provided_kwargs, resolved_kwargs, default_flags

    def log_model_creation(
        self,
        *,
        model_class,
        logger: Optional[logging.Logger] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a single structured summary describing how the model will be instantiated."""

        run_logger = logger or logging.getLogger(__name__)
        provided_kwargs = model_kwargs or self.get_model_kwargs(model_class)
        resolved_kwargs, default_flags = resolve_signature_defaults(
            model_class, provided_kwargs
        )

        def build_entries(
            field_names: Sequence[str],
        ) -> List[Tuple[str, str, Optional[str]]]:
            entries: List[Tuple[str, str, Optional[str]]] = []
            for field in field_names:
                if field == "model_name":
                    entries.append(
                        (
                            "model_name",
                            format_value(self.model_name or model_class.__name__),
                            None,
                        )
                    )
                    continue

                if field == "model_class":
                    entries.append(
                        ("model_class", format_value(model_class.__name__), None)
                    )
                    continue

                if field not in resolved_kwargs:
                    continue

                marker = "default" if default_flags.get(field, False) else None
                entries.append((field, format_value(resolved_kwargs[field]), marker))
            return entries

        section_fields = OrderedDict(
            [
                ("creating_model", ("model_name", "model_class")),
                (
                    "creating_model.architecture",
                    (
                        "input_dims",
                        "n_classes",
                        "n_timesteps",
                        "data_presentation_pattern",
                        "feedforward_only",
                        "idle_timesteps",
                    ),
                ),
                (
                    "creating_model.temporal",
                    (
                        "dt",
                        "tau",
                        "dynamics_solver",
                        "t_feedforward",
                        "t_recurrence",
                        "t_feedback",
                        "t_skip",
                        "truncated_bptt_timesteps",
                    ),
                ),
                (
                    "creating_model.connectivity",
                    (
                        "recurrence_type",
                        "recurrence_target",
                        "skip",
                        "feedback",
                        "supralinearity",
                        "feedforward_only",
                    ),
                ),
                (
                    "creating_model.training",
                    (
                        "learning_rate",
                        "optimizer",
                        "optimizer_kwargs",
                        "optimizer_configs",
                        "lr_parameter_groups",
                        "scheduler",
                        "scheduler_kwargs",
                        "scheduler_configs",
                        "loss_reaction_time",
                        "retain_graph",
                        "non_label_index",
                    ),
                ),
                (
                    "creating_model.storage",
                    (
                        "store_responses",
                        "store_responses_on_cpu",
                        "store_train_responses",
                        "store_val_responses",
                        "store_test_responses",
                        "store_train_records",
                        "store_val_records",
                        "store_test_records",
                        "early_test_stop",
                    ),
                ),
            ]
        )

        seen_fields = set()
        for fields in section_fields.values():
            seen_fields.update(fields)

        for section_name, fields in section_fields.items():
            entries = build_entries(fields)
            if entries:
                log_section(run_logger, section_name, entries)

        custom_entries = []
        for key, value in resolved_kwargs.items():
            if key in seen_fields:
                continue
            marker = "default" if default_flags.get(key, False) else None
            custom_entries.append((key, format_value(value), marker))

        if custom_entries:
            log_section(run_logger, "creating_model.custom", custom_entries)

    def get_optimizer_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration for PyTorch Lightning configure_optimizers.

        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        config = {
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "optimizer_configs": self.optimizer_configs,
            "learning_rate": self.learning_rate,
            "lr_parameter_groups": self.lr_parameter_groups,
            "scheduler": self.scheduler,
            "scheduler_kwargs": self.scheduler_kwargs,
            "scheduler_configs": self.scheduler_configs,
        }
        # Filter out None values to allow optimizer/scheduler class defaults
        return {k: v for k, v in config.items() if v is not None}


# Example usage and testing
if __name__ == "__main__":
    # Test basic instantiation with minimal required fields
    try:
        model_params = ModelParams(
            model_name="DyRCNNx4",
            n_classes=10,
            input_dims=(20, 3, 224, 224),
            supralinearity=2.0,
        )
        print("Default model params created successfully")
        print(f"Default criterion_params: {model_params.criterion_params}")
    except Exception as e:
        print(f"Error creating basic model params: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # Test with custom biological parameters
    try:
        bio_params = ModelParams(
            model_name="DyRCNNx4",
            n_classes=100,
            input_dims=(25, 3, 128, 128),
            dt=1.0,
            tau=15.0,
            recurrence_type="full",
            supralinearity=2.5,
            learning_rate=0.001,
            optimizer="Adam",
        )
        print("Biological model params created successfully")
    except Exception as e:
        print(f"Error creating biological model params: {e}")
        import traceback

        traceback.print_exc()

    # Test loss configuration
    try:
        # Simple string loss
        simple_loss = ModelParams(
            model_name="TestModel",
            n_classes=5,
            input_dims=(15, 3, 64, 64),
            supralinearity=1.8,
            loss="CrossEntropyLoss",
        )
        print(f"Simple loss criterion_params: {simple_loss.criterion_params}")

        # List of losses
        multi_loss = ModelParams(
            model_name="TestModel",
            n_classes=8,
            input_dims=(30, 3, 32, 32),
            supralinearity=2.2,
            loss=["CrossEntropyLoss", "MSELoss"],
            criterion_params=[
                ("CrossEntropyLoss", {"weight": 1.0}),
                ("MSELoss", {"weight": 0.1}),
            ],
        )
        print(f"Multi loss criterion_params: {multi_loss.criterion_params}")

    except Exception as e:
        print(f"Error testing loss configurations: {e}")
        import traceback

        traceback.print_exc()

    # Test configuration file loading
    import tempfile
    import yaml
    import os

    config_data = {
        "model_name": "DyRCNNx4",
        "n_classes": 50,
        "input_dims": [20, 3, 128, 128],
        "dt": 1.5,
        "tau": 10.0,
        "t_feedforward": 2.0,
        "t_recurrence": 1.5,
        "recurrence_type": "full",
        "supralinearity": True,
        "learning_rate": 0.003,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss",
        "criterion_params": [
            ["CrossEntropyLoss", {"weight": 1.0, "ignore_index": -1}],
            ["ActivityLoss", {"weight": 0.1}],
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_config = f.name

    try:
        file_params = ModelParams.from_cli_and_config(config_path=temp_config)
        print("Config file params created successfully")
        print(
            f"Loaded loss criterion_params: {len(file_params.criterion_params)} loss functions"
        )

    except Exception as e:
        print(f"Error with config file loading: {e}")
        import traceback

        traceback.print_exc()
    finally:
        os.unlink(temp_config)

    # Test with custom model parameters
    try:
        custom_model_params = ModelParams(
            model_name="CustomRCNN",
            n_classes=12,
            input_dims=(18, 3, 96, 96),
            supralinearity=1.9,
            model_kwargs={
                "custom_layer_size": 256,
                "use_attention": True,
                "dropout_rate": 0.1,
            },
        )

        model_kwargs = custom_model_params.get_model_kwargs()
        print(
            f"Custom model kwargs include: custom_layer_size={model_kwargs.get('custom_layer_size')}"
        )

    except Exception as e:
        print(f"Error with custom model params: {e}")
        import traceback

        traceback.print_exc()
