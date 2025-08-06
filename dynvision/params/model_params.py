"""
Model parameter handling for DynVision using Pydantic.

This module provides type-safe parameter management for neural network models,
including biological parameters, architecture settings, optimizer/scheduler 
configuration, and loss functions.
"""

from pydantic import Field, field_validator, model_validator, ConfigDict
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
import logging
from pathlib import Path

from dynvision.params.base_params import BaseParams, DynVisionValidationError


class ModelParams(BaseParams):
    """
    Model parameters for neural network architecture and training configuration.

    Handles biological parameters, architecture settings, optimizer/scheduler
    configuration, loss functions, and model-specific parameters.
    """

    # ===== CORE ARCHITECTURE =====
    model_name: str = Field(
        default="None", description="Name of the model architecture"
    )
    n_classes: int = Field(default=10, description="Number of output classes", ge=1)
    input_dims: Tuple[int, ...] = Field(
        default=(1, 3, 224, 224),
        description="Input dimensions (timesteps, channels, height, width)",
    )
    n_timesteps: int = Field(
        default=1, description="Number of timesteps for model processing", ge=1
    )
    data_presentation_pattern: List[int] = Field(
        default=[1],
        description="Pattern for data presentation across timesteps (1 = present, 0 = absent)",
    )

    # ===== BIOLOGICAL PARAMETERS =====
    dt: float = Field(default=1.0, description="Integration time step (ms)", gt=0.0)
    tau: float = Field(default=10.0, description="Neural time constant (ms)", gt=0.0)
    t_feedforward: float = Field(
        default=0.0, description="Feedforward delay (ms)", ge=0.0
    )
    t_recurrence: float = Field(
        default=1.0, description="Recurrent delay (ms)", ge=1.0
    )
    t_feedback: float = Field(default=1.0, description="Feedback delay (ms)", ge=1.0)
    t_skip: float = Field(default=1.0, description="Skip delay (ms)", ge=1.0)
    dynamics_solver: Literal["euler", "rk4"] = Field(
        default="euler", description="Dynamical systems solver"
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
    ] = Field(default="none", description="Type of recurrent connections")

    # ===== CONNECTIVITY =====
    skip: bool = Field(
        default=False, description="Enable skip connections between layers"
    )
    feedback: bool = Field(
        default=False,
        description="Enable feedback connections from higher to lower layers",
    )

    # ===== NONLINEARITIES =====
    supralinearity: float = Field(
        default=1, description="Supralinearity exponent", gt=0.0
    )

    # ===== RESPONSE STORAGE =====
    store_responses: int = Field(
        default=0,
        description="Number of responses to store during evaluation (0 = disabled)",
        ge=-1,
        le=1000,
    )
    store_responses_on_cpu: bool = Field(
        default=True, description="Store responses on CPU to save GPU memory"
    )
    classifier_name: str = Field(
        default="classifier", description="Name of the classifier layer"
    )

    # ===== LOSS CONFIGURATION =====
    loss: Union[str, List[str]] = Field(
        default="CrossEntropyLoss",
        description="Loss function name or list of loss functions",
    )
    loss_configs: Dict[str, Dict] = Field(
        default_factory=dict, description="Configurations for the loss function"
    )
    loss_reaction_time: float = Field(
        default=0.0, description="Reaction time for loss calculation (ms)", ge=0.0
    )
    # ===== OPTIMIZER CONFIGURATION =====
    learning_rate: float = Field(
        default=0.001, description="Learning rate for optimizer", gt=0.0
    )
    optimizer: str = Field(
        default="Adam", description="Optimizer type (Adam, SGD, AdamW, etc.)"
    )
    optimizer_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional optimizer arguments"
    )
    optimizer_configs: Dict[str, Any] = Field(
        default_factory=dict, description="Optimizer configuration dictionary"
    )

    # ===== LEARNING RATE PARAMETER GROUPS =====
    lr_parameter_groups: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Learning rate factors for different parameter groups",
    )

    # ===== SCHEDULER CONFIGURATION =====
    scheduler: str = Field(
        default="StepLR", description="Learning rate scheduler type"
    )
    scheduler_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Scheduler arguments"
    )
    scheduler_configs: Dict[str, Any] = Field(
        default_factory=dict, description="Scheduler configuration"
    )

    # ===== TRAINING BEHAVIOR =====
    retain_graph: bool = Field(
        default=False,
        description="Whether to retain computation graph for backward pass",
    )
    non_label_index: int = Field(
        default=-1, description="Index for non-label timesteps"
    )

    # ===== CUSTOM MODEL PARAMETERS =====
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific arguments"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,  # Validate fields when assigned after creation
        use_enum_values=True,  # Use enum values in serialization
        validate_by_name=True,  # Allow validation using field names and aliases
    )

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
            }
        )
        return aliases

    @field_validator("store_responses")
    def convert_store_responses(cls, v) -> int:
        """Convert store_responses to int, handling string inputs."""
        if isinstance(v, str):
            if v.lower() == "all":
                return -1  # Use -1 to indicate "all responses"
            return int(v)
        return int(v)

    @field_validator("recurrence_type")
    def validate_recurrence_type(cls, v):
        if v is None:
            return "none"
        return v

    @field_validator("loss")
    def validate_loss(cls, v) -> list:
        if isinstance(v, list):
            return v
        else:
            return [v]

    @field_validator("optimizer")
    def validate_optimizer(cls, v):
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
        valid_models = [
            "DyRCNNx4",
            "AlexNet",
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "CORnet-RT",
            "CordsNet",
            "BLT",
            "TwoLayerCNN",
        ]
        if v not in valid_models:
            logging.warning(
                f"Unknown model: {v}. Consider adding to valid_models list"
            )
        return v

    @field_validator("input_dims")
    def validate_input_dims(cls, v):
        """Validate input dimensions."""
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
        """Set up defaults and validate all constraints."""
        if not self.loss_configs:
            object.__setattr__(
                self,
                "loss_configs",
                {
                    "CrossEntropyLoss": {"weight": 1.0, "ignore_index": -1},
                    "EnergyLoss": {"weight": 100},
                },
            )
        for loss in self.loss:
            if loss not in self.loss_configs:
                logging.warning(f"Loss configuration for '{loss}' not found.")

        # Provide default lr_parameter_groups if empty
        if not self.lr_parameter_groups:
            object.__setattr__(
                self,
                "lr_parameter_groups",
                {
                    "regular": {"lr_factor": 1.0},
                    "recurrent": {"lr_factor": 1.0},
                    "feedback": {"lr_factor": 1.0},
                },
            )
            logging.info("Using default lr_parameter_groups")

        # Validate lr_parameter_groups
        for group_name, group_config in self.lr_parameter_groups.items():
            if "lr_factor" not in group_config:
                group_config["lr_factor"] = 1.0
                logging.info(f"Added missing lr_factor=1.0 to group '{group_name}'")

        # Provide default scheduler configs
        if not self.scheduler_configs:
            default_configs = {
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "learning_rate",
            }
            object.__setattr__(self, "scheduler_configs", default_configs)
            logging.info("Using default scheduler_configs")

        # Provide default scheduler kwargs based on scheduler type
        if not self.scheduler_kwargs:
            if self.scheduler == "StepLR":
                object.__setattr__(
                    self, "scheduler_kwargs", {"step_size": 30, "gamma": 0.1}
                )
                logging.info("Using default StepLR scheduler_kwargs")
            elif self.scheduler == "CosineAnnealingLR":
                object.__setattr__(self, "scheduler_kwargs", {"T_max": 100})
                logging.info("Using default CosineAnnealingLR scheduler_kwargs")

        # Scheduler configuration consistency
        if self.scheduler == "ReduceLROnPlateau":
            if "monitor" not in self.scheduler_configs:
                self.scheduler_configs["monitor"] = "val_loss"

        # Biological timing constraints (informational warnings only)
        if self.dt >= self.tau:
            logging.warning(
                f"dt ({self.dt}) should be much smaller than tau ({self.tau}) "
                "for numerical stability (typically dt < tau/10)"
            )

        # Delay constraints
        if self.t_feedforward < 0 or self.t_recurrence < 1:
            raise DynVisionValidationError(
                "t_feedforward must be non-negative, t_recurrence must be at least 1."
            )

        # Check if delays are exact multiples of dt (informational)
        if self.t_feedforward % self.dt != 0:
            steps = int(self.t_feedforward / self.dt)
            logging.info(
                f"t_feedforward ({self.t_feedforward}ms) rounds to {steps} timesteps "
                f"(dt={self.dt}ms)"
            )

        if self.t_recurrence % self.dt != 0:
            steps = int(self.t_recurrence / self.dt)
            logging.info(
                f"t_recurrence ({self.t_recurrence}ms) rounds to {steps} timesteps "
                f"(dt={self.dt}ms)"
            )

        # Architecture consistency checks
        if self.supralinearity != 1 and self.recurrence_type == "none":
            logging.warning(
                "Supralinearity enabled but no recurrence - may cause instability"
            )

        # Input dimensions consistency
        if len(self.input_dims) == 4:
            timesteps_from_dims = self.input_dims[0]
            if timesteps_from_dims > 1 and timesteps_from_dims != self.n_timesteps:
                logging.warning(
                    f"n_timesteps ({self.n_timesteps}) doesn't match "
                    f"input_dims timesteps ({timesteps_from_dims})"
                )

        # Input/output consistency
        if self.n_classes < 2:
            logging.warning(
                f"n_classes ({self.n_classes}) should be >= 2 for meaningful classification"
            )

        return self

    # ===== COMPUTED PROPERTIES (DERIVED PARAMETERS) =====

    @property
    def delay_feedforward(self) -> int:
        """Number of timesteps for feedforward delay."""
        return int(self.t_feedforward / self.dt)

    @property
    def delay_recurrence(self) -> int:
        """Number of timesteps for recurrence delay."""
        return int(self.t_recurrence / self.dt)

    @property
    def stability_ratio(self) -> float:
        """Ratio dt/tau for numerical stability assessment."""
        return self.dt / self.tau

    @property
    def criterion_params(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get the parameters for the criterion used in the model."""
        if not isinstance(self.loss, list):
            self.loss = [self.loss]
        return [(l, self.loss_configs[l]) for l in self.loss]

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

        Args:
            model_class: The model class to filter kwargs for

        Returns:
            Dictionary of kwargs suitable for model instantiation
        """
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
            "recurrence_type": self.recurrence_type,
            "skip": self.skip,
            "feedback": self.feedback,
            "supralinearity": self.supralinearity,
            "store_responses": self.store_responses,
            "store_responses_on_cpu": self.store_responses_on_cpu,
            "classifier_name": self.classifier_name,
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
        model_kwargs.update(self.model_kwargs)

        # Add extra fields if any
        if hasattr(self, "__pydantic_extra__"):
            model_kwargs.update(self.__pydantic_extra__)

        # Filter kwargs if model_class is provided
        if model_class is not None:
            try:
                from dynvision.utils import filter_kwargs

                known, unknown = filter_kwargs(model_class, model_kwargs)

                if known:
                    logging.info(
                        f"Filtered known model kwargs for {model_class.__name__}: {list(known.keys())}"
                    )

                return known

            except ImportError:
                logging.warning("filter_kwargs not available, returning all kwargs")
                return model_kwargs

        return model_kwargs

    def get_optimizer_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration for PyTorch Lightning configure_optimizers.

        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        return {
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "optimizer_configs": self.optimizer_configs,
            "learning_rate": self.learning_rate,
            "lr_parameter_groups": self.lr_parameter_groups,
            "scheduler": self.scheduler,
            "scheduler_kwargs": self.scheduler_kwargs,
            "scheduler_configs": self.scheduler_configs,
        }


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
        print(f"Default model params created successfully")
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
        print(f"Biological model params created successfully")
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
        "supralinearity": 2.5,
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
            ["EnergyLoss", {"weight": 0.1}],
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_config = f.name

    try:
        file_params = ModelParams.from_cli_and_config(config_path=temp_config)
        print(f"Config file params created successfully")
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
