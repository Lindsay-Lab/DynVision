"""
# Parameter Management Strategy

1. Base Parameters & Validation → Pydantic Classes

- Type checking and range validation
- Cross-parameter biological constraints
- Configuration file loading and CLI parsing

2. Derived Parameters & Warnings → Pydantic Computed Properties

- Calculated values like delay_ff = int(t_feedforward / dt)
- Consistency warnings (t_feedforward not multiple of dt)
- Parameter preprocessing and normalization

3. Architecture Implementation → Model Classes

- Layer creation and weight initialization
- Computational graph setup
- Model-specific parameter interpretation

4. Runtime Adaptation → Model Classes

- Dynamic adjustments based on actual data shapes
- Device-specific optimizations
- Context-dependent modifications
"""

from .base_params import (  # noqa: F401
    BaseParams,
    DynVisionValidationError,
    DynVisionConfigError,
)
from .composite_params import CompositeParams  # noqa: F401
from .model_params import ModelParams  # noqa: F401
from .data_params import DataParams  # noqa: F401
from .trainer_params import TrainerParams  # noqa: F401
from .init_params import InitParams  # noqa: F401
from .training_params import TrainingParams  # noqa: F401
from .testing_params import TestingParams  # noqa: F401
