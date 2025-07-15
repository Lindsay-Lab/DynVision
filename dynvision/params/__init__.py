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

from .base_params import BaseParams, DynVisionValidationError, DynVisionConfigError
from .model_params import ModelParams
from .data_params import DataParams
from .trainer_params import TrainerParams
from .init_params import InitParams
from .training_params import TrainingParams
from .testing_params import TestingParams
