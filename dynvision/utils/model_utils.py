"""Model utility functions for the DynVision toolbox.

This module provides model-related utilities:
- Error handling and logging
- Model stability checking
- Weight analysis and validation
"""

import logging
import traceback
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Tuple, Optional
from pathlib import Path
from .config_utils import filter_kwargs

import numpy as np
import torch

logger = logging.getLogger(__name__)


def handle_errors(verbose: bool = False) -> Callable:
    """Decorator to handle errors consistently across functions.

    Wraps functions with try-except that catches exceptions, logs them,
    and optionally prints full traceback before re-raising as ValueError.

    Args:
        verbose: Whether to print full Python traceback on error

    Returns:
        Decorated function with error handling

    Example:
        @handle_errors(verbose=True)
        def my_function():
            # Function code here
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if verbose:
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise ValueError(f"Error in {func.__name__}: {str(e)}") from e

        return wrapper

    return decorator


def check_stability(x: torch.Tensor, stage: str = "") -> None:
    """Check numerical stability of activations.

    Args:
        x: Tensor to check for stability
        stage: Optional name of the stage being checked (for logging)
    """
    if torch.isnan(x).any():
        logger.error(f"NaN values detected in {stage} output")
        # raise ValueError("Numerical instability detected")
    if torch.isinf(x).any():
        logger.error(f"Inf values detected in {stage} output")
        # raise ValueError("Numerical instability detected")


def check_weights(model, message="", min=-2, max=2):
    """Check model weights for potential issues.

    Analyzes weight distributions and identifies problematic values:
    - Values outside specified range
    - NaN values
    - Extremely small/large values

    Args:
        model: PyTorch model to check
        message: Optional message to display with results
        min: Minimum acceptable weight value
        max: Maximum acceptable weight value

    Returns:
        Tuple of (weight_info, contain_nan):
        - weight_info: Dictionary mapping layer names to weight statistics
        - contain_nan: Boolean indicating if any NaN values were found
    """
    layer_names = model.state_dict().keys()
    layer_names = [name.rstrip(".weight") for name in layer_names]
    weight_info = {}
    contain_nan = False

    for layer in model.state_dict().keys():
        layer_name = layer.rstrip(".weight")
        weights = model.state_dict()[layer].numpy().flatten()
        n_weights = len(weights)
        min_value = np.nanmin(weights)
        max_value = np.nanmax(weights)
        norm = np.linalg.norm(weights)
        weight_info[layer_name] = SimpleNamespace(
            min=min_value, max=max_value, norm=norm
        )

        is_nan = ~np.isfinite(weights)
        is_large = np.abs(weights) > max
        is_small = np.abs(weights) < min
        is_bad = is_nan.any() or is_small.any() or is_large.any()

        if is_bad:
            print(message)
            print(f"Layer: {layer_name}")
            if is_large.any():
                n_large = np.sum(is_large.astype(int))
                print(f"\t{n_large}/{n_weights} large weights (>{max:2f})")
            if is_small.any():
                n_small = np.sum(is_small.astype(int))
                print(f"\t{n_small}/{n_weights} small weights (<{min:3f})")
            if is_nan.any():
                n_nans = np.sum(is_nan.astype(int))
                print(f"\t{n_nans}/{n_weights} NaN weights")
                contain_nan = True

    return weight_info, contain_nan


@handle_errors(verbose=False)
def load_model_and_weights(
    model_name: str,
    state_dict_path: Path,
    config: Any,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, int]:
    """Load the model and its weights.

    Args:
        model_name: Name of the model class
        state_dict_path: Path to the saved model weights
        config: Configuration object containing model parameters

    Returns:
        Tuple containing:
            - Loaded model instance
            - Number of classes

    Raises:
        ValueError: If model loading fails
    """
    # Lazy import to avoid circular dependency
    from dynvision import models

    state_dict = torch.load(state_dict_path, map_location=device)
    if not len(state_dict):
        raise ValueError(f"State dict is empty: {state_dict_path}")

    last_key = next(reversed(state_dict))
    n_classes = len(state_dict[last_key])

    model_class = getattr(models, model_name)
    model_args, _ = filter_kwargs(model_class, vars(config))
    model_args.update({"n_classes": n_classes})

    model = model_class(**model_args)

    if device is not None:
        model = model.to(device)

    model.load_state_dict(state_dict)

    return model
