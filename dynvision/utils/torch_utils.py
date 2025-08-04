"""PyTorch utility functions for the DynVision toolbox.

This module provides PyTorch-specific utilities:
- Random seed setting
- Device management
- Dtype management
- Context management for device/dtype consistency
"""

import logging
import random
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Union, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.amp import autocast
from torch import nn


logger = logging.getLogger(__name__)


def set_seed(seed: Union[str, int]) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
    - PyTorch CPU and CUDA
    - NumPy
    - Python random
    - PyTorch Lightning

    Also configures CUDA for deterministic behavior.

    Args:
        seed: Random seed value (string or integer)
    """
    # Validate and convert seed
    if isinstance(seed, str):
        seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)
    return None


def ensure_same_device(
    *args: Union[torch.Tensor, torch.nn.Module],
    target_device: Optional[torch.device] = None,
    non_blocking: bool = True,
    label: Optional[str] = None,
    **kwargs: Union[torch.Tensor, torch.nn.Module],
) -> Tuple[tuple, dict]:
    """
    Ensure all variables (tensors or modules) are on the same device.

    Args:
        args: Positional tensor or module arguments to check and align.
        kwargs: Keyword tensor or module arguments to check and align.
        target_device: The target device to move tensors/modules to. If None, defaults to CUDA if available.
        non_blocking: Whether to perform non-blocking memory transfers.
        label: Optional label for logging context.

    Returns:
        Tuple of (args, kwargs) with variables moved to the same device.
    """
    # Collect devices
    devices = set()

    def process_device(var):
        if isinstance(var, torch.Tensor):
            devices.add(var.device)
        elif isinstance(var, torch.nn.Module):
            for param in var.parameters():
                devices.add(param.device)

    for var in args:
        process_device(var)
    for var in kwargs.values():
        process_device(var)

    def are_devices_same(device1: torch.device, device2: torch.device) -> bool:
        """Compare two devices considering that 'cuda' equals 'cuda:0'."""
        if device1.type != device2.type:
            return False
        if device1.type == "cuda":
            # If either device is just 'cuda', treat it as 'cuda:0'
            index1 = device1.index if device1.index is not None else 0
            index2 = device2.index if device2.index is not None else 0
            return index1 == index2
        if "cuda" in device1.type:
            return device1.index == device2.index
        return True

    # Determine and normalize target device
    if target_device is None:
        target_device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(target_device_name)
    elif isinstance(target_device, str):
        if target_device == "cuda":
            target_device = torch.device("cuda:0")
        else:
            target_device = torch.device(target_device)
    elif isinstance(target_device, torch.device):
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda:0")
    else:
        raise ValueError(f"Invalid target_device type: {type(target_device)}")

    # Handle device mismatches
    if len(devices) > 1:
        label_text = f" in {label}" if label else ""
        logger.warning(f"Variables are on different devices{label_text}")

        # Log device info for each variable
        for i, var in enumerate(args):
            if isinstance(var, torch.Tensor):
                logger.info(f"param_{i}: {var.device}")
            elif isinstance(var, torch.nn.Module) and len(list(var.parameters())) > 0:
                logger.info(f"param_{i}: {next(var.parameters()).device}")
        for name, var in kwargs.items():
            if isinstance(var, torch.Tensor):
                logger.info(f"{name}: {var.device}")
            elif isinstance(var, torch.nn.Module) and len(list(var.parameters())) > 0:
                logger.info(f"{name}: {next(var.parameters()).device}")

        logger.info(f"Moving variables to {target_device} device")

        def move_to_device(var):
            if isinstance(var, torch.Tensor):
                return var.to(device=target_device, non_blocking=non_blocking)
            elif isinstance(var, torch.nn.Module):
                # Force all parameters to the target device
                var = var.to(device=target_device)
                # Double check all parameters
                for param in var.parameters():
                    if param.device != target_device:
                        param.data = param.data.to(
                            device=target_device, non_blocking=non_blocking
                        )
                return var
            return var

        args = tuple(move_to_device(var) for var in args)
        kwargs = {name: move_to_device(var) for name, var in kwargs.items()}

        # Final consistency check
        inconsistent_vars = []

        def check_device(var, name=None):
            if isinstance(var, torch.Tensor):
                if not are_devices_same(var.device, target_device):
                    inconsistent_vars.append(
                        f"{'tensor' if name is None else name}: expected {target_device}, got {var.device}"
                    )
            elif isinstance(var, torch.nn.Module):
                for param_name, param in var.named_parameters():
                    if not are_devices_same(param.device, target_device):
                        inconsistent_vars.append(
                            f"parameter {param_name}: expected {target_device}, got {param.device}"
                        )

        for i, var in enumerate(args):
            check_device(var, f"arg_{i}")
        for name, var in kwargs.items():
            check_device(var, name)

        if inconsistent_vars:
            label_text = f" in {label}" if label else ""
            details = "\n  - " + "\n  - ".join(inconsistent_vars)
            logging.warning(
                f"After device alignment, some variables are still not on expected device{label_text}:"
                f"{details}",
                RuntimeWarning,
            )

    return args, kwargs


def determine_target_dtype(
    *args: Union[torch.Tensor, torch.nn.Module],
    target_dtype: Optional[torch.dtype] = None,
    label: Optional[str] = None,
    default_dtype: torch.dtype = torch.bfloat16,
    **kwargs: Union[torch.Tensor, torch.nn.Module],
) -> torch.dtype:
    """
    Determine the target dtype from a collection of tensors and modules.

    Args:
        args: Positional tensor or module arguments to analyze.
        kwargs: Keyword tensor or module arguments to analyze.
        target_dtype: If provided, returns this dtype directly.
        label: Optional label for logging context.

    Returns:
        torch.dtype: The determined target dtype.
    """
    # If target dtype is explicitly provided, return it
    if target_dtype is not None:
        return target_dtype

    for var in list(args) + list(kwargs.values()):
        if hasattr(var, "get_target_dtype"):
            try:
                lightning_dtype = var.get_target_dtype()
                logger.info(
                    f"Using Lightning target dtype: {lightning_dtype} for {label}"
                )
                return lightning_dtype
            except:
                pass

    return default_dtype


def ensure_same_dtype(
    *args: Union[torch.Tensor, torch.nn.Module],
    target_dtype: Optional[torch.dtype] = None,
    label: Optional[str] = None,
    **kwargs: Union[torch.Tensor, torch.nn.Module],
) -> Tuple[tuple, dict]:
    """
    Ensure all variables (tensors or modules) have the same dtype.

    Args:
        args: Positional tensor or module arguments to check and align.
        kwargs: Keyword tensor or module arguments to check and align.
        target_dtype: The target dtype to cast tensors to. If None, uses most common dtype.
        label: Optional label for logging context.

    Returns:
        Tuple of (args, kwargs) with variables cast to the same dtype.
    """
    # Determine the target dtype
    target_dtype = determine_target_dtype(
        *args, target_dtype=target_dtype, label=label, **kwargs
    )

    # Cast function
    def cast_to_dtype(var):
        if isinstance(var, torch.Tensor):
            if var.dtype != target_dtype:
                logger.info(f"Casting: {var.dtype} -> {target_dtype}")
                return var.to(dtype=target_dtype)
        elif isinstance(var, torch.nn.Module):
            if len(list(var.parameters())) > 0:
                param_dtype = next(var.parameters()).dtype
                if param_dtype != target_dtype:
                    logger.info(
                        f"Casting module parameters: {param_dtype} -> {target_dtype}"
                    )
                    return var.to(dtype=target_dtype)
        return var

    # Cast all variables
    new_args = tuple(cast_to_dtype(var) for var in args)
    new_kwargs = {name: cast_to_dtype(var) for name, var in kwargs.items()}

    return new_args, new_kwargs


@contextmanager
def on_same_device(
    *args: Union[torch.Tensor, torch.nn.Module],
    mixed_precision: bool = True,
    non_blocking: bool = True,
    target_device: Optional[torch.device] = None,
    target_dtype: Optional[torch.dtype] = None,
    label: Optional[str] = None,
    **kwargs: Union[torch.Tensor, torch.nn.Module],
):
    """
    Ensure all variables (tensors or modules) are on the same device and optionally the same dtype.
    Device consistency is always enforced, while dtype casting is optional.

    Args:
        args: Positional tensor or module arguments to check and align.
        kwargs: Keyword tensor or module arguments to check and align.
        mixed_precision: Whether to enable mixed precision autocasting.
        non_blocking: Whether to perform non-blocking memory transfers.
        target_device: The target device to move tensors/modules to. If None, defaults to CUDA if available.
        target_dtype: Optional target dtype to cast tensors to. If None, original dtypes are preserved.
        label: Optional label for logging context.

    Yields:
        Tuple of args and kwargs with variables moved to the same device and optionally same dtype.
    """
    # First ensure same device (mandatory)
    args, kwargs = ensure_same_device(
        *args,
        target_device=target_device,
        non_blocking=non_blocking,
        label=label,
        **kwargs,
    )

    # Then handle dtype if specified (optional)
    if target_dtype is not None:
        args, kwargs = ensure_same_dtype(
            *args, target_dtype=target_dtype, label=label, **kwargs
        )

    # Get the actual device name for autocast
    if target_device is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_name = str(target_device)

    with autocast(device_name, enabled=mixed_precision):
        yield args, kwargs


def apply_parametrization(
    module: nn.Module,
    parametrization: Optional[Union[Callable[[nn.Module], nn.Module], str]] = None,
) -> nn.Module:

    if parametrization is None:
        return module
    elif callable(parametrization):
        return parametrization(module)
    elif isinstance(parametrization, str):
        if parametrization == "identity":
            return module
        elif hasattr(nn.utils.parametrizations, parametrization):
            parametrization_fn = getattr(nn.utils.parametrizations, parametrization)
            return parametrization_fn(module)
        else:
            raise ValueError(f"Unknown parametrization: {parametrization}")
    else:
        raise TypeError("Parametrization must be a callable, string, or None.")


def get_effective_dtype_from_precision(precision: str) -> str:
    """
    Get the actual dtype that PyTorch Lightning will use for a given precision.

    This matches Lightning's internal logic to avoid dtype mismatches.
    """
    # Lightning's actual behavior for mixed precision
    if precision in ["16-mixed", "bf16-mixed"]:
        # Lightning prefers bfloat16 for mixed precision on supported hardware
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
        else:
            return "float16"

    # Direct mappings for non-mixed precision
    precision_to_dtype = {
        "16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "32": "float32",
        "64": "float64",
    }

    return precision_to_dtype.get(precision, "float32")

def calculate_conv_out_dim(in_dim: int, kernel_size: int, padding: int, stride: int = 1) -> int:
    out_dim = int((in_dim + 2 * padding - kernel_size) / stride) + 1
    return out_dim