# usage: integration_strategy: Union[Callable, str] = "additive",

from typing import Union, Callable
import torch

__all__ = ["setup_integration_strategy"]


def _additive(x, h):
    return x + h


def _multiplicative(x, h):
    return x * (1 + torch.tanh(h))


def _none(x, h):
    return x


def setup_integration_strategy(strategy: Union[Callable, str]) -> None:
    if isinstance(strategy, str):
        if strategy == "additive":
            return _additive
        elif strategy == "multiplicative":
            return _multiplicative
        elif strategy == "none" or strategy is None:
            return _none
        else:
            raise ValueError(f"Invalid recurrence strategy: {strategy}")
    else:
        return strategy
