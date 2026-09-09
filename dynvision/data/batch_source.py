"""BatchSource seam: single entry point for backend-specific data loading.

Collapses the ``use_ffcv`` branching that used to be duplicated across
:class:`dynvision.data.datamodule.DataModule` call sites into one seam, and
is the *only* place ``ffcv`` may be imported. All ``ffcv`` imports here are
deferred (inside methods/functions), so importing this module — and anything
that imports it, such as :mod:`dynvision.data.datamodule` — never requires
ffcv to be installed. ffcv is only required when :class:`FFCVAdapter` is
actually instantiated (i.e. when ``use_ffcv=True``).

See ``docs/development/planning/ffcv-batch-source-seam.md`` and
`Lindsay-Lab/DynVision#16 <https://github.com/Lindsay-Lab/DynVision/issues/16>`_.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Union

logger = logging.getLogger(__name__)

try:
    from ffcv.loader import OrderOption
except ImportError:  # pragma: no cover - ffcv is an optional dependency
    from enum import IntEnum

    class OrderOption(IntEnum):
        """Stand-in for ffcv.loader.OrderOption when ffcv is not installed.

        Mirrors ffcv's enum so callers (e.g. ``DataParams``) can
        validate/store an ``order`` field without requiring ffcv unless
        ``use_ffcv=True`` is actually used.
        """

        SEQUENTIAL = 0
        RANDOM = 1
        QUASI_RANDOM = 2


FFCV_INSTALL_HINT = (
    "use_ffcv=True requires the 'ffcv' package, which is not importable in this "
    "environment. Install it (see docs/development/dependencies for "
    "platform-specific build instructions) or set use_ffcv=False to use the "
    "PyTorch data loading backend instead."
)


class BatchSource(ABC):
    """Adapter interface for building a train/val/preview DataLoader.

    Implementations own exactly one backend (torch or ffcv) and are
    responsible for importing that backend lazily, so selecting a
    :class:`BatchSource` never requires backends other than the one selected.
    """

    @property
    @abstractmethod
    def loader_class(self) -> Callable[..., Any]:
        """Return the underlying loader-construction callable.

        Used for kwarg filtering/logging (``DataParams.get_dataloader_kwargs``,
        ``DataInterface._log_dataloader_creation``) without necessarily
        constructing a loader.
        """

    @abstractmethod
    def create_loader(self, path: Union[str, Path], **config: Any) -> Any:
        """Construct and return a DataLoader-like object for ``path``."""


class TorchAdapter(BatchSource):
    """BatchSource backed by :mod:`dynvision.data.dataloader` (always available)."""

    @property
    def loader_class(self) -> Callable[..., Any]:
        from dynvision.data.dataloader import get_data_loader

        return get_data_loader

    def create_loader(self, path: Union[str, Path], **config: Any) -> Any:
        from dynvision.data.dataloader import get_data_loader

        return get_data_loader(path, **config)


class FFCVAdapter(BatchSource):
    """BatchSource backed by :mod:`dynvision.data.ffcv_dataloader`.

    Raises a clear :class:`ImportError` at construction time if ffcv is not
    importable, instead of letting an unrelated crash surface deep inside
    ``ffcv_dataloader``/``ffcv_operations`` when the loader is later used.
    """

    def __init__(self) -> None:
        try:
            import ffcv  # noqa: F401
        except ImportError as exc:
            raise ImportError(FFCV_INSTALL_HINT) from exc

    @property
    def loader_class(self) -> Callable[..., Any]:
        from dynvision.data.ffcv_dataloader import get_ffcv_dataloader

        return get_ffcv_dataloader

    def create_loader(self, path: Union[str, Path], **config: Any) -> Any:
        from dynvision.data.ffcv_dataloader import get_ffcv_dataloader

        return get_ffcv_dataloader(path=path, **config)


def get_batch_source(use_ffcv: bool) -> BatchSource:
    """Select the batch-loading adapter for the given configuration.

    Args:
        use_ffcv: Whether FFCV-backed data loading was requested.

    Returns:
        A :class:`FFCVAdapter` if ``use_ffcv`` is True, else a
        :class:`TorchAdapter`.

    Raises:
        ImportError: if ``use_ffcv=True`` but ffcv is not importable.
    """
    if use_ffcv:
        return FFCVAdapter()
    return TorchAdapter()


__all__ = [
    "OrderOption",
    "BatchSource",
    "TorchAdapter",
    "FFCVAdapter",
    "get_batch_source",
    "FFCV_INSTALL_HINT",
]
