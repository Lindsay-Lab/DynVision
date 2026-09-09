"""Shared DataModule helpers for all DynVision workflows.

This module centralizes dataset/dataloader wiring for every runtime entry point:

* :class:`DataModule` powers training via ``runtime/train_model.py``.  It accepts a
    full :class:`dynvision.params.TrainingParams` instance, builds PyTorch or FFCV
    loaders from :class:`dynvision.params.data_params.DataParams`, and logs every
    dataset/dataloader mutation so Lightning sees the exact configuration used in
    fit/validation.
* :class:`SimpleDataModule` trims the interface down to a single dataset path and
    is used for lightweight preview/initialization flows (``runtime/init_model.py``)
    to infer tensor shapes directly from the same parameter-derived kwargs.
* :class:`TestingDataModule` extends ``SimpleDataModule`` with sampler handling and
    richer logging so ``runtime/test_model.py`` reuses the identical preview + active
    loader creation logic when validating checkpoints.

By routing every workflow through these classes, parameter updates happen inside
``DataParams`` helpers (``get_dataset_kwargs()``, ``get_preview_dataloader_kwargs()``,
``get_validation_dataloader_kwargs()``), ensuring the config diffs shown in logs
match the actual objects handed to Lightning/Snakerules and keeping dataset access
consistent between init/train/test steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from dynvision.data.batch_source import get_batch_source
from dynvision.data.dataloader import (
    _adjust_data_dimensions,
    _adjust_label_dimensions,
    get_data_loader,
    get_data_loader_class,
    get_train_val_loaders,
)
from dynvision.data.datasets import get_dataset
from dynvision.data import sampler
from dynvision.params import DynVisionConfigError
from dynvision.utils import format_value, log_section

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dynvision.params import TrainingParams

logger = logging.getLogger(__name__)


class DataInterface:
    """Shared dataset/dataloader logging utilities with preview diff tracking."""

    def __init__(self, config: Any):
        self.config = config
        self._preview_dataset_kwargs: Optional[Dict[str, Any]] = None
        self._preview_dataset_path: Optional[Path] = None
        self._preview_dataloader_kwargs: Optional[Dict[str, Any]] = None
        self._preview_dataloader_class: Optional[Any] = None

    def _log_dataloader_creation(
        self,
        *,
        dataloader_class: Any,
        dataloader_kwargs: Dict[str, Any],
        context: str,
    ) -> None:
        """Log dataloader metadata and capture preview kwargs for diffs."""

        previous_kwargs = None
        if context != "preview" and self._preview_dataloader_kwargs is not None:
            if self._preview_dataloader_class is dataloader_class:
                previous_kwargs = self._preview_dataloader_kwargs
            self._preview_dataloader_kwargs = None
            self._preview_dataloader_class = None

        self.config.log_dataloader_creation(
            dataloader_class=dataloader_class,
            dataloader_kwargs=dataloader_kwargs,
            logger=logger,
            context=context,
            previous_kwargs=previous_kwargs,
            level=logging.DEBUG if context == "preview" else logging.INFO,
        )

        if context == "preview":
            self._preview_dataloader_kwargs = dataloader_kwargs.copy()
            self._preview_dataloader_class = dataloader_class

    def _log_dataset_creation(
        self,
        *,
        dataset_path: Path,
        dataset_kwargs: Dict[str, Any],
        context: str,
    ) -> None:
        previous_kwargs = None
        previous_path = None
        if context != "preview" and self._preview_dataset_kwargs is not None:
            previous_kwargs = self._preview_dataset_kwargs
            previous_path = self._preview_dataset_path

        self.config.data.log_dataset_creation(
            dataset_path=dataset_path,
            dataset_kwargs=dataset_kwargs,
            logger=logger,
            context=context,
            previous_kwargs=previous_kwargs,
            previous_dataset_path=previous_path,
            level=logging.DEBUG if context == "preview" else logging.INFO,
        )

        if context == "preview":
            self._preview_dataset_kwargs = dataset_kwargs.copy()
            self._preview_dataset_path = dataset_path
        else:
            self._preview_dataset_kwargs = None
            self._preview_dataset_path = None

    def create_dataset(
        self,
        *,
        dataset_path: Path,
        dataset_kwargs: Dict[str, Any],
        context: str,
    ) -> Dataset:
        self._log_dataset_creation(
            dataset_path=dataset_path,
            dataset_kwargs=dataset_kwargs,
            context=context,
        )
        return get_dataset(
            data_path=dataset_path,
            **dataset_kwargs,
        )

    def create_dataloader(
        self,
        *,
        dataset: Dataset,
        dataloader: Optional[Union[str, type]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        context: str = "active",
    ) -> DataLoader:
        dataloader_kwargs = dataloader_kwargs or {}
        loader = get_data_loader(
            dataset=dataset,
            dataloader=dataloader,
            **dataloader_kwargs,
        )
        logging_class = (
            get_data_loader_class(dataloader)
            if isinstance(dataloader, str)
            else loader.__class__
        )
        self._log_dataloader_creation(
            dataloader_class=logging_class,
            dataloader_kwargs=dataloader_kwargs,
            context=context,
        )
        return loader


class DataModule(DataInterface, pl.LightningDataModule):
    """Enhanced DataModule with unified dataloader creation and distributed handling."""

    def __init__(self, config: "TrainingParams"):
        pl.LightningDataModule.__init__(self)
        DataInterface.__init__(self, config)
        self.train_loader = None
        self.val_loader = None
        self._preview_loader = None

        # Validate required parameters early
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.data.use_ffcv:
            required_paths = ["dataset_train", "dataset_val"]
            missing = [p for p in required_paths if not getattr(self.config, p, None)]
            if missing:
                raise DynVisionConfigError(f"FFCV mode requires: {missing}")
        else:
            if not getattr(self.config, "dataset_link", None):
                raise DynVisionConfigError("PyTorch mode requires dataset_link")

    def create_preview_loader(self) -> DataLoader:
        """Create a minimal loader for dimension inference before trainer setup."""
        if self._preview_loader is not None:
            return self._preview_loader

        # Create minimal config for preview
        preview_config = self.config.data.get_preview_dataloader_kwargs()

        if self.config.data.use_ffcv:
            self._preview_loader = self._create_ffcv_loader(
                self.config.dataset_train,
                preview_config,
                context="preview",
            )
        else:
            self._preview_loader = self._create_pytorch_loader(preview_config)

        return self._preview_loader

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up data loaders with proper distributed configuration."""
        if stage not in ["fit", None]:
            return

        # Get dataloader config - None values already filtered by get_dataloader_kwargs()
        # Additional filtering happens in get_data_loader() and get_train_val_loaders()
        dataloader_config = self.config.data.get_dataloader_kwargs()

        if self.config.data.use_ffcv:
            self._setup_ffcv_loaders(dataloader_config)
        else:
            self._setup_pytorch_loaders(dataloader_config)

    def _setup_ffcv_loaders(self, config: Dict[str, Any]) -> None:
        """Set up FFCV data loaders."""
        train_kwargs = dict(config)
        train_kwargs.setdefault("train", True)
        self.train_loader = self._create_ffcv_loader(
            self.config.dataset_train,
            train_kwargs,
            context="train",
        )

        val_kwargs = self.config.data.get_validation_dataloader_kwargs(config)
        self.val_loader = self._create_ffcv_loader(
            self.config.dataset_val,
            val_kwargs,
            context="val",
        )

    def _create_ffcv_loader(
        self, path: Path, config: Dict[str, Any], *, context: str = "active"
    ) -> Any:
        """Create a single FFCV data loader via the FFCV BatchSource adapter."""
        adapter = get_batch_source(use_ffcv=True)
        logging_kwargs = {"path": path, **config}
        self._log_dataloader_creation(
            dataloader_class=adapter.loader_class,
            dataloader_kwargs=logging_kwargs,
            context=context,
        )
        return adapter.create_loader(path, **config)

    def _setup_pytorch_loaders(self, config: Dict[str, Any]) -> None:
        """Set up standard PyTorch data loaders."""
        # Create dataset once and split
        dataset = self._create_pytorch_dataset(context="train", **config)

        self.train_loader, self.val_loader = get_train_val_loaders(
            dataset=dataset, **config
        )

        train_log_kwargs = dict(config)
        train_log_kwargs.update({"shuffle": True})
        self._log_dataloader_creation(
            dataloader_class=self.train_loader.__class__,
            dataloader_kwargs=train_log_kwargs,
            context="train",
        )

        val_log_kwargs = dict(config)
        val_log_kwargs.update({"shuffle": False, "train": False})
        self._log_dataloader_creation(
            dataloader_class=self.val_loader.__class__,
            dataloader_kwargs=val_log_kwargs,
            context="val",
        )

    def _create_pytorch_loader(self, config: Dict[str, Any]) -> DataLoader:
        """Create a single PyTorch loader for preview."""
        dataset = self._create_pytorch_dataset(context="preview", **config)
        dataloader = get_data_loader(dataset, **config)
        self._log_dataloader_creation(
            dataloader_class=dataloader.__class__,
            dataloader_kwargs=config.copy(),
            context="preview",
        )
        return dataloader

    def _create_pytorch_dataset(self, *, context: str = "active", **kwargs) -> Dataset:
        """Create PyTorch dataset with consistent configuration."""
        dataset_kwargs = {"data_name": self.config.data.data_name}
        dataset_kwargs.update(kwargs)
        return self.create_dataset(
            dataset_path=self.config.dataset_link,
            dataset_kwargs=dataset_kwargs,
            context=context,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_loader is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        if self.val_loader is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.val_loader


class SimpleDataModule(DataInterface):
    """Single-dataset data module for initialization and testing workflows."""

    def __init__(self, config: Any, dataset_path: Path):
        super().__init__(config)
        self.dataset_path = Path(dataset_path)
        self.dataset: Optional[Dataset] = None

    def setup_dataset(
        self,
        *,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        context: str = "active",
    ) -> Dataset:
        dataset_kwargs = dataset_kwargs or self.config.data.get_dataset_kwargs()
        self.dataset = self.create_dataset(
            dataset_path=self.dataset_path,
            dataset_kwargs=dataset_kwargs,
            context=context,
        )
        return self.dataset

    def create_dataloader(
        self,
        *,
        dataloader_class: Any,
        dataloader_kwargs: Dict[str, Any],
        context: str = "active",
        dataset: Optional[Dataset] = None,
    ) -> DataLoader:
        dataset = dataset or self.dataset or self.setup_dataset(context=context)
        dataloader = get_data_loader(
            dataset=dataset,
            dataloader=dataloader_class,
            **dataloader_kwargs,
        )
        self._log_dataloader_creation(
            dataloader_class=dataloader.__class__,
            dataloader_kwargs=dataloader_kwargs,
            context=context,
        )
        return dataloader

    def create_preview_loader(
        self,
        *,
        dataloader_class: Any,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        dataset = self.setup_dataset(
            context="preview",
            dataset_kwargs=dataset_kwargs,
        )
        effective_kwargs = (
            dataloader_kwargs or self.config.data.get_preview_dataloader_kwargs()
        )
        return self.create_dataloader(
            dataloader_class=dataloader_class,
            dataloader_kwargs=effective_kwargs,
            context="preview",
            dataset=dataset,
        )


class TestingDataModule(SimpleDataModule):
    """Data module leveraging shared logging utilities for testing workflows."""

    def __init__(self, config: Any, dataset_path: Path):
        super().__init__(config=config, dataset_path=dataset_path)
        self.dataloader: Optional[DataLoader] = None
        self.sampler = None

    def setup_dataset(
        self,
        *,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        context: str = "active",
    ) -> Dataset:
        dataset = super().setup_dataset(dataset_kwargs=dataset_kwargs, context=context)

        if self.sampler is None:
            sampler_name = self.config.data.sampler
            if sampler_name:
                try:
                    logger.info("Instantiating sampler: %s", sampler_name)
                    sampler_class = getattr(sampler, sampler_name)
                    self.sampler = sampler_class(dataset, seed=42)
                    logger.info(
                        "Sampler instantiated: %s", type(self.sampler).__name__
                    )
                except AttributeError as exc:
                    logger.error(
                        "Sampler '%s' not found in dynvision.data.sampler: %s",
                        sampler_name,
                        exc,
                    )
                    logger.warning("Continuing without sampler")
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "Failed to instantiate sampler '%s': %s",
                        sampler_name,
                        exc,
                    )
                    logger.warning("Continuing without sampler")

        logger.info("Test dataset loaded with %d samples", len(dataset))
        return dataset

    def setup_dataloader(self) -> DataLoader:
        if self.dataloader is not None:
            return self.dataloader

        # Use existing dataset if available, otherwise set it up
        dataset = self.dataset if self.dataset is not None else self.setup_dataset()
        dataloader_name = self.config.data.data_loader
        dataloader_class = get_data_loader_class(dataloader_name)
        dataloader_kwargs = self.config.get_dataloader_kwargs(
            dataloader_class=dataloader_class
        )

        if self.sampler is not None:
            dataloader_kwargs["sampler"] = self.sampler
            logger.info("Using sampler instance: %s", type(self.sampler).__name__)
        else:
            sampler_name = dataloader_kwargs.get("sampler")
            if sampler_name and isinstance(sampler_name, str):
                logger.info(
                    "Sampler '%s' specified but not instantiated in setup_dataset()",
                    sampler_name,
                )

        dataset_size = len(dataset)
        original_batch_size = dataloader_kwargs.get("batch_size", 1)
        if original_batch_size > dataset_size:
            dataloader_kwargs["batch_size"] = dataset_size
            logger.warning(
                "Adjusted batch_size from %s to dataset size %s to avoid empty loader",
                original_batch_size,
                dataset_size,
            )

        if logger.isEnabledFor(logging.DEBUG):
            log_section(
                logger,
                f"{dataloader_name} dataloader",
                [
                    (
                        "batch_size",
                        format_value(dataloader_kwargs.get("batch_size")),
                        None,
                    ),
                    (
                        "num_workers",
                        format_value(dataloader_kwargs.get("num_workers")),
                        None,
                    ),
                    (
                        "drop_last",
                        format_value(dataloader_kwargs.get("drop_last")),
                        None,
                    ),
                    (
                        "sampler",
                        format_value(dataloader_kwargs.get("sampler")),
                        None,
                    ),
                ],
                level=logging.DEBUG,
            )

        self.dataloader = self.create_dataloader(
            dataloader_class=dataloader_class,
            dataloader_kwargs=dataloader_kwargs,
            dataset=dataset,
        )

        logger.debug("Dataloader created: %d batches expected", len(self.dataloader))
        return self.dataloader

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dataloader = self.setup_dataloader()
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            raise RuntimeError(
                f"Dataloader is empty! Dataset has {len(self.dataset)} samples, "
                f"but dataloader produced no batches. Common causes:\n"
                f"  - batch_size ({self.config.data.batch_size}) > dataset size ({len(self.dataset)})\n"
                f"  - drop_last=True with small dataset\n"
                f"  - sampler: {self.config.data.sampler}\n"
                f"  - Data filtering excluding all samples\n"
                f"Fix: Reduce batch_size or set drop_last=False in configuration"
            )

        inputs, labels, *_ = batch
        inputs = _adjust_data_dimensions(inputs)
        labels = _adjust_label_dimensions(labels)
        return inputs, labels


__all__ = ["DataModule", "SimpleDataModule", "TestingDataModule"]
