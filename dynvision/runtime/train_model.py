"""Train a neural network model on a dataset with comprehensive Pydantic parameter management.

This script handles the complete training pipeline for DynVision models with type-safe,
validated parameter handling using composite Pydantic configuration classes.

Features:
- Composite configuration management (ModelParams + TrainerParams + DataParams)
- Automatic parameter consistency validation and optimization
- Learning rate scaling based on effective batch size
- Memory optimization for large datasets and distributed training
- Advanced error handling with informative feedback
- Configuration export for full reproducibility
"""

import logging
import math
import numbers
import wandb
import os
import sys
import multiprocessing
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from dynvision import models
from dynvision.data.dataloader import (
    _adjust_data_dimensions,
    _adjust_label_dimensions,
)
from dynvision.data.datamodule import DataModule
from dynvision.project_paths import project_paths
from dynvision.utils import (  # noqa: F401
    filter_kwargs,
    str_to_bool,
    handle_errors,
    log_section,
    format_value,
)
from dynvision.utils.checkpoint_to_statedict import get_best_checkpoint

# Import the Pydantic parameter classes
from dynvision.params import TrainingParams
from pytorch_lightning.loggers import Logger

logger = logging.getLogger(__name__)


class EarlyStoppingWithMin(pl.callbacks.EarlyStopping):
    """Enhanced early stopping with a minimum performance threshold."""

    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 5,
        min_val_accuracy: float = 0.7,
        mode: str = "max",
        verbose: bool = False,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            monitor: Metric to monitor (e.g., "val_accuracy").
            patience: Number of epochs to wait for improvement before stopping.
            min_val_accuracy: Minimum threshold for the monitored metric.
            mode: One of {"min", "max"}. In "min" mode, training will stop when the
                monitored quantity stops decreasing; in "max" mode, it will stop when
                the monitored quantity stops increasing.
            verbose: If True, logs a message for each validation improvement.
            strict: If True, will crash if the monitored metric is not available.
        """
        super().__init__(
            monitor=monitor,
            patience=patience,
            mode=mode,
            verbose=verbose,
            strict=strict,
            **kwargs,
        )
        self.min_val_accuracy = min_val_accuracy
        self.threshold_met = False  # Tracks if the threshold was met at least once

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Check validation metrics and update early stopping state."""
        # Get the current value of the monitored metric
        current_value = trainer.callback_metrics.get(self.monitor)

        if current_value is None:
            if self.strict:
                raise RuntimeError(
                    f"EarlyStoppingWithMin requires {self.monitor} available in metrics. "
                    "Make sure it is logged during validation."
                )
            return

        # Check if the threshold has been met at least once
        if current_value >= self.min_val_accuracy:
            self.threshold_met = True

        # Only apply early stopping logic if the threshold has been met
        if self.threshold_met:
            # Call the parent class's logic for early stopping
            super().on_validation_epoch_end(trainer, pl_module)
        else:
            if self.verbose:
                trainer.logger.log_metrics(
                    {"early_stopping_threshold_not_met": current_value},
                    step=trainer.global_step,
                )
                trainer.logger.log_text(
                    f"Early stopping threshold of {self.min_val_accuracy} not met yet. Current value: {current_value}"
                )


class SaveLastCheckpointCallback(pl.Callback):
    """Save the final checkpoint with DynVision's naming convention."""

    def __init__(
        self,
        dirpath: Path,
        base_stem: str,
        monitor: str,
        metric_precision: int = 2,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.base_stem = base_stem
        self.monitor = monitor
        self.metric_precision = metric_precision
        self._latest_metric: Optional[float] = None

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Track the most recent monitor metric after validation."""
        metric = trainer.callback_metrics.get(self.monitor)
        numeric = self._to_float(metric)
        if numeric is not None:
            self._latest_metric = numeric

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Persist the final model checkpoint on rank zero."""
        if not trainer.is_global_zero:
            return

        epoch_index = max(trainer.current_epoch - 1, 0)
        metric_value = self._latest_metric
        if metric_value is None:
            metric_value = self._to_float(trainer.callback_metrics.get(self.monitor))

        filename = self._build_filename(epoch_index, metric_value)
        checkpoint_path = self.dirpath / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        trainer.save_checkpoint(str(checkpoint_path))
        logging.getLogger(__name__).info(
            "Final checkpoint saved to %s", checkpoint_path
        )

    def _build_filename(self, epoch_index: int, metric_value: Optional[float]) -> str:
        metric_str = self._format_metric(metric_value)
        return f"{self.base_stem}-last-{epoch_index:02d}-{metric_str}.ckpt"

    def _format_metric(self, metric_value: Optional[float]) -> str:
        if metric_value is None or not math.isfinite(metric_value):
            return "nan"
        return f"{metric_value:.{self.metric_precision}f}"

    def _to_float(self, metric: Any) -> Optional[float]:
        if metric is None:
            return None
        if isinstance(metric, torch.Tensor):
            if metric.numel() == 0:
                return None
            metric = metric.detach().float().cpu()
            if metric.numel() == 1:
                return float(metric.item())
            return float(metric.mean().item())
        if isinstance(metric, numbers.Number):
            return float(metric)
        try:
            return float(metric)
        except (TypeError, ValueError):
            return None


class CallbackManager:
    """Enhanced callback management with configuration integration."""

    def __init__(self, config: TrainingParams):
        self.config = config

    def setup_callbacks(self) -> Tuple[List[pl.Callback], Path]:
        """Set up training callbacks based on configuration."""
        callbacks = []

        # Add model-specific callbacks when temporal depth is active
        # n_timesteps = self.config.model.n_timesteps
        # if n_timesteps is not None and n_timesteps > 1:
        #     callbacks.append(custom_callbacks.MonitorClassifierResponses())

        # callbacks.append(custom_callbacks.MonitorWeightDistributions())

        # Setup checkpointing
        checkpoint_path = self._setup_checkpointing(callbacks)

        # # Add early stopping if configured
        # early_stopping_kwargs = (
        #     self.config.trainer.get_early_stopping_callback_kwargs()
        # )
        # if early_stopping_kwargs:
        #     callbacks.append(EarlyStoppingWithMin(**early_stopping_kwargs))  # Todo: debug remove

        # Add learning rate monitor
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

        # Add additional trainer callbacks if configured
        if hasattr(self.config.trainer, "callbacks"):
            for callback_config in self.config.trainer.callbacks:
                callback = self._create_callback_from_config(callback_config)
                if callback:
                    callbacks.append(callback)

        return callbacks, checkpoint_path

    def _setup_checkpointing(self, callbacks: List[pl.Callback]) -> Path:
        """Setup model checkpointing with configuration."""
        # Generate checkpoint path and base filename
        if self.config.checkpoint_dir is not None:
            # Use provided checkpoint directory (hierarchical structure)
            # Checkpoints saved as: /models/{model_name}/{model_identifier}/{data_name}/{status}-{type}-{epoch}-{metric}.ckpt
            dirpath = Path(self.config.checkpoint_dir)
            # Use .name to get "trained.pt", then remove ".pt" suffix manually (preserves dots in model_identifier)
            status = self.config.output_model_state.name.removesuffix(".pt")
            base_stem = status
        else:
            # Legacy behavior: checkpoints in model-specific subdirectory
            # Checkpoints saved as: /models/{model_name}/{model_identifier}/{data_name}/checkpoints/{status}-{type}-{epoch}-{metric}.ckpt
            checkpoint_path = (
                self.config.output_model_state.parent
                / "checkpoints"
                / self.config.output_model_state.name
            )
            dirpath = checkpoint_path.parent
            # Use .name and remove suffix manually instead of .stem (preserves dots in filename)
            base_stem = checkpoint_path.name.removesuffix(".pt")

        # Ensure the directory exists
        dirpath.mkdir(parents=True, exist_ok=True)

        # Get checkpoint configuration
        checkpoint_kwargs = self.config.trainer.get_checkpoint_callback_kwargs()

        # Extract monitor metric from config for flexible filename formatting
        monitor_metric = checkpoint_kwargs.get("monitor") or "val_loss"

        # If monitoring val metric but validation won't run initially, use train metric
        # to avoid warnings about missing metrics
        check_val_freq = self.config.trainer.check_val_every_n_epoch
        if "val_" in monitor_metric and check_val_freq > 1:
            # Use train metric for early epochs, will switch to val once available
            initial_monitor = monitor_metric.replace("val_", "train_")
            logger.info(
                f"Validation runs every {check_val_freq} epochs. "
                f"Using '{initial_monitor}' for checkpointing until validation metrics are available."
            )
            monitor_metric = initial_monitor
        metric_precision = 2
        metric_format = f":.{metric_precision}f"
        metric_suffix = "{epoch:02d}-{" + monitor_metric + metric_format + "}"

        # Handle save_last separately (custom callback ensures consistent naming)
        save_last = checkpoint_kwargs.pop("save_last", False)
        every_n_epochs = checkpoint_kwargs.get("every_n_epochs")

        # Primary checkpoint: track top-k best models
        topk_kwargs = checkpoint_kwargs.copy()
        topk_kwargs.pop("every_n_epochs", None)
        topk_kwargs.update(
            {
                "dirpath": str(dirpath),
                "filename": f"{base_stem}-best-{metric_suffix}",
                "monitor": monitor_metric,
                "save_last": False,
            }
        )
        callbacks.append(pl.callbacks.ModelCheckpoint(**topk_kwargs))

        # Periodic checkpoint: save every n epochs with unified naming
        if every_n_epochs and every_n_epochs > 0:
            periodic_kwargs = checkpoint_kwargs.copy()
            periodic_kwargs.update(
                {
                    "dirpath": str(dirpath),
                    "filename": f"{base_stem}-epoch-{metric_suffix}",
                    "monitor": monitor_metric,
                    "save_top_k": -1,
                    "save_last": False,
                    "every_n_epochs": every_n_epochs,
                }
            )
            callbacks.append(pl.callbacks.ModelCheckpoint(**periodic_kwargs))

        # Final checkpoint: always persist the last state with shared pattern
        if save_last:
            callbacks.append(
                SaveLastCheckpointCallback(
                    dirpath=dirpath,
                    base_stem=base_stem,
                    monitor=monitor_metric,
                    metric_precision=metric_precision,
                )
            )

        logger.debug(
            "Configured checkpoint callbacks: dir=%s, monitor=%s, top_k=%s, every_n_epochs=%s, save_last=%s",
            str(dirpath),
            monitor_metric,
            checkpoint_kwargs.get("save_top_k"),
            every_n_epochs,
            save_last,
        )

        # Return a representative checkpoint path for get_best_checkpoint
        # The actual checkpoint filename will be determined by PyTorch Lightning
        checkpoint_path = dirpath / f"{base_stem}.ckpt"
        return checkpoint_path

    def _create_callback_from_config(
        self, callback_config: Dict[str, Any]
    ) -> Optional[pl.Callback]:
        """Create callback from configuration dictionary."""
        # This would be extended to support various callback types
        callback_type = callback_config.get("type")
        if callback_type == "ModelCheckpoint":
            return pl.callbacks.ModelCheckpoint(**callback_config.get("kwargs", {}))
        elif callback_type == "EarlyStopping":
            return pl.callbacks.EarlyStopping(**callback_config.get("kwargs", {}))
        else:
            logger.warning(f"Unknown callback type: {callback_type}")
            return None


class ModelManager:
    """Enhanced model management with configuration integration."""

    def __init__(self, config: TrainingParams):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cached_state_dict = (
            None  # Cache to avoid loading state dict multiple times
        )

    def load_and_initialize_model(self, load_state_dict=True) -> pl.LightningModule:
        """Load and initialize model with current configuration."""
        # Load state dict (use cached version if available to avoid loading twice)
        if self._cached_state_dict is None:
            self._cached_state_dict = torch.load(
                self.config.input_model_state,
                map_location=self.device,
                weights_only=True,
            )
        state_dict = self._cached_state_dict

        # Create model with current configuration
        model_class = getattr(models, self.config.model.model_name)
        model_kwargs = self.config.model.get_model_kwargs(model_class)

        self.config.model.log_model_creation(
            model_class=model_class,
            model_kwargs=model_kwargs,
        )

        model = model_class(**model_kwargs).to(self.device)

        # Load state dict with error handling
        if not load_state_dict:
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("Model state loaded successfully")
            except Exception as e:
                logger.warning(f"Strict loading failed: {e}. Trying non-strict...")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing keys: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys: {unexpected}")

        # Initialize model-specific settings
        if hasattr(model, "set_residual_timesteps"):
            model.set_residual_timesteps()

        return model

    def save_model(self, model: pl.LightningModule) -> None:
        """Save trained model with configuration."""
        torch.save(model.state_dict(), self.config.output_model_state)
        logger.info(f"Model saved to {self.config.output_model_state}")

        # Export configuration alongside model
        config_path = self.config.output_model_state.with_suffix(".config.yaml")
        self.config.export_full_config(config_path)
        logger.info(f"Configuration exported to {config_path}")


class TrainingOrchestrator:
    """Enhanced training orchestrator with comprehensive configuration management."""

    def __init__(self, config: TrainingParams):
        self.config = config
        self.datamodule = DataModule(config)
        self.callback_manager = CallbackManager(config)
        self.model_manager = ModelManager(config)

    @contextmanager
    def training_context(self):
        """Enhanced training context with configuration logging."""
        try:
            # Setup
            torch.set_float32_matmul_precision("medium")

            # Aggressive GPU cache clearing to prevent OOM from lingering allocations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                # Also try PyTorch's expandable segments if available
                if hasattr(torch.cuda, "memory"):
                    torch.cuda.reset_peak_memory_stats()

            # Log comprehensive configuration
            self._log_training_configuration()

            yield

        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _log_training_configuration(self) -> None:
        """Log key training configuration information."""
        self.config.log_training_overview(logger=logger)

    def _preview_log_level(self) -> int:
        """Return INFO for verbose runs, otherwise DEBUG for preview noise."""
        return (
            logging.INFO if getattr(self.config, "verbose", False) else logging.DEBUG
        )

    def _setup_trainer(
        self, callbacks: List[pl.Callback], pl_logger: Logger
    ) -> pl.Trainer:
        """Setup PyTorch Lightning trainer with full configuration."""
        # Get trainer kwargs from configuration
        trainer_kwargs = self.config.trainer.get_trainer_kwargs()

        # Add additional configuration
        trainer_kwargs.update(
            {
                "callbacks": callbacks,
                "logger": pl_logger,
            }
        )

        # Filter valid arguments for Trainer
        trainer_kwargs, unknown = filter_kwargs(pl.Trainer, trainer_kwargs)
        if unknown:
            logger.debug("Trainer kwargs set: %s", trainer_kwargs)

        self.config.log_trainer_creation(
            trainer_kwargs=trainer_kwargs,
            logger=logger,
        )

        return pl.Trainer(**trainer_kwargs)

    def infer_and_update_from_data(self, pl_logger: pl.loggers.Logger) -> None:
        """
        Infer model parameters from training data and update configuration.

        This method creates a preview loader, loads a sample batch, extracts dimensions,
        validates consistency, and updates the model configuration accordingly.
        """
        # Create preview loader (non-distributed for dimension inference)
        preview_loader = self.datamodule.create_preview_loader()

        # Load a sample from the preview dataloader
        inputs, label_indices, *_ = next(iter(preview_loader))
        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        # Extract actual dimensions
        batch_size, actual_n_timesteps, *spatial_dims = inputs.shape
        actual_input_dims = (actual_n_timesteps, *spatial_dims)

        pixel_mean = inputs.mean().item()
        pixel_std = inputs.std().item()
        log_section(
            logger,
            "Preview batch",
            [
                ("Batch shape", format_value(tuple(inputs.size())), None),
                ("Input dims", format_value(actual_input_dims), None),
                ("Pixel stats", f"{pixel_mean:.3f} ± {pixel_std:.3f}", None),
            ],
            level=self._preview_log_level(),
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Extract n_classes from state dict (use cached version to avoid loading twice)
        if self.model_manager._cached_state_dict is None:
            self.model_manager._cached_state_dict = torch.load(
                self.config.input_model_state, map_location=device, weights_only=True
            )
        state_dict = self.model_manager._cached_state_dict
        actual_n_classes = self._extract_n_classes_from_state_dict(state_dict)

        # Update model parameters using the dedicated method
        self.config.update_model_parameters_from_data(
            input_dims=actual_input_dims,
            n_classes=actual_n_classes,
            verbose=True,
        )
        # Log sample images
        self._log_sample_images(inputs, label_indices, pl_logger)

        # Explicitly delete preview data to free memory
        del inputs, label_indices, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_n_classes_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> int:
        """Extract number of classes from model state dict."""
        # Look for classifier layer first
        for key in state_dict.keys():
            if "classifier" in key and "weight" in key:
                n_classes = state_dict[key].shape[0]
                logger.info(f"Found n_classes={n_classes} from {key}")
                return n_classes

        # Fallback: use last weight layer
        weight_keys = [k for k in state_dict.keys() if "weight" in k]
        if weight_keys:
            last_key = weight_keys[-1]
            n_classes = state_dict[last_key].shape[0]
            logger.info(f"Found n_classes={n_classes} from {last_key}")
            return n_classes

        logger.warning("Could not extract n_classes from state dict")
        return self.config.model.n_classes

    def _log_sample_images(
        self, inputs: torch.Tensor, labels: torch.Tensor, pl_logger: Logger
    ) -> None:
        """Log sample images to the logger."""
        if not hasattr(pl_logger, "log_image"):
            return

        try:
            n_timesteps = inputs.shape[1]
            sample_images = [inputs[0, t] for t in range(n_timesteps)]
            sample_labels = [str(labels[0, t].item()) for t in range(n_timesteps)]

            pl_logger.log_image(
                key="input_samples", images=sample_images, caption=sample_labels
            )
        except Exception as e:
            logger.warning(f"Failed to log sample images: {e}")

    def run_training(self) -> int:
        """Run the complete training pipeline with comprehensive error handling.

        Loading Logic:
        - If Lightning checkpoint exists: Resume from checkpoint (includes model, optimizer, epoch)
        - If no checkpoint: Load from state dict (fresh training start)
        """
        with self.training_context():
            try:
                output_path = self.config.output_model_state
                try:
                    relative_path = output_path.relative_to(output_path.parents[2])
                    model_name = str(relative_path.with_suffix(""))
                except (IndexError, ValueError):
                    model_name = output_path.name.removesuffix(output_path.suffix)

                # Initialize logger
                pl_logger = pl.loggers.WandbLogger(
                    project=project_paths.project_name,
                    save_dir=project_paths.large_logs,
                    config=self.config.get_full_config(flat=True),
                    tags=["train"],
                    name=model_name,
                )

                # Setup trainer components
                callbacks, checkpoint_path = self.callback_manager.setup_callbacks()
                trainer = self._setup_trainer(callbacks, pl_logger)

                # Infer and update model parameters from preview data
                self.infer_and_update_from_data(pl_logger)

                # Persist resolved configuration before long-running training begins
                self.config.persist_resolved_config(
                    primary_output=self.config.output_model_state,
                    script_name=__file__,
                )

                # Check for existing Lightning checkpoint FIRST (before loading anything)
                existing_checkpoint = get_best_checkpoint(
                    checkpoint_path.parent, model_name, raise_error=False
                )

                if existing_checkpoint:
                    # Verify checkpoint accessibility across all ranks if distributed
                    existing_checkpoint = self._verify_checkpoint_across_ranks(
                        trainer, existing_checkpoint
                    )

                # ALWAYS load model from state dict - Lightning will override with checkpoint if needed
                logger.info("Loading model from state dict...")
                model = self.model_manager.load_and_initialize_model(
                    load_state_dict=existing_checkpoint is None
                )

                checkpoint_entries = []
                if existing_checkpoint:
                    ckpt_path = existing_checkpoint
                    checkpoint_entries.extend(
                        [
                            ("mode", "resume", None),
                            ("checkpoint", format_value(existing_checkpoint), None),
                            (
                                "weight_source",
                                "lightning_checkpoint",
                                "overrides state dict",
                            ),
                        ]
                    )
                else:
                    ckpt_path = None
                    checkpoint_entries.extend(
                        [
                            ("mode", "state_dict", None),
                            ("checkpoint", "–", "missing"),
                            (
                                "weight_source",
                                format_value(self.config.input_model_state),
                                None,
                            ),
                        ]
                    )
                log_section(logger, "checkpoint_selection", checkpoint_entries)

                # Synchronize after model loading in distributed mode
                if trainer.strategy and hasattr(trainer.strategy, "barrier"):
                    logger.debug("Synchronizing ranks after model initialization...")
                    trainer.strategy.barrier()
                    logger.debug("All ranks synchronized")

                # Hack to log histograms
                wandb.init(settings=wandb.Settings(init_timeout=120))

                # Final synchronization before training
                if trainer.strategy and hasattr(trainer.strategy, "barrier"):
                    logger.debug("Final synchronization before training start...")
                    trainer.strategy.barrier()
                    logger.debug("All ranks ready to train")

                # Clear GPU cache one more time right before training to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Log detailed memory stats
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    logger.info(
                        f"Pre-training GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
                    )
                    if allocated > 20.0:  # More than 20GB is suspicious
                        logger.error(
                            f"GPU memory usage is very high ({allocated:.2f} GB) before training starts!"
                        )
                        logger.error(
                            "This suggests a memory leak during model/data initialization."
                        )
                        logger.error("Attempting to continue, but OOM is likely...")

                # Train model using DataModule
                logger.info("Starting training...")
                trainer.fit(
                    model,  # Always pass the model - Lightning handles checkpoint loading
                    datamodule=self.datamodule,
                    ckpt_path=ckpt_path,
                )

                # Save trained model (get final model from trainer)
                final_model = trainer.model
                self.model_manager.save_model(final_model)

                logger.info("Training completed successfully!")
                return 0

            except Exception as e:
                logger.error(f"Training failed: {e}")
                self._handle_training_failure(trainer, checkpoint_path)
                return 1

    def _verify_checkpoint_across_ranks(
        self, trainer: pl.Trainer, checkpoint_path: Optional[str]
    ) -> Optional[str]:
        """Verify checkpoint exists and is accessible on all ranks.

        Args:
            trainer: PyTorch Lightning trainer
            checkpoint_path: Path to checkpoint to verify

        Returns:
            Verified checkpoint path if valid on all ranks, None otherwise
        """
        if checkpoint_path is None:
            return None

        # Check if we're in distributed mode
        is_distributed = (
            trainer.world_size > 1
            and hasattr(trainer, "strategy")
            and trainer.strategy is not None
        )

        if not is_distributed:
            # Single GPU/CPU: verify existence and readability
            if not os.path.exists(checkpoint_path):
                logger.info(f"Checkpoint not found: {checkpoint_path}")
                return None

            try:
                torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                logger.info(f"Checkpoint found and readable: {checkpoint_path}")
                return checkpoint_path
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "Checkpoint exists but failed to load (%s). Will start from state dict instead.",
                    err,
                )
                return None

        # Distributed mode: Check if distributed is actually initialized
        try:
            import torch.distributed as dist

            # CRITICAL FIX: Check if distributed is initialized AND available
            if not dist.is_available() or not dist.is_initialized():
                logger.info(
                    "Distributed backend not yet initialized by Lightning. "
                    "Using simple file existence check - Lightning will handle "
                    "checkpoint verification during trainer.fit()."
                )
                # Just check local file existence - Lightning handles the rest
                if os.path.exists(checkpoint_path):
                    logger.info(f"Checkpoint found locally: {checkpoint_path}")
                    return checkpoint_path
                else:
                    logger.info(f"Checkpoint not found locally: {checkpoint_path}")
                    return None

            # If we reach here, distributed is fully initialized
            rank = trainer.global_rank
            world_size = trainer.world_size

            logger.info(
                f"[Rank {rank}/{world_size}] Verifying checkpoint: {checkpoint_path}"
            )

            # Check if checkpoint exists on this rank
            checkpoint_exists = os.path.exists(checkpoint_path)
            logger.info(f"[Rank {rank}] Checkpoint exists: {checkpoint_exists}")

            # Convert to tensor for collective operations
            exists_tensor = torch.tensor(
                [1 if checkpoint_exists else 0],
                dtype=torch.long,
                device=trainer.strategy.root_device,
            )

            # Gather existence status from all ranks
            all_exists = [torch.zeros_like(exists_tensor) for _ in range(world_size)]
            dist.all_gather(all_exists, exists_tensor)

            # Check if ALL ranks can see the checkpoint
            visibility = [t.item() for t in all_exists]
            all_ranks_see_checkpoint = all(v == 1 for v in visibility)

            logger.info(
                f"[Rank {rank}] Checkpoint visibility across ranks: {visibility}"
            )

            if not all_ranks_see_checkpoint:
                logger.warning(
                    f"[Rank {rank}] Checkpoint not visible to all ranks. "
                    f"Will start fresh from state dict instead."
                )
                # Barrier to ensure all ranks agree
                dist.barrier()
                return None

            # Verify checkpoint is readable (lightweight check)
            try:
                ckpt = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                epoch = ckpt.get("epoch", "unknown")
                logger.info(f"[Rank {rank}] Checkpoint is readable, epoch: {epoch}")
                del ckpt  # Free memory
            except Exception as e:
                logger.error(f"[Rank {rank}] Failed to read checkpoint: {e}")
                dist.barrier()
                return None

            # Final barrier before confirming checkpoint
            logger.info(f"[Rank {rank}] Waiting at checkpoint verification barrier...")
            dist.barrier()
            logger.info(f"[Rank {rank}] Checkpoint verified successfully")

            return checkpoint_path

        except Exception as e:
            logger.info(f"Distributed operations not available: {e}")
            # Fallback to simple file existence check
            logger.info("Using simple checkpoint existence check")
            if os.path.exists(checkpoint_path):
                logger.info(f"Checkpoint found (fallback): {checkpoint_path}")
                return checkpoint_path
            else:
                logger.info(f"Checkpoint not found (fallback): {checkpoint_path}")
                return None

    def _handle_training_failure(
        self, trainer: pl.Trainer, checkpoint_path: Path
    ) -> None:
        """Handle training failure by saving a checkpoint with detailed filename."""
        import traceback

        # Print the traceback for debugging
        traceback.print_exc()

        # Ensure the checkpoint directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Get checkpoint configuration
        checkpoint_kwargs = self.config.trainer.get_checkpoint_callback_kwargs()

        # Extract monitor metric from config for flexible filename formatting
        monitor_metric = checkpoint_kwargs.get("monitor", "val_loss")

        # Extract the current epoch and validation loss
        current_epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics.get(monitor_metric, float("nan"))

        # Format the checkpoint filename
        checkpoint_filename = (
            f"{checkpoint_path.stem}-{current_epoch:02d}-{val_loss:.2f}.ckpt"
        )
        checkpoint_file = checkpoint_path.parent / checkpoint_filename

        # Save the checkpoint
        trainer.save_checkpoint(checkpoint_file)
        logger.info(f"Checkpoint before training failure saved to {checkpoint_file}")


@handle_errors(verbose=True)
def main() -> int:
    """Main entry point for training with comprehensive configuration management."""
    # Enable PyTorch's expandable segments to reduce memory fragmentation
    # This helps prevent OOM errors from fragmentation
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info(
            "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation"
        )

    # Check GPU memory status before starting
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            if (
                allocated > 1.0 or reserved > 1.0
            ):  # More than 1GB suggests leftover memory
                logger.warning(
                    f"GPU {i} has {allocated:.2f} GB allocated and {reserved:.2f} GB reserved "
                    f"before training start. This may indicate a previous process didn't clean up properly. "
                    f"Consider clearing GPU memory or restarting the job."
                )

    try:
        config = TrainingParams.from_cli_and_config()
        config.setup_logging()

    except Exception as e:
        logger.error(f"Failed to create training configuration: {e}")
        return 1

    try:
        orchestrator = TrainingOrchestrator(config)
        return orchestrator.run_training()
    except Exception as e:
        logger.error(f"Training orchestration failed: {e}")
        return 1


def run_main() -> int:
    """Entry point for distributed training."""
    return main()


if __name__ == "__main__":
    # Set process start method for SLURM compatibility
    if os.environ.get("SLURM_JOB_ID"):
        multiprocessing.set_start_method("spawn", force=True)

    # Run training
    sys.exit(run_main())
