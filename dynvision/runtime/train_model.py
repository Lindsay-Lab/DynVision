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

Example:
    $ python train_model.py --config_path configs/train_config.yaml --model_name DyRCNNx4
"""

import logging
import wandb
import os
import sys


def should_clean_distributed_env():
    """
    Robust detection of distributed training using standard PyTorch patterns.

    Returns True if we're definitely in distributed mode, False otherwise.
    """

    devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if isinstance(devices, list):
        devices = len(devices)
    elif isinstance(devices, str):
        # Handle comma-separated devices
        devices = devices.split(",")
        devices = len(devices)
    else:
        pass

    if devices is None or int(devices) <= 1:
        return True

    return False


def clean_distributed_env():
    """Clean distributed training environment variables for non-distributed training."""
    if not should_clean_distributed_env():
        return

    print("Non-distributed mode - cleaning problematic environment variables")

    distributed_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "GROUP_RANK",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NNODES",
    ]

    cleaned_vars = []
    for var in distributed_vars:
        if var in os.environ:
            # Remove empty strings or problematic values
            if os.environ[var] in ["", "0"] or not os.environ[var].strip():
                del os.environ[var]
                cleaned_vars.append(var)

    if cleaned_vars:
        print(f"Cleaned environment variables: {cleaned_vars}")

    # Set single-device mode explicitly for non-distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


# Clean environment conditionally before ANY imports
clean_distributed_env()

import multiprocessing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import ffcv

from dynvision import models
from dynvision.data.dataloader import (
    _adjust_data_dimensions,
    _adjust_label_dimensions,
    get_train_val_loaders,
)
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader
from dynvision.data.dataloader import get_train_val_loaders, get_data_loader
from dynvision.data.datasets import get_dataset
from dynvision.project_paths import project_paths
from dynvision.utils import (
    filter_kwargs,
    str_to_bool,
    handle_errors,
)
from dynvision.visualization import callbacks as custom_callbacks

# Import the Pydantic parameter classes
from dynvision.params import TrainingParams, DynVisionConfigError

logger = logging.getLogger(__name__)


class EarlyStoppingWithMin(pl.callbacks.EarlyStopping):
    """Enhanced early stopping with minimum performance threshold."""

    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 5,
        min_val_accuracy: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(monitor=monitor, patience=patience, **kwargs)
        self.min_val_accuracy = min_val_accuracy

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Check validation metrics and update early stopping state."""
        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is not None and current_value >= self.min_val_accuracy:
            super().on_validation_epoch_end(trainer, pl_module)
        else:
            # Reset wait count if minimum performance not met
            self.wait_count = 0


class DataModule(pl.LightningDataModule):
    """Enhanced DataModule with unified dataloader creation and distributed handling."""

    def __init__(self, config: TrainingParams):
        super().__init__()
        self.config = config
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
        preview_config = self._create_preview_config()

        if self.config.data.use_ffcv:
            self._preview_loader = self._create_ffcv_loader(
                self.config.dataset_train,
                preview_config | {"train": False, "distributed": False},
            )
        else:
            self._preview_loader = self._create_pytorch_loader(preview_config)

        return self._preview_loader

    def _create_preview_config(self) -> Dict[str, Any]:
        """Create minimal configuration for preview loader."""
        base_config = self.config.data.get_dataloader_kwargs()
        return base_config | {
            "distributed": False,
            "batch_size": min(base_config.get("batch_size", 32), 32),
            "num_workers": min(base_config.get("num_workers", 4), 1),
            "shuffle": False,  # No need to shuffle for preview
        }

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up data loaders with proper distributed configuration."""
        if stage not in ["fit", None]:
            return

        dataloader_config = self.config.data.get_dataloader_kwargs()

        if self.config.data.use_ffcv:
            self._setup_ffcv_loaders(dataloader_config)
        else:
            self._setup_pytorch_loaders(dataloader_config)

    def _setup_ffcv_loaders(self, config: Dict[str, Any]) -> None:
        """Set up FFCV data loaders."""
        self.train_loader = self._create_ffcv_loader(
            self.config.dataset_train,
            config | {"train": True},
        )
        self.val_loader = self._create_ffcv_loader(
            self.config.dataset_val,
            config | {"train": False},
        )

    def _create_ffcv_loader(
        self, path: Path, config: Dict[str, Any]
    ) -> ffcv.loader.Loader:
        """Create a single FFCV data loader."""
        return get_ffcv_dataloader(path=path, **config)

    def _setup_pytorch_loaders(self, config: Dict[str, Any]) -> None:
        """Set up standard PyTorch data loaders."""
        # Create dataset once and split
        dataset = self._create_pytorch_dataset(**config)

        self.train_loader, self.val_loader = get_train_val_loaders(
            dataset=dataset, **config
        )

    def _create_pytorch_loader(self, config: Dict[str, Any]) -> DataLoader:
        """Create a single PyTorch loader for preview."""
        dataset = self._create_pytorch_dataset(**config)
        return get_data_loader(dataset, **config)

    def _create_pytorch_dataset(self, **kwargs) -> Dataset:
        """Create PyTorch dataset with consistent configuration."""
        return get_dataset(
            data_path=self.config.dataset_link,
            data_name=self.config.data.data_name,
            **kwargs,
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


class CallbackManager:
    """Enhanced callback management with configuration integration."""

    def __init__(self, config: TrainingParams):
        self.config = config

    def setup_callbacks(self) -> Tuple[List[pl.Callback], Path]:
        """Set up training callbacks based on configuration."""
        callbacks = []

        # Add model-specific callbacks
        print(f"timesteps: {self.config.model.n_timesteps}")
        if self.config.model.n_timesteps > 1:
            callbacks.append(custom_callbacks.MonitorClassifierResponses())

        callbacks.append(custom_callbacks.MonitorWeightDistributions())

        # Setup checkpointing
        checkpoint_path = self._setup_checkpointing(callbacks)

        # Add early stopping if configured
        early_stopping_kwargs = (
            self.config.trainer.get_early_stopping_callback_kwargs()
        )
        if early_stopping_kwargs:
            callbacks.append(EarlyStoppingWithMin(**early_stopping_kwargs))

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
        # Generate checkpoint path
        checkpoint_path = (
            self.config.output_model_state.parent
            / "checkpoints"
            / self.config.output_model_state.stem
        )

        # Get checkpoint configuration
        checkpoint_kwargs = self.config.trainer.get_checkpoint_callback_kwargs()
        checkpoint_kwargs.update(
            {
                "dirpath": checkpoint_path.parent,
                "filename": checkpoint_path.name + "-{epoch:02d}-{val_loss:.2f}",
            }
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_kwargs)
        callbacks.append(checkpoint_callback)

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

    def load_and_initialize_model(self) -> pl.LightningModule:
        """Load and initialize model with current configuration."""
        # Load state dict
        state_dict = torch.load(
            self.config.input_model_state, map_location=self.device, weights_only=True
        )

        # Create model with current configuration
        model_class = getattr(models, self.config.model.model_name)
        model_kwargs = self.config.model.get_model_kwargs(model_class)

        logger.info(f"Creating {model_class.__name__} with:")
        logger.info(f"  - Input dims: {model_kwargs.get('input_dims')}")
        logger.info(f"  - N classes: {model_kwargs.get('n_classes')}")
        logger.info(f"  - N timesteps: {model_kwargs.get('n_timesteps')}")

        model = model_class(**model_kwargs).to(self.device)

        # Load state dict with error handling
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
            torch.cuda.empty_cache()

            # Log comprehensive configuration
            self._log_training_configuration()

            yield

        finally:
            # Cleanup
            torch.cuda.empty_cache()

    def _log_training_configuration(self) -> None:
        """Log key training configuration information."""
        logger.info("=" * 60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model.model_name}")
        logger.info(f"Dataset: {self.config.data.data_name}")
        logger.info(f"Data batch size: {self.config.data.batch_size}")
        logger.info(f"Global batch size: {self.config.global_batch_size}")
        logger.info(f"Effective batch size: {self.config.effective_batch_size}")
        logger.info(
            f"Effective Learning rate: {self.config.effective_learning_rate:.6f}"
        )
        logger.info(f"Epochs: {self.config.trainer.epochs}")
        logger.info(f"Precision: {self.config.trainer.precision}")
        logger.info(f"Optimizer: {self.config.model.optimizer}")

        if self.config.trainer.is_distributed:
            logger.info("Distributed training enabled")
            logger.info(f"World size: {self.config.trainer.world_size}")
            logger.info(f"Distributed strategy: {self.config.trainer.strategy}")
            logger.info(f"Number of nodes: {self.config.trainer.num_nodes}")
            logger.info(f"Devices: {self.config.trainer.devices}")

        logger.info("=" * 60)

    def _setup_trainer(
        self, callbacks: List[pl.Callback], pl_logger: pl.loggers.Logger
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
            logger.info(f"Filtered unknown trainer kwargs: {list(unknown.keys())}")

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
        inputs, label_indices, *paths = next(iter(preview_loader))
        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        # Extract actual dimensions
        batch_size, actual_n_timesteps, *spatial_dims = inputs.shape
        actual_input_dims = (actual_n_timesteps, *spatial_dims)

        logger.info(f"Extracted from training data:")
        logger.info(f"  - Input shape: {inputs.size()}")
        logger.info(f"  - Input dims: {actual_input_dims}")
        logger.info(f"  - Pixel stats: {inputs.mean():.3f} ± {inputs.std():.3f}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Extract n_classes from state dict
        state_dict = torch.load(
            self.config.input_model_state, map_location=device, weights_only=True
        )
        actual_n_classes = self._extract_n_classes_from_state_dict(state_dict)

        # Update model parameters using the dedicated method
        self.config.update_model_parameters_from_data(
            input_dims=actual_input_dims,
            n_classes=actual_n_classes,
            verbose=True,
        )
        # Log sample images
        self._log_sample_images(inputs, label_indices, pl_logger)

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
        self, inputs: torch.Tensor, labels: torch.Tensor, pl_logger: pl.loggers.Logger
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
        """Run the complete training pipeline with comprehensive error handling."""
        with self.training_context():
            try:
                # Initialize logger
                pl_logger = pl.loggers.WandbLogger(
                    project=project_paths.project_name,
                    save_dir=project_paths.large_logs,
                    config=self.config.get_full_config(flat=True),
                    tags=["train"],
                    name=f"{self.config.output_model_state.stem}",
                )
                # Setup trainer components
                callbacks, checkpoint_path = self.callback_manager.setup_callbacks()
                trainer = self._setup_trainer(callbacks, pl_logger)

                # Infer and update model parameters from preview data (before trainer setup)
                self.infer_and_update_from_data(pl_logger)

                # Load and initialize model with updated configuration
                model = self.model_manager.load_and_initialize_model()

                # Check for existing checkpoint
                existing_checkpoint = self._find_existing_checkpoint(checkpoint_path)

                # Hack to log histograms
                wandb.init()

                # Train model using DataModule (handles distributed setup properly)
                logger.info("Starting training...")
                trainer.fit(
                    model,
                    datamodule=self.datamodule,
                    ckpt_path=existing_checkpoint,
                )

                # Save trained model
                self.model_manager.save_model(model)

                logger.info("Training completed successfully!")
                return 0

            except Exception as e:
                logger.error(f"Training failed: {e}")
                # if logger.isEnabledFor(logging.DEBUG):
                import traceback

                traceback.print_exc()
                return 1

    def _find_existing_checkpoint(self, checkpoint_path: Path) -> Optional[Path]:
        """Find existing checkpoint for resuming training."""
        checkpoint_dir = checkpoint_path.parent
        if not checkpoint_dir.exists():
            return None

        # Look for checkpoint files
        checkpoint_files = list(checkpoint_dir.glob(f"{checkpoint_path.name}*.ckpt"))
        if checkpoint_files:
            # Return the most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            return latest_checkpoint

        return None


@handle_errors(verbose=True)
def main() -> int:
    """Main entry point for training with comprehensive configuration management."""
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
