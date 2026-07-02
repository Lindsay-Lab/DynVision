import io
from typing import Any, Dict, List, Optional, Tuple
import logging

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from dynvision.losses import lr_scheduler
from dynvision.utils import alias_kwargs
from dynvision.utils.performance_measures import (
    calculate_accuracy,
    calculate_confidence,
)
import gc

logger = logging.getLogger(__name__)


class LightningBase(pl.LightningModule):
    """PyTorch Lightning integration for DynVision models."""

    @alias_kwargs(
        lr="learning_rate",
        solver="dynamics_solver",
        activityloss="activity_loss_weight",
    )
    def __init__(
        self,
        # Training configuration
        retain_graph: bool = False,
        # Loss configuration
        criterion_params: List[Tuple[str, Dict[str, Any]]] = [
            ("CrossEntropyLoss", {"weight": 1.0})
        ],
        activity_loss_weight: Optional[float] = None,
        non_label_index: int = -1,
        # Optimizer configuration
        optimizer: str = "Adam",
        optimizer_kwargs: Dict[str, Any] = {"weight_decay": 0.0005},
        optimizer_configs: Dict[str, Dict[str, Any]] = {"monitor": "train_loss"},
        learning_rate: float = 0.0002,
        lr_parameter_groups: Dict[str, Dict[str, Any]] = {},
        # Scheduler configuration
        scheduler: str = "CosineAnnealingLR",
        scheduler_kwargs: Dict[str, Any] = {"T_max": 250},
        scheduler_configs: Dict[str, Dict[str, Any]] = {"monitor": "train_loss"},
        # Logging configuration
        log_level: str = "info",
        log_every_n_steps: int = 50,
        **kwargs: Any,
    ) -> None:
        # Store Lightning-specific attributes BEFORE calling super()
        # This ensures they're available if other classes need them
        self.retain_graph = retain_graph
        self.criterion_params = criterion_params
        self.non_label_index = non_label_index
        self.activity_loss_weight = (
            float(activity_loss_weight) if activity_loss_weight is not None else None
        )
        self.update_criterion_params(
            "ActivityLoss", {"weight": self.activity_loss_weight}
        )

        # Optimizer attributes
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_configs = optimizer_configs
        self.learning_rate = float(learning_rate)
        self.lr_parameter_groups = lr_parameter_groups

        # Scheduler attributes
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_configs = scheduler_configs

        # Logging attributes
        self.log_level = log_level
        self.log_every_n_steps = int(log_every_n_steps)

        # Call super().__init__() with kwargs to continue MRO chain
        super().__init__(**kwargs)

    # Core training steps
    #####################
    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        compute_confidence: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:

        if hasattr(self, "_process_batch"):
            batch = self._process_batch(batch, batch_idx)

        # inputs: (batch_size, n_timesteps, n_channels, height, width)
        inputs, label_index, *paths = batch
        batch_size, n_timesteps = label_index.shape

        # Create image indices for storage
        image_index = torch.arange(batch_size, device=inputs.device) + (
            batch_idx * batch_size
        )
        image_index = image_index.unsqueeze(1).expand(batch_size, n_timesteps)

        # Extract first non-negative label index for each sample in the batch
        first_label_index = label_index[
            torch.arange(label_index.size(0), device=label_index.device),
            torch.argmax((label_index >= 0).float(), dim=1),
        ]
        first_label_index = first_label_index.unsqueeze(1).expand(
            batch_size, n_timesteps
        )

        # forward
        outputs = self.forward(
            inputs, **kwargs
        )  # shape: batch_size, n_timesteps, n_classes

        # calculate loss
        loss = self.compute_loss(
            outputs,
            label_index=label_index,
        )
        # calculate performance metrics
        guess_index = self.predictor(outputs)
        accuracy = self.calculate_accuracy(guess_index, label_index)

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }

        # Extract confidences at guess and first label positions
        if compute_confidence:
            guess_confidence, first_label_confidence = self.calculate_confidence(
                outputs, [guess_index, first_label_index]
            )
            metrics.update(
                {
                    "guess_confidence": guess_confidence.mean(),
                    "first_label_confidence": first_label_confidence.mean(),
                }
            )
        else:
            guess_confidence = None
            first_label_confidence = None

        if hasattr(self, "storage"):
            # Store records with flexible extras - all additional fields go to extras dict
            storage_kwargs = {
                "guess_index": guess_index,
                "label_index": label_index,
                "image_index": image_index,
                "first_label_index": first_label_index,
            }
            if compute_confidence:
                storage_kwargs.update(
                    {
                        "guess_confidence": guess_confidence,
                        "first_label_confidence": first_label_confidence,
                    }
                )
            self.storage.store_records(**storage_kwargs)

        del outputs
        return metrics

    def predictor(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)

    # Lightning step methods
    ########################
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        batch_size = batch[0].size(0)
        metrics = self.model_step(batch, batch_idx, compute_confidence=False)

        train_metrics = {f"train_{k}": v for k, v in metrics.items()}
        self.log_dict(
            train_metrics,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
            rank_zero_only=True,
        )
        return metrics

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, float]: Validation loss and accuracy.
        """

        batch_size = batch[0].size(0)

        with torch.no_grad():
            metrics = self.model_step(batch, batch_idx)

        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        self.log_dict(
            val_metrics,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
            rank_zero_only=True,
        )

        torch.cuda.empty_cache()
        gc.collect()
        return metrics

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform a single test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, float]: Test loss and accuracy.
        """
        batch_size = batch[0].size(0)
        metrics = self.model_step(batch, batch_idx)

        test_metrics = {f"test_{k}": v for k, v in metrics.items()}
        self.log_dict(
            test_metrics,
            prog_bar=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
            rank_zero_only=True,
        )
        return metrics

    def compute_loss(
        self,
        outputs: torch.Tensor,
        label_index: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, *_, n_classes = outputs.shape

        # Flatten time dimension
        outputs = outputs.view(-1, n_classes)
        label_index = label_index.reshape(-1)

        # Quick validation
        invalid_mask = (label_index < 0) | (label_index >= n_classes)
        if invalid_mask.all():
            logger.warning(f"All labels invalid! \n {label_index}")
            # return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        # Calculate loss for each criterion
        loss_values = torch.zeros(len(self.criterion), device=outputs.device)

        for i, criterion_fn in enumerate(self.criterion):
            if isinstance(criterion_fn, tuple):
                criterion_fn, weight = criterion_fn
            else:
                weight = 1

            loss_value = weight * criterion_fn(outputs, label_index)
            loss_values[i] = loss_value

            self.log_dict(
                {f"loss/{criterion_fn.__class__.__name__}": loss_value},
                batch_size=batch_size,
                sync_dist=True,
                rank_zero_only=True,
            )

        loss = loss_values.sum()

        # Quick NaN check
        if torch.isnan(loss):
            logger.warning("⚠️  NaN loss detected")

        return loss

    def calculate_confidence(
        self,
        outputs: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate confidence scores for the given outputs."""
        return calculate_confidence(outputs, indices)

    def calculate_accuracy(
        self,
        label_index: torch.Tensor,
        guess_index: torch.Tensor,
    ) -> float:
        return calculate_accuracy(label_index, guess_index)

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: Any = None,
        optimizer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Perform backward pass with optional retain_graph.

        Args:
            loss: The loss tensor to backpropagate
            optimizer: The optimizer (unused, for Lightning compatibility)
            optimizer_idx: The optimizer index (unused, for Lightning compatibility)
            *args: Additional positional arguments (unused, for Lightning compatibility)
            **kwargs: Additional keyword arguments (unused, for Lightning compatibility)
        """
        if self.retain_graph:
            loss.backward(retain_graph=True)
        else:
            try:
                loss.backward()
            except RuntimeError as e:
                if "retain_graph" in str(e) or "computation graph" in str(e):
                    logger.warning(f"Retrying with retain_graph=True due to: {e}")
                    loss.backward(retain_graph=True)
                    self.retain_graph = True
                else:
                    raise e

    # Optimizer configuration
    #########################
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        This is a standard PyTorch Lightning hook that sets up the optimizer
        with appropriate parameter groups and learning rates. Learning rate scaling
        based on batch size is handled by the trainer configuration.

        Returns:
            Dict containing optimizer and scheduler configurations

        Raises:
            ValueError: If required optimizer or scheduler configurations are invalid
        """
        self.print_trainable_parameter_names()

        base_lr = self._get_base_learning_rate()

        optimizer = self._create_optimizer(base_lr)

        lr_scheduler_config = self._create_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
            **self.optimizer_configs,
        }

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.log_level.upper() == "DEBUG":
            with torch.no_grad():
                self._check_gradients()
                self._check_weights(raise_error=True)

    def _group_lr_parameters(self, base_lr: float) -> List[Dict[str, Any]]:
        """Group parameters with appropriate learning rates and gradient clipping.

        This helper method organizes parameters into groups based on their type
        (regular, recurrent, or feedback) and assigns appropriate learning rates
        and gradient clipping settings to each group.

        Args:
            base_lr: The base learning rate after batch size scaling
            recurrence_factor: Factor to scale recurrent weights' learning rate
            feedback_factor: Factor to scale feedback weights' learning rate

        Returns:
            List of parameter group dictionaries for the optimizer
        """
        params = {key: [] for key in self.lr_parameter_groups.keys()}
        params["regular"] = []

        # Categorize parameters
        for name, param in self.named_trainable_parameters():
            for key in params.keys():
                if key != "regular" and key in name:
                    params[key].append(param)
                    break
            else:
                params["regular"].append(param)

        param_groups = []

        for group_name, group_params in params.items():

            if group_params:  # Only create groups with parameters
                group_config = self.lr_parameter_groups.get(group_name, {})
                group_config["name"] = group_name
                group_config["params"] = group_params

                # Set learning rate based on group configuration
                lr_factor = group_config.pop("lr_factor", 1.0)
                if "lr" not in group_config:
                    group_config["lr"] = base_lr * lr_factor

                param_groups.append(group_config)

        return param_groups

    def _get_base_learning_rate(self) -> float:
        """Get the base learning rate from model attributes.

        Returns:
            float: Base learning rate value
        """
        base_lr = getattr(self, "lr", getattr(self, "learning_rate", None))
        if base_lr is None:
            raise ValueError("No learning rate specified in model attributes")
        return base_lr

    def _create_optimizer(self, scaled_lr: float) -> torch.optim.Optimizer:
        """Create and configure the optimizer.

        Args:
            scaled_lr: Scaled learning rate
            recurrence_lr_factor: Factor for scaling recurrent weights' learning rate
            feedback_lr_factor: Factor for scaling feedback weights' learning rate

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if isinstance(self.optimizer, str):
            if not hasattr(torch.optim, self.optimizer):
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

            param_groups = self._group_lr_parameters(scaled_lr)

            optimizer_class = getattr(torch.optim, self.optimizer)

            optimizer = optimizer_class(param_groups, **self.optimizer_kwargs)
            return optimizer
        else:
            return self.optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Create and configure the learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            Dict[str, Any]: Scheduler configuration
        """
        if hasattr(lr_scheduler, self.scheduler):
            scheduler = getattr(lr_scheduler, self.scheduler)(
                optimizer, **self.scheduler_kwargs
            )
            return {
                "scheduler": scheduler,
                **self.scheduler_configs,
            }
        elif hasattr(torch.optim.lr_scheduler, self.scheduler):
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)(
                optimizer, **self.scheduler_kwargs
            )
            return {
                "scheduler": scheduler,
                **self.scheduler_configs,
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

    def update_criterion_params(
        self, criterion_name: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Generalized updater for self.criterion_params.

        - Merges non-None entries from `config` into any existing criterion configs
            whose name matches `criterion_name` (case-insensitive).
        - If no matching criterion exists, will append a new (name, config) tuple
            only when `config` contains at least one non-None value.
        - Avoids mutating caller-owned dicts by copying configs.
        """
        if config is None:
            return

        # Ensure criterion_params exists and is a list
        if not hasattr(self, "criterion_params") or self.criterion_params is None:
            self.criterion_params = []

        name_lower = criterion_name.lower()
        updated = False

        for idx, (cname, cconfig) in enumerate(list(self.criterion_params)):
            if isinstance(cname, str) and cname.lower() == name_lower:
                # copy existing config (or start new) and merge non-None updates
                new_config = dict(cconfig) if isinstance(cconfig, dict) else {}
                for k, v in config.items():
                    if v is not None:
                        new_config[k] = v
                # replace tuple with copied config to avoid mutating caller objects
                self.criterion_params[idx] = (cname, dict(new_config))
                logger.debug(
                    f"Updated criterion_params[{idx}] ({cname}) with {config}"
                )
                updated = True

        # If nothing matched, append only if config contains meaningful values
        if not updated:
            if any(v is not None for v in config.values()):
                self.criterion_params.append((criterion_name, dict(config)))
                logger.debug(
                    f"Appended new criterion '{criterion_name}' with {config}"
                )
            else:
                logger.debug(
                    f"No update/appended for '{criterion_name}' because config values were all None"
                )

    # Logging
    #########
    def log_table(
        self,
        key: str,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        step: Optional[int] = None,
    ) -> None:
        if self.logger:
            self.logger.log_table(key=key, dataframe=dataframe, step=step)

    def log_figure(
        self, fig: plt.Figure, key: str, step: Optional[int] = None
    ) -> None:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        self.log(
            {f"{key}", wandb.Image(buffer, caption=key)},
            step=step,
            rank_zero_only=True,
        )

    # Hooks
    #######
    def on_fit_start(self) -> None:
        """Called at the start of fit."""
        super().on_fit_start() if hasattr(super(), "on_fit_start") else None

    def on_train_start(self) -> None:
        """Called at the start of training."""
        super().on_train_start() if hasattr(super(), "on_train_start") else None

    def on_train_end(self) -> None:
        """Called at the end of training."""
        super().on_train_end() if hasattr(super(), "on_train_end") else None

    def on_validation_start(self) -> None:
        """Called at the start of validation."""
        (
            super().on_validation_start()
            if hasattr(super(), "on_validation_start")
            else None
        )

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        (
            super().on_validation_epoch_end()
            if hasattr(super(), "on_validation_epoch_end")
            else None
        )

    def on_test_start(self) -> None:
        """Called at the start of testing."""
        super().on_test_start() if hasattr(super(), "on_test_start") else None

    def on_train_batch_start(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called before each training batch.

        Args:
            batch: Batch of input data and labels
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader (for multiple dataloaders)
        """
        if hasattr(super(), "on_train_batch_start"):
            super().on_train_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called before each validation batch.

        Args:
            batch: Batch of input data and labels
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader (for multiple dataloaders)
        """
        if hasattr(super(), "on_validation_batch_start"):
            super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called after each training batch.

        Args:
            outputs: Outputs from training_step
            batch: Batch of input data and labels
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader (for multiple dataloaders)
        """
        if hasattr(super(), "on_train_batch_end"):
            super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_before_optimizer_step(self, optimizer: Any, optimizer_idx: int = 0) -> None:
        """Called before optimizer step.

        Args:
            optimizer: The optimizer being stepped
            optimizer_idx: The optimizer index (for multiple optimizers)
        """
        if hasattr(super(), "on_before_optimizer_step"):
            super().on_before_optimizer_step(optimizer, optimizer_idx)
