import io
from typing import Any, Dict, List, Optional, Tuple
import logging

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from dynvision import losses
from dynvision.utils import alias_kwargs

logger = logging.getLogger(__name__)


class LightningBase(pl.LightningModule):
    """PyTorch Lightning integration for DynVision models."""

    @alias_kwargs(
        lr="learning_rate",
        solver="dynamics_solver",
        lossrt="loss_reaction_time",
    )
    def __init__(
        self,
        # Training configuration
        retain_graph: bool = False,
        # Loss configuration
        criterion_params: List[Tuple[str, Dict[str, Any]]] = [
            ("CrossEntropyLoss", {"weight": 1.0})
        ],
        loss_reaction_time: float = 0.0,
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
        super().__init__()
        pl.LightningModule.__init__(self)

        # Store Lightning-specific attributes
        self.retain_graph = retain_graph
        self.criterion_params = criterion_params
        self.loss_reaction_time = float(loss_reaction_time)
        self.non_label_index = non_label_index

        # Optimizer attributes
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_configs = optimizer_configs
        self.learning_rate = float(learning_rate)
        self.lr_parameter_groups = lr_parameter_groups
        self.loss_reaction_time = float(loss_reaction_time)

        # Scheduler attributes
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_configs = scheduler_configs

        # Logging attributes
        self.log_level = log_level
        self.log_every_n_steps = int(log_every_n_steps)

    # Core training steps
    #####################
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, **kwargs
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:

        if hasattr(self, "_process_batch"):
            batch = self._process_batch(batch, batch_idx)

        inputs, label_indices, *paths = batch

        # forward
        outputs = self.forward(inputs, **kwargs)

        # calculate loss
        loss = self.compute_loss(
            outputs,
            label_indices=label_indices,
        )
        # calculate accuracy
        guess_indices = self.predictor(outputs)
        accuracy = self.calc_accuracy(label_indices, guess_indices)

        if hasattr(self, "storage"):
            self.storage.store_records(
                guess_indices, label_indices, label_indices
            )  # todo: flexibly accept image_indices

        del outputs
        return loss, accuracy

    def predictor(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(outputs, dim=-1)

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
        loss, accuracy = self.model_step(batch, batch_idx)

        metrics = {"train_loss": loss, "train_accuracy": accuracy}
        self.log_dict(
            metrics,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
            rank_zero_only=True,
        )
        return loss

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
        loss, accuracy = self.model_step(batch, batch_idx)

        metrics = {"val_loss": loss, "val_accuracy": accuracy}
        self.log_dict(
            metrics,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
            rank_zero_only=True,
        )

        return loss, accuracy

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
        loss, accuracy = self.model_step(batch, batch_idx)

        metrics = {"test_loss": loss, "test_accuracy": accuracy}
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
            rank_zero_only=True,
        )
        return loss, accuracy

    # Loss and accuracy
    ###################
    def _init_loss(self) -> None:
        self.criterion = []

        if hasattr(self, "loss_reaction_time") and self.loss_reaction_time:
            self.ignore_initial_n_labels = self.n_residual_timesteps + int(
                self.loss_reaction_time / self.dt
            )
        else:
            self.ignore_initial_n_labels = 0

        for criterion_name, criterion_config in self.criterion_params:
            # Set criterion weight
            if "weight" in criterion_config.keys():
                criterion_weight = criterion_config.pop("weight")
            else:
                criterion_weight = 1

            # Add ignore_index for cross entropy loss if not specified
            if (
                criterion_name.lower() in ["crossentropyloss", "cross_entropy_loss"]
                and "ignore_index" not in criterion_config
            ):
                criterion_config["ignore_index"] = self.non_label_index
                logger.info(
                    f"Setting ignore_index={self.non_label_index} for {criterion_name}"
                )

            logger.info(
                f"Criterion: {criterion_name} with weight: {criterion_weight} and config: {criterion_config}"
            )

            # Init loss
            if hasattr(losses, criterion_name):
                criterion_fn = getattr(losses, criterion_name)(**criterion_config)
                self.criterion += [(criterion_fn, criterion_weight)]

            else:
                raise ValueError(f"Invalid loss function: {criterion_name}")

            # Init hooks if required
            try:
                criterion_fn.register_hooks(self)
                logger.debug(f"registered hook for {criterion_fn}")
            except Exception as e:
                pass

        return None

    def compute_loss(
        self,
        outputs: torch.Tensor,
        label_indices: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, *_, n_classes = outputs.shape

        # Apply loss reaction time
        if hasattr(self, "ignore_initial_n_labels") and self.ignore_initial_n_labels:
            label_indices[:, : self.ignore_initial_n_labels] = self.non_label_index

        # Flatten time dimension
        outputs = outputs.view(-1, n_classes)
        label_indices = label_indices.reshape(-1)

        # Quick validation
        invalid_mask = (label_indices < 0) | (label_indices >= n_classes)
        if invalid_mask.all():
            logger.warning(f"All labels invalid! \n {label_indices}")
            # return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        # Calculate loss for each criterion
        loss_values = torch.zeros(len(self.criterion), device=outputs.device)

        for i, criterion_fn in enumerate(self.criterion):
            if isinstance(criterion_fn, tuple):
                criterion_fn, weight = criterion_fn
            else:
                weight = 1

            loss_value = weight * criterion_fn(outputs, label_indices)
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
            logger.warning(f"⚠️  NaN loss detected")

        return loss

    def calc_accuracy(
        self,
        label_indices: torch.Tensor,
        guess_indices: torch.Tensor,
    ) -> float:

        # Create mask for valid labels (excluding non_label_index)
        valid_mask = (label_indices >= 0) & (label_indices < self.n_classes)
        if not valid_mask.all():
            if valid_mask.any():
                label_indices = label_indices[valid_mask]
                guess_indices = guess_indices[valid_mask]
            else:
                logger.warning("All labels invalid, returning zero accuracy")
                return 0.0

        accuracy = (guess_indices == label_indices).float().mean().item()
        return accuracy

    def backward(self, loss: torch.Tensor) -> None:
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
                if not "lr" in group_config:
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
        if not hasattr(torch.optim.lr_scheduler, self.scheduler):
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)(
            optimizer, **self.scheduler_kwargs
        )

        return {
            "scheduler": scheduler,
            **self.scheduler_configs,
        }

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
    def setup(self, stage: Optional[str] = None) -> None:
        """Lightning setup hook."""
        super().setup(stage) if hasattr(super(), "setup") else None

        # Initialize loss functions
        self._init_loss()

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
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Called before each training batch."""
        (
            super().on_train_batch_start(batch, batch_idx)
            if hasattr(super(), "on_train_batch_start")
            else None
        )

    def on_validation_batch_start(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Called before each validation batch."""
        (
            super().on_validation_batch_start(batch, batch_idx)
            if hasattr(super(), "on_validation_batch_start")
            else None
        )

    def on_train_batch_end(
        self, outputs: Any, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Called after each training batch."""
        (
            super().on_train_batch_end(outputs, batch, batch_idx)
            if hasattr(super(), "on_train_batch_end")
            else None
        )

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Called before optimizer step."""
        (
            super().on_before_optimizer_step(optimizer)
            if hasattr(super(), "on_before_optimizer_step")
            else None
        )
