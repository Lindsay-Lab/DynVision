"""Debugging, logging, and monitoring utilities."""

import logging
import gc
import psutil
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from pytorch_lightning import LightningModule
import wandb
from dynvision.data.operations import _adjust_data_dimensions, _adjust_label_dimensions

logger = logging.getLogger(__name__)


class Monitoring:
    """Core debugging and monitoring functionality."""

    # Tensor checking
    #################
    def _check_tensors(
        self,
        generator_name: str = "named_parameters",
        data_attr: str = "data",
        raise_error: bool = False,
    ) -> None:
        """
        Check for NaN/Inf values and dtype consistency in model parameters.
        """
        iterator = getattr(self, generator_name)
        if isinstance(iterator, dict):
            iterator = iterator.items()
        elif callable(iterator):
            iterator = iterator()
        else:
            raise ValueError(
                f"The attribute {generator_name} is neither a dict nor a generator."
            )

        nonfinite_detected = False
        model_dtype = next(self.parameters()).dtype
        dtype_mismatches = []

        logger.info(f"\nChecking {generator_name} {data_attr}:")
        logger.info("-" * 100)
        logger.info(
            f"{'Module Name':<30} {'Shape':<20} {'Type':<16} {'Device':<8} {'Min':>8} {'Max':>8} {'Norm':>8}"
        )
        logger.info("-" * 100)

        for name, tensor in iterator:
            if data_attr and self.hasattr(tensor, data_attr):
                tensor = self.getattr(tensor, data_attr)
            else:
                logger.debug(f"Attribute {data_attr} not found in {name}!")

            if isinstance(tensor, list):
                tensor = self._concatenate_tensors(tensor, dim=0)

            # Check dtype consistency
            if tensor.dtype != model_dtype:
                dtype_mismatches.append((name, tensor.dtype, model_dtype))

            if len(tensor.size()):
                valid_data = tensor[torch.isfinite(tensor)]
                if valid_data.numel() > 0:
                    shape_str = str(tensor.size()).replace("torch.Size", "")
                    logger.info(
                        f"{name:<30} {shape_str:<20} {str(tensor.dtype):<16} {str(tensor.device):<8} "
                        f"{valid_data.min().item():>8.3f} {valid_data.max().item():>8.3f} {valid_data.norm().item():>8.3f}"
                    )
                else:
                    logger.warning(
                        f"{name:<30} {'[NaN/Inf]':<20} {str(tensor.dtype):<16} {str(tensor.device):<8} {'---':>8} {'---':>8} {'---':>8}"
                    )
            else:
                logger.info(
                    f"{name:<30} {'[scalar]':<20} {str(tensor.dtype):<16} {str(tensor.device):<8} {tensor:>8.3f} {tensor:>8.3f} {tensor:>8.3f}"
                )

            if (torch.isnan(tensor)).any():
                logger.warning(f"\t NaN detected in {name}: ")
                logger.warning(
                    f"\t {(torch.isnan(tensor)).sum().item()} / {tensor.numel()}"
                )
            if torch.isinf(tensor).any():
                logger.warning(f"\t Inf detected in {name}: ")
                logger.warning(
                    f"\t {(torch.isinf(tensor)).sum().item()} / {tensor.numel()}"
                )
            if (~torch.isfinite(tensor)).any():
                nonfinite_detected = True

        # Report dtype mismatches
        if dtype_mismatches:
            logger.warning("Detected dtype mismatches:")
            for name, param_dtype, expected_dtype in dtype_mismatches:
                logger.warning(f"\t {name}: {param_dtype} (expected {expected_dtype})")

        if raise_error and (nonfinite_detected or dtype_mismatches):
            raise ValueError("NaN/Inf values or dtype mismatches detected!")

        return None

    def _check_gradients(self, raise_error: bool = False) -> None:
        self._check_tensors(
            generator_name="named_trainable_parameters",
            data_attr="grad.data",
            raise_error=raise_error,
        )

    def _check_weights(self, raise_error: bool = False) -> None:
        self._check_tensors(
            generator_name="named_parameters",
            data_attr="data",
            raise_error=raise_error,
        )

    def _check_responses(self, raise_error: bool = False) -> None:
        self._check_tensors(
            generator_name="get_responses",
            data_attr="",
            raise_error=raise_error,
        )

    # Parameter inspection
    ######################
    def safely_named_parameters(self, module=None, replace={".": "_"}):
        if module is None:
            module = self

        if isinstance(module, nn.Sequential):
            generator = (
                (f"Sequential.{child_name}.{param_name}", param)
                for child_name, child in module.named_children()
                if hasattr(child, "named_parameters")
                for param_name, param in child.named_parameters()
            )

        elif hasattr(module, "named_parameters"):
            generator = module.named_parameters()

        else:
            logger.warning(
                f"module {module} has no named parameters to safely rename!"
            )
            generator = iter([])

        for name, param in generator:
            for k, v in replace.items():
                name = name.replace(k, v)
            yield name, param

    def get_safely_named_parameters_dict(self, module=None, replace={".": "_"}):
        if module is None:
            module = self
        if not hasattr(module, "safely_named_parameters_dict"):
            setattr(
                module,
                "safely_named_parameters_dict",
                {
                    k: v
                    for k, v in self.safely_named_parameters(
                        module=module, replace=replace
                    )
                },
            )
        return module.safely_named_parameters_dict

    # Logging utilities
    ###################
    def log_param_stats(
        self,
        section: str = "params",
        metrics: List[str] = ["hist", "norm"],  # min, max
        log_only_trainable: bool = False,
    ) -> None:
        """
        Log statistics of model parameters.

        Args:
            section (str, optional): Section name for logging. Defaults to "params".
            metrics (List[str], optional): List of metrics to log. Defaults to ["hist", "norm"].
            log_only_trainable (bool, optional): Whether to log only trainable parameters. Defaults to False.
        """
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                metrics = [m for m in metrics if m != "hist"]
            else:
                return
        
        for name, param in self.named_parameters():
            if log_only_trainable and not param.requires_grad:
                continue

            if len(param.data.size()):
                for metric in metrics:
                    if hasattr(param.data, metric):
                        self.log(
                            f"{section}/{name}_{metric}",
                            getattr(param.detach().data, metric)(),
                            sync_dist=True,
                            rank_zero_only=True,
                        )
                    elif metric == "hist":
                        wandb.log(
                            {
                                f"{section}/{name}_{metric}": wandb.Histogram(
                                    param.detach().cpu().flatten()
                                ),
                            },
                        )
                    else:
                        logger.debug(f"Metric {metric} not available!")
            else:
                self.log(f"{section}/{name}", param.detach().data, sync_dist=True, rank_zero_only=True)

    # System monitoring
    ###################
    def _log_system_info(self) -> None:
        """Log essential system information at training start."""
        logger.info("=" * 60)
        logger.info("🚀 TRAINING STARTED")
        logger.info("=" * 60)

        # Model basics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Model: {self.__class__.__name__} | Params: {total_params:,} ({trainable_params:,} trainable)"
        )
        logger.info(
            f"Config: {self.n_classes} classes | {self.n_timesteps} timesteps | non_label_idx: {self.non_label_index}"
        )

        # System info
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            logger.info(f"Device: {device_name}")

    def _log_memory_usage(self) -> None:
        """Log detailed and consistent memory and CPU usage statistics."""
        process = psutil.Process()
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        process_mem_gb = process.memory_info().rss / (1024**3)

        logger.info(
            f"\t{'System RAM':<18}: {ram_used_gb:6.2f} GB / {ram_total_gb:.2f} GB ({ram.percent:.1f}%)"
        )
        logger.info(f"\t{'CPU Usage':<18}: {cpu_percent:6.1f}%")
        logger.info(f"\t{'Process Memory':<18}: {process_mem_gb:6.2f} GB")

        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
            gpu_mem_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(
                f"\t{'GPU Memory Used':<18}: {gpu_mem_gb:6.2f} GB / {gpu_total_gb:.2f} GB"
            )
            logger.info(
                f"\t{'GPU Memory Reserved':<18}: {gpu_mem_reserved_gb:6.2f} GB / {gpu_total_gb:.2f} GB"
            )

    def _log_training_summary(self) -> None:
        """Log training completion summary."""
        try:
            logger.info(
                f"✅ Training completed: {self.trainer.current_epoch} epochs | {self.trainer.global_step} steps"
            )
        except RuntimeError:
            return

    def count(self, label="counter:") -> None:
        if hasattr(self, "_count"):
            self._count += 1
        else:
            self._count = 1

        logger.info(f"{label} {self._count}")


class MonitoringMixin(Monitoring, LightningModule):
    """Monitoring with PyTorch Lightning integration."""

    # Lightning hooks
    #################
    def on_train_start(self) -> None:
        """Initialize training with weight check and basic system info."""
        try:
            super().on_train_start()
        except AttributeError:
            pass

        self._check_weights()
        self._log_system_info()

        if hasattr(self.logger, "watch") and not hasattr(self, "_wandb_watched"):
            try:
                self.logger.watch(self)
                self._wandb_watched = True  # Mark as watched
            except ValueError as e:
                if "can only call" in str(e) and "wandb.watch" in str(e):
                    # Already being watched, skip silently
                    pass
                else:
                    raise e

    def on_train_end(self) -> None:
        """Final diagnostics and cleanup."""
        try:
            super().on_train_end()
        except AttributeError:
            pass
        # self._check_responses()
        self._log_training_summary()

    def on_train_epoch_start(self) -> None:
        try:
            super().on_train_epoch_start()
        except AttributeError:
            pass

        self.count("Epoch")
        self._log_memory_usage()

    def on_train_batch_start(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Check for critical data issues early in training."""
        try:
            super().on_train_batch_start(batch, batch_idx)
        except AttributeError:
            pass
        if batch_idx < 2:  # Only check first few batches
            self._validate_batch_data(batch, batch_idx, "train")

    def on_validation_batch_start(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Check for critical data issues in validation."""
        try:
            super().on_validation_batch_start(batch, batch_idx)
        except AttributeError:
            pass
        if batch_idx == 0:  # Only check first validation batch
            self._validate_batch_data(batch, batch_idx, "val")
            self._log_memory_usage()

    def on_train_batch_end(
        self, outputs: Any, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Monitor training progress and check for issues."""
        try:
            super().on_train_batch_end(outputs, batch, batch_idx)
        except AttributeError:
            pass
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs.get("loss")
        self._check_training_health(loss, batch_idx)

        if batch_idx % self.log_every_n_steps == 0:
            self.log_param_stats()

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Check gradients before optimizer step."""
        try:
            super().on_before_optimizer_step(optimizer)
        except AttributeError:
            pass
        if self.log_level.upper() == "DEBUG":
            self._check_gradients()

    # Batch validation
    ##################
    def _validate_batch_data(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ) -> None:
        """Validate batch data for critical issues."""
        inputs, labels = batch[:2]

        # Adjust dimensions for validation
        inputs = _adjust_data_dimensions(inputs)
        labels = _adjust_label_dimensions(labels)

        # Check for NaN/Inf in inputs
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logger.warning(
                f"⚠️  [{stage.upper()}] Batch {batch_idx}: NaN/Inf detected in inputs"
            )

        # Validate label range
        label_min, label_max = labels.min().item(), labels.max().item()
        if label_min < 0 or label_max >= self.n_classes:
            invalid_labels = labels[(labels < 0) | (labels >= self.n_classes)]
            unique_invalid = torch.unique(invalid_labels).tolist()
            logger.warning(
                f"⚠️  [{stage.upper()}] Batch {batch_idx}: Invalid labels {unique_invalid} (expect 0-{self.n_classes-1})"
            )

        # Log first batch info
        if batch_idx == 0:
            logger.info(
                f"📦 [{stage.upper()}] First batch: {inputs.shape} | Labels: [{label_min}, {label_max}] | Device: {inputs.device}"
            )

    def _check_training_health(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Check for training health issues."""
        if loss is None:
            return

        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"⚠️  Batch {batch_idx}: Loss is {'NaN' if torch.isnan(loss) else 'Inf'}"
            )

        # Check for extremely high loss
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        if loss_val > 100:
            logger.warning(f"⚠️  Batch {batch_idx}: Very high loss {loss_val:.4f}")

    def _should_log_detailed(self, batch_idx: int, stage: str) -> bool:
        """Determine if detailed logging should occur."""
        try:
            epoch = self.trainer.current_epoch
            # Log more frequently early in training
            if epoch < 2:
                return batch_idx % 20 == 0
            elif epoch < 5:
                return batch_idx % 50 == 0
            else:
                return batch_idx % 100 == 0
        except RuntimeError:
            return batch_idx < 5
