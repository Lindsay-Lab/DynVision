"""Test a trained neural network model on a dataset with comprehensive Pydantic parameter management.

This script handles the complete testing pipeline for DynVision models with type-safe,
validated parameter handling using composite Pydantic configuration classes.

Features:
- Composite configuration management (ModelParams + TrainerParams + DataParams + TestingParams)
- Automatic parameter validation and memory optimization in TestingParams
- Comprehensive response storage and analysis
- Memory-conscious processing handled by configuration
- Advanced error handling with detailed feedback
- Results export in multiple formats (CSV, tensors)
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Dict

import pytorch_lightning as pl
import torch

from dynvision import models

from dynvision.data.datamodule import TestingDataModule
from dynvision.utils import (
    filter_kwargs,
    handle_errors,
    log_section,
    format_value,
)

# Import the Pydantic parameter classes
from dynvision.params.testing_params import (  # noqa: F401
    TestingParams,
    DynVisionValidationError,
)

logger = logging.getLogger(__name__)


class TestingModelManager:
    """Enhanced model management for testing with configuration integration."""

    def __init__(self, config: TestingParams):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_and_configure_model(self) -> pl.LightningModule:
        """Load and configure model for testing using TestingParams configuration."""
        logger.info(
            f"Loading model {self.config.model.model_name} from {self.config.input_model_state}"
        )

        # Load state dict
        state_dict = torch.load(
            self.config.input_model_state, map_location=self.device, weights_only=True
        )

        # Create model with current configuration (already optimized by TestingParams)
        model_class = getattr(models, self.config.model.model_name)
        model_kwargs = self.config.get_model_kwargs(model_class)

        model = model_class(**model_kwargs).to(self.device)

        # init dynamically created connections by running a forward pass with dummy data
        if hasattr(model, "_initialize_connections"):
            model._initialize_connections()

        logger.debug("Model kwargs: %s", model_kwargs)
        logger.debug("State dict keys: %s", list(model.state_dict().keys()))
        logger.debug("Checkpoint keys: %s", list(state_dict.keys()))

        # Load state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.warning(f"Strict loading failed: {e}. Trying non-strict...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected}")

        # Configure model for testing
        model.eval()  # Set to evaluation mode

        # Set residual timesteps if available
        if hasattr(model, "set_residual_timesteps"):
            model.set_residual_timesteps()

        return model

    def extract_n_classes_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> int:
        """Extract number of classes from model state dict."""
        # Look for classifier layer first
        for key in state_dict.keys():
            if "classifier" in key and "weight" in key and "source" not in key:
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


class TestingOrchestrator:
    """Enhanced testing orchestrator with comprehensive configuration management."""

    def __init__(self, config: TestingParams):
        self.config = config
        self.datamodule = TestingDataModule(
            config=config,
            dataset_path=config.dataset_path,
        )
        self.model_manager = TestingModelManager(config)

    @contextmanager
    def testing_context(self):
        """Enhanced testing context with configuration logging and memory management."""
        try:
            # Setup
            torch.set_float32_matmul_precision("medium")

            # Clear GPU cache before testing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            yield

        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_testing_configuration(self) -> None:
        """Log key testing configuration information."""
        self.config.log_testing_overview(logger=logger)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        entries = [
            ("n_classes", format_value(len(self.datamodule.dataset.classes)), None),
            ("device", format_value(str(device)), None),
        ]

        if torch.cuda.is_available():
            entries.extend(
                [
                    ("gpu_name", format_value(torch.cuda.get_device_name()), None),
                    (
                        "gpu_memory_gb",
                        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
                        None,
                    ),
                ]
            )

        log_section(logger, "testing_environment", entries)

    def _preview_log_level(self) -> int:
        """Use INFO for verbose testing runs, else keep preview noise at DEBUG."""
        return (
            logging.INFO if getattr(self.config, "verbose", False) else logging.DEBUG
        )

    def infer_and_update_from_data(self) -> None:
        """
        Infer model parameters from test data and update configuration.

        This method loads a sample batch, extracts dimensions, validates consistency,
        and updates the model configuration accordingly.
        """
        try:
            # Get sample batch for dimension inference
            inputs, labels = self.datamodule.get_sample_batch()

            # Extract actual dimensions
            batch_size, actual_n_timesteps, *spatial_dims = inputs.shape
            actual_input_dims = (actual_n_timesteps, *spatial_dims)

            preview_level = self._preview_log_level()
            label_str = " ".join(str(int(x)) for x in labels[0])
            pixel_mean = inputs.mean().item()
            pixel_std = inputs.std().item()
            log_section(
                logger,
                "Preview batch",
                [
                    ("Batch shape", format_value(tuple(inputs.size())), None),
                    ("Input dims", format_value(actual_input_dims), None),
                    ("Pixel stats", f"{pixel_mean:.3f} ± {pixel_std:.3f}", None),
                    ("Label presentation", label_str, None),
                ],
                level=preview_level,
            )

            # Extract n_classes from state dict
            state_dict = torch.load(
                self.config.input_model_state, map_location="cpu", weights_only=True
            )
            actual_n_classes = self.model_manager.extract_n_classes_from_state_dict(
                state_dict
            )

            # Update model parameters using the dedicated method
            self.config.update_model_parameters_from_data(
                input_dims=actual_input_dims,
                n_classes=actual_n_classes,
                dataset_size=len(self.datamodule.dataset),
                verbose=True,
            )

        except RuntimeError as e:
            # RuntimeError from get_sample_batch indicates critical dataloader issue
            logger.error(f"Critical error during parameter inference: {e}")
            raise  # Re-raise to stop execution
        except Exception as e:
            logger.error(f"Failed to infer parameters from data: {e}", exc_info=True)
            logger.warning("Continuing with configuration defaults")
            if self.config.verbose:
                import traceback

                traceback.print_exc()

    def setup_trainer(self) -> pl.Trainer:
        """Setup PyTorch Lightning trainer for testing using TestingParams configuration."""
        # Get trainer configuration (already optimized by TestingParams)
        trainer_kwargs = self.config.get_trainer_kwargs()

        # # Optional: Setup logger if needed
        # if hasattr(self.config, "logger") and getattr(self.config, "logger", None):
        #     trainer_kwargs["logger"] = pl.loggers.WandbLogger(
        #         project=project_paths.project_name,
        #         save_dir=project_paths.large_logs,
        #         config=(
        #             self.config.get_full_config(flat=True)
        #             if hasattr(self.config, "get_full_config")
        #             else {}
        #         ),
        #         tags=["test"],
        #         name=f"test_{self.config.input_model_state.name}",
        #     )

        # Filter valid arguments for Trainer
        trainer_kwargs, unknown = filter_kwargs(pl.Trainer, trainer_kwargs)
        if unknown:
            logger.debug(f"Filtered unknown trainer kwargs: {list(unknown.keys())}")

        self.config.log_trainer_creation(
            trainer_kwargs=trainer_kwargs,
            logger=logger,
        )

        # wandb.init(settings=wandb.Settings(init_timeout=120))  # hack to log histograms, but testing is not logged
        return pl.Trainer(**trainer_kwargs)

    def save_results(self, model: pl.LightningModule, precision: int = 16) -> None:
        """
        Save test results and model responses.

        Args:
            model: The model to extract results from
            precision: Bit precision to save response tensors (16 or 32)
        """
        logger.info(f"Saving results to {self.config.output_results}")

        # Save test results (CSV)
        try:
            results_df = model.storage.get_dataframe()
            if not results_df.empty:
                results_df.to_csv(self.config.output_results, index=False)
                logger.info(f"Test results saved to {self.config.output_results}")
                logger.info(f"Results shape: {results_df.shape}")
            else:
                logger.warning("No test results to save (empty DataFrame)")

            # Free memory from results_df
            del results_df
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

        # Save model responses (tensors)
        try:
            if hasattr(model, "storage") and hasattr(model.storage, "responses"):
                response_data = model.storage.responses.get_all()

                # Check if we have any responses
                if not response_data or len(response_data) == 0:
                    logger.warning(
                        "No model responses recorded - skipping response file"
                    )
                    return

                logger.info(
                    f"Saving model responses with precision={precision} bits..."
                )
                total_size_mb = 0
                responses = {}

                # Target dtype based on precision parameter
                target_dtype = torch.float16 if precision == 16 else torch.float32

                for layer in response_data[0].keys():
                    layer_responses = [item[layer] for item in response_data]
                    tensor = torch.cat(layer_responses, dim=0)

                    # Move to CPU and convert precision if needed
                    if tensor.is_cuda:
                        tensor = tensor.cpu()
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(dtype=target_dtype)

                    responses[layer] = tensor
                    size_mb = responses[layer].nbytes / (1024 * 1024)
                    total_size_mb += size_mb
                    logger.info(
                        f"  Layer {layer}: {responses[layer].shape}, {responses[layer].dtype} -> {size_mb:.2f} MB"
                    )

                # Inject metadata about storage format
                response_resolution = getattr(
                    getattr(model, "storage", None), "response_resolution", "unit"
                )
                responses["_metadata"] = {
                    "response_resolution": response_resolution,
                    "version": 1,
                }

                # Save responses
                logger.info("Writing response tensors to disk...")
                torch.save(responses, self.config.output_responses)
                logger.info(f"Model responses saved to {self.config.output_responses}")
                logger.info(f"Total response size: {total_size_mb:.2f} MB")

                # Explicit cleanup to prevent OOM and resource leaks
                logger.debug("Cleaning up response tensors from memory...")
                del responses
                del response_data
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.debug("Response cleanup completed")
            else:
                logger.warning(
                    "No storage or responses available - skipping response file"
                )
        except MemoryError as e:
            logger.error(f"Out of memory while saving model responses: {e}")
            logger.error("Consider reducing store_responses or batch_size")
            # Don't save empty dict - let Snakemake detect failure by missing output
            raise
        except Exception as e:
            logger.error(f"Failed to save model responses: {e}")
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                logger.error(
                    "GPU/CPU out of memory - reduce batch_size or store_responses"
                )
            # Don't save empty dict - let Snakemake detect failure by missing output
            raise

    def run_testing(self) -> int:
        """Run the complete testing pipeline with comprehensive error handling."""
        with self.testing_context():
            try:
                # Setup data (TestingParams already optimized all parameters)
                self.datamodule.setup_dataset()
                dataloader = self.datamodule.setup_dataloader()
                self.config.data.log_configuration(dataloader=dataloader)

                # Infer and update model parameters from data
                self.infer_and_update_from_data()

                # Persist resolved configuration before long-running evaluation begins
                self.config.persist_resolved_config(
                    primary_output=self.config.output_results,
                    script_name=__file__,
                )

                # Load and configure model
                model_class = getattr(models, self.config.model.model_name)
                model_kwargs = self.config.model.get_model_kwargs(model_class)
                self.config.model.log_model_creation(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                )
                model = self.model_manager.load_and_configure_model()

                # Setup trainer
                trainer = self.setup_trainer()

                # Log testing configuration
                self._log_testing_configuration()

                # Log memory usage before testing
                if torch.cuda.is_available():
                    logger.info(
                        f"GPU memory before testing: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
                    )

                # Run testing with early stopping exception handling
                logger.info("Starting model testing...")
                try:
                    trainer.test(model, dataloader)
                except pl.utilities.exceptions._TunerExitException as e:
                    logger.info(f"Testing stopped early: {e}")
                    logger.info(
                        "This is expected behavior when using early_test_stop=True"
                    )
                    logger.info(
                        f"Collected {len(model.storage.responses)} response samples"
                    )
                except Exception as e:
                    # Handle other exceptions
                    logger.error(f"Testing encountered an error: {e}")
                    if self.config.verbose:
                        import traceback

                        traceback.print_exc()

                # Log peak memory usage
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1e6
                    logger.info(f"Peak GPU memory usage: {peak_memory:.2f}MB")

                    if peak_memory > 1000:  # More than 1GB
                        logger.warning(
                            "High GPU memory usage detected. Consider reducing batch_size or "
                            "store_responses for future runs."
                        )

                # Save results - will happen even if testing was stopped early
                self.save_results(model)

                # Cleanup model, trainer, and dataloader to free memory and prevent resource leaks
                del model
                del trainer
                del dataloader
                import gc
                import time

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Small delay to allow multiprocessing workers to clean up properly
                time.sleep(0.5)

                logger.debug("Cleaned up model, trainer, and dataloader resources")
                logger.info("Testing completed successfully!")
                return 0

            except Exception as e:
                logger.error(f"Testing failed: {e}")
                if self.config.verbose:
                    import traceback

                    traceback.print_exc()
                return 1


@handle_errors(verbose=True)
def main() -> int:
    """Main entry point for testing with comprehensive configuration management."""
    config = TestingParams.from_cli_and_config()
    config.setup_logging()

    orchestrator = TestingOrchestrator(config)
    return orchestrator.run_testing()


if __name__ == "__main__":
    # Set process start method for compatibility
    if os.environ.get("SLURM_JOB_ID"):
        import multiprocessing

        multiprocessing.set_start_method("spawn", force=True)

    # Run testing
    sys.exit(main())
