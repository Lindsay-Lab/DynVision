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

Example:
    $ python test_model.py --config_path configs/test_config.yaml --model_name DyRCNNx4
"""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import wandb

from dynvision import models
from dynvision.data.dataloader import (
    StandardDataLoader,
    _adjust_data_dimensions,
    _adjust_label_dimensions,
)
from dynvision.data.dataloader import get_data_loader
from dynvision.data.datasets import get_dataset
from dynvision.project_paths import project_paths
from dynvision.utils import (
    filter_kwargs,
    handle_errors,
)

# Import the Pydantic parameter classes
from dynvision.params.testing_params import (
    TestingParams,
    DynVisionValidationError,
)

logger = logging.getLogger(__name__)


class TestingDataModule:
    """Simplified data module optimized for testing workflows."""

    def __init__(self, config: TestingParams):
        self.config = config
        self.dataset = None
        self.dataloader = None

    def setup_dataset(self) -> Dataset:
        """Set up the test dataset with configuration from TestingParams."""
        if self.dataset is not None:
            return self.dataset

        logger.info(f"Loading test dataset from {self.config.dataset}")

        # Get dataset configuration from TestingParams (all optimizations already applied)
        dataset_kwargs = self.config.data.get_dataset_kwargs()

        self.dataset = get_dataset(self.config.dataset, **dataset_kwargs)

        total_samples = len(self.dataset)
        logger.info(f"Test dataset loaded with {total_samples} samples")

        return self.dataset

    def setup_dataloader(self) -> DataLoader:
        """Set up the test data loader using configuration from TestingParams."""
        if self.dataloader is not None:
            return self.dataloader

        if self.dataset is None:
            self.setup_dataset()

        # Get dataloader configuration (already optimized by TestingParams)
        dataloader_name = self.config.data.data_loader
        dataloader_config = self.config.get_dataloader_kwargs()

        logger.info(
            f"Creating DataLoader {dataloader_name} "
            f"with batch_size={dataloader_config['batch_size']}, "
            f"num_workers={dataloader_config['num_workers']}, "
            f"shuffle={dataloader_config['shuffle']}, "
            f"non_label_index{dataloader_config['non_label_index']}, "
            f"non_input_value{dataloader_config['non_input_value']}, "
        )

        self.dataloader = get_data_loader(
            dataset=self.dataset, dataloader=dataloader_name, **dataloader_config
        )

        return self.dataloader

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch for dimension inference."""
        if self.dataloader is None:
            self.setup_dataloader()

        inputs, labels, *paths = next(iter(self.dataloader))
        inputs = _adjust_data_dimensions(inputs)
        labels = _adjust_label_dimensions(labels)

        return inputs, labels


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

        logger.info(f"Creating {model_class.__name__} with:")
        logger.info(f"  - Input dims: {model_kwargs.get('input_dims')}")
        logger.info(f"  - N classes: {model_kwargs.get('n_classes')}")
        logger.info(f"  - N timesteps: {model_kwargs.get('n_timesteps')}")
        logger.info(f"  - Store responses: {model_kwargs.get('store_responses')}")

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


class TestingOrchestrator:
    """Enhanced testing orchestrator with comprehensive configuration management."""

    def __init__(self, config: TestingParams):
        self.config = config
        self.datamodule = TestingDataModule(config)
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
        logger.info("=" * 60)
        logger.info("TESTING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model.model_name}")
        logger.info(f"Dataset: {self.config.data.data_name}")
        logger.info(f"Data group: {self.config.data_group}")
        logger.info(f"Data classes: {len(self.datamodule.dataset.classes)}")
        logger.info(f"Model classes: {self.config.model.n_classes}")
        logger.info(f"Batch size: {self.config.data.batch_size}")
        logger.info(f"Store responses: {self.config.model.store_responses}")
        logger.info(f"Precision: {self.config.trainer.precision}")
        logger.info(
            f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
        )

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            )

        logger.info("=" * 60)

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

            logger.info(f"Extracted from test data:")
            logger.info(f"  - Input shape: {inputs.size()}")
            logger.info(f"  - Input dims: {actual_input_dims}")
            logger.info(f"  - Pixel stats: {inputs.mean():.3f} ± {inputs.std():.3f}")

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

        except Exception as e:
            logger.error(f"Failed to infer parameters from data: {e}")
            logger.warning("Continuing with configuration defaults")

    def setup_trainer(self) -> pl.Trainer:
        """Setup PyTorch Lightning trainer for testing using TestingParams configuration."""
        # Get trainer configuration (already optimized by TestingParams)
        trainer_kwargs = self.config.get_trainer_kwargs()

        # Optional: Setup logger if needed
        if hasattr(self.config, "logger") and getattr(self.config, "logger", None):
            trainer_kwargs["logger"] = pl.loggers.WandbLogger(
                project=project_paths.project_name,
                save_dir=project_paths.large_logs,
                config=(
                    self.config.get_full_config(flat=True)
                    if hasattr(self.config, "get_full_config")
                    else {}
                ),
                tags=["test"],
                name=f"test_{self.config.input_model_state.stem}",
            )

        # Filter valid arguments for Trainer
        trainer_kwargs, unknown = filter_kwargs(pl.Trainer, trainer_kwargs)
        if unknown:
            logger.debug(f"Filtered unknown trainer kwargs: {list(unknown.keys())}")

        wandb.init()  # hack to log histograms
        return pl.Trainer(**trainer_kwargs)

    def save_results(self, model: pl.LightningModule) -> None:
        """Save test results and model responses."""
        logger.info(f"Saving results to {self.config.output_results}")

        # Save test results (CSV)
        try:
            results_df = model.storage.get_dataframe()
            results_df.to_csv(self.config.output_results, index=False)
            logger.info(f"Test results saved to {self.config.output_results}")
            logger.info(f"Results shape: {results_df.shape}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

        # Save model responses (tensors)
        try:
            if hasattr(model, "storage"):
                logger.info("Saving model responses...")
                total_size_mb = 0
                response_data = model.storage.responses.get_all()
                responses = {}
                for layer in response_data[0].keys():
                    layer_responses = [item[layer] for item in response_data]
                    responses[layer] = torch.cat(layer_responses, dim=0)
                    size_mb = responses[layer].nbytes / (1024 * 1024)
                    total_size_mb += size_mb
                    logger.info(
                        f"  Layer {layer}: {responses[layer].shape} -> {size_mb:.2f} MB"
                    )

                torch.save(responses, self.config.output_responses)
                logger.info(f"Model responses saved to {self.config.output_responses}")
                logger.info(f"Total response size: {total_size_mb:.2f} MB")
            else:
                torch.save({}, self.config.output_responses)
                logger.warning("No model responses to save")
        except Exception as e:
            logger.error(f"Failed to save model responses: {e}")

    def run_testing(self) -> int:
        """Run the complete testing pipeline with comprehensive error handling."""
        with self.testing_context():
            try:
                # Setup data (TestingParams already optimized all parameters)
                self.datamodule.setup_dataset()
                dataloader = self.datamodule.setup_dataloader()

                # Infer and update model parameters from data
                self.infer_and_update_from_data()

                # Load and configure model
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

                # Run testing
                logger.info("Starting model testing...")
                trainer.test(model, dataloader)

                # Log peak memory usage
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1e6
                    logger.info(f"Peak GPU memory usage: {peak_memory:.2f}MB")

                    if peak_memory > 1000:  # More than 1GB
                        logger.warning(
                            "High GPU memory usage detected. Consider reducing batch_size or "
                            "store_responses for future runs."
                        )

                # Save results
                self.save_results(model)

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
