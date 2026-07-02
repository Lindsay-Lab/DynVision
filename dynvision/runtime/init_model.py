"""
Initialize a neural network model with type-safe parameter management.

This script creates and saves initialized models with automatic dimension inference
from datasets and optional pretrained weight loading.

Example:
    $ python init_model.py --model_name DyRCNNx4 --data_name cifar100 --output model.pt
"""

import logging

import torch

from dynvision import models
from dynvision.data.dataloader import StandardDataLoader
from dynvision.data.datamodule import SimpleDataModule
from dynvision.params.init_params import InitParams
from dynvision.utils import (
    set_seed,
    handle_errors,
    log_section,
    format_value,
)

logger = logging.getLogger(__name__)


# def find_similar_trained_model(
#     directory: Path, model_name: str, data_name: str, seed: int, model_args: str = ""
# ) -> Optional[Path]:
#     """
#     Find the most similar trained model in the directory by comparing model arguments and seeds.

#     Chooses the model with the most similar arguments, fewest arguments, and largest seed,
#     in that priority order.

#     Args:
#         directory: Directory to search for trained models
#         model_name: Name of the model architecture
#         data_name: Name of the dataset
#         seed: Random seed
#         model_args: Model arguments string for comparison

#     Returns:
#         Path to the most similar trained model, or None if none found
#     """
#     pattern = re.compile(
#         rf"{re.escape(model_name)}(:.*?)?_(\d+)_{re.escape(data_name)}_trained\.pt$"
#     )

#     pretrained_candidates = []
#     for candidate in directory.glob("*.pt"):
#         match = pattern.match(candidate.name)
#         if match:
#             cmodel_args, cseed = match.groups()
#             cmodel_args = cmodel_args.lstrip(":") if cmodel_args else ""
#             pretrained_candidates.append((candidate, cmodel_args, cseed))

#     if pretrained_candidates:
#         best_candidate = max(
#             pretrained_candidates,
#             key=lambda x: (arg_similarity(x[1], model_args), -len(x[1]), int(x[2])),
#         )
#         return best_candidate[0]

#     return None


# def arg_similarity(a: str, b: str) -> int:
#     """Calculate similarity between two argument strings."""
#     a_dict = str2dict(a, assigner="=") if a else {}
#     b_dict = str2dict(b, assigner="=") if b else {}
#     return sum(1 for key in a_dict if key in b_dict and a_dict[key] == b_dict[key])


def infer_dimensions_from_dataset(config: InitParams) -> None:
    """
    Infer model dimensions from dataset and update configuration.

    Args:
        config: InitParams instance to update
    """
    if not config.dataset_path or not config.dataset_path.exists():
        missing_path = config.dataset_path or "<unspecified>"
        logger.warning(
            "No dataset provided or dataset not found at %s. Using config defaults.",
            missing_path,
        )
        return

    try:
        data_module = SimpleDataModule(config=config, dataset_path=config.dataset_path)
        preview_loader = data_module.create_preview_loader(
            dataloader_class=StandardDataLoader,
        )
        dataset = data_module.dataset
        if dataset is None:
            raise RuntimeError("Failed to load dataset for dimension inference")

        # Get sample batch
        inputs, _labels, *_paths = next(iter(preview_loader))
        input_dims = inputs.shape[1:]  # Exclude batch dimension
        n_classes = len(dataset.classes)

        log_section(
            logger,
            "dataset_inference",
            [
                ("dataset_path", format_value(config.dataset_path), None),
                ("input_dims", format_value(input_dims), None),
                ("n_classes", format_value(n_classes), None),
            ],
        )

        # Update model configuration
        previous_updates = set(getattr(config.model, "_dynamic_updates", set()))
        config.update_model_parameters_from_dataset(
            input_dims=input_dims, n_classes=n_classes, verbose=True
        )
        current_updates = set(getattr(config.model, "_dynamic_updates", set()))

        if current_updates - previous_updates:
            legend_requested = bool(
                config.show_provenance_legend or getattr(config, "verbose", False)
            )
            config.model.log_summary(
                logger=logger.getChild("model"),
                title="Model parameters after dataset inference",
                include_defaults=False,
                force_provenance_legend=legend_requested,
            )

    except Exception as e:
        logger.warning(f"Failed to infer dimensions from dataset: {e}")
        logger.warning("Continuing with configuration defaults")


def create_and_initialize_model(config: InitParams) -> torch.nn.Module:
    """
    Create and initialize model with configuration.

    Args:
        config: InitParams instance with model configuration

    Returns:
        Initialized model
    """
    # Get model class
    if not hasattr(models, config.model.model_name):
        raise ValueError(f"Invalid model name: {config.model.model_name}")

    model_class = getattr(models, config.model.model_name)

    # Get filtered kwargs for model creation
    model_kwargs = config.get_model_kwargs(model_class)
    config.log_model_creation(
        model_class=model_class,
        model_kwargs=model_kwargs,
        logger=logger,
    )

    # Create model
    model = model_class(**model_kwargs)

    if hasattr(model, "_initialize_connections"):
        model._initialize_connections()

    if hasattr(model, "_init_parameters"):
        model._init_parameters()

    return model


# def load_pretrained_weights(model: torch.nn.Module, config: InitParams) -> None:
#     """
#     Load pretrained weights if available.

#     Args:
#         model: Model to load weights into
#         config: InitParams with pretrained configuration
#     """
#     # Extract model args from output filename
#     model_args = ""
#     if ":" in config.output.name:
#         model_args = config.output.name.split("_")[0].split(":")[1]

#     # Find similar pretrained model
#     pretrained_file = find_similar_trained_model(
#         directory=config.output.parent,
#         model_name=config.model.model_name,
#         data_name=config.data.data_name,
#         seed=config.seed,
#         model_args=model_args,
#     )

#     if pretrained_file:
#         log_section(
#             logger,
#             "Pretrained initialization",
#             [
#                 ("Source", format_value(pretrained_file), None),
#                 (
#                     "Match args",
#                     format_value(model_args or None),
#                     None,
#                 ),
#                 ("Seed", format_value(config.seed), None),
#             ],
#         )
#         try:
#             state_dict = torch.load(pretrained_file, weights_only=True)
#             model.load_state_dict(state_dict)
#         except Exception as e:
#             logger.warning(f"Failed to load pretrained weights: {e}")
#             logger.info("Proceeding with random initialization")
#             if hasattr(model, "_init_parameters"):
#                 model._init_parameters()
#     else:
#         log_section(
#             logger,
#             "Pretrained initialization",
#             [
#                 ("Source", "Not found", None),
#                 ("Seed", format_value(config.seed), None),
#             ],
#         )
#         if hasattr(model, "_init_parameters"):
#             model._init_parameters()


@handle_errors(verbose=True)
def main() -> int:
    """Main entry point for model initialization."""
    # Load configuration
    config = InitParams.from_cli_and_config()
    config.setup_logging()

    config.log_initialization_overview(logger=logger)

    # Set random seed
    set_seed(config.seed)

    # Infer dimensions from dataset if provided
    infer_dimensions_from_dataset(config)

    config.persist_resolved_config(
        primary_output=config.output,
        script_name=__file__,
    )

    # Create and initialize model
    model = create_and_initialize_model(config)

    # Calculate and log the number of trainable parameters in the model
    num_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    log_section(
        logger,
        "Model statistics",
        [("Trainable parameters", f"{num_trainable_params:,}", None)],
    )

    # Validate model state
    if not len(model.state_dict()):
        raise ValueError("Model state dict is empty. Check model initialization.")

    # Save initialized model
    torch.save(model.state_dict(), config.output)
    log_section(
        logger,
        "Artifacts",
        [("State dict", format_value(config.output), None)],
    )
    return 0


if __name__ == "__main__":
    exit(main())
