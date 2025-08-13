"""Checkpoint conversion utility for DynVision models.

This module provides functionality to:
- Convert PyTorch Lightning checkpoints to state dictionaries
- Select best checkpoint based on validation loss
- Handle checkpoint conversion and saving
- Validate checkpoint compatibility

Usage:
    python checkpoint_to_statedict.py \
        --checkpoint_dir checkpoints/ \
        --output models/model.pt
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

import torch
import argparse


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Convert PyTorch Lightning checkpoint to state_dict .pt file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--checkpoint_dir", type=Path, help="Path to checkpoint directory", required=True
)
parser.add_argument(
    "--output", type=Path, help="Path to output .pt file", required=True
)


def get_best_checkpoint(
    checkpoint_dir: Path, model_identifier: str, pattern: Optional[str] = None
) -> Path:
    """Find best checkpoint based on validation loss.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_identifier: Model name to match
        pattern: Custom regex pattern for matching

    Returns:
        Path to best checkpoint

    Raises:
        FileNotFoundError: If no matching checkpoints found
        ValueError: If checkpoint directory does not exist
    """
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Updated pattern: model_identifier-<epch>-<val_loss>.ckpt
    # Example: mymodel-12-0.34.ckpt
    regex = pattern or rf"{re.escape(model_identifier)}-(\d+)-(\d+\.\d{{2}})\.ckpt"
    compiled_pattern = re.compile(regex)
    best_loss = float("inf")
    best_checkpoint = None

    for checkpoint_path in checkpoint_dir.glob("*.ckpt"):
        match = compiled_pattern.fullmatch(checkpoint_path.name)
        if match:
            epch = int(match.group(1))
            val_loss = float(match.group(2))
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = checkpoint_path
                logger.info(
                    f"Found better checkpoint (epoch {epch}) with loss {val_loss:.4f}"
                )

    if best_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint files found for model identifier: {model_identifier}"
        )

    logger.info(f"Selected best checkpoint: {best_checkpoint}")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    return best_checkpoint


def save_state_dict(
    checkpoint_path: Path, output_path: Path, device: Union[str, torch.device] = "cpu"
) -> None:
    """Save checkpoint state dictionary.

    Args:
        checkpoint_path: Path to checkpoint file
        output_path: Path to save state dictionary
        device: Device to load checkpoint on

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        KeyError: If state dict not found in checkpoint
        RuntimeError: If checkpoint loading fails
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    if "state_dict" not in checkpoint:
        raise KeyError("No state_dict found in checkpoint")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["state_dict"], output_path)
    logger.info(f"Model state saved to {output_path}")


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    model_identifier = args.output.stem
    output_path = args.output.with_suffix(".pt")

    try:
        # Find best checkpoint
        best_checkpoint = get_best_checkpoint(
            args.checkpoint_dir, model_identifier=model_identifier
        )

        # Save state dictionary if output doesn't exist
        if output_path.exists():
            logger.warning(f"Output file already exists: {output_path}")
        else:
            save_state_dict(best_checkpoint, output_path)

        # Touch output file to mark completion
        args.output.touch()

    except Exception as e:
        logger.error(f"Failed to convert checkpoint: {e}")
        raise
