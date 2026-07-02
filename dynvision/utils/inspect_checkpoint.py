#!/usr/bin/env python3
"""
Checkpoint Inspector for PyTorch Lightning files
Usage: python inspect_checkpoint.py path/to/checkpoint.ckpt
"""

import torch
import sys
from pathlib import Path
from datetime import datetime
import glob


def extract_model_info(hparams, state_dict):
    """Extract detailed model information."""
    model_info = {}

    # Try to get actual model class name
    if "__class__" in hparams:
        model_info["class"] = hparams["__class__"]
    elif "model" in hparams and hasattr(hparams["model"], "__class__"):
        model_info["class"] = hparams["model"].__class__.__name__
    else:
        # Try to infer from state_dict keys
        if any("efficientnet" in k.lower() for k in state_dict.keys()):
            model_info["class"] = "EfficientNet-based"
        elif any("resnet" in k.lower() for k in state_dict.keys()):
            model_info["class"] = "ResNet-based"
        elif any(
            "vit" in k.lower() or "transformer" in k.lower() for k in state_dict.keys()
        ):
            model_info["class"] = "Vision Transformer"
        else:
            model_info["class"] = "Unknown"

    # Extract architecture details from state_dict
    layer_types = set()
    for key in state_dict.keys():
        if "conv" in key.lower():
            layer_types.add("Convolutional")
        elif "linear" in key.lower() or "fc" in key.lower():
            layer_types.add("Linear")
        elif "attention" in key.lower() or "attn" in key.lower():
            layer_types.add("Attention")
        elif "norm" in key.lower() or "bn" in key.lower():
            layer_types.add("Normalization")
        elif (
            "recurrent" in key.lower() or "rnn" in key.lower() or "lstm" in key.lower()
        ):
            layer_types.add("Recurrent")

    model_info["layer_types"] = sorted(list(layer_types))

    return model_info


def extract_dataset_info(hparams):
    """Extract dataset information from hyperparameters."""
    dataset_info = {}

    # Common dataset parameter names
    dataset_keys = [
        "dataset",
        "dataset_name",
        "data_name",
        "data_path",
        "dataset_path",
        "train_dataset",
        "val_dataset",
        "test_dataset",
        "data_dir",
        "data_root",
    ]

    for key in dataset_keys:
        if key in hparams:
            dataset_info[key] = hparams[key]

    # Try to infer dataset from other parameters
    if "n_classes" in hparams:
        n_classes = hparams["n_classes"]
        common_datasets = {
            10: ["CIFAR-10", "MNIST"],
            100: ["CIFAR-100"],
            1000: ["ImageNet-1k"],
            21841: ["ImageNet-21k"],
        }
        if n_classes in common_datasets:
            dataset_info["likely_datasets"] = common_datasets[n_classes]

    # Check input dimensions for dataset hints
    if "input_dims" in hparams:
        dims = hparams["input_dims"]
        if isinstance(dims, (list, tuple)) and len(dims) >= 3:
            if dims[-2:] == [224, 224] or dims[-2:] == (224, 224):
                dataset_info["input_resolution"] = "224x224 (ImageNet-style)"
            elif dims[-2:] == [32, 32] or dims[-2:] == (32, 32):
                dataset_info["input_resolution"] = "32x32 (CIFAR-style)"
            elif dims[-2:] == [28, 28] or dims[-2:] == (28, 28):
                dataset_info["input_resolution"] = "28x28 (MNIST-style)"

    return dataset_info


def format_training_metrics(ckpt):
    """Extract and format training metrics."""
    metrics = {}

    # Look for metrics in various places
    if "lr_schedulers" in ckpt and ckpt["lr_schedulers"]:
        try:
            lr_state = ckpt["lr_schedulers"][0]
            if "last_lr" in lr_state:
                metrics["current_lr"] = lr_state["last_lr"]
        except Exception:
            pass

    # Look for logged metrics
    if "callbacks" in ckpt:
        callbacks = ckpt["callbacks"]
        for callback_name, callback_state in callbacks.items():
            if "best_model_score" in str(callback_state):
                try:
                    if hasattr(callback_state, "best_model_score"):
                        metrics["best_score"] = callback_state.best_model_score
                    if hasattr(callback_state, "best_model_path"):
                        metrics["best_model_path"] = Path(
                            callback_state.best_model_path
                        ).name
                except Exception:
                    pass

    return metrics


def inspect_checkpoint(checkpoint_path):
    """Extract and display key information from a Lightning checkpoint."""
    try:
        # Load checkpoint (map to CPU to avoid GPU memory issues)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        print(f"\n{'='*70}")
        print(f"CHECKPOINT INSPECTION: {Path(checkpoint_path).name}")
        print(f"{'='*70}")

        # Basic training info
        print(f"🔹 Epoch: {ckpt.get('epoch', 'Unknown')}")
        print(f"🔹 Global Step: {ckpt.get('global_step', 'Unknown')}")
        print(
            f"🔹 PyTorch Lightning Version: {ckpt.get('pytorch-lightning_version', 'Unknown')}"
        )

        # File info
        file_size = Path(checkpoint_path).stat().st_size / (1024**2)  # MB
        mod_time = datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime)
        print(f"🔹 File Size: {file_size:.1f} MB")
        print(f"🔹 Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Hyperparameters (this is where DynVision stores model config)
        if "hyper_parameters" in ckpt:
            hparams = ckpt["hyper_parameters"]

            # Extract model and dataset info
            model_info = extract_model_info(hparams, ckpt.get("state_dict", {}))
            dataset_info = extract_dataset_info(hparams)

            print("\n🤖 MODEL INFORMATION:")
            print(f"   Model Class: {model_info['class']}")
            if model_info["layer_types"]:
                print(f"   Architecture: {' + '.join(model_info['layer_types'])}")

            print(f"   Classes: {hparams.get('n_classes', 'Unknown')}")
            if "n_timesteps" in hparams:
                print(f"   Timesteps: {hparams['n_timesteps']}")

            # DynVision specific parameters
            dynvision_params = {
                "dt": "Time Step",
                "tau": "Time Constant",
                "t_feedforward": "Feedforward Time",
                "t_recurrence": "Recurrence Time",
                "recurrence_type": "Recurrence Type",
            }

            for param, description in dynvision_params.items():
                if param in hparams:
                    print(f"   {description}: {hparams[param]}")

            print("\n📊 DATASET INFORMATION:")
            if dataset_info:
                for key, value in dataset_info.items():
                    if key == "likely_datasets":
                        print(f"   Likely Dataset: {' or '.join(value)}")
                    else:
                        print(f"   {key.replace('_', ' ').title()}: {value}")
            else:
                print(f"   Input Dimensions: {hparams.get('input_dims', 'Unknown')}")
                print(f"   Classes: {hparams.get('n_classes', 'Unknown')}")

            print("\n⚙️  TRAINING CONFIGURATION:")
            print(f"   Learning Rate: {hparams.get('learning_rate', 'Unknown')}")
            print(f"   Optimizer: {hparams.get('optimizer', 'Unknown')}")
            print(f"   Loss Function: {hparams.get('loss', 'Unknown')}")
            if "batch_size" in hparams:
                print(f"   Batch Size: {hparams['batch_size']}")

            # Additional model-specific parameters
            other_params = [
                k
                for k in hparams.keys()
                if k
                not in [
                    "__class__",
                    "n_classes",
                    "n_timesteps",
                    "learning_rate",
                    "dt",
                    "tau",
                    "t_feedforward",
                    "t_recurrence",
                    "recurrence_type",
                    "input_dims",
                    "optimizer",
                    "loss",
                    "batch_size",
                ]
                and not k.startswith("_")
            ]
            if other_params:
                print(f"   Other Parameters: {len(other_params)} additional")
                if len(other_params) <= 3:
                    for param in other_params:
                        print(f"     {param}: {hparams[param]}")

        # Training metrics
        metrics = format_training_metrics(ckpt)
        if metrics:
            print("\n📈 TRAINING METRICS:")
            for key, value in metrics.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")

        # Model architecture info
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            total_params = sum(
                p.numel() for p in state_dict.values() if hasattr(p, "numel")
            )
            _ = sum(
                p.numel()
                for p in state_dict.values()
                if hasattr(p, "numel") and "weight" in str(p)
            )

            print("\n🧠 MODEL ARCHITECTURE:")
            print(f"   Total Parameters: {total_params:,}")
            print(f"   Model Size: {total_params * 4 / (1024**2):.1f} MB (float32)")

            # Count different layer types
            layer_counts = {}
            for key in state_dict.keys():
                if "weight" in key:
                    layer_type = key.split(".")[0] if "." in key else key
                    layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1

            if layer_counts:
                print("   Layer Distribution:")
                for layer_type, count in sorted(layer_counts.items()):
                    print(f"     {layer_type}: {count} layers")

        # Training state
        if "lr_schedulers" in ckpt or "optimizer_states" in ckpt:
            print("\n🎯 TRAINING STATE:")
            if "lr_schedulers" in ckpt:
                print(f"   Learning Rate Schedulers: {len(ckpt['lr_schedulers'])}")
            if "optimizer_states" in ckpt:
                print(f"   Optimizers: {len(ckpt['optimizer_states'])}")

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"❌ Error inspecting {checkpoint_path}: {e}")


def inspect_multiple_checkpoints(checkpoint_dir):
    """Inspect all checkpoint files in a directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoint files
    ckpt_files = list(checkpoint_dir.glob("*.ckpt")) + list(
        checkpoint_dir.glob("**/last*.ckpt")
    )

    if not ckpt_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(ckpt_files)} checkpoint files:")

    # Sort by modification time (newest first)
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for ckpt_file in ckpt_files:
        inspect_checkpoint(ckpt_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path(s)_or_directory>")
        print("Examples:")
        print("  python inspect_checkpoint.py checkpoint.ckpt")
        print("  python inspect_checkpoint.py checkpoint1.ckpt checkpoint2.ckpt")
        print("  python inspect_checkpoint.py /path/to/checkpoints/")
        print("  python inspect_checkpoint.py /path/to/*.ckpt")
        sys.exit(1)

    # Collect all paths to process
    paths_to_process = []

    for arg in sys.argv[1:]:
        path = Path(arg)

        # Handle wildcard patterns
        if "*" in arg:
            matched_files = glob.glob(arg)
            if matched_files:
                paths_to_process.extend([Path(f) for f in matched_files])
            else:
                print(f"No files match pattern: {arg}")
        # Handle directories
        elif path.is_dir():
            ckpt_files = list(path.glob("*.ckpt")) + list(path.glob("**/last*.ckpt"))
            if ckpt_files:
                paths_to_process.extend(ckpt_files)
            else:
                print(f"No checkpoint files found in directory: {path}")
        # Handle single files
        elif path.is_file():
            paths_to_process.append(path)
        else:
            print(f"Path does not exist: {path}")

    if not paths_to_process:
        print("No valid checkpoint files found")
        sys.exit(1)

    # Remove duplicates and sort by modification time (newest first)
    unique_paths = list(set(paths_to_process))
    unique_paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"Processing {len(unique_paths)} checkpoint file(s):")
    for path in unique_paths:
        inspect_checkpoint(path)
