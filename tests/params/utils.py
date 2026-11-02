"""Shared helpers for parameter tests."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import yaml


def create_temp_config(config_data: Dict[str, Any]) -> Path:
    """Create a temporary YAML config file merged with default parameters."""

    defaults = {
        # Base params
        "seed": 0,
        "log_level": "info",
        "learning_rate": 1e-3,
        # Mode toggles disabled so tests control activation explicitly
        "use_local_mode": False,
        "use_debug_mode": False,
        "use_large_dataset_mode": False,
        "use_distributed_mode": False,
        # Data params
        "data_name": "mnist",
        "data_group": "all",
        "train": True,
        "use_ffcv": True,
        "batch_size": 256,
        "num_workers": 4,
        "persistent_workers": True,
        "drop_last": True,
        "train_ratio": 0.9,
        "pixel_range": "0-1",
        # data_timesteps: defaults to 1 in DataParams unless t_sequence or dsteps is given
        "non_input_value": 0,
        "non_label_index": -1,
        "use_distributed": False,
        "encoding": "image",
        "writer_mode": "proportion",
        "max_resolution": 224,
        "compress_probability": 0.25,
        "jpeg_quality": 60,
        "chunksize": 1000,
        "page_size": 4194304,
        "batches_ahead": 3,
        "order": "QUASI_RANDOM",
        "pin_memory": False,
        "shuffle": True,
        "prefetch_factor": 1,
        # Trainer params
        "epochs": 200,
        "check_val_every_n_epoch": 10,
        "log_every_n_steps": 20,
        "num_sanity_val_steps": 0,
        "accumulate_grad_batches": 4,
        "precision": "bf16-mixed",
        "deterministic": False,
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "auto",
        "enable_progress_bar": False,
        "benchmark": True,
        "gradient_clip_algorithm": "norm",
        "limit_val_batches": 0.2,
        "reload_dataloaders_every_n_epochs": 0,
        "early_stopping_min_delta": 0.0,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "save_top_k": 2,
        "monitor_checkpoint": "val_loss",
        "checkpoint_mode": "min",
        "save_last": True,
        "every_n_epochs": 50,
    }

    full_config = {**defaults, **config_data}

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(full_config, temp_file)
    temp_file.close()
    return Path(temp_file.name)


@contextmanager
def temporary_init_paths():
    """Provide filesystem artifacts required by InitParams validators."""

    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset_path = temp_dir / "dataset"
        dataset_path.mkdir()
        output_path = temp_dir / "init.pt"
        yield {
            "dataset_path": dataset_path,
            "output": output_path,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def temporary_training_paths():
    """Provide filesystem artifacts required by TrainingParams validators."""

    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset_link = temp_dir / "dataset"
        dataset_link.mkdir()
        dataset_train = temp_dir / "train.ffcv"
        dataset_train.write_bytes(b"0")
        dataset_val = temp_dir / "val.ffcv"
        dataset_val.write_bytes(b"0")
        input_model_state = temp_dir / "input.pt"
        input_model_state.write_bytes(b"0")
        output_model_state = temp_dir / "output.pt"
        yield {
            "dataset_link": dataset_link,
            "dataset_train": dataset_train,
            "dataset_val": dataset_val,
            "input_model_state": input_model_state,
            "output_model_state": output_model_state,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def temporary_testing_paths():
    """Provide filesystem artifacts required by TestingParams validators."""

    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset_path = temp_dir / "dataset"
        dataset_path.mkdir()
        input_model_state = temp_dir / "input.pt"
        input_model_state.write_bytes(b"0")
        output_results = temp_dir / "results.csv"
        output_responses = temp_dir / "responses.pt"
        yield {
            "dataset_path": dataset_path,
            "input_model_state": input_model_state,
            "output_results": output_results,
            "output_responses": output_responses,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
