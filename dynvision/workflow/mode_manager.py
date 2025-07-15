"""
Simple modes system for DynVision parameter management.

Handles large_dataset, distributed, debug, and other operational modes
through config file overrides and automatic detection.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigModeManager:
    """Manages operational modes and parameter overrides."""

    def __init__(self, config: Any, local=True):
        if isinstance(config, dict):
            self.config = config
        elif hasattr(config, "__dict__"):
            self.config = vars(config)
        else:
            raise TypeError("config must be a dictionary or an object with __dict__.")

        self.local = local
        self._detect_available_modes()
        self.active_modes = []

    def _detect_available_modes(self) -> None:
        """detect modes with namespace `use_<mode>_mode` in config."""

        use_mode_keys = [
            key
            for key in self.config.keys()
            if key.startswith("use_") and key.endswith("_mode")
        ]
        self.available_modes = {
            k.replace("use_", "").replace("_mode", ""): self.config[k]
            for k in use_mode_keys
        }

    def log_modes(self) -> None:
        if not len(self.available_modes):
            print("No configuration modes available.")
            return

        max_mode_length = max(
            len(mode_name) for mode_name in self.available_modes.keys()
        )

        print("=" * 40)
        print(" Configuration Modes ".center(40, "="))
        print("=" * 40)
        for mode_name, initial_status in self.available_modes.items():
            detected_status = mode_name in self.active_modes
            print(
                f"\t {mode_name:<{max_mode_length}} | {str(initial_status):<5} -> {str(detected_status)}"
            )

    def log_config(self) -> None:
        """Log the current configuration with active modes applied."""
        print("=" * 40)
        print(" Current Configuration ".center(40, "="))
        print("=" * 40)
        for key, value in self.config.items():
            print(f"\t{key}: {value}")

    def apply_modes(self) -> None:
        self.active_modes = []

        for mode_name in self.available_modes.keys():
            if self.config[f"use_{mode_name}_mode"] == "auto":
                if hasattr(self, f"_detect_{mode_name}_mode"):
                    active = getattr(self, f"_detect_{mode_name}_mode")()
                else:
                    active = False
            else:
                active = self.config[f"use_{mode_name}_mode"]

            if active:
                self._apply_mode(mode_name)
                self.active_modes.append(mode_name)

    def _apply_mode(self, mode_name) -> None:
        if mode_name in self.config:
            self.config.update(self.config[mode_name])
        else:
            print(
                f"Mode '{mode_name}' is active but not defined in config. "
                "Skipping application."
            )

    def get_config(self, return_namespace: bool = True) -> Dict[str, Any]:
        """Return the current configuration with active modes applied."""
        if return_namespace:
            from types import SimpleNamespace

            return SimpleNamespace(**self.config)
        else:
            return self.config

    def save_config(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            f.write(
                "# This is an automatically compiled file. Do not edit manually!\n"
            )
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    # detection functions
    def _detect_local_mode(self) -> bool:
        return self.local

    def _detect_debug_mode(self) -> bool:
        """Auto-detect if debug mode should be enabled."""
        log_level = self.config.get("log_level", "INFO").upper()
        epochs = self.config.get("epochs", 100)

        return log_level == "DEBUG" or epochs <= 5

    def _detect_large_dataset_mode(self) -> bool:
        """Auto-detect if large dataset optimizations should be enabled."""
        data_name = self.config.get("data_name", "").lower()

        # Known large datasets
        large_datasets = ["imagenet", "coco", "openimages"]
        for large_name in large_datasets:
            if large_name in data_name:
                return True
        return False
