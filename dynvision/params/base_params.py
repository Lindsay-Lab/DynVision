"""
Base parameter handling system for DynVision using Pydantic.

This module provides the foundation for type-safe, validated parameter management
across all DynVision scripts with support for CLI parsing, config file loading,
and alias resolution.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import Dict, Any, Optional, List, Union, Type
from pathlib import Path
import argparse
import yaml
import json
import logging
import sys
from dynvision.utils import str_to_bool

logger = logging.getLogger(__name__)


class DynVisionValidationError(ValueError):
    """Custom validation error for DynVision parameter validation."""

    def __init__(self, message: str, field_name: Optional[str] = None):
        self.field_name = field_name
        super().__init__(f"{field_name}: {message}" if field_name else message)


class DynVisionConfigError(Exception):
    """Configuration-specific error with context information."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(f"{message}\nContext: {self.context}")


class BaseParams(BaseModel):
    """
    Base parameter class providing common functionality for all DynVision configs.

    Parameter Precedence Hierarchy (lowest to highest priority):
    1. Pydantic field defaults
    2. Configuration file values
    3. Command line arguments
    4. Direct override kwargs (programmatic use)
    """

    # Common parameters that appear across scripts
    seed: int = Field(default=42, description="Random seed for reproducibility")
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,  # Validate fields when assigned after creation
        use_enum_values=True,  # Use enum values in serialization
        validate_by_name=True,  # Allow validation using field names and aliases
    )

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """
        Return mapping of aliases to full parameter names.

        Override in child classes to add specific aliases.

        Returns:
            Dict mapping alias names to full parameter names
        """
        return {
            "log": "log_level",
        }

    @classmethod
    def get_params_from_cli_and_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create config instance with explicit precedence hierarchy.

        Precedence (lowest to highest priority):
        1. Pydantic field defaults
        2. Configuration file values
        3. Command line arguments
        4. Direct override kwargs

        Config path can be provided either as function argument or via CLI --config_path.
        Function argument takes precedence over CLI argument.

        Args:
            config_path: Path to YAML/JSON configuration file (takes precedence over CLI)
            override_kwargs: Direct parameter overrides (highest priority)
            args: CLI arguments list (None to use sys.argv)

        Returns:
            Configured parameter instance

        Raises:
            DynVisionValidationError: If parameter validation fails
            DynVisionConfigError: If config file loading fails
        """
        # Start with empty dict - Pydantic will provide field defaults
        params = {}

        # Determine config path: explicit argument takes precedence over CLI
        final_config_path = config_path
        if not final_config_path:
            # Check if config_path was provided via CLI
            cli_config_path = cls._extract_config_path_from_cli(args)
            final_config_path = cli_config_path

        # Priority 1: Load config file values (if we have a path)
        if final_config_path:
            try:
                config_data = cls._load_config_file(final_config_path)
                params.update(config_data)
                source = "explicit argument" if config_path else "CLI argument"
                logger.debug(
                    f"Loaded {len(config_data)} parameters from config file ({source}): {final_config_path}"
                )
            except Exception as e:
                raise DynVisionConfigError(
                    f"Failed to load config file: {final_config_path}",
                    {"error": str(e)},
                )

        # Priority 2: Parse and apply CLI arguments
        try:
            cli_data = cls._parse_cli_args(args)
            # Remove config_path from CLI data since it's not a model parameter
            cli_data.pop("config_path", None)
            params.update(cli_data)
            logger.debug(f"Parsed {len(cli_data)} parameters from CLI")
        except Exception as e:
            raise DynVisionConfigError(
                f"Failed to parse CLI arguments", {"error": str(e), "args": args}
            )

        # Priority 3: Apply direct overrides (highest priority)
        if override_kwargs:
            params.update(override_kwargs)
            logger.debug(f"Applied {len(override_kwargs)} direct parameter overrides")

        # Resolve aliases after all sources are merged
        params = cls._resolve_aliases(params)

        # Log final parameter summary
        config_source_info = (
            f" (config: {final_config_path})" if final_config_path else ""
        )
        logger.debug(
            f"Creating {cls.__name__} with {len(params)} total parameters{config_source_info}"
        )
        return params

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> "BaseParams":

        params = cls.get_params_from_cli_and_config(
            config_path=config_path,
            override_kwargs=override_kwargs,
            args=args,
        )

        try:
            return cls(**params)
        except Exception as e:
            # Provide detailed validation error with parameter context
            error_context = {
                "provided_params": list(params.keys()),
                "config_path": str(config_path) if config_path else None,
                "has_overrides": override_kwargs is not None,
            }
            logger.error(f"Error Context: {error_context}")
            raise DynVisionValidationError(
                f"Parameter validation failed: {e}", "validation"
            )

    @classmethod
    def _extract_config_path_from_cli(
        cls, args: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Extract config_path from CLI arguments without full parsing.

        Supports both formats: --config_path /path/to/config and --config_path=/path/to/config

        Args:
            args: CLI arguments list (None to use sys.argv)

        Returns:
            Config path if found, None otherwise
        """
        if args is None:
            args = sys.argv[1:]

        for i, arg in enumerate(args):
            if arg == "--config_path" and i + 1 < len(args):
                # Format: --config_path /path/to/config
                return args[i + 1]
            elif arg.startswith("--config_path="):
                # Format: --config_path=/path/to/config
                return arg.split("=", 1)[1]

        return None

    @classmethod
    def _load_config_file(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary of configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            DynVisionConfigError: If file format is unsupported or parsing fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yml", ".yaml"]:
                    data = yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == ".json":
                    data = json.load(f) or {}
                else:
                    raise DynVisionConfigError(
                        f"Unsupported config file format: {config_path.suffix}. "
                        f"Supported formats: .yml, .yaml, .json"
                    )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise DynVisionConfigError(f"Failed to parse {config_path}: {e}")

        return data

    @classmethod
    def _parse_cli_args(cls, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse CLI arguments, returning only explicitly provided values.

        This ensures proper precedence by only overriding defaults and config
        values when arguments are actually provided on the command line.

        Args:
            args: CLI arguments list (None to use sys.argv)

        Returns:
            Dictionary containing only explicitly provided CLI arguments
        """
        parser = argparse.ArgumentParser(
            description=f"Parameters for {cls.__name__}",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Add special config_path argument (not a model field, but useful for CLI)
        parser.add_argument(
            "--config_path",
            type=str,
            help="Path to configuration file (YAML or JSON)",
            default=argparse.SUPPRESS,
        )

        # Add arguments for each field
        for field_name, field_info in cls.model_fields.items():
            cls._add_field_argument(parser, field_name, field_info)

        # Add alias arguments
        for alias, full_name in cls.get_aliases().items():
            if full_name in cls.model_fields:
                field_info = cls.model_fields[full_name]
                cls._add_field_argument(parser, alias, field_info, is_alias=True)

        # Parse arguments
        if args is None:
            args = sys.argv[1:]

        parsed_args, unknown_args = parser.parse_known_args(args)

        # Extract only provided arguments (argparse.SUPPRESS ensures missing args aren't included)
        provided_params = {k: v for k, v in vars(parsed_args).items() if v is not None}

        # Handle unknown arguments if any
        if unknown_args:
            unknown_params = cls._parse_unknown_args(unknown_args)
            provided_params.update(unknown_params)
            logging.debug(
                f"Parsed {len(unknown_params)} unknown CLI arguments: {list(unknown_params.keys())}"
            )

        return provided_params

    @classmethod
    def _add_field_argument(
        cls,
        parser: argparse.ArgumentParser,
        name: str,
        field_info,
        is_alias: bool = False,
    ):
        """Add a single field as CLI argument."""
        arg_name = f"--{name}"

        # Base argument configuration
        kwargs = {
            "help": field_info.description or f"Parameter {name}",
            "dest": name,
            "default": argparse.SUPPRESS,  # Let Pydantic handle defaults
        }

        # Type-specific handling
        if hasattr(field_info, "type_"):
            field_type = field_info.type_
        elif hasattr(field_info, "annotation"):
            field_type = field_info.annotation
        else:
            field_type = str

        # Handle Optional types
        is_optional = False
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            args = field_type.__args__
            if len(args) == 2 and type(None) in args:
                is_optional = True
                field_type = args[0] if args[1] is type(None) else args[1]

        # Configure argument based on type
        if field_type == bool:
            kwargs["type"] = str_to_bool
        elif field_type in [int, float, str, Path]:
            kwargs["type"] = field_type
            if is_optional:
                kwargs["nargs"] = "?"
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            kwargs["nargs"] = "+"
            if len(field_type.__args__) > 0:
                kwargs["type"] = field_type.__args__[0]

        parser.add_argument(arg_name, **kwargs)

    @classmethod
    def _parse_unknown_args(cls, unknown_args: List[str]) -> Dict[str, Any]:
        """
        Parse unknown command line arguments into parameter dictionary.

        Converts arguments of the form:
        --param1 value1 --param2 value2 --flag --list_param val1 val2
        into:
        {'param1': 'value1', 'param2': 'value2', 'flag': True, 'list_param': ['val1', 'val2']}

        Args:
            unknown_args: List of unknown command line arguments

        Returns:
            Dictionary of parameter names and values

        Raises:
            DynVisionValidationError: If parsing fails
        """
        if not unknown_args:
            return {}

        params = {}
        current_key = None
        current_values = []

        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]

            if arg.startswith("-"):
                # Save previous parameter if exists
                if current_key is not None:
                    params[current_key] = cls._process_parameter_value(current_values)

                # Start new parameter
                current_key = arg.lstrip("-")
                current_values = []

                # Check if next arg is also a flag (boolean parameter)
                if i + 1 >= len(unknown_args) or unknown_args[i + 1].startswith("-"):
                    params[current_key] = True
                    current_key = None

            elif current_key is not None:
                # Collect values for current parameter
                current_values.append(arg)
            else:
                # Orphaned value - log warning but continue
                logging.warning(f"Orphaned CLI argument ignored: {arg}")

            i += 1

        # Handle last parameter
        if current_key is not None:
            params[current_key] = cls._process_parameter_value(current_values)

        return params

    @classmethod
    def _process_parameter_value(cls, values: List[str]) -> Any:
        """
        Process parameter values from CLI parsing.
        Type conversion is handled by convert_string_values validator.
        """
        if not values:
            return True  # Boolean flag with no value

        if len(values) == 1:
            return values[0]  # Single value
        else:
            return values  # Multiple values as list

    @classmethod
    def _resolve_aliases(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve parameter aliases to full parameter names.

        If both an alias and its full name are provided, the alias value takes precedence
        (since aliases are typically provided later in the precedence chain).

        Additionally, if an alias resolves to a more specific (dotted) parameter name
        (e.g., 'model.n_timesteps'), and a less specific parameter (e.g., 'n_timesteps')
        also exists, the more specific one prevails and the less specific is removed.
        If there are multiple specific parameters (e.g., 'model.n_timesteps', 'data.n_timesteps'),
        they are all kept.

        Args:
            params: Dictionary of parameters with potential aliases

        Returns:
            Dictionary with aliases resolved to full parameter names
        """
        aliases = cls.get_aliases()
        resolved = {}

        # First, add all non-alias parameters
        for key, value in params.items():
            if key not in aliases:
                resolved[key] = value

        # Then, resolve aliases (potentially overriding full names)
        for key, value in params.items():
            if key in aliases:
                full_name = aliases[key]
                resolved[full_name] = value
                logging.debug(f"Resolved alias '{key}' -> '{full_name}' = {value}")

        # Remove less specific keys if a more specific one exists
        # For example, if both 'n_timesteps' and 'model.n_timesteps' exist, remove 'n_timesteps'
        specific_keys = [k for k in resolved if "." in k]
        for spec_key in specific_keys:
            base = spec_key.split(".")[-1]
            if base in resolved:
                # Only remove the base key if a more specific one exists
                logging.debug(
                    f"Removing less specific parameter '{base}' because '{spec_key}' exists"
                )
                resolved.pop(base)

        return resolved

    @classmethod
    def update_kwargs(cls, config: dict, updates: dict, verbose: bool = True) -> dict:
        """
        Update a configuration dictionary with another dictionary.

        Logs all changes to values if verbose=True.

        Args:
            config: The original configuration dictionary to update.
            updates: A dictionary of updates to apply.
            verbose: Whether to log changes.

        Returns:
            The updated configuration dictionary.
        """
        for key, value in updates.items():
            old_value = config.get(key, None)
            if old_value != value:
                if verbose:
                    logging.info(
                        f"Parameter '{key}' changed from {old_value} to {value}."
                    )
                config[key] = value
        return config

    def update_field(
        self,
        field_name: str,
        new_value: Any,
        verbose: bool = False,
        validate: bool = False,
    ) -> None:
        """
        Update a single field in-place with automatic validation.

        Args:
            field_name: Name of field to update (supports dot notation)
            new_value: New value to set
            verbose: Whether to log the change
        """
        if "." in field_name:
            # Handle nested fields like "model.n_classes"
            parts = field_name.split(".", 1)
            obj = getattr(self, parts[0])
            obj.update_field(parts[1], new_value, verbose)
            return

        if verbose:
            old_value = getattr(self, field_name, None)
            logging.info(f"Updating {field_name}: {old_value} -> {new_value}")

        if validate:
            setattr(self, field_name, new_value)
        else:
            object.__setattr__(self, field_name, new_value)

    @model_validator(mode="before")
    @classmethod
    def convert_string_values(cls, values):
        """
        Convert string representations to appropriate Python types.

        Handles:
        - 'None', 'none', 'null', '' -> None
        - 'True', 'true', 'False', 'false' -> bool
        - String numbers -> int/float (if field expects numeric type)

        This applies to ALL parameter sources (CLI, config files, direct instantiation).
        """
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, str):
                    values[key] = cls._convert_string_value(value, key)
        return values

    @classmethod
    def _convert_string_value(cls, value: str, field_name: str = None) -> Any:
        """Convert a single string value to appropriate type."""

        # Handle None representations
        if value.lower() in ("none", "null", ""):
            return None

        # Handle boolean representations
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Handle numeric representations (only if field expects numeric type)
        if field_name and field_name in cls.model_fields:
            field_info = cls.model_fields[field_name]
            expected_type = getattr(field_info, "annotation", str)

            # Handle Optional types
            if (
                hasattr(expected_type, "__origin__")
                and expected_type.__origin__ is Union
            ):
                args = expected_type.__args__
                if len(args) == 2 and type(None) in args:
                    expected_type = args[0] if args[1] is type(None) else args[1]

            # Try to convert to expected numeric type
            if expected_type in (int, float):
                try:
                    return expected_type(value)
                except ValueError:
                    pass  # Keep as string if conversion fails

        # Return as string if no conversion applies
        return value

    def setup_logging(self) -> None:
        """Configure logging based on log_level parameter."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # Override existing configuration
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured at {self.log_level} level")

    @classmethod
    def get_script_context(cls) -> Optional[str]:
        """
        Determine which script is currently running for context-aware validation.

        Returns:
            Script context ('init_model', 'train_model', 'test_model', or None)
        """
        if len(sys.argv) > 0:
            script_name = Path(sys.argv[0]).stem.lower()
            if "init" in script_name:
                return "init_model"
            elif "train" in script_name:
                return "train_model"
            elif "test" in script_name:
                return "test_model"
        return None

    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Ensure log_level is valid."""
        if isinstance(v, str):
            v = v.upper()
            if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(f"Invalid log level: {v}")
        return v

    def validate_consistency_with(
        self, other: "BaseParams", fields_to_check: Optional[List[str]] = None
    ) -> List[str]:
        """
        Check parameter consistency with another config instance.

        Args:
            other: Another parameter instance to compare with
            fields_to_check: Specific fields to check (None for common fields)

        Returns:
            List of inconsistent field names
        """
        if fields_to_check is None:
            # Check common fields between instances
            fields_to_check = set(self.model_fields.keys()) & set(
                other.model_fields.keys()
            )

        inconsistent = []
        for field_name in fields_to_check:
            if hasattr(self, field_name) and hasattr(other, field_name):
                self_value = getattr(self, field_name)
                other_value = getattr(other, field_name)

                if self_value != other_value:
                    inconsistent.append(field_name)
                    logging.warning(
                        f"Parameter inconsistency in '{field_name}': "
                        f"{self_value} vs {other_value}"
                    )

        return inconsistent

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding None values."""
        return self.dict(exclude_none=exclude_none)


# Example usage and testing
if __name__ == "__main__":
    # Example of how BaseParams would be used

    # Test basic instantiation
    params = BaseParams(seed=123, log_level="DEBUG")
    print(f"Basic params: {params}")

    # Test CLI parsing with known and unknown args
    try:
        # This would work with actual CLI args like: --seed 456 --log DEBUG --custom_param value --flag
        test_args = [
            "--seed",
            "456",
            "--log",
            "DEBUG",
            "--custom_param",
            "value",
            "--flag",
            "--list_param",
            "a",
            "b",
            "c",
        ]
        cli_params = BaseParams.from_cli_and_config(args=test_args)
        print(f"CLI params with unknown args: {cli_params}")
    except DynVisionValidationError as e:
        print(f"CLI parsing (expected to fail without valid CLI context): {e}")

    # Test unknown args parsing directly
    unknown_args = [
        "--model_type",
        "RCNN",
        "--learning_rate",
        "0.001",
        "--use_cuda",
        "--layers",
        "64",
        "128",
        "256",
    ]
    parsed_unknown = BaseParams._parse_unknown_args(unknown_args)
    print(f"Parsed unknown args: {parsed_unknown}")

    # Test type inference
    type_test_args = [
        "--int_param",
        "42",
        "--float_param",
        "3.14",
        "--bool_param",
        "true",
        "--string_param",
        "hello",
    ]
    parsed_types = BaseParams._parse_unknown_args(type_test_args)
    print(f"Type inference test: {parsed_types}")
    print(
        f"Types: int={type(parsed_types['int_param'])}, float={type(parsed_types['float_param'])}, bool={type(parsed_types['bool_param'])}, str={type(parsed_types['string_param'])}"
    )

    # Test config file simulation
    import tempfile
    import os

    # Create temporary config file
    config_data = {"seed": 789, "log_level": "WARNING"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_config = f.name

    try:
        file_params = BaseParams.from_cli_and_config(config_path=temp_config)
        print(f"Config file params: {file_params}")
    finally:
        os.unlink(temp_config)

    # Test validation
    params.setup_logging()
    print("Logging configured successfully")

    print(f"Script context: {BaseParams.get_script_context()}")
