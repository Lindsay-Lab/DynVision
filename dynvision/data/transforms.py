"""Data transforms with YAML-driven presets.

Usage:
    from dynvision.data.transforms import get_data_transform

    # Resolve transform by backend, context, and dataset/preset name
    transforms = get_data_transform(
        backend='torch',
        context='train',
        dataset_or_preset='imagenette',
    )
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import yaml
import torchvision.transforms.v2 as tv2
import ffcv.transforms
from .operations import IndexToLabel

logger = logging.getLogger(__name__)

# Backend type hint
Backend = Literal["torch", "ffcv"]


# ===== Transform String Parser =====


def parse_transform_string(
    transform_str: str,
    backend: Backend = "torch",
) -> Optional[Callable]:
    """Parse a transform specification string into a callable transform.

    Supports two formats:
    - Bare module name: "RandomHorizontalFlip"
    - With arguments: "RandomBrightness(0.2)" or "RandomAffine(0, translate=(0.1, 0.1))"

    Args:
        transform_str: String specification of transform (e.g., "RandomHorizontalFlip()")
        backend: Transform backend - "torch" for torchvision.transforms.v2 or "ffcv" for ffcv.transforms

    Returns:
        Instantiated transform object, or None if parsing fails

    Raises:
        ValueError: If backend is invalid or transform string cannot be parsed
        AttributeError: If transform module does not exist in the specified backend
    """
    if not transform_str or not transform_str.strip():
        logger.warning("Empty transform string provided")
        return None

    transform_str = transform_str.strip()

    # Select backend module
    if backend == "torch":
        module = tv2
    elif backend == "ffcv":
        module = ffcv.transforms
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'torch' or 'ffcv'")

    # Extract module name and arguments
    # Pattern: "ModuleName" or "ModuleName(args)"
    match = re.match(r"^(\w+)(?:\((.*)\))?$", transform_str, re.DOTALL)

    if not match:
        raise ValueError(
            f"Invalid transform string format: '{transform_str}'. "
            f"Expected 'ModuleName' or 'ModuleName(args)'"
        )

    module_name = match.group(1)
    args_str = match.group(2)

    # Get the transform class from the module
    if not hasattr(module, module_name):
        raise AttributeError(
            f"Transform '{module_name}' not found in {backend} backend. "
            f"Available transforms: {', '.join(dir(module))}"
        )

    transform_class = getattr(module, module_name)

    # Parse arguments if present
    if args_str:
        try:
            # Separate positional and keyword arguments
            # Strategy: parse the full call expression via ast.parse, then extract args
            full_expr = f"{module_name}({args_str})"
            tree = ast.parse(full_expr, mode="eval")

            # The tree is an Expression with a Call node
            call_node = tree.body
            if not isinstance(call_node, ast.Call):
                raise ValueError(f"Expected Call node, got {type(call_node)}")

            # Extract positional args
            positional_args = []
            for arg in call_node.args:
                positional_args.append(ast.literal_eval(arg))

            # Extract keyword args
            keyword_args = {}
            for keyword in call_node.keywords:
                keyword_args[keyword.arg] = ast.literal_eval(keyword.value)

            # Instantiate with parsed args
            instance = transform_class(*positional_args, **keyword_args)

        except (SyntaxError, ValueError, AttributeError) as e:
            raise ValueError(
                f"Failed to parse arguments for {module_name}: '{args_str}'. Error: {e}"
            ) from e
    else:
        # No arguments - call with defaults
        instance = transform_class()

    logger.debug(
        f"Parsed {backend} transform: {module_name} -> {type(instance).__name__}"
    )

    return instance


def parse_transform_list(
    transform_strings: List[str],
    backend: Backend = "torch",
) -> List[Callable]:
    """Parse a list of transform specification strings.

    Args:
        transform_strings: List of transform specification strings
        backend: Transform backend - "torch" or "ffcv"

    Returns:
        List of instantiated transform objects

    Raises:
        ValueError: If any transform string fails to parse
    """
    transforms = []

    for i, transform_str in enumerate(transform_strings):
        try:
            transform = parse_transform_string(transform_str, backend=backend)
            if transform is not None:
                transforms.append(transform)
        except (ValueError, AttributeError) as e:
            logger.error(
                f"Failed to parse transform at index {i}: '{transform_str}'. Error: {e}"
            )
            raise

    return transforms


def validate_transform_string(
    transform_str: str,
    backend: Backend = "torch",
) -> Tuple[bool, Optional[str]]:
    """Validate a transform specification string without instantiating it.

    Args:
        transform_str: String specification of transform
        backend: Transform backend - "torch" or "ffcv"

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        parse_transform_string(transform_str, backend=backend)
        return True, None
    except (ValueError, AttributeError) as e:
        return False, str(e)


# ===== Transform Preset Resolution =====

# Cache for loaded YAML config
_transform_presets_cache: Optional[Dict[str, Any]] = None


def _load_transform_presets() -> Dict[str, Any]:
    """Load transform presets from YAML configuration.

    Returns:
        Dict with structure: {backend: {context: {preset_name: [transform_strings]}}}

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML structure is invalid
    """
    global _transform_presets_cache

    if _transform_presets_cache is not None:
        return _transform_presets_cache

    # Locate config file relative to this module
    config_path = Path(__file__).parent.parent / "configs" / "config_data.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Transform config not found: {config_path}")

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if "transform_presets" not in full_config:
        raise ValueError(
            f"'transform_presets' section missing from {config_path}. "
            "Expected structure: transform_presets -> backend -> context -> preset_name"
        )

    presets = full_config["transform_presets"]

    # Validate structure
    for backend in presets:
        if backend not in ("torch", "ffcv"):
            logger.warning(
                f"Unknown backend '{backend}' in transform_presets. "
                "Supported backends: 'torch', 'ffcv'"
            )
        for context in presets.get(backend, {}):
            if context not in ("train", "test"):
                logger.warning(
                    f"Unknown context '{context}' in transform_presets[{backend}]. "
                    "Typical contexts: 'train', 'test'"
                )

    _transform_presets_cache = presets
    logger.debug(f"Loaded transform presets from {config_path}")

    return presets


def resolve_transform_preset(
    backend: Backend,
    context: str,
    dataset_or_preset: Optional[str] = None,
) -> Optional[List[str]]:
    """Resolve transform preset strings for given backend, context, and dataset/preset.

    Selection logic:
    1. If dataset_or_preset is specified and exists as a key, use it
    2. Otherwise, fall back to 'base' preset
    3. If neither exists, return None

    Args:
        backend: Transform backend - "torch" or "ffcv"
        context: Context - typically "train" or "test"
        dataset_or_preset: Dataset name (e.g., "imagenette") or named preset (e.g., "auto")

    Returns:
        List of transform specification strings, or None if no preset found

    Raises:
        ValueError: If backend/context combination not found in config
    """
    presets = _load_transform_presets()

    if backend not in presets:
        raise ValueError(
            f"Backend '{backend}' not found in transform_presets. "
            f"Available backends: {list(presets.keys())}"
        )

    backend_presets = presets[backend]

    if context not in backend_presets:
        raise ValueError(
            f"Context '{context}' not found in transform_presets[{backend}]. "
            f"Available contexts: {list(backend_presets.keys())}"
        )

    context_presets = backend_presets[context]

    # Try dataset_or_preset first, then fall back to 'base'
    if dataset_or_preset and dataset_or_preset in context_presets:
        selected = dataset_or_preset
    elif "base" in context_presets:
        selected = "base"
    else:
        logger.debug(
            f"No preset found for {backend}/{context}/{dataset_or_preset or 'base'}"
        )
        return None

    transform_strings = context_presets[selected]

    logger.debug(
        f"Resolved preset: {backend}/{context}/{selected} -> {len(transform_strings)} transforms"
    )

    return transform_strings


def get_data_transform(
    backend: Backend,
    context: str,
    dataset_or_preset: Optional[str] = None,
) -> Optional[List[Any]]:
    """Get data transforms for image preprocessing.

    Args:
        backend: Transform backend - "torch" or "ffcv"
        context: Context - "train" or "test"
        dataset_or_preset: Dataset name or named preset to select

    Returns:
        List of transform objects, or None if no transforms specified

    Raises:
        ValueError: If preset resolution fails
    """
    # Resolve preset and parse
    transform_strings = resolve_transform_preset(backend, context, dataset_or_preset)

    if not transform_strings:
        return None

    # Parse strings into callable transforms
    try:
        transforms = parse_transform_list(transform_strings, backend=backend)
        logger.info(
            f"Loaded {len(transforms)} {backend} transforms for {context}/{dataset_or_preset or 'base'}"
        )
        return transforms
    except (ValueError, AttributeError) as e:
        logger.error(
            f"Failed to parse transforms for {backend}/{context}/{dataset_or_preset}: {e}"
        )
        raise


def get_target_transform(
    data_name: str,
    data_group: str = "all",
) -> Optional[List[Any]]:
    """Get target (label) transforms for dataset preprocessing.

    Creates IndexToLabel transformer if data_group is not "all".

    Args:
        data_name: Name of the dataset (e.g., "imagenette", "mnist")
        data_group: Data group to select (e.g., "all", "invertebrates", "one")
                    If "all", no transformation is applied

    Returns:
        List of target transforms or None if no transforms are needed

    Raises:
        ValueError: If data_name or data_group is invalid
    """
    if not data_name or not data_name.strip():
        raise ValueError("data_name cannot be empty")

    if not data_group or not data_group.strip():
        raise ValueError("data_group cannot be empty")

    # Only apply IndexToLabel if not using all classes
    if data_group.lower() != "all":
        return [IndexToLabel(data_name.lower(), data_group.lower())]

    return None
