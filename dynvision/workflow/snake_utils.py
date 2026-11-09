"""Workflow utilities for testing.

This module provides access to workflow functions for unit testing.
"""
import re

def key_value_pattern(
    key: str, assigner: str = "=", key_chars: str = "A-Za-z0-9"
) -> str:
    """Return a regex that matches ``{key}{assigner}`` as a whole key.

    Identifiers join ``key=value`` tokens with ``+``, ``:``, ``/`` and ``_``, and
    keys never contain those characters. A bare ``re.search(f"{key}=")`` also
    matches inside a longer key: ``stim`` inside ``tstim=10`` (a millisecond
    value read as steps) and ``pattern`` inside ``shufflepattern=true``. The
    lookbehind requires that the key is not preceded by a key character.

    Args:
        key: Parameter key, matched literally.
        assigner: Character between key and value.
        key_chars: Regex character-class body of the characters keys consist
            of. Pass ``"A-Za-z0-9_"`` for snake_case keys such as W&B column
            names, where ``_`` belongs to the key.

    Returns:
        The regex source for the key and assigner, without the value group.
    """
    return rf"(?<![{key_chars}]){re.escape(key)}{re.escape(assigner)}"


def compute_hash(*args, length: int = 8) -> str:
    """Compute deterministic hash from model_args and seed.

    Creates a short hash representation to avoid filesystem length limits.
    Idempotent - returns input unchanged if already a hash.

    Args:
        *args: Components to hash (model_args, seed - NOT data_name)
        length: Hash length in hex characters (default: 8 = 32 bits)

    Returns:
        Hash-prefixed string (e.g., ':hash=a7f3c9d4')

    Examples:
        >>> compute_hash('tsteps=20+dt=2+...', '42')
        ':hash=a7f3c9d4'

        >>> compute_hash(':hash=a7f3c9d4')  # Idempotent
        ':hash=a7f3c9d4'

    Notes:
        - Uses MD5 for speed (cryptographic strength not required)
        - Deterministic (same inputs → same output)
        - 8 hex chars = ~4 billion combinations
        - Collision probability negligible for typical use (<1000 models)
        - Hash does NOT include data_name (training data in subfolder)
    """
    import hashlib

    # Idempotent: if any arg already contains 'hash=', return first such arg
    for arg in args:
        if "hash=" in str(arg):
            return str(arg)

    # Combine all arguments
    combined = "_".join(str(arg).lstrip(":") for arg in args)

    # Compute MD5 hash
    hash_obj = hashlib.md5(combined.encode())
    hash_val = hash_obj.hexdigest()[:length]

    return f":hash={hash_val}"


def args_product(
    args_dict=None,
    delimiter: str = "+",
    assigner: str = "=",
    prefix: str = ":",
):
    """Generate the product of argument combinations as identifier strings.

    This is the workflow's identifier format: ``{prefix}k1=v1{delimiter}k2=v2``,
    keys in the order of ``args_dict``, each value written with ``str()``.
    ``snake_utils.smk`` imports it from here, so the workflow and the tools that
    reconstruct its paths (``scripts/link_legacy_temporal_paths.py``) share one
    implementation.

    Args:
        args_dict: Mapping of key to a value or a list of values.
        delimiter: Character separating arguments.
        assigner: Character separating key and value.
        prefix: Prefix character for the argument string.

    Returns:
        One identifier string per combination; ``['']`` for no arguments.
    """
    from itertools import product

    if not args_dict:
        return [""]

    args_dict = {
        key: [value] if not isinstance(value, list) else value
        for key, value in args_dict.items()
    }
    return [
        prefix
        + delimiter.join(
            f"{key}{assigner}{value}" for key, value in zip(args_dict.keys(), combo)
        )
        for combo in product(*args_dict.values())
    ]


def test_spec_hash(data_loader: str, data_args: str, length: int = 8) -> str:
    """Return the directory name of a test spec in compressed mode.

    Reproduces the workflow exactly: MD5 of ``"{loader}_{args}"`` (leading
    ``:`` stripped), first ``length`` hex characters, then
    ``.lstrip(":hash=")``. That last call strips a character *set*, not a
    prefix, so a hash starting with ``a`` loses its leading ``a`` characters.
    Every existing result directory is named this way, so the quirk is kept.

    Args:
        data_loader: Loader name, e.g. ``"StimulusInterval"``.
        data_args: Spec string as built by :func:`args_product`.
        length: Hash length before stripping.

    Returns:
        The test identifier used as directory name.
    """
    import hashlib

    combined = "_".join(str(arg).lstrip(":") for arg in (data_loader, data_args))
    return hashlib.md5(combined.encode()).hexdigest()[:length].lstrip(":hash=")


# Not a pytest test despite the name.
test_spec_hash.__test__ = False
