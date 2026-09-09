"""Central dtype policy for DynVision.

This module is the single source of truth for mapping a PyTorch Lightning
``precision`` value to the ``torch.dtype`` that consuming components should use.

The key semantic is the **stored-parameter dtype**: under Lightning's
``16-mixed`` / ``bf16-mixed`` precision the optimizer keeps master weights in
float32 and only autocasts activations during the forward pass. The data loader
must therefore emit float32 for mixed precision, and only the full-precision
modes (``16`` / ``bf16`` / ``32`` / ``64``) pre-cast tensors to the corresponding
dtype. Resolving data dtype to the compute (autocast) dtype instead was the source
of the original dtype-mismatch crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Canonical precision -> stored-parameter dtype map.
#
# Lightning normalizes trainer.precision: e.g. "32" -> "32-true", and integer
# precisions 16/32/64 are accepted as well. Both spellings are included so a
# single map covers every form the config layer may hand us.
PRECISION_TO_DTYPE: dict[str, torch.dtype] = {
    "32": torch.float32,
    "32-true": torch.float32,
    "16": torch.float16,
    "16-true": torch.float16,
    "64": torch.float64,
    "64-true": torch.float64,
    "bf16": torch.bfloat16,
    "bf16-true": torch.bfloat16,
    # Mixed precision keeps fp32 master weights.
    "16-mixed": torch.float32,
    "bf16-mixed": torch.float32,
}

DEFAULT_DTYPE: torch.dtype = torch.float32


def normalize_precision(precision: Any) -> str:
    """Normalize a precision value to the canonical string key.

    Accepts strings, integers (16/32/64), ``torch.dtype``, bools, and None.
    ``None`` normalizes to the empty string, which ``resolve_dtype`` treats as
    unknown (warn + default).
    """
    if isinstance(precision, bool):
        return str(int(precision))
    if isinstance(precision, torch.dtype):
        return _dtype_to_precision_key(precision)
    if precision is None:
        return ""
    if isinstance(precision, int):
        return str(precision)
    return str(precision).strip().lower()


def _dtype_to_precision_key(dtype: torch.dtype) -> str:
    for key, value in PRECISION_TO_DTYPE.items():
        if value == dtype:
            return key
    return ""


def resolve_dtype(
    precision: Any,
    *,
    warn_fallback: bool = True,
    default: torch.dtype = DEFAULT_DTYPE,
) -> torch.dtype:
    """Resolve a precision value to the stored-parameter ``torch.dtype``.

    Unknown or missing precision emits a warning (unless ``warn_fallback`` is
    False) and returns ``default`` (float32). This replaces the previous silent
    float16 fallback that caused data/model dtype mismatches.
    """
    key = normalize_precision(precision)

    resolved = PRECISION_TO_DTYPE.get(key)
    if resolved is not None:
        return resolved

    if warn_fallback:
        logger.warning(
            "Unknown precision %r; falling back to %s",
            precision,
            default,
        )
    return default


@dataclass(frozen=True)
class DtypePolicy:
    """Resolved dtype policy for all components that consume a dtype.

    ``precision`` is the Lightning trainer concept and the single source of
    truth. ``trainer_dtype`` / ``model_dtype`` / ``data_dtype`` /
    ``checkpoint_dtype`` are independently overridable slots so future workflows
    can intentionally diverge (e.g. fp16 compute with fp32 storage). Today only
    ``data_dtype`` is exposed as a user-settable override; the others track the
    trainer dtype.
    """

    trainer_dtype: torch.dtype
    model_dtype: torch.dtype
    data_dtype: torch.dtype
    checkpoint_dtype: torch.dtype

    @classmethod
    def from_precision(
        cls,
        precision: Any,
        *,
        data_dtype: Optional[torch.dtype] = None,
    ) -> "DtypePolicy":
        trainer_dtype = resolve_dtype(precision)
        model_dtype = trainer_dtype
        checkpoint_dtype = trainer_dtype
        resolved_data_dtype = trainer_dtype if data_dtype is None else data_dtype
        return cls(
            trainer_dtype=trainer_dtype,
            model_dtype=model_dtype,
            data_dtype=resolved_data_dtype,
            checkpoint_dtype=checkpoint_dtype,
        )


def coordinate_component_dtypes(config: Any) -> Optional[torch.dtype]:
    """Align ``config.data.dtype`` to the trainer's effective dtype.

    Shared by ``TrainingParams`` and ``TestingParams`` so both contexts derive the
    data dtype from the trainer identically. If ``data.dtype`` was explicitly set
    to a different value, it is force-aligned with a warning.

    Returns the coordinated dtype, or ``None`` when the config has no trainer
    component (e.g. ``InitParams``).
    """
    trainer = getattr(config, "trainer", None)
    if trainer is None or not hasattr(trainer, "get_effective_dtype"):
        return None

    data = getattr(config, "data", None)
    trainer_dtype = trainer.get_effective_dtype()

    if data is None:
        return trainer_dtype

    if data.dtype is None or data.dtype != trainer_dtype:
        if data.dtype is not None:
            logger.warning(
                "Data dtype (%s) differs from trainer dtype (%s). "
                "Aligning data dtype to trainer.",
                data.dtype,
                trainer_dtype,
            )
        data.update_field("dtype", trainer_dtype, mutation_tag="derived")

    return trainer_dtype
