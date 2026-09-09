"""Provenance primitives shared by the config-resolution and validation layers.

This module is deliberately dependency-free (standard library only). It is imported
by :mod:`dynvision.params.resolution`, which must stay usable without pydantic or
torch so that parameter-resolution rules can be tested directly. See
``docs/development/planning/config-resolution-refactor.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = ["ProvenanceRecord", "ParamsDict"]


@dataclass(frozen=True)
class ProvenanceRecord:
    """Track where a value originated and how it was mutated."""

    source: str
    scope: Optional[str] = None
    mutations: Tuple[str, ...] = ()

    def with_scope(self, scope: Optional[str]) -> "ProvenanceRecord":
        if not scope or scope == self.scope:
            return self
        return ProvenanceRecord(self.source, scope, self.mutations)

    def add_mutation(self, tag: str) -> "ProvenanceRecord":
        if tag in self.mutations:
            return self
        return ProvenanceRecord(
            self.source,
            self.scope,
            self.mutations + (tag,),
        )

    def is_default(self) -> bool:
        return self.source == "default" and not self.mutations and not self.scope

    def format(self) -> Optional[str]:
        if self.is_default():
            return None
        base = self.source if not self.scope else f"{self.source}:{self.scope}"
        segments = [base] if base else []
        segments.extend(self.mutations)
        return "; ".join(segments) if segments else None


class ParamsDict(dict):
    """Dictionary that carries provenance metadata for each key."""

    def __init__(
        self, *args, provenance: Optional[Dict[str, ProvenanceRecord]] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.provenance: Dict[str, ProvenanceRecord] = provenance or {}

    def copy(self) -> "ParamsDict":
        return ParamsDict(super().copy(), provenance=self.provenance.copy())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParamsDict({dict(self)!r}, provenance={self.provenance!r})"
