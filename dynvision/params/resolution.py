"""Pure configuration resolution, decoupled from pydantic validation.

This module answers one question: *given a set of labelled parameter sources and a
description of the target composite, what value does each component field end up
with, and where did it come from?*

It does so over plain dictionaries. It imports no pydantic, no torch, and nothing
else from ``dynvision`` except the equally dependency-free
:mod:`dynvision.params.provenance`. That constraint is the point of the module and
is enforced by ``tests/params/test_config_resolution.py``: resolution rules must be
assertable without constructing a fully valid params object.

See ``docs/development/planning/config-resolution-refactor.md`` and
`issue #13 <https://github.com/Lindsay-Lab/DynVision/issues/13>`_.

Precedence
----------

Two independent axes, applied in this order:

1. **Across sources** (outer): later sources win outright. The conventional order
   is ``config`` → ``modes`` → ``cli``. Scope does *not* rank across sources: an
   unscoped CLI value beats a scoped config value.
2. **Within a source** (inner), highest to lowest:

   =====  ============================  ===================================
   Level  Pattern                       Example
   =====  ============================  ===================================
   2      ``mode.component.param``      ``test.data.batch_size``
   3      ``mode.param``                ``test.batch_size``
   4      ``component.param``           ``data.batch_size``
   5      ``param``                     ``batch_size``
   =====  ============================  ===================================

Aliases are resolved first, per source, before any of the above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from dynvision.params.provenance import ProvenanceRecord

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigSchema",
    "ConfigSource",
    "ResolvedConfig",
    "SeparatedSource",
    "deep_merge",
    "flatten_component_sections",
    "flatten_nested_payload",
    "merge_mode_sections",
    "remove_conflicting_base_keys",
    "resolve",
    "resolve_aliases",
    "separate_source",
]


ComponentDict = Dict[str, Any]

#: Mutates and/or returns a component's parameter dict just before validation.
Preprocessor = Callable[[ComponentDict], Optional[ComponentDict]]

#: Hook for keys that match no component field. Receives the key, its value, the
#: per-component dicts, and the composite base dict; mutates them in place.
UnscopedHandler = Callable[[str, Any, Dict[str, ComponentDict], Dict[str, Any]], None]

#: Marker key under which a source's composite-level values are carried.
COMPOSITE_BASE_KEY = "_composite_base"


# ---------------------------------------------------------------------------
# Plain-data interface types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigSchema:
    """A plain-data description of the composite that parameters resolve against.

    This is the adapter boundary: :class:`~dynvision.params.composite_params.CompositeParams`
    projects its pydantic class graph onto this structure, and the resolver never
    sees a model class. Subclass extension points (``preprocessors``,
    ``unscoped_handler``) are explicit fields rather than inherited methods, so the
    available hooks are visible in the type.

    Args:
        component_fields: Component name -> the field names it declares.
        base_fields: Field names belonging to the composite itself.
        aliases: Alias name -> dotted target name.
        mode_name: Active mode prefix (e.g. ``"test"``), or ``None``.
        component_order: Component precedence order for unscoped assignment.
            Defaults to ``component_fields`` insertion order.
        preprocessors: Component name -> callable applied after merging.
        unscoped_handler: Fallback for keys matching no component field.
    """

    component_fields: Mapping[str, frozenset]
    base_fields: frozenset = frozenset()
    aliases: Mapping[str, str] = field(default_factory=dict)
    mode_name: Optional[str] = None
    component_order: Tuple[str, ...] = ()
    preprocessors: Mapping[str, Preprocessor] = field(default_factory=dict)
    unscoped_handler: Optional[UnscopedHandler] = None

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Component names in assignment-precedence order."""
        if self.component_order:
            known = set(self.component_fields)
            ordered = [name for name in self.component_order if name in known]
            ordered.extend(name for name in self.component_fields if name not in ordered)
            return tuple(ordered)
        return tuple(self.component_fields)


@dataclass(frozen=True)
class ConfigSource:
    """One labelled bundle of raw parameters with per-key provenance.

    Args:
        label: Source name, used as the default provenance source
            (``"config"``, ``"modes"``, ``"cli"``, ``"override"``).
        params: Raw, possibly dotted, possibly aliased parameters.
        provenance: Per-key provenance. Keys absent here fall back to ``label``.
    """

    label: str
    params: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, ProvenanceRecord] = field(default_factory=dict)

    def record_for(self, key: str) -> ProvenanceRecord:
        """Provenance for ``key``, defaulting to this source's label."""
        return self.provenance.get(key, ProvenanceRecord(source=self.label))


@dataclass(frozen=True)
class SeparatedSource:
    """One source after alias resolution and scope-precedence routing."""

    components: Dict[str, ComponentDict]
    composite: Dict[str, Any]
    component_provenance: Dict[str, Dict[str, ProvenanceRecord]]
    composite_provenance: Dict[str, ProvenanceRecord]


@dataclass(frozen=True)
class ResolvedConfig:
    """A fully resolved but *unvalidated* configuration.

    ``components`` maps each component name to a kwargs dict ready to be passed to
    its params class. Nothing here has been type-checked or validated -- that is
    deliberately the caller's next, separate step.
    """

    components: Dict[str, ComponentDict]
    composite: Dict[str, Any]
    component_provenance: Dict[str, Dict[str, ProvenanceRecord]]
    composite_provenance: Dict[str, ProvenanceRecord]


# ---------------------------------------------------------------------------
# Dict helpers (pure)
# ---------------------------------------------------------------------------


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` in place.

    Nested dictionaries are merged key-by-key; any other value replaces its
    counterpart wholesale.

    Returns:
        ``base``, for convenience. It has already been modified in place.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def flatten_component_sections(
    config: Mapping[str, Any], component_names: Iterable[str]
) -> Dict[str, Any]:
    """Convert nested component sections into dotted keys.

    ``{"trainer": {"devices": 1}, "seed": 42}`` becomes
    ``{"trainer.devices": 1, "seed": 42}``. Nested dicts that are *not* component
    sections (e.g. transform specifications) are left untouched.
    """
    known = set(component_names)
    flattened: Dict[str, Any] = {}

    for key, value in config.items():
        if key in known and isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f"{key}.{subkey}"] = subvalue
        else:
            flattened[key] = value

    return flattened


def flatten_nested_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten arbitrarily nested dictionaries into dotted keys."""
    flattened: Dict[str, Any] = {}

    def _recurse(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                _recurse(f"{prefix}.{subkey}" if prefix else subkey, subvalue)
        else:
            flattened[prefix] = value

    for key, value in payload.items():
        _recurse(key, value)

    return flattened


def remove_conflicting_base_keys(
    config: Dict[str, Any], mode_overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """Resolve base-level vs. scoped conflicts introduced by a mode merge.

    When a mode block writes ``trainer.epochs`` while the config also carries a
    base-level ``epochs``, the base-level key would later be routed to *every*
    component that declares it. This lifts the base-level value into the scoped
    section as a fallback and drops the base-level key.

    Returns:
        ``config``, modified in place.
    """

    def _resolve(overrides: Mapping[str, Any], scope: str = "") -> None:
        for key, value in overrides.items():
            if isinstance(value, dict):
                if key not in config or not isinstance(config[key], dict):
                    config[key] = {}
                _resolve(value, key)
                continue

            if not scope:
                continue

            if scope not in config or not isinstance(config[scope], dict):
                config[scope] = {}

            if key in config and not isinstance(config[key], dict):
                # A base-level key shadows the scoped one: demote it to a fallback.
                base_value = config[key]
                config[scope].setdefault(key, base_value)
                config.pop(key)
            else:
                config[scope].setdefault(key, value)

    _resolve(mode_overrides)
    return config


def merge_mode_sections(
    config: Mapping[str, Any],
    mode_name: Optional[str],
    component_names: Iterable[str],
) -> Dict[str, Any]:
    """Fold a config file's ``<mode>:`` block into the base config.

    Given ``mode_name="test"`` and::

        train: true
        test:
          data:
            train: false

    this yields ``{"data.train": False}``.

    A non-dict value under the mode key (the legacy ``test: true`` toggle) is left
    in place as an ordinary parameter, since it is a toggle rather than a block.

    Returns:
        A new flattened config dict. The input is not modified.
    """
    merged = dict(config)

    if not mode_name or mode_name not in merged:
        return merged

    mode_entry = merged[mode_name]
    if not isinstance(mode_entry, dict):
        return merged

    mode_overrides = merged.pop(mode_name)
    deep_merge(merged, mode_overrides)
    remove_conflicting_base_keys(merged, mode_overrides)
    return flatten_component_sections(merged, component_names)


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def resolve_aliases(
    params: Mapping[str, Any],
    provenance: Optional[Mapping[str, ProvenanceRecord]] = None,
    aliases: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, ProvenanceRecord]]:
    """Rewrite alias keys to their targets, respecting scope precedence.

    Rules, in order:

    1. **Same scope depth** (``tff`` vs ``t_feedforward``, or ``model.tff`` vs
       ``model.t_feedforward``): the alias wins.
    2. **Different scope depth**: the deeper-scoped key wins, whether that is the
       alias or the target.
    3. If the target is absent, the alias always resolves to it.

    The alias key is removed in every branch, and its provenance travels to the
    target whenever the alias supplies the winning value.

    .. warning::
       Because the alias key is *always* removed, an alias whose target is scoped to
       one component diverts that key away from sibling components that declare the
       same field. This is the shape of the ``c10::Half`` crash described in the
       planning doc; it is intended behaviour, but it is easy to declare by accident.

    Returns:
        ``(resolved_params, resolved_provenance)`` as new dicts.
    """
    resolved: Dict[str, Any] = dict(params)
    resolved_provenance: Dict[str, ProvenanceRecord] = dict(provenance or {})

    if not aliases:
        return resolved, resolved_provenance

    def _promote(alias: str, target: str) -> None:
        """Move the alias' value and provenance onto the target."""
        resolved[target] = resolved[alias]
        if alias in resolved_provenance:
            resolved_provenance[target] = resolved_provenance[alias]

    def _drop(alias: str) -> None:
        resolved_provenance.pop(alias, None)
        del resolved[alias]

    same_scope: List[Tuple[str, str, int]] = []
    cross_scope: List[Tuple[str, str, int, int]] = []
    for alias, target in aliases.items():
        alias_depth = alias.count(".")
        target_depth = target.count(".")
        if alias_depth == target_depth:
            same_scope.append((alias, target, alias_depth))
        else:
            cross_scope.append((alias, target, alias_depth, target_depth))

    # Phase 1: same scope depth -- the alias always wins.
    for alias, target, depth in same_scope:
        if alias not in resolved:
            continue
        if target in resolved:
            logger.debug(
                "Alias '%s'=%s overrides '%s'=%s (same scope depth %d, alias wins)",
                alias, resolved[alias], target, resolved[target], depth,
            )
        else:
            logger.debug(
                "Alias '%s'=%s resolves to '%s' (scope depth %d)",
                alias, resolved[alias], target, depth,
            )
        _promote(alias, target)
        _drop(alias)

    # Phase 2: differing scope depth -- the deeper-scoped key wins.
    for alias, target, alias_depth, target_depth in cross_scope:
        if alias not in resolved:
            continue
        if target in resolved and target_depth > alias_depth:
            logger.debug(
                "Scoped target '%s'=%s beats unscoped alias '%s'=%s (%d > %d)",
                target, resolved[target], alias, resolved[alias],
                target_depth, alias_depth,
            )
        else:
            logger.debug(
                "Alias '%s'=%s resolves to '%s' (alias depth %d, target depth %d)",
                alias, resolved[alias], target, alias_depth, target_depth,
            )
            _promote(alias, target)
        _drop(alias)

    return resolved, resolved_provenance


# ---------------------------------------------------------------------------
# Per-source separation
# ---------------------------------------------------------------------------


def separate_source(source: ConfigSource, schema: ConfigSchema) -> SeparatedSource:
    """Route one source's parameters into per-component dictionaries.

    Applies alias resolution, then the four-level scope precedence documented in the
    module docstring, entirely within this source. Cross-source precedence is
    :func:`resolve`'s job.
    """
    component_fields = schema.component_fields
    component_names = schema.component_names
    mode = schema.mode_name

    params, provenance = resolve_aliases(
        source.params, dict(source.provenance), schema.aliases
    )

    def record_for(key: str) -> ProvenanceRecord:
        return provenance.get(key, ProvenanceRecord(source=source.label))

    # --- Phase 1: classify each key by scope depth ---------------------------
    unscoped: Dict[str, Any] = {}
    unscoped_prov: Dict[str, ProvenanceRecord] = {}
    scoped = {name: {} for name in component_names}
    scoped_prov = {name: {} for name in component_names}
    mode_level: Dict[str, Any] = {}
    mode_level_prov: Dict[str, ProvenanceRecord] = {}
    mode_scoped = {name: {} for name in component_names}
    mode_scoped_prov = {name: {} for name in component_names}
    composite: Dict[str, Any] = {}
    composite_prov: Dict[str, ProvenanceRecord] = {}

    for key, value in params.items():
        parts = key.split(".")
        record = record_for(key)

        if len(parts) == 3:
            mode_prefix, comp_name, param_name = parts
            if mode and mode_prefix == mode and comp_name in component_fields:
                mode_scoped[comp_name][param_name] = value
                mode_scoped_prov[comp_name][param_name] = record.with_scope(
                    f"{mode}.{comp_name}"
                )
                logger.debug(
                    "Mode+Component: %s.%s=%s [mode=%s]", comp_name, param_name, value, mode
                )
                continue

        elif len(parts) == 2:
            prefix, param_name = parts
            if mode and prefix == mode:
                mode_level[param_name] = value
                mode_level_prov[param_name] = record.with_scope(mode)
                logger.debug("Mode: %s=%s [mode=%s]", param_name, value, mode)
                continue

            if prefix in component_fields:
                scoped[prefix][param_name] = value
                scoped_prov[prefix][param_name] = record.with_scope(prefix)
                logger.debug("Component: %s.%s=%s", prefix, param_name, value)
                continue

        elif key in schema.base_fields:
            composite[key] = value
            composite_prov[key] = record

        unscoped[key] = value
        unscoped_prov[key] = record

    # --- Phase 2: apply the precedence hierarchy -----------------------------
    explicitly_scoped: Set[Tuple[str, str]] = {
        (comp_name, key)
        for comp_name in component_names
        for source_map in (scoped[comp_name], mode_scoped[comp_name])
        for key in source_map
    }

    components: Dict[str, ComponentDict] = {}
    component_provenance: Dict[str, Dict[str, ProvenanceRecord]] = {}

    for comp_name in component_names:
        fields = component_fields[comp_name]
        config: ComponentDict = {}
        prov: Dict[str, ProvenanceRecord] = {}

        # Level 5: unscoped keys, unless this source scoped them explicitly.
        for key, value in unscoped.items():
            if key in fields and (comp_name, key) not in explicitly_scoped:
                config[key] = value
                prov[key] = unscoped_prov[key].with_scope(comp_name)
                logger.debug("Unscoped '%s' routed to %s", key, comp_name)

        # Level 4: component.param
        config.update(scoped[comp_name])
        prov.update(scoped_prov[comp_name])

        # Level 3: mode.param
        for key, value in mode_level.items():
            if key in fields:
                config[key] = value
                prov[key] = mode_level_prov[key]

        # Level 2: mode.component.param
        config.update(mode_scoped[comp_name])
        prov.update(mode_scoped_prov[comp_name])

        components[comp_name] = config
        component_provenance[comp_name] = prov

    # --- Phase 3: keys matching no component field ---------------------------
    for key, value in unscoped.items():
        if any(key in component_fields[name] for name in component_names):
            continue

        if schema.unscoped_handler is not None:
            schema.unscoped_handler(key, value, components, composite)
        else:
            logger.debug("Unscoped parameter '%s' assigned to composite base", key)
            composite[key] = value

        composite_prov.setdefault(key, unscoped_prov[key])
        for comp_name in component_names:
            if key in components.get(comp_name, {}):
                component_provenance[comp_name].setdefault(
                    key, unscoped_prov[key].with_scope(comp_name)
                )

    # --- Phase 4: make composite base fields visible to components -----------
    for comp_name in component_names:
        for base_key, base_value in composite.items():
            if base_key not in components[comp_name]:
                components[comp_name][base_key] = base_value
                record = composite_prov.get(base_key, ProvenanceRecord(source="default"))
                component_provenance[comp_name][base_key] = record.with_scope(comp_name)

    return SeparatedSource(
        components=components,
        composite=composite,
        component_provenance=component_provenance,
        composite_provenance=composite_prov,
    )


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


def resolve(
    sources: Sequence[ConfigSource], schema: ConfigSchema
) -> ResolvedConfig:
    """Resolve labelled parameter sources into per-component parameter dicts.

    This is the single entry point of the resolution layer, and the seam that
    separates *working out what the values are* from *validating them*. The result
    contains plain dicts; no field has been type-checked.

    Args:
        sources: Ordered sources, lowest precedence first (conventionally
            ``config`` → ``modes`` → ``cli``). Empty sources are skipped.
        schema: Description of the composite to resolve against.

    Returns:
        A :class:`ResolvedConfig` whose ``components`` entries are ready to be
        passed as kwargs to the corresponding params classes.
    """
    component_names = schema.component_names

    separated = [
        separate_source(source, schema) for source in sources if source.params
    ]

    # Composite base values merge across sources, last source winning.
    composite: Dict[str, Any] = {}
    composite_provenance: Dict[str, ProvenanceRecord] = {}
    for part in separated:
        composite.update(part.composite)
        composite_provenance.update(part.composite_provenance)

    # Seed every component with the composite base, then layer sources in order.
    components: Dict[str, ComponentDict] = {}
    component_provenance: Dict[str, Dict[str, ProvenanceRecord]] = {}
    for comp_name in component_names:
        components[comp_name] = dict(composite)
        component_provenance[comp_name] = {
            key: composite_provenance[key]
            for key in composite
            if key in composite_provenance
        }

    for part in separated:
        for comp_name in component_names:
            values = part.components.get(comp_name)
            if values:
                components[comp_name].update(values)
            if comp_name in part.component_provenance:
                component_provenance[comp_name].update(part.component_provenance[comp_name])

    _apply_preprocessors(components, component_provenance, schema)

    return ResolvedConfig(
        components=components,
        composite=composite,
        component_provenance=component_provenance,
        composite_provenance=composite_provenance,
    )


def _apply_preprocessors(
    components: Dict[str, ComponentDict],
    component_provenance: Dict[str, Dict[str, ProvenanceRecord]],
    schema: ConfigSchema,
) -> None:
    """Run each component's preprocessor, tagging changed values as ``derived``."""
    for comp_name, preprocessor in schema.preprocessors.items():
        if preprocessor is None or comp_name not in components:
            continue

        original = components[comp_name].copy()
        updated = preprocessor(components[comp_name])
        if updated is None:
            # Preprocessors are permitted to mutate in place and return nothing.
            updated = components[comp_name]
        components[comp_name] = updated

        provenance = component_provenance.setdefault(comp_name, {})
        for key, value in updated.items():
            if key in original and original[key] == value:
                continue
            record = provenance.get(key, ProvenanceRecord(source="default"))
            provenance[key] = record.add_mutation("derived")
