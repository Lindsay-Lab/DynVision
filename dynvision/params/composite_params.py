"""Shared utilities for composite parameter classes."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum

import yaml

try:  # pragma: no cover - optional dependency guard
    import torch
except Exception:  # pragma: no cover - torch unavailable in some doc builds
    torch = None

from dynvision.params.base_params import (
    BaseParams,
    DynVisionValidationError,
    ParamsDict,
    ProvenanceRecord,
)
from dynvision.params.mode_registry import ModeRegistry, ModeResolution
from dynvision.params.resolution import (
    ConfigSchema,
    ConfigSource,
    ResolvedConfig,
    deep_merge,
    flatten_component_sections,
    flatten_nested_payload,
    merge_mode_sections,
    remove_conflicting_base_keys,
    resolve,
    resolve_aliases,
    separate_source,
)

logger = logging.getLogger(__name__)


ComponentDict = Dict[str, Any]
Preprocessor = Callable[[ComponentDict], ComponentDict]


class CompositeParams(BaseParams):
    """
    Base class for parameter objects composed of multiple component parameter sets.

    Subclasses must define :pyattr:`component_classes` mapping field names to the
    underlying ``BaseParams`` subclasses. Optional preprocessors can mutate the raw
    dictionaries before component instantiation. Unknown parameters can be handled
    by overriding :py:meth:`_handle_unscoped_param` or
    :py:meth:`_assign_to_components`.

    Mode-Specific Parameter Precedence:

    Subclasses can define :pyattr:`mode_name` to enable mode-specific parameter
    precedence. Parameters are resolved with the following hierarchy (highest to lowest):

    1. CLI arguments (handled upstream)
    2. mode.component.param (e.g., test.data.batch_size)
    3. mode.param (e.g., test.batch_size)
    4. component.param (e.g., data.batch_size)
    5. param (base defaults)

    Example:
        class TestingParams(CompositeParams):
            mode_name = "test"
            component_classes = {"data": DataParams, "trainer": TrainerParams}
    """

    # Mapping of component field name -> BaseParams subclass
    component_classes: ClassVar[Dict[str, Type[BaseParams]]] = {}

    # Optional mode name for mode-specific parameter precedence
    # ClassVar indicates this is a class variable, not a Pydantic instance field
    mode_name: ClassVar[Optional[str]] = None

    @classmethod
    def get_component_classes(cls) -> Dict[str, Type[BaseParams]]:
        """Return component mapping for the composite configuration."""
        if not cls.component_classes:
            raise NotImplementedError(
                f"{cls.__name__} must define component_classes to use CompositeParams"
            )
        return cls.component_classes

    @classmethod
    def get_component_preprocessors(cls) -> Dict[str, Preprocessor]:
        """
        Optional per-component preprocessors applied before instantiation.

        Returns a mapping of component name to callable that receives the component
        parameter dictionary and returns the possibly modified dictionary.
        """
        return {}

    @classmethod
    def get_component_assignment_order(cls) -> Iterable[str]:
        """
        Ordering used when attributing unscoped keys to components.

        Default ordering is the declaration order of ``component_classes``.
        """
        return cls.get_component_classes().keys()

    def iter_components(self) -> Iterable[Tuple[str, BaseParams]]:
        """Iterate over instantiated component parameter objects."""

        for name in self.get_component_classes().keys():
            component = getattr(self, name, None)
            if isinstance(component, BaseParams):
                yield name, component

    def log_overview(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        include_components: bool = True,
        include_defaults: bool = False,
    ) -> None:
        """Log a summarized view of the composite and its components."""

        logger = logger or logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        legend_requested = bool(
            self.show_provenance_legend or getattr(self, "verbose", False)
        )

        self.log_summary(
            logger=logger,
            title=f"{self.__class__.__name__} parameters",
            include_defaults=include_defaults,
            force_provenance_legend=legend_requested,
        )

        if include_components:
            for name, component in self.iter_components():
                component.log_summary(
                    logger=logger.getChild(name),
                    title=f"{name.capitalize()} parameters",
                    include_defaults=include_defaults,
                    force_provenance_legend=legend_requested,
                )

    @classmethod
    def _get_active_mode(cls) -> Optional[str]:
        """Return the active mode name if defined."""
        return cls.mode_name

    # ------------------------------------------------------------------
    # Adapter onto the pure resolution layer
    # ------------------------------------------------------------------

    @classmethod
    def build_config_schema(cls) -> ConfigSchema:
        """Project this pydantic class graph onto a plain :class:`ConfigSchema`.

        This is the adapter between the validation layer (pydantic classes, which
        own field declarations and validators) and the resolution layer
        (:mod:`dynvision.params.resolution`, which owns precedence rules and knows
        nothing about pydantic).

        Subclass extension points -- ``get_component_preprocessors`` and
        ``_handle_unscoped_param`` -- are passed through as explicit callables, so
        overriding them keeps working exactly as before.
        """
        component_classes = cls.get_component_classes()
        return ConfigSchema(
            component_fields={
                name: frozenset(comp.model_fields.keys())
                for name, comp in component_classes.items()
            },
            base_fields=frozenset(cls.model_fields.keys()) - set(component_classes),
            aliases=cls.get_aliases(),
            mode_name=cls._get_active_mode(),
            component_order=tuple(cls.get_component_assignment_order()),
            preprocessors=cls.get_component_preprocessors(),
            unscoped_handler=cls._handle_unscoped_param,
        )

    @classmethod
    def _load_config_file(cls, config_path) -> Dict[str, Any]:
        """
        Load config file and apply mode-specific overrides.

        Merges mode-specific sections (e.g., test.data.train) into base config.
        Mode name comes from cls.mode_name (e.g., "test" for TestingParams).

        Example config structure:
            train: true          # base default
            test:
              data:
                train: false    # test mode override

        For TestingParams (mode_name="test"), this merges test.data.train into data.train.
        """
        # Load base config using parent method, then fold in the `<mode>:` block.
        config = super()._load_config_file(config_path)
        return merge_mode_sections(
            config,
            cls._get_active_mode(),
            cls.get_component_classes().keys(),
        )

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deprecated alias for :func:`dynvision.params.resolution.deep_merge`.

        Retained because existing tests call it as a classmethod.
        """
        deep_merge(base, override)

    @classmethod
    def _flatten_component_sections(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested component sections back to dotted notation.

        _separate_single_source expects flat keys like "trainer.devices",
        but after deep_merge and conflict resolution we have nested dicts
        like config['trainer']['devices'].

        This method converts:
            {'trainer': {'devices': 1, 'num_nodes': 1}, 'seed': 42}
        To:
            {'trainer.devices': 1, 'trainer.num_nodes': 1, 'seed': 42}

        Only flattens known component sections, leaves other nested dicts intact.

        Args:
            config: Config dict with potentially nested component sections

        Returns:
            Flattened config dict with dotted keys
        """
        return flatten_component_sections(config, cls.get_component_classes().keys())

    @classmethod
    def _remove_conflicting_base_keys(
        cls, config: Dict[str, Any], mode_overrides: Dict[str, Any]
    ) -> None:
        """Deprecated alias for :func:`resolution.remove_conflicting_base_keys`.

        Retained because existing tests call it as a classmethod.
        """
        remove_conflicting_base_keys(config, mode_overrides)

    @classmethod
    def _resolve_aliases_with_precedence(
        cls,
        params: Dict[str, Any],
        sources: Optional[Dict[str, ProvenanceRecord]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, ProvenanceRecord]]:
        """Resolve aliases with scope-aware precedence.

        Thin wrapper over :func:`dynvision.params.resolution.resolve_aliases`,
        supplying this class's alias table. See that function for the precedence
        rules and for the alias-shadowing caveat.
        """
        return resolve_aliases(params, sources or {}, cls.get_aliases())

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[str] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> "CompositeParams":
        """Create an instance from config files, CLI arguments and modes.

        Runs in three clearly separated stages:

        1. **Gather** (I/O) -- parse CLI args, load the config file, resolve which
           modes are active.
        2. **Resolve** (pure) -- :func:`dynvision.params.resolution.resolve` turns
           the labelled sources into per-component dictionaries. No validation,
           no pydantic; see ``tests/params/test_config_resolution.py``.
        3. **Validate** -- instantiate the components and the composite.

        Precedence:
        - Within a source: scoped > unscoped (``model.model_name`` > ``model_name``)
        - Between sources: ALL CLI args > modes > ALL config values
        """
        # --- Stage 1: gather sources (I/O) ---
        config_params, cli_params = cls._get_config_and_cli_params_separate(
            config_path=config_path, override_kwargs=override_kwargs, args=args
        )

        # Resolve mode overrides after both sources are parsed
        mode_params, mode_resolution = cls._resolve_mode_overrides(
            config_params, cli_params
        )

        # Remove toggle keys so they do not leak into component configs
        cls._strip_mode_toggle_keys(config_params, cli_params)

        # --- Stages 2 & 3: resolve, then validate ---
        separated = cls._separate_component_configs_two_sources(
            config_params=config_params,
            cli_params=cli_params,
            mode_params=mode_params,
        )

        try:
            instance = cls(**separated)
        except Exception as exc:  # pragma: no cover - covered via runtime errors
            raise DynVisionValidationError(
                f"{cls.__name__} creation failed: {exc}"
            ) from exc

        provenance_map = getattr(separated, "provenance", {})
        object.__setattr__(instance, "_value_provenance", provenance_map)
        object.__setattr__(instance, "_mode_resolution", mode_resolution)
        return instance

    @classmethod
    def _get_config_and_cli_params_separate(
        cls,
        config_path: Optional[str] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> Tuple[ParamsDict, ParamsDict]:
        """Get config params and CLI params as separate dicts.

        Returns:
            Tuple of (config_params, cli_params)
        """
        config_params: Dict[str, Any] = {}
        config_sources: Dict[str, ProvenanceRecord] = {}
        cli_params: Dict[str, Any] = {}
        cli_sources: Dict[str, ProvenanceRecord] = {}

        # Parse CLI args first to extract config_path if present
        # Note: args=None means use sys.argv, which is the common case
        cli_args = cls._parse_cli_args(args)
        # Extract config_path from CLI if not provided as parameter
        if not config_path and "config_path" in cli_args:
            config_path = cli_args.pop("config_path")
        else:
            cli_args.pop("config_path", None)  # Remove if present
        cli_params.update(cli_args)
        for key in cli_args:
            cli_sources[key] = ProvenanceRecord(source="cli")
        logger.debug(f"Parsed {len(cli_args)} parameters from CLI")

        # Load config file (may come from parameter or extracted from CLI args)
        if config_path:
            config_file_params = cls._load_config_file(config_path)
            for key, value in config_file_params.items():
                config_params[key] = value
                config_sources[key] = ProvenanceRecord(source="config")
            logger.debug(f"Loaded {len(config_params)} parameters from config file")

        # Add override kwargs (highest priority within CLI)
        if override_kwargs:
            cli_params.update(override_kwargs)
            for key in override_kwargs:
                cli_sources[key] = ProvenanceRecord(source="override")
            logger.debug(f"Applied {len(override_kwargs)} direct overrides")

        # DO NOT resolve aliases here!
        # Alias resolution converts unscoped to scoped (e.g., model_name -> model.model_name)
        # which would incorrectly override explicitly scoped values within the same source.
        # Aliases are resolved in _separate_single_source after scoped/unscoped precedence.

        return ParamsDict(config_params, provenance=config_sources), ParamsDict(
            cli_params, provenance=cli_sources
        )

    @classmethod
    def _resolve_mode_overrides(
        cls, config_params: ParamsDict, cli_params: ParamsDict
    ) -> Tuple[Optional[ParamsDict], ModeResolution]:
        """Resolve activated mode patches and return them as a ParamsDict."""

        toggle_values = cls._gather_mode_toggle_values(config_params, cli_params)
        context = cls._build_mode_context(config_params, cli_params)
        resolution = ModeRegistry.resolve_modes(toggle_values, context)

        flattened, provenance = cls._flatten_mode_patches(resolution)
        if not flattened:
            return None, resolution

        return ParamsDict(flattened, provenance=provenance), resolution

    @classmethod
    def _strip_mode_toggle_keys(
        cls,
        config_params: ParamsDict,
        cli_params: ParamsDict,
    ) -> None:
        """Remove canonical and shortcut mode toggles from raw parameter dicts."""

        for mode_name in ModeRegistry.list_modes():
            definition = ModeRegistry.get_definition(mode_name)
            if not definition:
                continue
            shortcut_keys = (definition.toggle_key, mode_name)
            for key in shortcut_keys:
                if key in config_params:
                    config_params.pop(key, None)
                    config_params.provenance.pop(key, None)
                if key in cli_params:
                    cli_params.pop(key, None)
                    cli_params.provenance.pop(key, None)

    @classmethod
    def _gather_mode_toggle_values(
        cls, config_params: Mapping[str, Any], cli_params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Collect toggle values with CLI taking precedence over config."""

        toggle_values: Dict[str, Any] = {}
        config_dict = dict(config_params)
        cli_dict = dict(cli_params)
        for mode_name in ModeRegistry.list_modes():
            definition = ModeRegistry.get_definition(mode_name)
            if not definition:
                continue
            value = definition.default_toggle
            shortcut_keys = (definition.toggle_key, mode_name)
            for key in shortcut_keys:
                if key in config_dict:
                    value = config_dict[key]
            for key in shortcut_keys:
                if key in cli_dict:
                    value = cli_dict[key]
            toggle_values[definition.toggle_key] = value
        return toggle_values

    @staticmethod
    def _build_mode_context(
        config_params: Mapping[str, Any], cli_params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Build context dictionary combining config and CLI values."""

        context = dict(config_params)
        context.update(dict(cli_params))
        return context

    @classmethod
    def _flatten_mode_patches(
        cls, resolution: ModeResolution
    ) -> Tuple[Dict[str, Any], Dict[str, ProvenanceRecord]]:
        """Flatten active mode payloads into dotted keys with provenance."""

        flattened: Dict[str, Any] = {}
        provenance: Dict[str, ProvenanceRecord] = {}

        for mode_name in resolution.active_modes:
            payload = resolution.patches.get(mode_name, {})
            flattened_payload = cls._flatten_nested_payload(payload)
            for key, value in flattened_payload.items():
                flattened[key] = value
                provenance[key] = ProvenanceRecord(source=f"mode:{mode_name}")

        return flattened, provenance

    @staticmethod
    def _flatten_nested_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Deprecated alias for :func:`resolution.flatten_nested_payload`."""
        return flatten_nested_payload(payload)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def persist_resolved_config(
        self,
        primary_output: Path | str,
        script_name: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write the fully resolved parameter set alongside an output artifact."""

        target = Path(f"{primary_output}.config.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)

        mode_resolution = getattr(
            self,
            "_mode_resolution",
            ModeResolution(
                toggles={}, raw_values={}, active_modes=tuple(), patches={}
            ),
        )

        flat_params = self._build_flat_parameter_map()
        flat_params["_active_modes"] = list(mode_resolution.active_modes)

        provenance_map = self._build_flat_provenance_map()
        if provenance_map:
            flat_params["_provenance"] = provenance_map

        metadata = {
            "generated_at": datetime.utcnow().isoformat(),
            "script": script_name,
            "primary_output": str(primary_output),
            "source_precedence": "config -> modes -> cli",
            "active_modes": ", ".join(mode_resolution.active_modes) or "none",
        }
        if mode_resolution.raw_values:
            metadata["mode_toggles"] = mode_resolution.raw_values
        if extra_metadata:
            metadata.update(extra_metadata)

        header_lines = ["# DynVision resolved configuration"] + [
            f"# {key.replace('_', ' ').title()}: {value}"
            for key, value in metadata.items()
        ]
        header = "\n".join(header_lines) + "\n\n"

        serializable_params = {
            key: self._prepare_yaml_value(value) for key, value in flat_params.items()
        }

        with target.open("w", encoding="utf-8") as handle:
            handle.write(header)
            yaml.safe_dump(
                serializable_params,
                handle,
                default_flow_style=False,
                sort_keys=False,
            )

        return target

    def _build_flat_parameter_map(self) -> Dict[str, Any]:
        """Flatten composite and component parameters into dotted keys."""

        payload = self.model_dump()
        component_names = set(self.get_component_classes().keys())
        flattened: Dict[str, Any] = {}

        for key, value in payload.items():
            if key in component_names and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}.{subkey}"] = subvalue
            else:
                flattened[key] = value

        return flattened

    def _build_flat_provenance_map(self) -> Dict[str, str]:
        """Create a dotted-key provenance map suitable for YAML serialization."""

        flattened: Dict[str, str] = {}

        base_provenance = getattr(self, "_value_provenance", {})
        for key, record in base_provenance.items():
            flattened[key] = record.format() or "default"

        for comp_name, component in self.iter_components():
            comp_provenance = getattr(component, "_value_provenance", {})
            for key, record in comp_provenance.items():
                flattened[f"{comp_name}.{key}"] = record.format() or "default"

        return flattened

    def _prepare_yaml_value(self, value: Any) -> Any:
        """Recursively coerce values to YAML-safe primitives."""

        if isinstance(value, dict):
            return {k: self._prepare_yaml_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._prepare_yaml_value(v) for v in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if torch is not None:
            if isinstance(value, torch.Tensor):  # Avoid device refs in YAML
                tensor = value.detach().cpu()
                if tensor.numel() == 1:
                    return tensor.item()
                return tensor.tolist()
            if isinstance(value, (torch.dtype, torch.device, torch.Size)):
                return str(value)
        return value

    @classmethod
    def _separate_component_configs_two_sources(
        cls,
        config_params: ParamsDict,
        cli_params: ParamsDict,
        mode_params: Optional[ParamsDict] = None,
    ) -> ParamsDict:
        """Resolve all sources, then instantiate (and thereby validate) components.

        The two halves are now explicit: :func:`resolution.resolve` works out the
        values over plain dicts, and :meth:`_instantiate_resolved` validates them.

        Args:
            config_params: Parameters from config files
            cli_params: Parameters from CLI arguments
            mode_params: Parameters injected by active modes (optional)

        Returns:
            ParamsDict of instantiated components plus composite base fields.
        """
        sources = cls._build_config_sources(config_params, cli_params, mode_params)
        resolved = resolve(sources, cls.build_config_schema())
        return cls._instantiate_resolved(resolved)

    @classmethod
    def _build_config_sources(
        cls,
        config_params: Optional[ParamsDict],
        cli_params: Optional[ParamsDict],
        mode_params: Optional[ParamsDict] = None,
    ) -> List[ConfigSource]:
        """Wrap the raw parameter dicts as ordered, labelled resolution sources.

        Order encodes cross-source precedence: config -> modes -> cli.
        """
        specs = (
            ("config", config_params),
            ("modes", mode_params),
            ("cli", cli_params),
        )
        return [
            ConfigSource(
                label=label,
                params=dict(params),
                provenance=dict(getattr(params, "provenance", {}) or {}),
            )
            for label, params in specs
            if params
        ]

    @classmethod
    def _instantiate_resolved(cls, resolved: ResolvedConfig) -> ParamsDict:
        """Validate a :class:`ResolvedConfig` by instantiating each component.

        This is the validation half of the split. Everything upstream of it operates
        on plain dictionaries and can be tested without pydantic.
        """
        instantiated: Dict[str, Any] = {}
        for comp_name, comp_cls in cls.get_component_classes().items():
            try:
                component_instance = comp_cls(**resolved.components.get(comp_name, {}))
            except Exception as exc:
                raise DynVisionValidationError(
                    f"{cls.__name__} component '{comp_name}' validation failed: {exc}"
                ) from exc
            object.__setattr__(
                component_instance,
                "_value_provenance",
                resolved.component_provenance.get(comp_name, {}),
            )
            instantiated[comp_name] = component_instance

        instantiated.update(resolved.composite)
        return ParamsDict(instantiated, provenance=resolved.composite_provenance)

    @classmethod
    def _separate_single_source(
        cls,
        params: Dict[str, Any],
        sources: Optional[Dict[str, ProvenanceRecord]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, ProvenanceRecord]]]:
        """Split one source's parameters into component configs.

        Thin wrapper over :func:`dynvision.params.resolution.separate_source`,
        preserving the legacy return shape in which composite-level values are
        carried under the ``_composite_base`` key.
        """
        separated = separate_source(
            ConfigSource(label="config", params=params, provenance=sources or {}),
            cls.build_config_schema(),
        )

        component_configs = dict(separated.components)
        component_provenance = dict(separated.component_provenance)
        component_configs["_composite_base"] = separated.composite
        component_provenance["_composite_base"] = separated.composite_provenance
        return component_configs, component_provenance

    @classmethod
    def _separate_component_configs(cls, params: Dict[str, Any]) -> ParamsDict:
        """Split flat parameters into component configs (legacy single-source path).

        For backward compatibility with direct instantiation. Applies scoped >
        unscoped precedence within the single parameter dict. For proper config vs
        CLI precedence, use :meth:`from_cli_and_config` instead.
        """
        resolved = resolve(
            [ConfigSource(label="config", params=params)],
            cls.build_config_schema(),
        )
        return cls._instantiate_resolved(resolved)

    @staticmethod
    def _find_component_targets(
        key: str,
        component_field_sets: Dict[str, set],
        order: Iterable[str],
    ) -> Tuple[str, ...]:
        """Return component names whose schemas contain ``key`` in precedence order."""
        targets: List[str] = []
        for component_name in order:
            if key in component_field_sets.get(component_name, set()):
                targets.append(component_name)
        return tuple(targets)

    @classmethod
    def _assign_to_components(
        cls,
        key: str,
        value: Any,
        component_data: Dict[str, ComponentDict],
        targets: Tuple[str, ...],
    ) -> None:
        """Assign value to one or more component dictionaries."""
        for component_name in targets:
            component_data[component_name][key] = value

    @classmethod
    def _handle_unscoped_param(
        cls,
        key: str,
        value: Any,
        component_data: Dict[str, ComponentDict],
        base_params: Dict[str, Any],
    ) -> None:
        """Fallback handler for keys that do not match any component field."""
        logger.debug(
            "Unscoped parameter '%s' assigned to composite base namespace", key
        )
        base_params[key] = value
