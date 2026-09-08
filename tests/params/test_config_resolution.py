"""Tests for the pure config-resolution layer (``dynvision.params.resolution``).

These tests exercise the seam that issue #13 introduces: resolution of parameter
sources into per-component dictionaries, *without* pydantic validation. Every test
in this module operates on plain dicts, so a precedence rule can be asserted
without constructing a valid params object (which would require torch, a model
checkpoint, and a dataset on disk).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dynvision.params.provenance import ProvenanceRecord
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

# ---------------------------------------------------------------------------
# Fixtures: a miniature composite schema with no pydantic behind it
# ---------------------------------------------------------------------------


def make_schema(**overrides) -> ConfigSchema:
    """Build a small three-component schema for precedence tests."""
    defaults = dict(
        component_fields={
            "model": frozenset({"model_name", "n_timesteps", "precision"}),
            "data": frozenset({"batch_size", "data_name", "precision"}),
            "trainer": frozenset({"devices", "epochs", "precision"}),
        },
        base_fields=frozenset({"output", "seed"}),
        aliases={"tsteps": "model.n_timesteps", "bs": "data.batch_size"},
        mode_name="test",
        component_order=("model", "data", "trainer"),
    )
    defaults.update(overrides)
    return ConfigSchema(**defaults)


def source(label, params, provenance=None) -> ConfigSource:
    prov = provenance or {key: ProvenanceRecord(source=label) for key in params}
    return ConfigSource(label=label, params=params, provenance=prov)


# ---------------------------------------------------------------------------
# The architectural invariant
# ---------------------------------------------------------------------------


BANNED_IMPORT_ROOTS = {"pydantic", "torch", "lightning", "numpy"}

PURE_MODULES = [
    "dynvision/params/resolution.py",
    "dynvision/params/provenance.py",
]


def _imported_roots(path: Path) -> set:
    """Return the root package name of every import in a module, at any depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("rel_path", PURE_MODULES)
def test_resolution_layer_has_no_validation_dependencies(rel_path):
    """The resolution layer must stay importable without pydantic or torch.

    This is the architectural invariant behind issue #13: if resolution can only be
    reached through pydantic, then resolution can only be tested by constructing a
    fully valid params object -- which is exactly the seam we are removing.
    """
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / rel_path
    assert module_path.exists(), f"expected pure module missing: {rel_path}"

    offenders = _imported_roots(module_path) & BANNED_IMPORT_ROOTS
    assert not offenders, f"{rel_path} must not import {sorted(offenders)}"


def test_resolution_layer_only_depends_on_provenance_within_dynvision():
    """resolution.py may only reach into the equally-pure provenance module."""
    repo_root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (repo_root / "dynvision/params/resolution.py").read_text(encoding="utf-8")
    )
    dynvision_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "dynvision"
        ):
            dynvision_imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("dynvision"):
                    dynvision_imports.add(alias.name)

    assert dynvision_imports <= {
        "dynvision.params.provenance"
    }, f"resolution.py must only depend on provenance, got {dynvision_imports}"


# ---------------------------------------------------------------------------
# Regression: the c10::Half alias-shadowing shape
# ---------------------------------------------------------------------------


def test_alias_diverts_unscoped_key_away_from_sibling_components():
    """Regression for the ``c10::Half`` crash (see planning doc section 1.2).

    ``TestingParams`` declared ``precision -> trainer.precision``. Alias resolution
    rewrites the alias to its target and deletes the alias key, so by the time
    component routing runs the unscoped ``precision`` no longer exists and
    ``DataParams.precision`` silently stays unset -- float16 data against an fp32
    model.

    This test pins that behaviour as *observable*. Previously it was only reachable
    as a runtime crash deep inside training.
    """
    schema = make_schema(aliases={"precision": "trainer.precision"})
    resolved = resolve([source("config", {"precision": "16-mixed"})], schema)

    assert resolved.components["trainer"]["precision"] == "16-mixed"
    # The shadowing: siblings declaring the same field receive nothing.
    assert "precision" not in resolved.components["data"]
    assert "precision" not in resolved.components["model"]


def test_unscoped_key_without_alias_reaches_every_declaring_component():
    """Without the alias, the same key fans out to all components that declare it.

    This is the fixed behaviour on ``refactor/dtype-policy`` -- deleting the
    ``precision`` alias is what let ``DataParams`` see the value at all.
    """
    schema = make_schema(aliases={})
    resolved = resolve([source("config", {"precision": "16-mixed"})], schema)

    for component in ("model", "data", "trainer"):
        assert resolved.components[component]["precision"] == "16-mixed"


# ---------------------------------------------------------------------------
# Alias precedence
# ---------------------------------------------------------------------------


def test_same_scope_alias_wins_over_target():
    params = {"model.tff": 3, "model.t_feedforward": 9}
    resolved, prov = resolve_aliases(params, {}, {"model.tff": "model.t_feedforward"})
    assert resolved == {"model.t_feedforward": 3}


def test_unscoped_same_scope_alias_wins_over_target():
    resolved, _ = resolve_aliases(
        {"tff": 3, "t_feedforward": 9}, {}, {"tff": "t_feedforward"}
    )
    assert resolved == {"t_feedforward": 3}


def test_cross_scope_scoped_target_beats_unscoped_alias():
    """``model.t_feedforward`` (scoped) outranks the unscoped alias ``tff``."""
    resolved, _ = resolve_aliases(
        {"tff": 3, "model.t_feedforward": 9}, {}, {"tff": "model.t_feedforward"}
    )
    assert resolved == {"model.t_feedforward": 9}


def test_cross_scope_alias_resolves_when_target_absent():
    resolved, _ = resolve_aliases({"tsteps": 12}, {}, {"tsteps": "model.n_timesteps"})
    assert resolved == {"model.n_timesteps": 12}


def test_cross_scope_scoped_alias_beats_unscoped_target():
    """A scoped alias outranks a less-scoped target."""
    resolved, _ = resolve_aliases(
        {"model.tsteps": 12, "n_timesteps": 4}, {}, {"model.tsteps": "n_timesteps"}
    )
    assert resolved == {"n_timesteps": 12}


def test_alias_resolution_is_a_noop_without_aliases():
    params = {"batch_size": 8}
    resolved, prov = resolve_aliases(
        params, {"batch_size": ProvenanceRecord("cli")}, {}
    )
    assert resolved == params
    assert prov["batch_size"].source == "cli"


def test_alias_resolution_transfers_provenance_to_the_target():
    prov_in = {"tsteps": ProvenanceRecord(source="cli")}
    resolved, prov = resolve_aliases(
        {"tsteps": 12}, prov_in, {"tsteps": "model.n_timesteps"}
    )
    assert resolved == {"model.n_timesteps": 12}
    assert prov["model.n_timesteps"].source == "cli"
    assert "tsteps" not in prov


def test_alias_resolution_does_not_mutate_its_input():
    params = {"tsteps": 12}
    resolve_aliases(params, {}, {"tsteps": "model.n_timesteps"})
    assert params == {"tsteps": 12}


# ---------------------------------------------------------------------------
# Scoped / unscoped precedence within a single source
# ---------------------------------------------------------------------------


def test_scoped_value_beats_unscoped_within_one_source():
    schema = make_schema()
    resolved = resolve(
        [source("config", {"precision": "32-true", "data.precision": "16-mixed"})],
        schema,
    )
    assert resolved.components["data"]["precision"] == "16-mixed"
    # Other components still see the unscoped fallback.
    assert resolved.components["trainer"]["precision"] == "32-true"


def test_component_scoped_key_routes_only_to_that_component():
    schema = make_schema()
    resolved = resolve([source("config", {"data.batch_size": 64})], schema)
    assert resolved.components["data"]["batch_size"] == 64
    assert "batch_size" not in resolved.components["model"]


def test_unknown_scoped_prefix_is_not_treated_as_a_component():
    """``foo.bar`` where ``foo`` is neither a mode nor a component stays unscoped."""
    schema = make_schema(component_order=("model", "data", "trainer"))
    resolved = resolve([source("config", {"foo.bar": 1})], schema)
    assert resolved.composite["foo.bar"] == 1


# ---------------------------------------------------------------------------
# Mode precedence
# ---------------------------------------------------------------------------


def test_mode_scoped_key_routes_to_component():
    """``test.batch_size`` reaches every component declaring ``batch_size``."""
    schema = make_schema()
    resolved = resolve([source("config", {"test.batch_size": 16})], schema)
    assert resolved.components["data"]["batch_size"] == 16


def test_mode_and_component_scoped_key_beats_component_scoped():
    schema = make_schema()
    resolved = resolve(
        [source("config", {"data.batch_size": 8, "test.data.batch_size": 16})],
        schema,
    )
    assert resolved.components["data"]["batch_size"] == 16


def test_mode_scoped_key_beats_component_scoped_key():
    """Level 3 (mode.param) outranks level 4 (component.param)."""
    schema = make_schema()
    resolved = resolve(
        [source("config", {"data.batch_size": 8, "test.batch_size": 16})], schema
    )
    assert resolved.components["data"]["batch_size"] == 16


def test_foreign_mode_prefix_is_left_alone():
    """A ``train.*`` key is inert when the schema's mode is ``test``."""
    schema = make_schema(mode_name="test")
    resolved = resolve([source("config", {"train.data.batch_size": 8})], schema)
    assert "batch_size" not in resolved.components["data"]
    assert resolved.composite["train.data.batch_size"] == 8


# ---------------------------------------------------------------------------
# Source precedence: config -> modes -> cli
# ---------------------------------------------------------------------------


def test_cli_source_overrides_config_source():
    schema = make_schema()
    resolved = resolve(
        [
            source("config", {"data.batch_size": 8}),
            source("cli", {"data.batch_size": 64}),
        ],
        schema,
    )
    assert resolved.components["data"]["batch_size"] == 64


def test_unscoped_cli_beats_scoped_config():
    """Between sources, *all* CLI beats *all* config -- scope only ranks within a source."""
    schema = make_schema()
    resolved = resolve(
        [source("config", {"data.batch_size": 8}), source("cli", {"batch_size": 64})],
        schema,
    )
    assert resolved.components["data"]["batch_size"] == 64


def test_mode_source_sits_between_config_and_cli():
    schema = make_schema()
    resolved = resolve(
        [
            source("config", {"data.batch_size": 8}),
            source("modes", {"data.batch_size": 32}),
            source("cli", {"data.batch_size": 64}),
        ],
        schema,
    )
    assert resolved.components["data"]["batch_size"] == 64

    without_cli = resolve(
        [
            source("config", {"data.batch_size": 8}),
            source("modes", {"data.batch_size": 32}),
        ],
        schema,
    )
    assert without_cli.components["data"]["batch_size"] == 32


def test_empty_sources_are_skipped():
    schema = make_schema()
    resolved = resolve(
        [source("config", {}), source("cli", {"data.batch_size": 4})], schema
    )
    assert resolved.components["data"]["batch_size"] == 4


def test_resolve_with_no_sources_yields_empty_components():
    schema = make_schema()
    resolved = resolve([], schema)
    assert set(resolved.components) == {"model", "data", "trainer"}
    assert all(config == {} for config in resolved.components.values())


# ---------------------------------------------------------------------------
# Composite base fields and the unscoped hook
# ---------------------------------------------------------------------------


def test_composite_base_field_is_recorded_and_propagated():
    schema = make_schema()
    resolved = resolve([source("config", {"seed": 42})], schema)
    assert resolved.composite["seed"] == 42
    # Base fields are made visible to components too (extra="allow" downstream).
    for component in ("model", "data", "trainer"):
        assert resolved.components[component]["seed"] == 42


def test_unknown_key_falls_through_to_the_composite_base():
    schema = make_schema()
    resolved = resolve([source("config", {"totally_unknown": "x"})], schema)
    assert resolved.composite["totally_unknown"] == "x"


def test_unscoped_handler_hook_can_redirect_unknown_keys():
    """``TestingParams`` fans unknown keys out to model *and* data; the resolver
    must accept that as injected behaviour rather than inheriting it."""

    def handler(key, value, component_data, base_params):
        component_data.setdefault("model", {})[key] = value
        component_data.setdefault("data", {})[key] = value

    schema = make_schema(unscoped_handler=handler)
    resolved = resolve([source("config", {"mystery": 7})], schema)

    assert resolved.components["model"]["mystery"] == 7
    assert resolved.components["data"]["mystery"] == 7
    assert "mystery" not in resolved.components["trainer"]


# ---------------------------------------------------------------------------
# Preprocessors
# ---------------------------------------------------------------------------


def test_preprocessor_runs_on_the_resolved_component_dict():
    def double_batch(config):
        if "batch_size" in config:
            config["batch_size"] *= 2
        return config

    schema = make_schema(preprocessors={"data": double_batch})
    resolved = resolve([source("config", {"data.batch_size": 8})], schema)
    assert resolved.components["data"]["batch_size"] == 16


def test_preprocessor_returning_none_keeps_the_mutated_dict():
    def mutate_in_place(config):
        config["data_name"] = "patched"
        return None

    schema = make_schema(preprocessors={"data": mutate_in_place})
    resolved = resolve([source("config", {"data.batch_size": 8})], schema)
    assert resolved.components["data"]["data_name"] == "patched"


def test_preprocessor_changes_are_tagged_as_derived_in_provenance():
    def override(config):
        config["batch_size"] = 999
        return config

    schema = make_schema(preprocessors={"data": override})
    resolved = resolve([source("config", {"data.batch_size": 8})], schema)

    record = resolved.component_provenance["data"]["batch_size"]
    assert "derived" in record.mutations


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_records_the_originating_source():
    schema = make_schema()
    resolved = resolve(
        [
            source("config", {"data.batch_size": 8}),
            source("cli", {"trainer.devices": 2}),
        ],
        schema,
    )
    assert resolved.component_provenance["data"]["batch_size"].source == "config"
    assert resolved.component_provenance["trainer"]["devices"].source == "cli"


def test_provenance_is_scoped_when_an_unscoped_key_is_routed():
    schema = make_schema(aliases={})
    resolved = resolve([source("cli", {"precision": "32-true"})], schema)
    assert resolved.component_provenance["data"]["precision"].scope == "data"
    assert resolved.component_provenance["model"]["precision"].scope == "model"


def test_provenance_reflects_the_winning_source():
    schema = make_schema()
    resolved = resolve(
        [
            source("config", {"data.batch_size": 8}),
            source("cli", {"data.batch_size": 64}),
        ],
        schema,
    )
    assert resolved.component_provenance["data"]["batch_size"].source == "cli"


# ---------------------------------------------------------------------------
# Pure dict helpers (previously CompositeParams private classmethods)
# ---------------------------------------------------------------------------


def test_deep_merge_overrides_scalars_and_recurses_into_dicts():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    deep_merge(base, {"a": 2, "nested": {"y": 3, "z": 4}})
    assert base == {"a": 2, "nested": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_creates_missing_sections():
    base = {}
    deep_merge(base, {"trainer": {"devices": 1}})
    assert base == {"trainer": {"devices": 1}}


def test_remove_conflicting_base_keys_demotes_a_colliding_base_key():
    """A base-level key is demoted into the scope only when the mode block names it."""
    config = {"devices": 4, "trainer": {}}
    remove_conflicting_base_keys(config, {"trainer": {"devices": 1}})
    assert "devices" not in config
    assert config["trainer"]["devices"] == 4


def test_remove_conflicting_base_keys_leaves_non_colliding_base_keys_alone():
    """``devices`` survives at base level because the mode block only sets ``epochs``.

    Pins a real limitation of the existing algorithm: conflict resolution is keyed
    on names present in the override, so an unrelated base-level key still fans out
    to every component that declares it.
    """
    config = {"devices": 4, "trainer": {}}
    remove_conflicting_base_keys(config, {"trainer": {"epochs": 10}})
    assert config == {"devices": 4, "trainer": {"epochs": 10}}


def test_flatten_component_sections_produces_dotted_keys():
    flat = flatten_component_sections(
        {"trainer": {"devices": 1, "epochs": 2}, "seed": 42}, {"trainer", "data"}
    )
    assert flat == {"trainer.devices": 1, "trainer.epochs": 2, "seed": 42}


def test_flatten_component_sections_leaves_unknown_nested_dicts_intact():
    payload = {"transforms": {"train": ["a"]}}
    assert flatten_component_sections(payload, {"trainer"}) == payload


def test_flatten_nested_payload_handles_arbitrary_depth():
    assert flatten_nested_payload({"a": {"b": {"c": 1}}, "d": 2}) == {
        "a.b.c": 1,
        "d": 2,
    }


def test_merge_mode_sections_folds_the_mode_block_into_the_base_config():
    config = {"train": True, "test": {"data": {"train": False}}}
    merged = merge_mode_sections(config, "test", {"data", "trainer"})
    assert merged["data.train"] is False


def test_merge_mode_sections_ignores_a_non_dict_mode_key():
    """A primitive ``test: true`` toggle must survive as a plain parameter."""
    config = {"test": True, "seed": 1}
    assert merge_mode_sections(config, "test", {"data"}) == config


def test_merge_mode_sections_without_a_mode_is_a_noop():
    config = {"seed": 1}
    assert merge_mode_sections(config, None, {"data"}) == config


# ---------------------------------------------------------------------------
# separate_source: the per-source seam
# ---------------------------------------------------------------------------


def test_separate_source_returns_plain_dicts():
    schema = make_schema()
    separated = separate_source(
        source("config", {"data.batch_size": 8, "seed": 1}), schema
    )

    assert separated.components["data"]["batch_size"] == 8
    assert separated.composite["seed"] == 1
    assert isinstance(separated.components["data"], dict)


def test_separate_source_does_not_mutate_the_source():
    schema = make_schema()
    params = {"tsteps": 12}
    separate_source(source("config", params), schema)
    assert params == {"tsteps": 12}


def test_resolve_returns_a_resolved_config():
    schema = make_schema()
    assert isinstance(resolve([source("config", {"seed": 1})], schema), ResolvedConfig)
