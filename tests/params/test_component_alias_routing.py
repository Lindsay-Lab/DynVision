"""Tests for mapping component aliases through a composite configuration.

Each parameter component declares its own aliases (``ModelParams`` maps
``trc -> t_recurrence``). A composite such as ``TrainingParams`` needs those
aliases scoped to the owning component (``trc -> model.t_recurrence``) and
previously listed them by hand. That hand list drifted: ``TestingParams``
carried the model timing aliases while ``TrainingParams`` and ``InitParams`` did
not.

Aliases missing from the composite map were not dropped outright, which is what
made the gap hard to see. ``--trc 4`` arrived as a bare key, was attributed to
the model component *unscoped*, and ``ModelParams`` stored it as an inert extra
attribute (``extra="allow"``). ``get_model_kwargs`` then merged
``__pydantic_extra__`` into the model kwargs, and the model classes re-open the
alias in their own ``@alias_kwargs`` decorator, so the value did reach the model
by a second, unintended route. Two things were wrong with that route:

* it bypassed parameter validation entirely -- ``--dt 3 --lossrt 4`` passed the
  params layer, and only the model layer noticed the delay was off the dt grid;
* it worked only where a composite already happened to carry the alias
  (testing) and silently did not elsewhere (training, init).

The tests below assert the properties that keep the gap closed:

1. Completeness: every alias a component declares resolves through every
   composite that contains it, scoped to that component.
2. Precedence: an alias already claimed by the composite is not overridden by a
   component's own meaning of the same name.
3. Directness: the value lands on the component *field*, so the params layer
   validates it, rather than surviving as an unvalidated extra.
"""

from __future__ import annotations

import pytest

from dynvision.params import InitParams, TestingParams, TrainingParams
from dynvision.params.composite_params import CompositeParams

from .utils import create_temp_config, temporary_training_paths

#: The composites that carry the full model/trainer/data component set.
COMPOSITES = (TrainingParams, TestingParams, InitParams)

#: Alias targets that are not declared fields of their component, and so reach
#: their consumer only via ``extra="allow"`` plus the component's kwargs builder
#: merging ``__pydantic_extra__``. Empty: both entries this once held
#: (``model.recurrence_target``, ``data.data_loader``) are now declared fields.
#: A new entry here means a parameter that validation, provenance and the summary
#: cannot see, so it should be added deliberately.
EXTRA_FORWARDED_TARGETS = frozenset()


def component_alias_pairs(composite):
    """Yield ``(component, alias, scoped_target)`` for one composite."""
    for name, component_cls in composite.get_component_classes().items():
        for alias, target in component_cls.get_aliases().items():
            yield name, alias, f"{name}.{target}"


class TestComponentAliasCompleteness:
    """Every component alias must be reachable from every containing composite."""

    @pytest.mark.parametrize("composite", COMPOSITES)
    def test_every_component_alias_is_mapped(self, composite):
        """An alias a component declares must appear in the composite's map."""
        aliases = composite.get_aliases()
        missing = sorted(
            (alias, target)
            for _, alias, target in component_alias_pairs(composite)
            if alias not in aliases
        )
        assert not missing, (
            f"{composite.__name__}.get_aliases() drops component aliases: "
            f"{missing}. A bare key for these reaches its component unscoped and "
            f"is ignored."
        )

    @pytest.mark.parametrize("composite", COMPOSITES)
    def test_component_aliases_point_at_their_own_component(self, composite):
        """An alias must resolve into the component that declared it.

        The only permitted deviation is an alias the composite already claims for
        itself, which must then resolve to a composite-level field (no dot) --
        ``log`` is claimed by all three components and by ``BaseParams``. A
        deviation pointing into a *different* component would be misrouting.
        """
        aliases = composite.get_aliases()
        wrong = []
        for _, alias, scoped_target in component_alias_pairs(composite):
            actual = aliases[alias]
            if actual != scoped_target and "." in actual:
                wrong.append((alias, actual, scoped_target))
        assert (
            not wrong
        ), f"{composite.__name__} routes an alias into the wrong component: {wrong}"

    @pytest.mark.parametrize("composite", COMPOSITES)
    def test_no_new_alias_targets_an_undeclared_field(self, composite):
        """Ratchet the alias targets that are not declared component fields.

        A target may be an undeclared field and still function: ``extra="allow"``
        stores it, ``get_model_kwargs``/``get_dataloader_kwargs`` merge
        ``__pydantic_extra__`` into the kwargs the component is built with, and
        the model classes re-open the name in ``@alias_kwargs``. That works but
        keeps the parameter invisible to validation, provenance and the summary.
        Declaring them would make them first-class.
        """
        classes = composite.get_component_classes()
        undeclared = set()
        for _, target in composite.get_aliases().items():
            component, _, field = target.partition(".")
            if not field or component not in classes:
                continue  # composite-level alias, e.g. log -> log_level
            if field not in classes[component].model_fields:
                undeclared.add(target)
        assert undeclared <= EXTRA_FORWARDED_TARGETS, (
            f"{composite.__name__} has aliases pointing at fields that are neither "
            f"declared nor on the forwarded-extras allow-list: "
            f"{sorted(undeclared - EXTRA_FORWARDED_TARGETS)}"
        )

    def test_the_reported_gap_is_closed(self):
        """The four wildcards that were silently dropped now route to the model."""
        aliases = TrainingParams.get_aliases()
        assert aliases["trc"] == "model.t_recurrence"
        assert aliases["tff"] == "model.t_feedforward"
        assert aliases["tsk"] == "model.t_skip"
        assert aliases["lossrt"] == "model.loss_reaction_time"

    def test_init_params_carries_the_model_aliases(self):
        aliases = InitParams.get_aliases()
        assert aliases["trc"] == "model.t_recurrence"
        assert aliases["tff"] == "model.t_feedforward"
        assert aliases["tsk"] == "model.t_skip"

    def test_composite_subclass_without_components_raises(self):
        """``get_component_aliases`` needs a declared component mapping."""

        class NoComponents(CompositeParams):
            pass

        with pytest.raises(NotImplementedError):
            NoComponents.get_component_aliases()


class TestComponentAliasPrecedence:
    """An alias the composite already claims must not be overridden."""

    def test_log_resolves_to_the_composite_log_level(self):
        """All three components alias 'log' to their own log_level.

        The composite's own map wins, so 'log' must stay unscoped rather than
        being hijacked by whichever component is declared first.
        """
        for composite in COMPOSITES:
            assert composite.get_aliases()["log"] == "log_level"

    @pytest.mark.parametrize("composite", COMPOSITES)
    def test_get_component_aliases_does_not_mutate_shared_state(self, composite):
        """Two calls return equal, independent maps."""
        first = composite.get_component_aliases()
        first["trc"] = "mutated"
        assert composite.get_component_aliases()["trc"] != "mutated"

    def test_first_component_wins_on_a_shared_alias_name(self):
        """Component-order tie-break for a name two components both declare."""

        class FakeComposite(CompositeParams):
            component_classes = TrainingParams.get_component_classes()

        scoped = FakeComposite.get_component_aliases()
        assert scoped["log"] == "model.log_level"


class TestAliasReachesTheComponent:
    """End-to-end: a bare alias key must land on the component's field."""

    def _train(self, config_data, overrides):
        config_file = create_temp_config(config_data)
        try:
            with temporary_training_paths() as training_paths:
                return TrainingParams.from_cli_and_config(
                    config_path=str(config_file),
                    override_kwargs={**training_paths, **overrides},
                )
        finally:
            config_file.unlink()

    @pytest.mark.parametrize(
        "alias,field,value",
        [
            ("trc", "t_recurrence", 6.0),
            ("tff", "t_feedforward", 0.0),
            ("tsk", "t_skip", 0.0),
        ],
    )
    def test_bare_alias_reaches_the_model_field(self, alias, field, value):
        params = self._train({"dt": 2}, {alias: value})
        assert getattr(params.model, field) == value
        # The bare key must not also survive as an inert extra attribute.
        extras = getattr(params.model, "__pydantic_extra__", None) or {}
        assert alias not in extras, (
            f"{alias!r} was stored as an unused extra instead of being routed "
            f"to {field!r}"
        )

    def test_reaction_time_alias_reaches_the_model_field(self):
        params = self._train({"dt": 2}, {"lossrt": 4.0})
        assert params.model.loss_reaction_time == 4.0

    def test_routed_delay_is_then_validated(self):
        """Once the alias routes, the dt-multiple constraint applies to it.

        Before the alias fix this configuration was accepted, because
        ``lossrt`` never reached ``loss_reaction_time`` at all.
        """
        with pytest.raises(Exception, match="loss_reaction_time=4 ms"):
            self._train({"dt": 3}, {"trc": 6.0, "lossrt": 4.0})

    def test_routing_does_not_break_a_valid_configuration(self):
        params = self._train(
            {"dt": 2},
            {"trc": 6.0, "tff": 0.0, "tsk": 0.0, "lossrt": 4.0},
        )
        assert params.model.t_recurrence == 6.0
        assert params.model.delay_recurrence == 3
        assert params.model.loss_reaction_time == 4.0
