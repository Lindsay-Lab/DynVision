"""Single authority for temporal units: milliseconds at the interface, steps inside.

DynVision specifies every quantity with a biological meaning in milliseconds:
the propagation delays (``t_feedforward``, ``t_recurrence``, ``t_feedback``,
``t_skip``), the loss reaction window (``loss_reaction_time``), the simulation
and idle windows, and the stimulus protocol (``t_stimulus``, ``t_intro``,
``t_interval``, ``t_sequence``). Models integrate on a fixed grid of ``dt``
millisecond steps, and the delay buffers, presentation patterns and stimulus
tensors that implement those quantities are integer-indexed. So every one of
them must be a whole number of steps, and it is converted exactly once, here.

A duration that is not an exact multiple cannot be represented on the grid. It
used to be converted silently, and the conversion sites disagreed with each
other -- ``int(value / dt)`` truncated toward zero for the feedforward and
recurrent delays while ``math.ceil`` rounded up for the reaction time, so
``dt=3, t_recurrence=4, loss_reaction_time=4`` ran with a 3 ms recurrence and a
6 ms reaction window. :func:`to_timesteps` therefore raises instead of rounding.

``tau`` is the one temporal parameter without this constraint: it enters the
dynamics only as the ratio ``dt / tau`` and never indexes anything.

The registry :data:`TEMPORAL_QUANTITIES` names each windowed quantity in both of
its forms: the millisecond field with its file-name key (``t_stimulus`` /
``tstim``), and the integer step field the engine consumes with the legacy
step-valued keys that existing configurations and paths still use (``stim``).
A ``t`` prefix on a file-name key always means milliseconds. A legacy key keeps
its step meaning and is translated with :func:`to_ms`, which is always exact.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Final, Literal, Mapping, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tolerance for the multiple test. Durations are milliseconds in the 0.1-100
# range, so 1e-9 sits far above float64 division error (~1e-16 relative) and far
# below any fraction of a step that carries biological meaning.
_RTOL: Final[float] = 1e-9
_ATOL: Final[float] = 1e-9

# Decimal places kept when a millisecond value is computed as steps * dt.
# 3 * 0.1 is 0.30000000000000004 in binary floating point; rounding to 9
# places restores the value a user would have written (0.3), and no meaningful
# duration has a finer resolution than a nanosecond.
_MS_DECIMALS: Final[int] = 9


def is_integer_multiple(
    value: float,
    dt: float,
    *,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> bool:
    """Return True when ``value`` is a whole number of ``dt`` steps.

    The test is tolerant because it is applied to user-supplied floats.
    ``0.3 % 0.1`` evaluates to ``0.09999999999999998`` in binary floating point,
    so an exact remainder test would reject the valid configuration
    ``dt=0.1, t_recurrence=0.3``. Comparing ``value / dt`` against its nearest
    integer avoids that without accepting a materially different duration.

    Args:
        value: Duration in milliseconds. Must be non-negative.
        dt: Integration step in milliseconds. Must be positive.
        rtol: Relative tolerance passed to :func:`math.isclose`.
        atol: Absolute tolerance passed to :func:`math.isclose`.

    Returns:
        True if ``value`` is an integer number of ``dt`` steps.

    Raises:
        ValueError: If ``dt`` is not positive, or ``value`` is negative.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive to test for multiples, got dt={dt:g}")
    if value < 0:
        raise ValueError(f"duration must be non-negative, got value={value:g}")
    ratio = value / dt
    return math.isclose(ratio, round(ratio), rel_tol=rtol, abs_tol=atol)


def to_timesteps(value: float, dt: float, name: str = "duration") -> int:
    """Convert a millisecond duration to a whole number of ``dt`` steps.

    The conversion is exact or it fails: a duration off the ``dt`` grid raises
    instead of being rounded, because a rounded delay or window silently runs a
    different experiment than the one configured. This makes the function safe
    for every entry point, including model classes constructed directly
    without parameter validation.

    Args:
        value: Duration in milliseconds. Must be non-negative.
        dt: Integration step in milliseconds. Must be positive.
        name: Parameter name used in the error message, e.g.
            ``"t_recurrence"``.

    Returns:
        The duration expressed in integration steps.

    Raises:
        ValueError: If ``dt`` is not positive, ``value`` is negative, or
            ``value`` is not an integer multiple of ``dt``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive to convert to timesteps, got dt={dt:g}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {name}={value:g}")
    if not is_integer_multiple(value, dt):
        raise ValueError(multiple_error_message(name, value, dt))
    return int(round(value / dt))


def to_ms(steps: int, dt: float) -> float:
    """Convert a step count to milliseconds.

    The inverse of :func:`to_timesteps`, and always exact on the grid. It is
    how a legacy step-valued key (``stim=5``) is translated into its
    millisecond counterpart (``t_stimulus=10`` at ``dt=2``).

    Args:
        steps: Number of integration steps. Must be a non-negative integer.
        dt: Integration step in milliseconds. Must be positive.

    Returns:
        The duration in milliseconds, free of binary rounding noise.

    Raises:
        ValueError: If ``dt`` is not positive, or ``steps`` is negative or not
            a whole number.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive to convert to ms, got dt={dt:g}")
    if steps < 0 or not float(steps).is_integer():
        raise ValueError(f"steps must be a non-negative integer, got steps={steps}")
    return round(int(steps) * float(dt), _MS_DECIMALS)


def step_end_ms(times_index: Any, dt: Any) -> Any:
    """Return the time in ms at which the state of recorded step ``times_index`` is taken.

    The engine integrates step ``k`` (0-based, counted after the idle period)
    over ``[k*dt, (k+1)*dt)`` and records the state at its end, so the state of
    step ``k`` belongs to ``(k + 1) * dt``. ``t = 0`` is the end of the idle
    period, when the first input arrives.

    Args:
        times_index: Recorded step index, a scalar or an array-like (numpy
            array, pandas Series). Must be non-negative.
        dt: Integration step in milliseconds, a scalar or an array-like of
            the same length (one ``dt`` per row). Must be positive.

    Returns:
        The time in milliseconds, of the same shape as the inputs, free of
        binary rounding noise.

    Raises:
        ValueError: If a ``dt`` is not positive or an index is negative.
    """
    index = np.asarray(times_index, dtype=float)
    step = np.asarray(dt, dtype=float)
    if np.any(step <= 0) or np.any(np.isnan(step)):
        raise ValueError(f"dt must be positive to convert steps to ms, got dt={dt}")
    if np.any(index < 0):
        raise ValueError("times_index must be non-negative")
    result = np.round((index + 1) * step, _MS_DECIMALS)
    if hasattr(times_index, "index") and hasattr(times_index, "to_numpy"):
        return type(times_index)(result, index=times_index.index)
    if result.ndim == 0:
        return float(result)
    return result


def format_ms(value: float) -> str:
    """Return the canonical text of a millisecond value in an identifier.

    Integral values print without a decimal point (``10``), others in their
    shortest form (``2.5``). This is exactly ``str()`` of the number that YAML
    parses from ``10`` or ``2.5`` in a config file, so an identifier built from
    a translated legacy value equals the one the workflow builds from the
    config.

    Args:
        value: Duration in milliseconds.

    Returns:
        The value as identifier text.
    """
    value = round(float(value), _MS_DECIMALS)
    if value.is_integer():
        return str(int(value))
    return repr(value)


def require_min_steps(
    name: str, value: float, dt: float, min_steps: int, reason: str
) -> None:
    """Reject a duration shorter than ``min_steps`` integration steps.

    A recurrent or feedback connection reads its source's state from an earlier
    step, because the source has not been updated yet in the current one. Its
    delay is therefore at least one step; a configured value below that cannot
    be represented and used to be raised to one step silently.

    Args:
        name: Parameter name, e.g. ``"t_recurrence"``.
        value: Duration in milliseconds.
        dt: Integration step in milliseconds.
        min_steps: Smallest number of steps the duration may span.
        reason: Why the minimum holds, appended to the message.

    Raises:
        ValueError: If ``value`` spans fewer than ``min_steps`` steps.
    """
    if value < to_ms(min_steps, dt) and not math.isclose(value, to_ms(min_steps, dt)):
        raise ValueError(
            f"{name}={value:g} ms is shorter than {min_steps} step(s) of "
            f"dt={dt:g} ms. {reason} Use {name} >= {format_ms(to_ms(min_steps, dt))} ms."
        )


def multiple_error_message(name: str, value: float, dt: float) -> str:
    """Build the error message for a duration that is not a multiple of ``dt``.

    The message names the offending parameter, reports the ratio that could not
    be represented, and gives both neighbouring valid values so the caller can
    pick one without recomputing them.

    Args:
        name: Parameter name as it appears in the configuration, e.g.
            ``"t_recurrence"``.
        value: The offending duration in milliseconds.
        dt: Integration step in milliseconds.

    Returns:
        A message suitable for raising as a ``ValueError``.
    """
    ratio = value / dt
    lower_steps = math.floor(ratio)
    upper_steps = math.ceil(ratio)
    return (
        f"{name}={value:g} ms is not an integer multiple of dt={dt:g} ms "
        f"({value:g}/{dt:g} = {ratio:.6g} timesteps). Durations are integrated "
        f"on the dt grid, so they must be whole numbers of steps. "
        f"Use {lower_steps * dt:g} ms ({lower_steps} step(s)) or "
        f"{upper_steps * dt:g} ms ({upper_steps} step(s))."
    )


@dataclass(frozen=True)
class TemporalQuantity:
    """One windowed quantity, named in milliseconds and in steps.

    Attributes:
        field: Canonical millisecond field in configs, e.g. ``"t_stimulus"``.
        key: File-name key for the millisecond value, e.g. ``"tstim"``. File
            names forbid ``_`` in keys, and a ``t`` prefix marks milliseconds.
        steps_field: Integer step field the engine consumes, e.g.
            ``"stimulus_duration"``.
        legacy_keys: Step-valued keys that existing configs and paths use for
            the same quantity, e.g. ``("stim",)``. They keep their step
            meaning.
        component: Parameter component that owns the quantity.
        aliases: Further accepted millisecond names, e.g. ``("tisi",)``.
    """

    field: str
    key: str
    steps_field: str
    legacy_keys: Tuple[str, ...]
    component: Literal["model", "data"]
    aliases: Tuple[str, ...] = ()

    @property
    def ms_names(self) -> Tuple[str, ...]:
        """Every name that carries this quantity in milliseconds."""
        return (self.field, self.key, *self.aliases)

    @property
    def step_names(self) -> Tuple[str, ...]:
        """Every name that carries this quantity in steps."""
        return (self.steps_field, *self.legacy_keys)


#: The windowed quantities that exist in both a millisecond and a step form.
#: Delays (``t_feedforward`` etc.) and ``loss_reaction_time`` are milliseconds
#: only and are not listed here.
TEMPORAL_QUANTITIES: Final[Tuple[TemporalQuantity, ...]] = (
    TemporalQuantity(
        field="t_simulation",
        key="tsim",
        steps_field="n_timesteps",
        legacy_keys=("tsteps", "steps"),
        component="model",
    ),
    TemporalQuantity(
        field="t_idle",
        key="tidle",
        steps_field="idle_timesteps",
        legacy_keys=("idle",),
        component="model",
    ),
    TemporalQuantity(
        field="t_sequence",
        key="tseq",
        steps_field="data_timesteps",
        legacy_keys=("dsteps",),
        component="data",
    ),
    TemporalQuantity(
        field="t_stimulus",
        key="tstim",
        steps_field="stimulus_duration",
        legacy_keys=("stim",),
        component="data",
    ),
    TemporalQuantity(
        field="t_intro",
        key="tintro",
        steps_field="intro_duration",
        legacy_keys=("intro",),
        component="data",
    ),
    TemporalQuantity(
        field="t_interval",
        key="tinterval",
        steps_field="interval_duration",
        legacy_keys=("interval",),
        component="data",
        aliases=("tisi",),
    ),
)


def _build_name_index() -> Dict[str, TemporalQuantity]:
    index: Dict[str, TemporalQuantity] = {}
    for quantity in TEMPORAL_QUANTITIES:
        for name in (*quantity.ms_names, *quantity.step_names):
            if name in index:
                raise ValueError(
                    f"temporal name {name!r} is claimed by both "
                    f"{index[name].field!r} and {quantity.field!r}"
                )
            index[name] = quantity
    return index


_BY_NAME: Final[Dict[str, TemporalQuantity]] = _build_name_index()


def quantity_for(name: str) -> Optional[TemporalQuantity]:
    """Return the quantity a name refers to, in either unit, or None."""
    return _BY_NAME.get(name)


def is_ms_name(name: str) -> bool:
    """Return True when ``name`` carries a windowed quantity in milliseconds."""
    quantity = _BY_NAME.get(name)
    return quantity is not None and name in quantity.ms_names


def is_legacy_step_key(name: str) -> bool:
    """Return True when ``name`` is a legacy step-valued key (``stim``, ``tsteps``)."""
    quantity = _BY_NAME.get(name)
    return quantity is not None and name in quantity.legacy_keys


def resolve_time_window(
    component: Any, ms_field: str, steps_field: str, dt: Optional[float]
) -> None:
    """Derive a window's missing form on ``component``, in place.

    Whichever of the millisecond and step values is given derives the other;
    both given must agree. A step value without ``dt`` stays as given; an ms
    value without ``dt`` cannot be converted and is rejected. Derived values
    are written with ``update_field(..., mutation_tag="derived")`` so the
    provenance shows they were not configured.

    Args:
        component: Parameter component holding both fields.
        ms_field: Name of the millisecond field, e.g. ``"t_sequence"``.
        steps_field: Name of the step field, e.g. ``"data_timesteps"``.
        dt: Integration step in milliseconds, or None if unknown.

    Raises:
        ValueError: If the ms value is off the grid or lacks ``dt``, or the two
            forms disagree.
    """
    ms_value = getattr(component, ms_field, None)
    steps = getattr(component, steps_field, None)
    if ms_value is None and steps is None:
        return
    if dt is None:
        if ms_value is not None:
            raise ValueError(
                f"{ms_field}={ms_value:g} ms needs dt to be converted to "
                f"{steps_field}; set dt."
            )
        return
    if ms_value is None:
        component.update_field(ms_field, to_ms(steps, dt), mutation_tag="derived")
        return
    expected = to_timesteps(ms_value, dt, ms_field)
    if steps is None:
        component.update_field(steps_field, expected, mutation_tag="derived")
    elif steps != expected:
        raise ValueError(
            f"{ms_field}={ms_value:g} ms and {steps_field}={steps} disagree at "
            f"dt={dt:g} ms ({ms_value:g} ms = {expected} steps). Give only one "
            f"of them."
        )


def coordinate_component_timing(config: Any) -> None:
    """Convert the data component's windows with the model's ``dt``.

    The data component holds the stimulus protocol (``t_sequence``,
    ``t_stimulus``, ``t_intro``, ``t_interval``) but not ``dt``, which belongs
    to the model. Shared by ``TrainingParams``, ``TestingParams`` and
    ``InitParams``, like ``coordinate_component_dtypes``. Afterwards each
    window holds both forms and the loaders receive step counts.

    Args:
        config: A composite with ``model`` and ``data`` components.

    Raises:
        ValueError: As :func:`resolve_time_window`.
    """
    model = getattr(config, "model", None)
    data = getattr(config, "data", None)
    if model is None or data is None:
        return
    for quantity in TEMPORAL_QUANTITIES:
        if quantity.component == "data" and hasattr(data, quantity.field):
            resolve_time_window(
                data, quantity.field, quantity.steps_field, getattr(model, "dt", None)
            )


_WARNED_LEGACY_KEYS: Set[str] = set()


def warn_legacy_keys(keys) -> None:
    """Log a deprecation warning, once per process, for each legacy step key.

    Called where configuration and CLI keys are resolved, so it fires for keys
    a user wrote, not for the step fields the engine uses internally. Keys may
    be scoped (``model.tsteps``); only the last component is inspected.

    Args:
        keys: Iterable of parameter keys as given by the user.
    """
    for key in keys:
        name = str(key).rsplit(".", 1)[-1]
        if name in _WARNED_LEGACY_KEYS or not is_legacy_step_key(name):
            continue
        _WARNED_LEGACY_KEYS.add(name)
        quantity = _BY_NAME[name]
        logger.warning(
            "Legacy step key '%s': its value counts dt steps, so the duration it "
            "names changes with dt. It keeps its step meaning but is deprecated; "
            "use '%s' (milliseconds) instead.",
            name,
            quantity.key,
        )


def translate_legacy_args(args: Mapping[str, Any], dt: float) -> Dict[str, Any]:
    """Rewrite legacy step-valued keys to their millisecond keys.

    Order is preserved: each translated key takes the position of the legacy
    key it replaces, so an identifier built from the result lines up with one
    built from a config that lists the millisecond keys in the same order.
    Keys that are not legacy step keys pass through unchanged.

    Args:
        args: Mapping of key to value, e.g. parsed from an identifier.
        dt: Integration step in milliseconds that the step values refer to.

    Returns:
        A new dict with ``{key}=ms`` in place of every ``{legacy_key}=steps``.
        Values are floats in milliseconds (format them with
        :func:`format_ms`).

    Raises:
        ValueError: If a legacy value is not a non-negative whole number of
            steps, or the result would contain the same quantity twice.
    """
    translated: Dict[str, Any] = {}
    for name, value in args.items():
        if is_legacy_step_key(name):
            quantity = _BY_NAME[name]
            new_name = quantity.key
            new_value: Any = to_ms(_as_steps(name, value), dt)
        else:
            new_name, new_value = name, value
        if new_name in translated:
            raise ValueError(
                f"{new_name!r} appears twice after translating legacy keys in {dict(args)}"
            )
        translated[new_name] = new_value
    return translated


def _as_steps(name: str, value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}={value!r} is not a step count") from None
    if number < 0 or not number.is_integer():
        raise ValueError(f"{name}={value!r} is not a non-negative whole number of steps")
    return int(number)
