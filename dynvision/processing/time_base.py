"""Time base of test results: ``dt``, ``idle_steps`` and ``time_ms`` per row.

A test records one row per sample and recorded step, indexed by
``times_index``. That index counts integration steps after the idle period, so
it only becomes a time with the run's ``dt``. Results used to carry no ``dt``
(except when ``dt`` was the swept category), and every plot converted them with
one configured ``dt``. This module attaches the time base to each row:

- ``dt``: the run's integration step in ms (``model.dt`` of the resolved
  config ``test_outputs.csv.config.yaml``).
- ``idle_steps``: idle steps run before the input (``model.idle_timesteps``).
  They are not recorded; the column only documents them.
- ``time_ms``: ``(times_index + 1) * dt``, see
  :func:`dynvision.utils.timing.step_end_ms`. ``t = 0`` is the end of the idle
  period.

It also gives every legacy step-valued parameter column (``stim``,
``interval``, ``idle``, ``dsteps``, ...) its millisecond column (``tstim``,
``tinterval``, ...), converted with each row's own ``dt``. Results tested with
legacy step keys and results tested with millisecond keys then aggregate on one
millisecond column.

The time base is never guessed: without a ``dt`` from the config, a column or
the path, :func:`ensure_time_base` raises instead of assuming one.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import pandas as pd
import yaml

from dynvision.utils.string_utils import extract_param_from_string
from dynvision.utils.timing import (
    TEMPORAL_QUANTITIES,
    is_legacy_step_key,
    is_ms_name,
    quantity_for,
    step_end_ms,
    to_ms,
    to_timesteps,
)

logger = logging.getLogger(__name__)

TIME_INDEX = "times_index"
DT = "dt"
IDLE_STEPS = "idle_steps"
TIME_MS = "time_ms"

#: Prefix the aggregation gives a category column that collides with the
#: experiment parameter (``train_idle`` next to ``idle``).
_CATEGORY_PREFIX = "train_"

_NAMESPACES = ("data.", "model.", "")


def load_resolved_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a resolved config written by ``persist_resolved_config``.

    Args:
        path: Path to a ``*.config.yaml`` file. Its keys are dotted
            (``model.dt``, ``data.t_stimulus``).

    Returns:
        The flat mapping of dotted keys to values.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _lookup(config: Mapping[str, Any], *names: str) -> Any:
    """Return the first value found for any name, in data, model or no scope."""
    for name in names:
        for namespace in _NAMESPACES:
            value = config.get(f"{namespace}{name}")
            if value is not None:
                return value
    return None


def config_time_base(
    config: Mapping[str, Any], source: Optional[Union[str, Path]] = None
) -> Tuple[float, int]:
    """Read ``(dt, idle_steps)`` from a resolved config.

    Both the current format (``model.t_idle`` with ``model.idle_timesteps``)
    and configs written before the millisecond fields existed (only
    ``model.idle_timesteps``) are read.

    Args:
        config: Flat mapping as returned by :func:`load_resolved_config`.
        source: Where the mapping came from, for the error message.

    Returns:
        The integration step in ms and the number of idle steps (0 when the
        config names none, which is the engine's default).

    Raises:
        ValueError: If the config holds no ``dt``.
    """
    dt = _lookup(config, "dt")
    if dt is None:
        raise ValueError(
            f"No model.dt in the resolved config {source or ''}; the time base "
            f"of the results cannot be determined."
        )
    dt = float(dt)
    idle_steps = _lookup(config, "idle_timesteps", "idle")
    if idle_steps is None:
        t_idle = _lookup(config, "t_idle", "tidle")
        idle_steps = 0 if t_idle is None else to_timesteps(float(t_idle), dt, "t_idle")
    return dt, int(idle_steps)


def config_parameter(config: Mapping[str, Any], key: str) -> Any:
    """Read a parameter from a resolved config by any of its names.

    A windowed quantity is stored under its field names (``data.t_stimulus``,
    ``data.stimulus_duration``), while experiments name it by a file-name key
    (``tstim``) or a legacy step key (``stim``). The value is returned in the
    unit the key names: a millisecond key gives ms, a step key gives steps,
    converted with the config's ``dt`` when only the other form is stored.
    Every other key is looked up as given, in the data, then model, then
    top-level scope.

    Args:
        config: Flat mapping as returned by :func:`load_resolved_config`.
        key: Parameter name, e.g. ``"tstim"``, ``"stim"``, ``"contrast"``.

    Returns:
        The value, or None if the config does not hold it.
    """
    quantity = quantity_for(key)
    if quantity is None:
        return _lookup(config, key)
    ms_value = _lookup(config, *quantity.ms_names)
    steps = _lookup(config, *quantity.step_names)
    if is_ms_name(key):
        if ms_value is not None or steps is None:
            return ms_value
        return to_ms(steps, config_time_base(config)[0])
    if steps is not None or ms_value is None:
        return steps
    return to_timesteps(float(ms_value), config_time_base(config)[0], quantity.field)


def add_time_base(
    df: pd.DataFrame, dt: Any, idle_steps: Optional[Any] = None
) -> pd.DataFrame:
    """Set the ``dt``, ``idle_steps`` and ``time_ms`` columns, in place.

    Args:
        df: Result table with a ``times_index`` column.
        dt: Integration step in ms, a scalar or one value per row.
        idle_steps: Idle steps, a scalar or one value per row; the column is
            left out when None.

    Returns:
        ``df``, for chaining.
    """
    df[DT] = dt
    if idle_steps is not None:
        df[IDLE_STEPS] = idle_steps
    if TIME_INDEX in df.columns:
        df[TIME_MS] = step_end_ms(df[TIME_INDEX], df[DT])
    return df


def _step_column_quantity(column: str):
    """Return (quantity, prefix) when ``column`` is a legacy step parameter.

    Only legacy keys are swept parameters. The step fields of the engine
    (``idle_timesteps``) describe the run itself, which ``idle_steps`` covers.
    """
    prefix = ""
    name = column
    if column.startswith(_CATEGORY_PREFIX):
        prefix, name = _CATEGORY_PREFIX, column[len(_CATEGORY_PREFIX) :]
    if not is_legacy_step_key(name):
        return None, prefix
    return quantity_for(name), prefix


def add_ms_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a millisecond column for every legacy step parameter column, in place.

    ``stim`` gets ``tstim``, ``interval`` gets ``tinterval``, ``tsteps`` gets
    ``tsim``, and a renamed category column ``train_idle`` gets
    ``train_tidle``. Each row is converted with its own
    ``dt``, so rows of different ``dt`` land on one millisecond axis. A
    millisecond column that already exists is kept; rows where it is empty
    are filled from the step column.

    Args:
        df: Result table with a ``dt`` column.

    Returns:
        ``df``, for chaining.

    Raises:
        KeyError: If a step column exists but ``dt`` does not.
    """
    for column in list(df.columns):
        quantity, prefix = _step_column_quantity(column)
        if quantity is None:
            continue
        if DT not in df.columns:
            raise KeyError(
                f"Column '{column}' counts steps, but the table has no '{DT}' "
                f"column to convert it with."
            )
        steps = pd.to_numeric(df[column], errors="coerce")
        pairs = pd.DataFrame({"steps": steps, DT: df[DT]}).dropna()
        lookup = {
            (s, d): to_ms(int(s), float(d))
            for s, d in pairs.drop_duplicates().itertuples(index=False)
        }
        converted = pd.Series(
            [lookup.get((s, d)) for s, d in zip(steps, df[DT])],
            index=df.index,
            dtype=float,
        )
        target = f"{prefix}{quantity.key}"
        if target in df.columns:
            df[target] = pd.to_numeric(df[target], errors="coerce").fillna(converted)
        else:
            df[target] = converted
    return df


def _dt_from_path(source_path: Optional[Union[str, Path]]) -> Optional[float]:
    if source_path is None:
        return None
    try:
        return float(extract_param_from_string(str(source_path), key=DT, value_type=float))
    except ValueError:
        return None


def _idle_steps_from_columns(df: pd.DataFrame) -> Optional[pd.Series]:
    """Idle steps per row from the columns a result may carry, or None."""
    for column in (IDLE_STEPS, "idle_timesteps", "idle"):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().all():
                return values.astype(int)
    idle_ms = quantity_for("tidle")
    for column in idle_ms.ms_names:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().all():
                return pd.Series(
                    [
                        to_timesteps(float(t), float(d), "t_idle")
                        for t, d in zip(values, df[DT])
                    ],
                    index=df.index,
                )
    return None


def ensure_time_base(
    df: pd.DataFrame,
    source_path: Optional[Union[str, Path]] = None,
    *,
    dt: Optional[float] = None,
    idle_steps: Optional[int] = None,
    t_idle: Optional[float] = None,
) -> pd.DataFrame:
    """Give a result table its time base, whatever format it was written in.

    Results processed before this module existed hold only ``times_index``,
    plus a ``dt`` column when ``dt`` was the swept category. ``dt`` is taken,
    in this order, from a complete ``dt`` column, the ``dt`` argument, or a
    ``dt=`` key in ``source_path`` (the model identifier). There is no default:
    a missing ``dt`` raises.

    ``idle_steps`` is taken from the table's columns (``idle_steps``,
    ``idle_timesteps``, ``idle``, ``tidle``), else from the ``idle_steps`` or
    ``t_idle`` arguments; when none of them gives it, the column is left out.

    Args:
        df: Result table with a ``times_index`` column.
        source_path: Path the table was read from.
        dt: Integration step in ms, when known from elsewhere.
        idle_steps: Idle steps, when known from elsewhere.
        t_idle: Idle period in ms, when known from elsewhere.

    Returns:
        A copy of ``df`` with ``dt``, ``time_ms``, the ms columns of its step
        parameters and, if determinable, ``idle_steps``.

    Raises:
        ValueError: If ``dt`` cannot be determined.
    """
    df = df.copy()
    if DT in df.columns and pd.to_numeric(df[DT], errors="coerce").notna().all():
        dt_values: Any = pd.to_numeric(df[DT]).astype(float)
    elif dt is not None:
        dt_values = float(dt)
    else:
        dt_values = _dt_from_path(source_path)
        if dt_values is None:
            raise ValueError(
                f"Cannot determine dt for {source_path or 'the result table'}: no "
                f"complete '{DT}' column, no dt argument, and no 'dt=' in the path. "
                f"Pass dt explicitly; it is never assumed."
            )
    df[DT] = dt_values

    idle = _idle_steps_from_columns(df)
    if idle is None and idle_steps is not None:
        idle = int(idle_steps)
    if idle is None and t_idle is not None:
        if isinstance(dt_values, float):
            idle = to_timesteps(float(t_idle), dt_values, "t_idle")
        else:
            idle = pd.Series(
                [to_timesteps(float(t_idle), float(d), "t_idle") for d in dt_values],
                index=df.index,
            )
    if idle is None:
        logger.info(
            "No idle period recorded for %s; leaving out '%s'.",
            source_path or "the result table",
            IDLE_STEPS,
        )
    add_time_base(df, df[DT], idle)
    return add_ms_columns(df)


__all__ = [
    "DT",
    "IDLE_STEPS",
    "TEMPORAL_QUANTITIES",
    "TIME_INDEX",
    "TIME_MS",
    "add_ms_columns",
    "add_time_base",
    "config_parameter",
    "config_time_base",
    "ensure_time_base",
    "load_resolved_config",
]
