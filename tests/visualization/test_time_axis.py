"""Plots read time from the data, and ridge plots scale once per model.

Plots used to convert ``times_index`` with one configured ``dt`` (default 2)
and an idle offset from the experiment's step-valued ``idle``. Since results
carry their own time base (``ensure_time_base``), every plot draws on
``time_ms``: rows of different ``dt`` share one millisecond axis, and t = 0 is
the end of the idle period.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from dynvision.processing.time_base import ensure_time_base  # noqa: E402
from dynvision.utils.visualization_utils import (  # noqa: E402
    calculate_label_indicator,
    format_parameter_value,
    stimulus_indicators,
    load_result_data,
    ms_parameter_key,
    require_time_ms,
)
from dynvision.visualization import plot_dynamics as pd_mod  # noqa: E402
from dynvision.visualization import plot_responses as pr  # noqa: E402
from dynvision.visualization import plot_unrolling as pu  # noqa: E402

CONFIG = {"palette": {}, "naming": {}, "ordering": {}}
WORKFLOW = Path(__file__).resolve().parents[2] / "dynvision" / "workflow"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _write(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _legacy_table(n_steps: int, interval_steps=(2,)) -> pd.DataFrame:
    """A processed result as written before results carried their time base."""
    rows = []
    for interval in interval_steps:
        for k in range(n_steps):
            rows.append(
                {
                    "times_index": k,
                    "sample_index": 0,
                    "interval": interval,
                    "rctype": "full",
                    "V1_response_avg": float(k),
                    "label_valid": k < 5,
                }
            )
    return pd.DataFrame(rows)


# --- loading -----------------------------------------------------------------


def test_load_result_data_takes_dt_from_each_path(tmp_path):
    """Files of different dt land on one ms axis that ends at the same time."""
    paths = [
        _write(
            tmp_path / f"m:tsteps=20+dt={dt}_7000" / "test_data.csv", _legacy_table(n)
        )
        for dt, n in ((1, 40), (2, 20))
    ]
    df = load_result_data(paths)

    for dt, n in ((1.0, 40), (2.0, 20)):
        rows = df[df["dt"] == dt]
        assert rows["time_ms"].min() == dt
        assert rows["time_ms"].max() == 40.0
        assert len(rows) == n
    # The legacy step parameter gets its ms column, converted per row.
    assert sorted(df.groupby("dt")["tinterval"].first().tolist()) == [2.0, 4.0]


def test_load_result_data_never_assumes_dt(tmp_path):
    path = _write(tmp_path / "no_time_base" / "test_data.csv", _legacy_table(4))
    with pytest.raises(ValueError, match="Cannot determine dt"):
        load_result_data(path)


def test_require_time_ms_rejects_a_table_without_time_base():
    with pytest.raises(KeyError, match="time_ms"):
        require_time_ms(_legacy_table(3))


def test_ms_parameter_key_prefers_the_ms_column():
    df = ensure_time_base(_legacy_table(3), dt=2.0)
    assert ms_parameter_key(df, "interval") == "tinterval"
    assert ms_parameter_key(df, "rctype") == "rctype"
    assert ms_parameter_key(df, "tinterval") == "tinterval"
    assert ms_parameter_key(_legacy_table(3), "interval") == "interval"


# --- labels ------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, parameter, expected",
    [
        ("10", "tstim", "10 ms"),
        ("2.5", "tinterval", "2.5 ms"),
        ("6", "trc", "6 ms"),
        ("5", "tau", "5 ms"),
        ("0.5", "dt", "0.5 ms"),
        ("0.40", "contrast", "0.40"),
        ("true", "skip", "True"),
    ],
)
def test_format_parameter_value_never_converts(value, parameter, expected):
    assert format_parameter_value(value, parameter) == expected


def test_legend_label_names_units_without_dt():
    assert pr._format_legend_label("tsim", "40", CONFIG) == "40 ms"
    assert pr._format_legend_label("tidle", "10", CONFIG) == "10 ms"
    # A step column without a ms counterpart is labelled as what it is.
    assert pr._format_legend_label("tsteps", "20", CONFIG) == "20 steps"


def test_label_indicator_runs_on_time_ms():
    df = ensure_time_base(_legacy_table(8), dt=2.0)
    indicator = calculate_label_indicator(df, "rctype", (0.0, 1.0), 0.25)
    assert list(indicator.columns) == ["time_ms", "label_indicator"]
    assert indicator["time_ms"].tolist() == [2.0 * (k + 1) for k in range(8)]


def _stimulus_table(dt: float, stim_ms: float, n_steps: int) -> pd.DataFrame:
    """One model at ``dt``: stimulus on for the first ``stim_ms``."""
    table = _legacy_table(n_steps)
    table["label_valid"] = (table["times_index"] + 1) * dt <= stim_ms
    table["dt_model"] = str(dt)
    return ensure_time_base(table, dt=dt)


def test_stimulus_indicator_shared_by_models_of_one_protocol():
    """dt 1 and 2 with a 10 ms stimulus: one indicator on the 1 ms grid from t = 0."""
    df = pd.concat([_stimulus_table(1.0, 10, 40), _stimulus_table(2.0, 10, 20)])
    indicators = stimulus_indicators(df, "dt_model", (0.0, 1.0), 0.25)
    assert len(indicators) == 1
    hue, indicator = indicators[0]
    assert hue is None
    assert indicator["time_ms"].tolist() == [0.0] + [float(t) for t in range(1, 41)]
    on = indicator.loc[indicator["label_indicator"] > 0, "time_ms"]
    assert (on.min(), on.max()) == (0.0, 10.0)


def test_stimulus_indicator_per_model_when_protocols_differ():
    """A legacy 5-step stimulus is 5 ms at dt=1 and 10 ms at dt=2: two indicators."""
    df = pd.concat([_stimulus_table(1.0, 5, 40), _stimulus_table(2.0, 10, 20)])
    indicators = dict(stimulus_indicators(df, "dt_model", (0.0, 1.0), 0.25))
    assert set(indicators) == {"1.0", "2.0"}
    assert indicators["2.0"]["time_ms"].tolist() == [0.0] + [
        2.0 * (k + 1) for k in range(20)
    ]


def test_panel_a_value_is_one_every_model_has():
    """Intervals of 1-10 steps are 0.5-5 ms at dt=0.5 and 2-20 ms at dt=2."""
    rows = [
        {"dt": dt, "tinterval": steps * dt}
        for dt in (0.5, 1.0, 2.0)
        for steps in (1, 2, 3, 4, 6, 8, 10)
    ]
    df = pd.DataFrame(rows)
    values = pr._extract_dimension_values(df, "parameter", "tinterval", CONFIG)
    assert pr.shared_middle_value(df, "tinterval", "dt", values) in {"2.0", "4.0"}
    # Without a category column, the middle of all values
    assert (
        pr.shared_middle_value(df, "tinterval", None, values)
        == values[len(values) // 2]
    )


# --- metrics -----------------------------------------------------------------


def test_peak_time_is_read_from_time_ms():
    """Peak at step 4 is 10 ms at dt=2 and 5 ms at dt=1."""
    frames = []
    for dt in (1.0, 2.0):
        table = _legacy_table(10)
        table["V1_response_avg"] = -((table["times_index"] - 4.0) ** 2)
        table["dt"] = dt
        frames.append(ensure_time_base(table))
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = df["dt"].astype(str)

    result = pd_mod.calculate_metrics(
        df, "V1", parameter="tinterval", category="dt", metric="peak_time"
    )
    peaks = result.groupby("dt")["V1_peak_time"].first().to_dict()
    assert peaks == {"1.0": 5.0, "2.0": 10.0}


# --- ridge plots -------------------------------------------------------------


def _pulse_table() -> pd.DataFrame:
    """Two models; a short interval sums the second pulse to twice the first.

    Per-ridge max normalization made the first pulse of the short-interval
    ridge half as high as that of the long-interval ridge.
    """
    rows = []
    for model, gain in (("full", 1.0), ("self", 3.0)):
        for interval, second in ((4.0, 2.0), (12.0, 1.0)):
            for k in range(12):
                first_pulse = 1.0 if k == 1 else 0.0
                second_pulse = second if k == 1 + int(interval) // 2 else 0.0
                rows.append(
                    {
                        "times_index": k,
                        "tinterval": interval,
                        "rctype": model,
                        "V1_response_avg": gain * (first_pulse + second_pulse),
                        "label_valid": True,
                    }
                )
    return ensure_time_base(pd.DataFrame(rows), dt=2.0)


def _ridge_lines(normalize):
    fig = plt.figure(figsize=(4, 3))
    axes = pr._plot_response_ridges(
        fig=fig,
        column_left=0.1,
        column_width=0.8,
        data=_pulse_table(),
        subplot_var="parameter",
        subplot_key="tinterval",
        subplot_values=["4.0", "12.0"],
        hue_var="category",
        hue_key="rctype",
        hue_values=["full", "self"],
        colors={"full": "#111111", "self": "#222222"},
        response_layer="V1",
        normalize=normalize,
        config=CONFIG,
    )
    # lines[0] is the zero reference line, then one line per hue value.
    return [ax.lines[1:3] for ax in axes]


def test_ridges_scale_once_per_model():
    ridges = _ridge_lines("max")
    for hue_idx in range(2):
        first_pulses = [lines[hue_idx].get_ydata()[1] for lines in ridges]
        # One scale across parameter values: the first pulse is identical.
        assert first_pulses[0] == pytest.approx(first_pulses[1])
        # The model's largest mean value is 1.
        peak = max(lines[hue_idx].get_ydata().max() for lines in ridges)
        assert peak == pytest.approx(1.0)
        assert first_pulses[0] == pytest.approx(0.5)


def test_ridges_draw_on_time_ms():
    ridges = _ridge_lines(False)
    x = ridges[0][0].get_xdata()
    np.testing.assert_allclose(x, 2.0 * (np.arange(12) + 1))


# --- workflow and unrolling --------------------------------------------------


def test_visualization_rules_pass_no_time_base():
    """The rules used a config dt (default 2) and the step-valued 'idle'."""
    text = (WORKFLOW / "snake_visualizations.smk").read_text()
    assert "--dt" not in text
    assert "--idle-timesteps" not in text
    assert "get('idle'" not in text


def test_unrolling_reads_dt_from_the_run_config(tmp_path):
    run = tmp_path / "m:dt=2_7000" / "StimulusDuration:tseq=40"
    run.mkdir(parents=True)
    (run / pu.RUN_CONFIG).write_text(
        yaml.safe_dump({"model.dt": 0.5, "model.idle_timesteps": 0, "data.stim": 8})
    )
    assert pu.run_time_base(run / "test_responses.pt") == (0.5, 8)
    # Without a config, the path names dt.
    assert pu.run_time_base(tmp_path / "x:dt=2_1" / "r.pt") == (2.0, None)
    with pytest.raises(ValueError, match="Cannot determine dt"):
        pu.run_time_base(tmp_path / "plain" / "r.pt")
