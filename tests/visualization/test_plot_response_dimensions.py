"""Tests for the flexible dimension mapping of the response ridge plotter.

'layers' is a special dimension: it selects the wide ``{layer}_response_avg``
column rather than filtering rows on a dataframe column. These tests pin down
that it is handled for each of the three dimension roles (subplot, hue, column)
and that it is rejected where it cannot be honoured.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dynvision.processing.time_base import ensure_time_base  # noqa: E402
from dynvision.visualization import plot_responses as pr  # noqa: E402

LAYERS = ["V1", "V2", "V4", "IT"]
LAYER_OFFSETS = {"V1": 1.0, "V2": 2.0, "V4": 4.0, "IT": 8.0}
PARAMETERS = [0.1, 0.4]
CATEGORIES = ["full", "self"]
N_TIMES = 4
N_SAMPLES = 3


def make_df() -> pd.DataFrame:
    """Response dataframe with per-layer, per-timestep constant responses."""
    rows = []
    for parameter in PARAMETERS:
        for category in CATEGORIES:
            for sample in range(N_SAMPLES):
                for t in range(N_TIMES):
                    row = {
                        "times_index": t,
                        "first_label_index": sample,
                        "label_index": 0,
                        "label_valid": True,
                        "contrast": parameter,
                        "rctype": category,
                        "accuracy": 0.5,
                        "first_label_confidence": 0.5,
                    }
                    for layer in LAYERS:
                        # Response depends on layer and timestep only, so the
                        # plotted mean trace is exactly reprovable.
                        row[f"{layer}_response_avg"] = LAYER_OFFSETS[layer] + t
                    rows.append(row)
    return ensure_time_base(pd.DataFrame(rows), dt=1.0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_find_layer_dimension():
    assert pr._find_layer_dimension("layers", "category", "parameter") == "subplot"
    assert pr._find_layer_dimension("parameter", "layers", "category") == "hue"
    assert pr._find_layer_dimension("parameter", "category", "layers") == "column"
    assert pr._find_layer_dimension("parameter", "category", None) is None


def test_validate_dimensions_rejects_duplicates():
    with pytest.raises(ValueError, match="must be unique"):
        pr._validate_dimensions("layers", "category", "layers")


def test_validate_dimension_roles_rejects_layers_with_classifier_topk():
    with pytest.raises(ValueError, match="cannot be combined"):
        pr._validate_dimension_roles("classifier_topk", "layers", None)
    with pytest.raises(ValueError, match="cannot be combined"):
        pr._validate_dimension_roles("layers", "classifier_topk", None)


def test_validate_dimension_roles_rejects_classifier_topk_column():
    with pytest.raises(ValueError, match="not supported for columns"):
        pr._validate_dimension_roles("parameter", "category", "classifier_topk")


def test_validate_dimension_roles_accepts_layers_in_one_role():
    pr._validate_dimension_roles("layers", "category", "parameter")
    pr._validate_dimension_roles("parameter", "layers", "category")
    pr._validate_dimension_roles("parameter", "category", "layers")


def test_pinned_layer_fixes_the_response_column():
    """A pinned layer draws its own response column, coloured by hue."""
    df = make_df()
    for layer in LAYERS:
        fig = plt.figure(figsize=(4, 3))
        axes = pr._plot_response_ridges(
            fig=fig,
            column_left=0.1,
            column_width=0.8,
            data=df,
            subplot_var="parameter",
            subplot_key="contrast",
            subplot_values=["0.1", "0.4"],
            hue_var="category",
            hue_key="rctype",
            hue_values=CATEGORIES,
            colors={"full": "#111111", "self": "#222222"},
            response_layer=layer,
            config={"palette": {}, "naming": {}, "ordering": {}},
        )
        assert len(axes) == len(PARAMETERS)
        for ax in axes:
            # lines[0] is the zero reference line, then one line per hue value.
            for hue_line in ax.lines[1 : 1 + len(CATEGORIES)]:
                np.testing.assert_allclose(
                    hue_line.get_ydata(), LAYER_OFFSETS[layer] + np.arange(N_TIMES)
                )


def test_layers_as_column_pins_each_column_to_its_layer(monkeypatch, tmp_path):
    """column='layers' must hand each column its own layer and full data."""
    calls = []

    def fake_ridges(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(pr, "_plot_response_ridges", fake_ridges)
    monkeypatch.setattr(pr, "_plot_accuracy_panel", lambda **kwargs: None)

    df = make_df()
    config = {"palette": {}, "naming": {"layer": "Layer"}, "ordering": {}}
    pr.plot_temporal_ridge_responses(
        data=df,
        output=tmp_path / "layers_as_column.png",
        subplot_var="parameter",
        hue_var="category",
        column_var="layers",
        category_key="rctype",
        parameter_key="contrast",
        config=config,
    )

    expected_order = pr._extract_dimension_values(df, "layers", "layers", config)
    assert [call["response_layer"] for call in calls] == expected_order
    assert sorted(expected_order) == sorted(LAYERS)
    for call in calls:
        # No row filtering happens for the layer dimension.
        assert len(call["data"]) == len(df)


def test_layers_as_column_renders(tmp_path):
    df = make_df()
    fig = pr.plot_temporal_ridge_responses(
        data=df,
        output=tmp_path / "layers_as_column.png",
        subplot_var="parameter",
        hue_var="category",
        column_var="layers",
        category_key="rctype",
        parameter_key="contrast",
        config={"palette": {}, "naming": {"layer": "Layer"}, "ordering": {}},
    )
    titles = [
        text.get_text()
        for ax in fig.axes
        for text in ax.texts
        if text.get_text().startswith("Layer =")
    ]
    assert sorted(titles) == sorted(f"Layer = {layer}" for layer in LAYERS)
    assert (tmp_path / "layers_as_column.png").exists()


def test_layers_as_hue_renders(tmp_path):
    df = make_df()
    fig = pr.plot_temporal_ridge_responses(
        data=df,
        output=tmp_path / "layers_as_hue.png",
        subplot_var="parameter",
        hue_var="layers",
        category_key="rctype",
        parameter_key="contrast",
        config={"palette": {}, "naming": {}, "ordering": {}},
    )
    assert fig is not None
    assert (tmp_path / "layers_as_hue.png").exists()


def test_layers_as_subplot_renders(tmp_path):
    df = make_df()
    fig = pr.plot_temporal_ridge_responses(
        data=df,
        output=tmp_path / "layers_as_subplot.png",
        subplot_var="layers",
        hue_var="category",
        column_var="parameter",
        category_key="rctype",
        parameter_key="contrast",
        config={"palette": {}, "naming": {}, "ordering": {}},
    )
    assert fig is not None
    assert (tmp_path / "layers_as_subplot.png").exists()


def test_plot_rejects_incompatible_classifier_dimension(tmp_path):
    """The role validation is wired into the plotting entry point."""
    with pytest.raises(ValueError, match="not supported for columns"):
        pr.plot_temporal_ridge_responses(
            data=make_df(),
            output=tmp_path / "rejected.png",
            subplot_var="parameter",
            hue_var="category",
            column_var="classifier_topk",
            category_key="rctype",
            parameter_key="contrast",
            config={"palette": {}, "naming": {}, "ordering": {}},
        )
