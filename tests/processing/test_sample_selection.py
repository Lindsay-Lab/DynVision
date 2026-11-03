"""Compared test runs must rest on the same samples, or a warning says so.

In the local dt sweep, some dt=2 runs recorded layer responses for 32 samples
and others for 384; averaging across them moved the first-pulse height by 2%
without any sign in the figure.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from dynvision.processing import aggregate_experiment_data as agg
from dynvision.processing.sample_selection import (
    run_columns,
    sample_selection,
    warn_if_selections_differ,
    warn_on_sample_mismatch,
)
from dynvision.utils.visualization_utils import load_result_data

LOGGER = "dynvision.processing.sample_selection"


def _run(n_samples=8, n_recorded=8, n_steps=3, first_image=0, **params):
    """One test run: responses recorded for the first ``n_recorded`` samples."""
    rows = []
    for sample in range(n_samples):
        for k in range(n_steps):
            rows.append(
                {
                    "sample_index": sample,
                    "image_index": first_image + sample,
                    "times_index": k,
                    "label_index": 0,
                    "accuracy": 1.0,
                    "V1_response_avg": float(k) if sample < n_recorded else np.nan,
                    **params,
                }
            )
    return pd.DataFrame(rows)


def _warnings(caplog):
    return [
        r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING
    ]


def test_selection_counts_evaluated_and_recorded_samples():
    selection = sample_selection(_run(n_samples=8, n_recorded=3))
    assert selection.n_samples == 8
    assert [(layer, n) for layer, n, _ in selection.responses] == [("V1", 3)]


def test_layers_recorded_for_different_samples_are_told_apart():
    """The local dt=2 runs: V1 for 32 samples, the other layers for 512."""
    run = _run(n_samples=8, n_recorded=8)
    run["V2_response_avg"] = run["V1_response_avg"]
    run.loc[run["sample_index"] >= 2, "V1_response_avg"] = np.nan
    selection = sample_selection(run)
    assert selection.describe() == "8 samples, responses recorded for V1 2, V2 8"


def test_class_level_tables_have_no_selection():
    assert sample_selection(_run().drop(columns=["sample_index"])) is None


def test_run_columns_are_the_swept_parameters():
    df = pd.concat([_run(interval=i, dt=2.0, seed=7000) for i in (1, 2)])
    assert run_columns(df) == ["interval", "dt", "seed"]


def test_equal_selections_do_not_warn(caplog):
    df = pd.concat([_run(interval=i, dt=2.0) for i in (1, 2, 3)])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert not warn_on_sample_mismatch(df)
    assert not _warnings(caplog)


@pytest.mark.parametrize(
    "other",
    [
        dict(n_samples=16, n_recorded=2),  # the dt=2 case: other counts
        dict(n_recorded=4),  # same samples, fewer responses
        dict(first_image=100),  # same counts, other images
    ],
)
def test_differing_selections_warn_once(caplog, other):
    df = pd.concat(
        [_run(interval=1, dt=2.0), _run(interval=2, dt=2.0, **other)],
        ignore_index=True,
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert warn_on_sample_mismatch(df, context="test")
    (record,) = _warnings(caplog)
    assert "2 selections among 2 runs" in record.getMessage()
    assert "interval=2" in record.getMessage()


def test_same_counts_other_samples_are_told_apart(caplog):
    selections = {
        "a": sample_selection(_run()),
        "b": sample_selection(_run(first_image=100)),
    }
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert warn_if_selections_differ(selections)
    assert "[samples " in _warnings(caplog)[0].getMessage()


def test_plot_loader_warns_across_files(tmp_path, caplog):
    paths = []
    for seed, recorded in ((7000, 8), (7001, 2)):
        path = tmp_path / f"m:tsim=40+dt=2_{seed}" / "test_data.csv"
        path.parent.mkdir()
        _run(n_recorded=recorded, interval=1, seed=seed).to_csv(path, index=False)
        paths.append(path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        load_result_data(paths)
    assert len(_warnings(caplog)) == 1


def test_aggregation_warns_across_test_runs(tmp_path, caplog, monkeypatch):
    files, configs = [], []
    for i, recorded in enumerate((8, 2)):
        path = tmp_path / f"run{i}" / "test_data.csv"
        path.parent.mkdir()
        _run(n_recorded=recorded, interval=1).to_csv(path, index=False)
        files.append(path)
        configs.append(path.with_suffix(".yaml"))

    def _load(test_data_file, **kwargs):
        return pd.read_csv(test_data_file)

    monkeypatch.setattr(agg, "load_test_data_with_metadata", _load)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        agg.aggregate_test_data(files, configs, "interval", "", [])
    assert len(_warnings(caplog)) == 1
