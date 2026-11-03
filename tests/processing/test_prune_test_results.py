"""Pruning test results that rest on different samples than their experiment.

Within one experiment every test must store the same number of responses, per
layer and in total. The pruner finds the runs that do not, so they can be
rerun; it must not touch an experiment whose runs already agree, and must not
confuse stage-2 aggregates (one file per model directory) with stage-1 results
(one per test).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prune_test_results.py"


@pytest.fixture(scope="module")
def prune():
    # Registered in sys.modules: the module's dataclasses need it to resolve
    # their own module during creation.
    spec = importlib.util.spec_from_file_location("prune_test_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _test_data(path: Path, n_recorded: int, n_samples: int = 8, layers=("V1", "V2")):
    """One stage-1 result: responses stored for the first ``n_recorded`` samples."""
    rows = []
    for sample in range(n_samples):
        for k in range(3):
            row = {
                "sample_index": sample,
                "image_index": sample,
                "times_index": k,
                "accuracy": 1.0,
            }
            for layer in layers:
                row[f"{layer}_response_avg"] = (
                    float(k) if sample < n_recorded else np.nan
                )
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _tree(root: Path, experiment: str, model: str, tests):
    data = (
        root / "reports" / experiment / f"{model}_7000" / "imagenette:all_trained-best"
    )
    files = {
        name: _test_data(data / name / "test_data.csv", recorded)
        for name, recorded in tests.items()
    }
    return data, files


def test_finds_stage_one_results_only(prune, tmp_path):
    data, files = _tree(tmp_path, "interval", "m", {"a": 8, "b": 8})
    (data / "test_data.csv").write_text("sample_index\n0\n")  # stage-2 aggregate
    assert prune.find_test_files(tmp_path / "reports") == sorted(files.values())


def test_scope_selects_experiments_models_and_seeds(prune, tmp_path):
    _, a = _tree(tmp_path, "interval", "DyRCNNx8:tsteps=20", {"t": 8})
    _, b = _tree(tmp_path, "interval", "DyRCNNx8:tsteps=40", {"t": 8})
    _, c = _tree(tmp_path, "duration", "DyRCNNx8:tsteps=20", {"t": 8})
    other_seed = _test_data(
        tmp_path
        / "reports"
        / "interval"
        / "DyRCNNx8:tsteps=20_7001"
        / "imagenette:all_trained-best"
        / "t"
        / "test_data.csv",
        8,
    )
    reports = tmp_path / "reports"
    assert len(prune.find_test_files(reports)) == 4
    assert prune.find_test_files(reports, experiments=["interval"]) == sorted(
        [a["t"], b["t"], other_seed]
    )
    assert prune.find_test_files(
        reports, models=["*tsteps=20*"], seeds=["7000"]
    ) == sorted([a["t"], c["t"]])
    assert prune.find_test_files(reports, experiments=["dur*"], seeds=["7001"]) == []


def test_consistent_experiment_has_no_deviations(prune, tmp_path):
    _tree(tmp_path, "interval", "m", {"a": 8, "b": 8, "c": 8})
    files = prune.find_test_files(tmp_path / "reports")
    selections = {path: prune.read_selection(path) for path in files}
    targets, deviations = prune.plan(selections)
    assert deviations == []
    assert targets["interval"].describe() == "8 samples, responses recorded for 8"


def test_minority_selection_is_flagged(prune, tmp_path):
    _tree(tmp_path, "interval", "m", {"a": 8, "b": 8, "c": 2})
    files = prune.find_test_files(tmp_path / "reports")
    selections = {path: prune.read_selection(path) for path in files}
    _, deviations = prune.plan(selections)
    assert len(deviations) == 1
    assert deviations[0].test_dir.name == "c"


def test_layers_must_agree_within_a_run(prune, tmp_path):
    """V1 recorded for fewer samples than V2 is a deviation, not a partial run."""
    data = tmp_path / "reports" / "interval" / "m_7000" / "d" / "a"
    _test_data(data / "test_data.csv", 8)
    lenient = pd.read_csv(data / "test_data.csv")
    lenient.loc[lenient["sample_index"] >= 4, "V1_response_avg"] = np.nan
    lenient.to_csv(data / "test_data.csv", index=False)
    _test_data(data.parent / "b" / "test_data.csv", 8)
    files = prune.find_test_files(tmp_path / "reports")
    selections = {path: prune.read_selection(path) for path in files}
    _, deviations = prune.plan(selections)
    assert [d.test_dir.name for d in deviations] == ["a"]


def test_runs_without_responses_need_the_flag(prune, tmp_path):
    _tree(tmp_path, "noise", "m", {"a": 8, "b": 8, "c": 0})
    files = prune.find_test_files(tmp_path / "reports")
    selections = {path: prune.read_selection(path) for path in files}
    assert prune.plan(selections)[1] == []
    assert len(prune.plan(selections, include_empty=True)[1]) == 1


def test_apply_deletes_only_the_deviating_results(prune, tmp_path):
    data, files = _tree(tmp_path, "interval", "m", {"a": 8, "b": 8, "c": 2})
    aggregate = data / "test_data.csv"
    aggregate.write_text("sample_index\n0\n")
    selections = {path: prune.read_selection(path) for path in files.values()}
    targets, deviations = prune.plan(selections)
    removed, aggregates = prune.prune(deviations)
    assert (removed, aggregates) == (1, 1)
    assert not files["c"].exists() and files["a"].exists()
    assert not aggregate.exists()


def test_script_runs_from_outside_the_repository(tmp_path):
    """No PYTHONPATH and an unrelated cwd: the script imports dynvision itself."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert "is not a directory" in result.stderr or "No stage-1" in result.stdout
