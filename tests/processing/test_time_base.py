"""Results carry their time base: dt, idle_steps, time_ms and ms parameter columns.

The engine records the state at the end of each step after the idle period, so
the state of ``times_index`` k belongs to ``(k + 1) * dt`` with ``t = 0`` at the
end of the idle period. Processing reads ``dt`` from each run's resolved config
(never from a default) and converts legacy step parameters with that run's own
``dt``, so results of different ``dt`` and of either key generation share one
millisecond column.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dynvision.processing.aggregate_experiment_data import (
    aggregate_test_data,
    load_test_data_with_metadata,
)
from dynvision.processing.process_single_test import process_single_test
from dynvision.processing.process_test_data import (
    _extract_metadata,
    build_measure_config,
)
from dynvision.processing.time_base import (
    add_ms_columns,
    config_parameter,
    config_time_base,
    ensure_time_base,
)
from dynvision.utils.memory_efficient_loading import MemoryMonitor
from dynvision.utils.timing import step_end_ms

MODEL_DIR = "DyRCNNx8:tsteps=20+dt={dt}+tau=5+tff=0+trc=6_7000"

# Resolved config as TestingParams writes it since the ms fields exist.
NEW_CONFIG = {
    "model.dt": 2.0,
    "model.n_timesteps": 20,
    "model.t_simulation": 40.0,
    "model.idle_timesteps": 20,
    "model.t_idle": 40.0,
    "data.data_timesteps": 40,
    "data.t_sequence": 80.0,
    "data.t_stimulus": 10.0,
    "data.stimulus_duration": 5,
    "data.t_intro": 2.0,
    "data.intro_duration": 1,
    "data.t_interval": 6.0,
    "data.interval_duration": 3,
    "data.contrast": 0.5,
    "seed": 7000,
}

# Resolved config written before: loader durations as undeclared step extras.
LEGACY_CONFIG = {
    "model.dt": 2.0,
    "model.n_timesteps": 20,
    "model.idle_timesteps": 20,
    "data.data_timesteps": 40,
    "data.stim": 5,
    "data.intro": 1,
    "data.interval": 3,
    "seed": 7000,
}


def _write_config(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# DynVision resolved configuration\n\n")
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def _test_outputs(n_samples: int = 2, n_steps: int = 4) -> pd.DataFrame:
    sample, step = np.meshgrid(np.arange(n_samples), np.arange(n_steps), indexing="ij")
    return pd.DataFrame(
        {
            "sample_index": sample.ravel(),
            "times_index": step.ravel(),
            "label_index": np.where(step.ravel() > 0, 3, -1),
            "guess_index": 3,
            "image_index": sample.ravel(),
        }
    )


def _run(tmp_path: Path, dt: float, config: dict, test_id: str = "abcd1234"):
    """One run's Stage 1 file and config, laid out like reports/."""
    run_dir = (
        tmp_path / "reports" / "interval" / MODEL_DIR.format(dt=dt)
        / "imagenette:all_trained-best" / test_id
    )
    config_file = _write_config(
        run_dir / "test_outputs.csv.config.yaml", {**config, "model.dt": dt}
    )
    data_file = run_dir / "test_data.csv"
    _test_outputs().to_csv(data_file, index=False)
    return data_file, config_file


class TestStepEndMs:
    def test_state_of_step_k_belongs_to_the_end_of_the_step(self):
        assert step_end_ms(0, 2) == 2.0
        assert step_end_ms(9, 0.5) == 5.0

    def test_series_keeps_its_index_and_takes_one_dt_per_row(self):
        index = pd.Series([0, 1, 2], index=[10, 11, 12])
        result = step_end_ms(index, pd.Series([0.5, 1.0, 2.0], index=index.index))
        assert list(result.index) == [10, 11, 12]
        assert list(result) == [0.5, 2.0, 6.0]

    def test_no_binary_rounding_noise(self):
        assert step_end_ms(2, 0.1) == 0.3

    @pytest.mark.parametrize("dt", [0, -1, float("nan")])
    def test_rejects_an_invalid_dt(self, dt):
        with pytest.raises(ValueError, match="dt must be positive"):
            step_end_ms(1, dt)


class TestConfigReaders:
    def test_time_base_from_a_new_config(self):
        assert config_time_base(NEW_CONFIG) == (2.0, 20)

    def test_time_base_from_a_legacy_config(self):
        assert config_time_base(LEGACY_CONFIG) == (2.0, 20)

    def test_idle_steps_derived_from_t_idle_alone(self):
        assert config_time_base({"model.dt": 0.5, "model.t_idle": 40.0}) == (0.5, 80)

    def test_missing_dt_raises_instead_of_assuming_one(self):
        with pytest.raises(ValueError, match="model.dt"):
            config_time_base({"model.idle_timesteps": 20})

    @pytest.mark.parametrize(
        "key, expected",
        [
            ("tstim", 10.0),
            ("stim", 5),
            ("tinterval", 6.0),
            ("tisi", 6.0),
            ("interval", 3),
            ("tidle", 40.0),
            ("idle", 20),
            ("tseq", 80.0),
            ("contrast", 0.5),
            ("seed", 7000),
        ],
    )
    def test_parameter_by_any_name_in_a_new_config(self, key, expected):
        assert config_parameter(NEW_CONFIG, key) == expected

    @pytest.mark.parametrize(
        "key, expected",
        [("tstim", 10.0), ("stim", 5), ("tinterval", 6.0), ("tidle", 40.0), ("tseq", 80.0)],
    )
    def test_ms_keys_read_a_legacy_config_through_its_dt(self, key, expected):
        assert config_parameter(LEGACY_CONFIG, key) == expected

    def test_steps_from_an_ms_only_config(self):
        assert config_parameter({"model.dt": 0.5, "data.t_stimulus": 10.0}, "stim") == 20

    def test_absent_parameter_is_none(self):
        assert config_parameter(NEW_CONFIG, "ssnr") is None

    def test_keys_of_a_persisted_testing_config(self, tmp_path):
        """The readers name the keys persist_resolved_config actually writes."""
        from dynvision.params import TestingParams
        from tests.params.utils import create_temp_config, temporary_testing_paths

        config_path = create_temp_config({})
        with temporary_testing_paths() as paths:
            params = TestingParams.from_cli_and_config(
                config_path=str(config_path),
                override_kwargs={
                    **paths,
                    "verbose": False,
                    "dt": 0.5,
                    "tsim": 40,
                    "tidle": 40,
                    "tseq": 80,
                    "tintro": 2,
                    "tstim": 10,
                    "data_loader": "StimulusDuration",
                },
            )
            persisted = params.persist_resolved_config(tmp_path / "out.csv", "test")
        metadata = _extract_metadata(
            persisted, data_arg_key="tstim", category="", extra_parameters=["stim"]
        )
        assert metadata.parameter_value == "10"
        assert metadata.extra_values["stim"] == "20"
        assert (metadata.dt, metadata.idle_steps) == (0.5, 80)


class TestMsColumns:
    def test_each_row_converts_with_its_own_dt(self):
        df = pd.DataFrame({"interval": [3, 6, 3], "dt": [2.0, 1.0, 0.5]})
        add_ms_columns(df)
        assert list(df["tinterval"]) == [6.0, 6.0, 1.5]

    def test_renamed_category_column_gets_a_renamed_ms_column(self):
        df = pd.DataFrame({"idle": [10], "train_idle": [20], "dt": [2.0]})
        add_ms_columns(df)
        assert (df["tidle"].item(), df["train_tidle"].item()) == (20.0, 40.0)

    def test_existing_ms_values_are_kept_and_gaps_filled(self):
        df = pd.DataFrame({"stim": [5, 5], "tstim": [10.0, None], "dt": [2.0, 1.0]})
        add_ms_columns(df)
        assert list(df["tstim"]) == [10.0, 5.0]

    def test_engine_step_fields_are_not_parameters(self):
        df = pd.DataFrame({"idle_timesteps": [20], "dt": [2.0]})
        add_ms_columns(df)
        assert "tidle" not in df.columns

    def test_step_column_without_dt_raises(self):
        with pytest.raises(KeyError, match="dt"):
            add_ms_columns(pd.DataFrame({"stim": [5]}))


class TestEnsureTimeBase:
    """Already-processed tables: only times_index, sometimes a dt column."""

    def test_dt_column_of_a_dt_sweep(self):
        df = pd.DataFrame({"times_index": [0, 1, 0], "dt": [2, 2, 0.5]})
        result = ensure_time_base(df, "reports/x/DyRCNNx8:tsteps=20+dt=*+tau=5_7000")
        assert list(result["time_ms"]) == [2.0, 4.0, 0.5]

    def test_dt_parsed_from_the_model_identifier(self):
        df = pd.DataFrame({"times_index": [0, 1], "stim": [5, 5]})
        result = ensure_time_base(df, Path("reports/x") / MODEL_DIR.format(dt=0.5) / "t.csv")
        assert list(result["dt"]) == [0.5, 0.5]
        assert list(result["time_ms"]) == [0.5, 1.0]
        assert list(result["tstim"]) == [2.5, 2.5]

    def test_a_dt_key_inside_another_key_is_not_dt(self):
        df = pd.DataFrame({"times_index": [0]})
        with pytest.raises(ValueError, match="never assumed"):
            ensure_time_base(df, "reports/x/Model:adt=2+tau=5_7000/t.csv")

    def test_no_dt_anywhere_raises(self):
        df = pd.DataFrame({"times_index": [0]})
        with pytest.raises(ValueError, match="never assumed"):
            ensure_time_base(df, "reports/x/DyRCNNx8:tsteps=20+dt=*+tau=5_7000")

    def test_explicit_dt_and_t_idle(self):
        df = pd.DataFrame({"times_index": [0, 3]})
        result = ensure_time_base(df, dt=1.0, t_idle=40)
        assert list(result["time_ms"]) == [1.0, 4.0]
        assert list(result["idle_steps"]) == [40, 40]

    def test_idle_from_a_tidle_parameter_column(self):
        df = pd.DataFrame({"times_index": [0, 0], "tidle": [10, 40], "dt": [2.0, 2.0]})
        result = ensure_time_base(df)
        assert list(result["idle_steps"]) == [5, 20]

    def test_unknown_idle_is_left_out(self):
        result = ensure_time_base(pd.DataFrame({"times_index": [0]}), dt=2)
        assert "idle_steps" not in result.columns

    def test_input_is_not_modified(self):
        df = pd.DataFrame({"times_index": [0]})
        ensure_time_base(df, dt=2)
        assert list(df.columns) == ["times_index"]


class TestStage1:
    def test_single_test_carries_its_time_base(self, tmp_path):
        test_output = tmp_path / "test_outputs.csv"
        _test_outputs().to_csv(test_output, index=False)
        _write_config(Path(f"{test_output}.config.yaml"), {**NEW_CONFIG, "model.dt": 0.5})
        df = process_single_test(
            response_file=tmp_path / "missing_responses.pt",
            test_output_file=test_output,
            measure_config=build_measure_config([]),
            memory_monitor=MemoryMonitor(memory_limit_gb=8.0),
        )
        assert set(df["dt"]) == {0.5}
        assert set(df["idle_steps"]) == {20}
        expected = (df["times_index"] + 1) * 0.5
        assert np.allclose(df["time_ms"], expected)


class TestStage2:
    def test_legacy_and_new_runs_share_one_ms_column(self, tmp_path):
        """A dt sweep: legacy steps at dt=2, ms fields at dt=0.5, both 10 ms."""
        legacy = _run(tmp_path, 2, LEGACY_CONFIG)
        new = _run(
            tmp_path,
            0.5,
            {**NEW_CONFIG, "data.stimulus_duration": 20, "data.interval_duration": 12,
             "model.idle_timesteps": 80},
        )
        combined, loaded = aggregate_test_data(
            test_data_files=[legacy[0], new[0]],
            config_files=[legacy[1], new[1]],
            data_arg_key="tinterval",
            category="dt",
            extra_parameters=["tstim"],
        )
        assert len(loaded) == 2
        assert set(pd.to_numeric(combined["tinterval"])) == {6.0}
        assert set(pd.to_numeric(combined["tstim"])) == {10.0}
        by_dt = combined.groupby("dt")
        assert by_dt["idle_steps"].first().to_dict() == {0.5: 80, 2.0: 20}
        last = by_dt["time_ms"].max().to_dict()
        assert last == {0.5: 2.0, 2.0: 8.0}

    def test_legacy_parameter_key_gets_its_ms_column(self, tmp_path):
        data_file, config_file = _run(tmp_path, 0.5, NEW_CONFIG)
        df = load_test_data_with_metadata(
            data_file, config_file, data_arg_key="interval", category="dt",
            extra_parameters=[],
        )
        assert set(df["interval"]) == {"3"}
        assert set(df["tinterval"]) == {1.5}

    def test_the_run_dt_wins_over_the_trained_dt(self, tmp_path):
        """A test spec may override dt; the time base follows the run."""
        data_file, config_file = _run(tmp_path, 2, NEW_CONFIG)
        _write_config(config_file, {**NEW_CONFIG, "model.dt": 1.0})
        df = load_test_data_with_metadata(
            data_file, config_file, data_arg_key="tstim", category="dt",
            extra_parameters=[],
        )
        assert set(df["dt"]) == {1.0}

    def test_new_ms_parameter_value_reads_like_the_identifier(self, tmp_path):
        data_file, config_file = _run(tmp_path, 2, NEW_CONFIG)
        df = load_test_data_with_metadata(
            data_file, config_file, data_arg_key="tstim", category="",
            extra_parameters=[],
        )
        assert set(df["tstim"]) == {"10"}
