"""The experiment configs are written in ms and equal the legacy ones at dt=2.

``config_experiments.yaml`` used to give every window in steps (``dsteps``,
``intro``, ``stim``, ``interval``, ``idle``), so a model at ``dt=0.5`` saw a
stimulus four times shorter than one at ``dt=2``. It now gives them in ms. The
existing results were all run at the default ``dt=2``, and for them nothing may
change: translating a legacy spec at ``dt=2`` must reproduce the new spec
character for character. That equality is what lets the link helper reuse
every ``dt=2`` result under its new name without re-running a test.

The frozen pre-migration copy in ``configs/legacy/`` is pinned by checksum,
because the helper recomputes hashed test identifiers from it.
"""

import hashlib
from itertools import product
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from dynvision.utils.timing import (
    format_ms,
    is_integer_multiple,
    is_legacy_step_key,
    is_ms_name,
    translate_legacy_args,
)

CONFIGS = Path(__file__).resolve().parents[2] / "dynvision" / "configs"
LEGACY_FILE = CONFIGS / "legacy" / "config_experiments_steps.yaml"
LEGACY_SHA256 = "faf259f8581c502afbb0d1023dbe5cc1c19cd28c49caa72a735c580fb000bf57"

#: dt of the default model_args, at which every existing result was produced.
RUN_DT = 2.0


def _experiments(path: Path) -> Dict[str, dict]:
    return yaml.safe_load(path.read_text())["experiment_config"]


LEGACY = _experiments(LEGACY_FILE)
CURRENT = _experiments(CONFIGS / "config_experiments.yaml")
WORKFLOW = yaml.safe_load((CONFIGS / "config_workflow.yaml").read_text())

TESTED = sorted(
    name for name, entry in CURRENT.items() if isinstance(entry, dict) and "data_loader" in entry
)


def args_product(args: Dict) -> List[str]:
    """The workflow's spec format (snake_utils.smk args_product)."""
    if not args:
        return [""]
    as_lists = {key: value if isinstance(value, list) else [value] for key, value in args.items()}
    return [
        ":" + "+".join(f"{key}={value}" for key, value in zip(as_lists, combination))
        for combination in product(*as_lists.values())
    ]


def translate_spec(spec: str, dt: float) -> str:
    """Rewrite one legacy spec string to ms keys, keeping order and format."""
    pairs = dict(token.split("=", 1) for token in spec.lstrip(":").split("+") if token)
    translated = translate_legacy_args(pairs, dt)
    return ":" + "+".join(
        f"{key}={format_ms(value) if is_ms_name(key) else value}"
        for key, value in translated.items()
    )


class TestNoStepKeysRemain:
    @pytest.mark.parametrize("name", TESTED)
    def test_data_args_are_in_ms(self, name):
        legacy_keys = [key for key in CURRENT[name]["data_args"] if is_legacy_step_key(key)]
        assert not legacy_keys

    @pytest.mark.parametrize("name", TESTED)
    def test_swept_parameter_is_not_a_step_key(self, name):
        assert not is_legacy_step_key(CURRENT[name]["parameter"])

    def test_categories_are_in_ms(self):
        assert not [key for key in CURRENT["categories"] if is_legacy_step_key(key)]

    def test_workflow_model_args_are_in_ms(self):
        assert not [key for key in WORKFLOW["model_args"] if is_legacy_step_key(key)]
        assert not [key for key in WORKFLOW["category"] if is_legacy_step_key(key)]

    def test_default_model_window_is_unchanged_at_its_dt(self):
        model_args = WORKFLOW["model_args"]
        assert model_args["tsim"] / model_args["dt"] == 20  # was tsteps: 20


class TestIdenticalAtRunDt:
    def test_same_experiments_and_loaders(self):
        assert sorted(LEGACY) == sorted(CURRENT)
        for name in TESTED:
            assert CURRENT[name]["data_loader"] == LEGACY[name]["data_loader"]

    @pytest.mark.parametrize("name", TESTED)
    def test_specs_translate_character_for_character(self, name):
        legacy_specs = args_product(LEGACY[name]["data_args"])
        current_specs = args_product(CURRENT[name]["data_args"])
        assert [translate_spec(spec, RUN_DT) for spec in legacy_specs] == current_specs

    @pytest.mark.parametrize("name", TESTED)
    def test_swept_parameter_translates(self, name):
        legacy_parameter = LEGACY[name]["parameter"]
        expected = translate_legacy_args({legacy_parameter: 0}, RUN_DT)
        assert CURRENT[name]["parameter"] == next(iter(expected))


class TestOnTheGrid:
    DTS = CURRENT["categories"]["dt"]

    def test_dt_sweep_values(self):
        assert self.DTS == [0.5, 1, 2]

    @pytest.mark.parametrize("name", TESTED)
    def test_every_window_is_whole_steps_at_every_swept_dt(self, name):
        for key, value in CURRENT[name]["data_args"].items():
            if not is_ms_name(key):
                continue
            for ms in value if isinstance(value, list) else [value]:
                for dt in self.DTS:
                    assert is_integer_multiple(ms, dt), f"{name}: {key}={ms} at dt={dt}"


def test_legacy_copy_is_frozen():
    """The link helper re-hashes legacy specs from this file; it must not drift."""
    assert hashlib.sha256(LEGACY_FILE.read_bytes()).hexdigest() == LEGACY_SHA256
