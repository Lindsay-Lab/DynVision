"""Tests for the temporal registry and the millisecond helpers in utils.timing.

The registry is what lets a legacy step-valued key (``stim=5``) and its
millisecond counterpart (``tstim=10`` at ``dt=2``) name the same quantity. These
tests pin the claims the rest of the codebase relies on:

1. Every name maps to exactly one quantity, and a ``t`` prefix on a file-name
   key always means milliseconds.
2. Translating steps to milliseconds is exact and order-preserving.
3. The canonical text of a millisecond value equals ``str()`` of the number
   YAML parses from a config, so translated and generated identifiers match.
"""

import pytest
import yaml

from dynvision.utils.timing import (
    TEMPORAL_QUANTITIES,
    format_ms,
    is_legacy_step_key,
    is_ms_name,
    quantity_for,
    to_ms,
    to_timesteps,
    translate_legacy_args,
)


class TestRegistry:
    def test_every_name_is_unique(self):
        names = [
            name
            for quantity in TEMPORAL_QUANTITIES
            for name in (*quantity.ms_names, *quantity.step_names)
        ]
        assert len(names) == len(set(names))

    def test_file_name_keys_carry_no_underscore(self):
        """Identifier parsing splits on '_', so keys and aliases must not contain it."""
        for quantity in TEMPORAL_QUANTITIES:
            for name in (quantity.key, *quantity.aliases, *quantity.legacy_keys):
                assert "_" not in name, name

    def test_millisecond_keys_start_with_t(self):
        for quantity in TEMPORAL_QUANTITIES:
            for name in (quantity.key, *quantity.aliases):
                assert name.startswith("t"), name

    def test_no_legacy_key_is_written_with_step_in_a_new_key(self):
        """New keys must not contain 'step' (tsteps and dsteps are deprecated)."""
        for quantity in TEMPORAL_QUANTITIES:
            assert "step" not in quantity.key

    @pytest.mark.parametrize(
        "name, field",
        [
            ("tsim", "t_simulation"),
            ("tsteps", "t_simulation"),
            ("n_timesteps", "t_simulation"),
            ("tidle", "t_idle"),
            ("idle", "t_idle"),
            ("tseq", "t_sequence"),
            ("dsteps", "t_sequence"),
            ("tstim", "t_stimulus"),
            ("stim", "t_stimulus"),
            ("tintro", "t_intro"),
            ("intro", "t_intro"),
            ("tinterval", "t_interval"),
            ("tisi", "t_interval"),
            ("interval", "t_interval"),
        ],
    )
    def test_quantity_for_resolves_every_name(self, name, field):
        assert quantity_for(name).field == field

    def test_unknown_names_resolve_to_none(self):
        for name in ("tff", "trc", "dt", "tau", "contrast", "ssnr"):
            assert quantity_for(name) is None

    def test_unit_of_each_name(self):
        assert is_ms_name("tstim") and is_ms_name("t_stimulus") and is_ms_name("tisi")
        assert not is_ms_name("stim") and not is_ms_name("stimulus_duration")
        assert is_legacy_step_key("stim") and is_legacy_step_key("tsteps")
        assert not is_legacy_step_key("tstim")
        assert not is_legacy_step_key("stimulus_duration")


class TestToMs:
    def test_exact_values(self):
        assert to_ms(5, 2.0) == 10.0
        assert to_ms(5, 0.5) == 2.5
        assert to_ms(0, 2.0) == 0.0

    def test_strips_binary_rounding_noise(self):
        assert 3 * 0.1 != 0.3, "precondition: the product is inexact"
        assert to_ms(3, 0.1) == 0.3

    def test_round_trip_with_to_timesteps(self):
        for dt in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0):
            for steps in (0, 1, 3, 7, 20, 80):
                assert to_timesteps(to_ms(steps, dt), dt) == steps

    @pytest.mark.parametrize("steps", [-1, 2.5])
    def test_rejects_non_step_counts(self, steps):
        with pytest.raises(ValueError, match="non-negative integer"):
            to_ms(steps, 2.0)

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            to_ms(5, 0.0)


class TestFormatMs:
    @pytest.mark.parametrize(
        "value, text", [(10.0, "10"), (10, "10"), (2.5, "2.5"), (0.0, "0"), (0.3, "0.3")]
    )
    def test_canonical_text(self, value, text):
        assert format_ms(value) == text

    @pytest.mark.parametrize("literal", ["10", "2.5", "0.3", "40", "0.25"])
    def test_matches_str_of_the_yaml_number(self, literal):
        """args_product writes str(value) of the YAML-parsed config value."""
        parsed = yaml.safe_load(f"value: {literal}")["value"]
        assert format_ms(float(parsed)) == str(parsed)

    def test_formats_translated_values_like_the_config(self):
        assert format_ms(to_ms(3, 0.1)) == "0.3"
        assert format_ms(to_ms(20, 2.0)) == "40"


class TestTranslateLegacyArgs:
    def test_translates_values_and_keeps_order(self):
        args = {"dsteps": 40, "intro": 1, "stim": 5, "interval": 4, "idle": 20}
        translated = translate_legacy_args(args, dt=2.0)
        assert list(translated) == ["tseq", "tintro", "tstim", "tinterval", "tidle"]
        assert translated == {
            "tseq": 80.0,
            "tintro": 2.0,
            "tstim": 10.0,
            "tinterval": 8.0,
            "tidle": 40.0,
        }

    def test_the_same_steps_mean_less_time_at_smaller_dt(self):
        assert translate_legacy_args({"stim": 5}, dt=0.5) == {"tstim": 2.5}

    def test_other_keys_pass_through(self):
        args = {"dsteps": 20, "noisetype": "uniform", "ssnr": 0.2, "tff": 4}
        assert translate_legacy_args(args, dt=2.0) == {
            "tseq": 40.0,
            "noisetype": "uniform",
            "ssnr": 0.2,
            "tff": 4,
        }

    def test_accepts_string_values_from_identifiers(self):
        assert translate_legacy_args({"tsteps": "20"}, dt=2.0) == {"tsim": 40.0}

    def test_rejects_fractional_steps(self):
        with pytest.raises(ValueError, match="whole number of steps"):
            translate_legacy_args({"stim": "2.5"}, dt=2.0)

    def test_rejects_a_quantity_given_twice(self):
        with pytest.raises(ValueError, match="'tstim' appears twice"):
            translate_legacy_args({"stim": 5, "tstim": 10}, dt=2.0)
