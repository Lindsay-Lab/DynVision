"""Path readers match a key only as a whole key.

Identifiers join ``key=value`` tokens with ``+``, ``:``, ``/`` and ``_``. An
unanchored ``re.search(f"{key}=")`` also matches inside a longer key. With the
millisecond keys this is a units error, not just a parsing slip: ``stim`` read
from ``...+tstim=10+...`` returns 10 ms as a step count. The pre-existing pair
``shufflepattern``/``pattern`` had the same flaw.

Every reader is pinned here: ``string_utils`` (extract and replace), the
workflow's own ``replace_param_in_string`` in ``snake_utils.smk``, the W&B
column parser, and the category inference of the weight-distribution plots.
"""

import re
import textwrap
from pathlib import Path
from typing import Optional

import pytest

from dynvision.processing.process_wandb_data import extract_param_value
from dynvision.utils.string_utils import (
    extract_param_from_string,
    key_value_pattern,
    replace_param_in_string,
)
from dynvision.visualization.plot_weight_distributions import _infer_category_value

SNAKE_UTILS = Path(__file__).resolve().parents[2] / "dynvision/workflow/snake_utils.smk"


def _load_workflow_replace():
    """Execute the .smk definition of replace_param_in_string in isolation."""
    source = SNAKE_UTILS.read_text()
    start = source.index("def replace_param_in_string(")
    end = source.index("\ndef ", start + 1)
    namespace = {
        "re": re,
        "Optional": Optional,
        "key_value_pattern": key_value_pattern,
    }
    exec(textwrap.dedent(source[start:end]), namespace)
    return namespace["replace_param_in_string"]


workflow_replace = _load_workflow_replace()

LEGACY_AND_MS = "StimulusInterval:dsteps=40+tstim=10+stim=5+tinterval=8+interval=4"


class TestKeyValuePattern:
    @pytest.mark.parametrize(
        "text, key, matches",
        [
            ("tstim=10", "stim", False),
            ("+tstim=10", "stim", False),
            ("+stim=5", "stim", True),
            (":stim=5", "stim", True),
            ("stim=5", "stim", True),
            ("model_contrast=0.5", "contrast", True),
            ("shufflepattern=true", "pattern", False),
            ("activityloss=1.0", "loss", False),
            ("/dt=2+", "dt", True),
        ],
    )
    def test_whole_key_only(self, text, key, matches):
        assert bool(re.search(key_value_pattern(key), text)) is matches

    def test_snake_case_keys_treat_underscore_as_part_of_the_key(self):
        pattern = key_value_pattern("loss_weight", ":", key_chars="A-Za-z0-9_")
        assert not re.search(pattern, "activity_loss_weight: 0.05")
        assert re.search(pattern, "loss_weight: 0.05")

    def test_key_is_matched_literally(self):
        assert not re.search(key_value_pattern("a.b"), "axb=1")


class TestExtract:
    def test_legacy_key_is_not_read_from_its_ms_key(self):
        assert extract_param_from_string(LEGACY_AND_MS, "stim", int) == 5
        assert extract_param_from_string(LEGACY_AND_MS, "interval", int) == 4

    def test_ms_key_reads_its_own_value(self):
        assert extract_param_from_string(LEGACY_AND_MS, "tstim", int) == 10

    def test_missing_legacy_key_is_reported_not_substituted(self):
        with pytest.raises(ValueError, match="No stim value found"):
            extract_param_from_string("StimulusDuration:tseq=60+tstim=20", "stim", int)

    def test_shufflepattern_does_not_shadow_pattern(self):
        identifier = ":tsim=40+shufflepattern=true+pattern=1011"
        assert extract_param_from_string(identifier, "pattern") == 1011

    def test_underscore_separated_key_still_matches(self):
        assert extract_param_from_string("model_contrast=0.5", "contrast", float) == 0.5


class TestReplace:
    def test_replaces_only_the_named_key(self):
        assert (
            replace_param_in_string("tstim=5+stim=5", "stim", int, "*")
            == "tstim=5+stim=*"
        )

    def test_does_not_touch_the_same_text_elsewhere(self):
        """str.replace() rewrote every 'stim=5', including the one in 'tstim=5'."""
        assert (
            replace_param_in_string("stim=5+tstim=5", "stim", int, "*")
            == "stim=*+tstim=5"
        )

    def test_workflow_copy_behaves_the_same(self):
        assert workflow_replace("tstim=5+stim=5", "stim", int, "*") == "tstim=5+stim=*"
        assert (
            workflow_replace(":tsim=40+dt=2+tau=5", "dt", None, "*")
            == ":tsim=40+dt=*+tau=5"
        )

    def test_workflow_copy_reports_a_missing_key(self):
        with pytest.raises(ValueError, match="No stim value found"):
            workflow_replace("tstim=10", "stim", int, "*")


class TestOtherReaders:
    def test_wandb_equals_format(self):
        assert extract_param_value("rctype=full+trc=6", "rctype") == "full"
        assert extract_param_value("rctype=full", "type") is None

    def test_wandb_colon_format_uses_snake_case_keys(self):
        column = "activity_loss_weight: 0.05"
        assert extract_param_value(column, "activity_loss_weight") == "0.05"
        assert extract_param_value(column, "loss_weight") is None

    def test_weight_plots_infer_whole_keys(self):
        path = Path("/models/DyRCNNx8:tsim=40+tstim=10+dt=2_7000/trained.pt")
        assert _infer_category_value(path, "stim") is None
        assert _infer_category_value(path, "tstim") == "10"
