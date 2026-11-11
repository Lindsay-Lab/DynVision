"""Storage caps must not silently cut the evaluated test set short.

Two settings used to truncate what a test run covers without saying so:

* ``store_test_records`` keeps the first n record batches, and the records are
  what carry the predictions -- so every metric of such a run describes a
  prefix of the test set.
* ``early_test_stop`` ends the evaluation as soon as the buffers are full, so
  a cap on either buffer stopped the test.

Both are fixed by defaulting to ``-1`` (unlimited) and by naming the
truncation when a cap is configured. Unlimited per-unit responses stay
rejected: that combination cannot fit.
"""

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dynvision.params.model_params import ModelParams

CONFIGS = Path(__file__).resolve().parents[2] / "dynvision" / "configs"
WARNING = "dynvision.params.model_params"


def make_params(**overrides) -> ModelParams:
    return ModelParams(**{"seed": 0, "log_level": "INFO", **overrides})


def test_shipment_defaults_store_every_sample():
    defaults = yaml.safe_load((CONFIGS / "config_defaults.yaml").read_text())
    assert defaults["store_test_responses"] == -1
    assert defaults["store_test_records"] == -1
    assert defaults["response_resolution"] == "layer"


def test_unlimited_testing_storage_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        make_params(store_test_responses=-1, store_test_records=-1)
    assert not [r for r in caplog.records if r.name == "root"]


@pytest.mark.parametrize(
    "setting, message",
    [
        ("store_test_records", "Every metric of this run then describes a prefix"),
        ("store_test_responses", "rest on a prefix of the samples"),
    ],
)
def test_a_positive_cap_names_the_truncation(caplog, setting, message):
    with caplog.at_level(logging.WARNING):
        make_params(**{setting: 12})
    (record,) = [r for r in caplog.records if setting in r.getMessage()]
    assert "the first 12 batches" in record.getMessage()
    assert message in record.getMessage()


def test_unlimited_unit_responses_are_rejected():
    with pytest.raises(ValidationError, match="response_resolution='layer'"):
        make_params(store_test_responses=-1, response_resolution="unit")


def test_a_cap_with_unit_responses_is_allowed():
    params = make_params(store_test_responses=4, response_resolution="unit")
    assert params.store_test_responses == 4


class _Fillable:
    """The attributes ``_check_buffer_filled`` reads, without a Lightning model."""

    def __init__(self, max_responses, max_records, early_test_stop=True):
        from dynvision.base.storage import StorageBuffer

        self.storage = StorageBuffer(
            max_responses=max_responses,
            max_records=max_records,
            response_strategy="fixed",
            record_strategy="fixed",
        )
        self.early_test_stop = early_test_stop
        self._stop_testing_early = False


def test_a_full_response_buffer_stops_the_evaluation():
    from dynvision.base.storage import StorageBufferMixin

    model = _Fillable(max_responses=2, max_records=2)
    for _ in range(2):
        model.storage.responses.append({"V1": 0})
    assert StorageBufferMixin._check_buffer_filled(model) is False  # records not full
    for _ in range(2):
        model.storage.records.append({"guess_index": 0})
    assert StorageBufferMixin._check_buffer_filled(model) is True


def test_unlimited_buffers_never_stop_the_evaluation():
    """With -1 the records never fill, so a capped response buffer cannot end
    the run early and the metrics still cover the whole test set."""
    from dynvision.base.storage import StorageBufferMixin

    model = _Fillable(max_responses=1, max_records=-1)
    for _ in range(3):
        model.storage.responses.append({"V1": 0})
    assert StorageBufferMixin._check_buffer_filled(model) is False


def test_early_stop_off_never_stops():
    from dynvision.base.storage import StorageBufferMixin

    model = _Fillable(max_responses=1, max_records=1, early_test_stop=False)
    model.storage.responses.append({"V1": 0})
    model.storage.records.append({"guess_index": 0})
    assert StorageBufferMixin._check_buffer_filled(model) is False
