"""Golden test: on-grid configurations behave exactly as before the units fix.

Routing every ms -> step conversion through ``utils.timing.to_timesteps`` must
not change any model whose delays are whole numbers of ``dt`` steps -- which is
every model trained so far. This test pins, for each case, the integer delay
indices, the buffer lengths, the reaction mask and the forward output. The
indices, buffer lengths and the mask must match exactly. The forward output is
compared within a floating-point tolerance, because a different torch build
rounds the same computation differently; a difference that is not
bit-identical is reported as a warning.

Regenerate the golden file only for a deliberate behaviour change::

    python tests/models/test_dt_golden.py --write
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

from dynvision.models.cordsnet import CordsNet
from dynvision.models.cornet_rt import CorNetRT
from dynvision.models.dyrcnn import DyRCNNx4, DyRCNNx8

GOLDEN = Path(__file__).parent / "golden" / "dt_golden.pt"

#: Tolerance for the forward output. A changed delay moves the output by at
#: least 4e-6 in every recorded case (measured with a one-step shift of
#: t_recurrence), so these values catch a shift everywhere, including
#: dyrcnnx8_dt2, whose output is small (max 5e-3). The cost: a build that
#: evaluates the same arithmetic in another order can exceed them -- 4.7e-5 was
#: measured once between two CI runs of torch 2.15.0+cpu. Such a failure is
#: reported with both differences by the warning below, and the golden file is
#: regenerated for that platform only after checking the delay indices.
OUTPUT_RTOL = 1e-6
OUTPUT_ATOL = 1e-7

N_TIMESTEPS = 24  # long enough for a 12-step recurrence (6 ms at dt=0.5) to act
IMAGE_SIZE = 32

_COMMON = dict(
    n_timesteps=N_TIMESTEPS,
    n_classes=10,
    input_dims=(N_TIMESTEPS, 3, IMAGE_SIZE, IMAGE_SIZE),
    shuffle_presentation_pattern=False,
)

#: case id -> (model class, constructor kwargs). The DyRCNNx8 cases match the
#: trained models (tau=5, tff=0, trc=6, tsk=0, lossrt=4); the DyRCNNx4 cases
#: switch on every delay path, including the feedback connection.
CASES: Dict[str, Tuple[type, Dict[str, Any]]] = {
    **{
        f"dyrcnnx8_dt{dt:g}": (
            DyRCNNx8,
            dict(
                dt=dt,
                tau=5.0,
                t_feedforward=0.0,
                t_recurrence=6.0,
                t_skip=0.0,
                skip=True,
                feedback=False,
                loss_reaction_time=4.0,
                idle_timesteps=4,
            ),
        )
        for dt in (0.5, 1.0, 2.0)
    },
    **{
        f"dyrcnnx4_all_delays_dt{dt:g}": (
            DyRCNNx4,
            dict(
                dt=dt,
                tau=5.0,
                t_feedforward=2.0,
                t_recurrence=6.0,
                t_skip=4.0,
                skip=True,
                feedback="add",
                t_feedback=10.0,
                loss_reaction_time=2.0,
            ),
        )
        for dt in (1.0, 2.0)
    },
    "cornetrt_dt2": (CorNetRT, dict(dt=2.0)),
    "cordsnet_dt2": (CordsNet, dict(dt=2.0, init_with_pretrained=False)),
    "cordsnet_dt1_tsk2": (
        CordsNet,
        dict(
            dt=1.0,
            t_feedforward=2.0,
            t_recurrence=2.0,
            t_skip=2.0,
            init_with_pretrained=False,
        ),
    ),
}


#: Index values that changed on purpose since the golden file was recorded,
#: as (case id, index name) -> (recorded value, current value). The forward
#: outputs of these cases are unchanged; only bookkeeping differs.
DELIBERATE_CHANGES: Dict[Tuple[str, str], Tuple[Any, Any]] = {
    # CordsNet built layer0 without dt, so its buffer was sized with the
    # layer default dt=1 ms (2 states at dt=2). It now uses the model's dt.
    # layer0 has no recurrence and no feedforward delay, so it never reads the
    # buffer beyond its newest state.
    ("cordsnet_dt2", "layer0.n_hidden_states"): (2, 1),
    # A disabled feedback connection gets no buffer index: its t_feedback is
    # inert and may lie off the dt grid, so it is no longer converted.
    ("dyrcnnx8_dt0.5", "delay_index_feedback"): (60, None),
    ("dyrcnnx8_dt1", "delay_index_feedback"): (30, None),
    ("dyrcnnx8_dt2", "delay_index_feedback"): (15, None),
}


def run_case(case_id: str) -> Dict[str, Any]:
    """Build the case's model from a fixed seed and record its timing state."""
    model_class, kwargs = CASES[case_id]
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(0)
            model = model_class(**{**_COMMON, **kwargs})
            model.eval()
            generator = torch.Generator().manual_seed(1)
            x = torch.rand(
                2, N_TIMESTEPS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator
            )
            with torch.no_grad():
                output = model(x)
    finally:
        logging.disable(logging.NOTSET)

    indices: Dict[str, Any] = {
        "delay_feedforward": getattr(model, "delay_feedforward", None),
        "delay_index_skip": getattr(model, "delay_index_skip", None),
        "delay_index_feedback": getattr(model, "delay_index_feedback", None),
    }
    for name, module in model.named_modules():
        for attribute in ("delay_recurrence", "n_hidden_states", "delay_index"):
            if hasattr(module, attribute) and name:
                indices[f"{name}.{attribute}"] = getattr(module, attribute)

    pattern = model._get_presentation_pattern()
    return {
        "indices": indices,
        "reaction_mask": model._compute_reaction_mask(pattern).clone(),
        "output": output.detach().clone(),
    }


def _assert_output_matches(current, expected, case_id: str) -> None:
    """Compare a forward output with the golden one, and warn if not bit-identical.

    Raises:
        AssertionError: if the output differs by more than the tolerance, which
            is a changed computation rather than another build's rounding.
    """
    torch.testing.assert_close(current, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    if torch.equal(current, expected):
        return
    difference = (current - expected).abs()
    index = difference.argmax().item()
    warnings.warn(
        f"{case_id}: the forward output is not bit-identical to the golden file. "
        f"Greatest absolute difference {difference.flatten()[index].item():.3e}, "
        f"greatest relative {((difference / expected.abs().clamp(min=1e-30)).max()).item():.3e}; "
        f"the check allows rtol={OUTPUT_RTOL:g}, atol={OUTPUT_ATOL:g}. Another torch "
        "build rounds differently; a delay that moved a step would exceed the check.",
        stacklevel=2,
    )


def write_golden() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    torch.save({case_id: run_case(case_id) for case_id in CASES}, GOLDEN)


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    return torch.load(GOLDEN, weights_only=False)


def _expected_indices(golden: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    expected = dict(golden[case_id]["indices"])
    for (changed_case, name), (recorded, current) in DELIBERATE_CHANGES.items():
        if changed_case == case_id:
            assert expected[name] == recorded, (
                f"DELIBERATE_CHANGES records {name}={recorded!r} for {case_id}, "
                f"but the golden file holds {expected[name]!r}"
            )
            expected[name] = current
    return expected


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_delay_indices_are_unchanged(golden, case_id):
    assert run_case(case_id)["indices"] == _expected_indices(golden, case_id)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_reaction_mask_is_unchanged(golden, case_id):
    assert torch.equal(
        run_case(case_id)["reaction_mask"], golden[case_id]["reaction_mask"]
    )


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_forward_output_is_unchanged(golden, case_id):
    _assert_output_matches(
        run_case(case_id)["output"], golden[case_id]["output"], case_id
    )


def test_a_tiny_difference_passes_and_warns(golden):
    """Rounding noise of another build passes the check, and is reported."""
    expected = golden["dyrcnnx8_dt2"]["output"]
    with pytest.warns(UserWarning, match="not bit-identical"):
        _assert_output_matches(expected + 1e-8, expected, "dyrcnnx8_dt2")


def test_a_one_step_delay_shift_fails(golden):
    """The tolerance is below the output change a one-step delay shift makes."""
    expected = golden["dyrcnnx8_dt2"]["output"]
    with pytest.raises(AssertionError):
        _assert_output_matches(expected + 5e-6, expected, "dyrcnnx8_dt2")


def test_a_changed_output_fails(golden):
    """A difference beyond the tolerance is a changed computation, not noise."""
    expected = golden["dyrcnnx8_dt2"]["output"]
    with pytest.raises(AssertionError):
        _assert_output_matches(expected + 1e-2, expected, "dyrcnnx8_dt2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the golden file")
    if parser.parse_args().write:
        write_golden()
        print(f"wrote {GOLDEN}")
