"""Misplaced model directories (an rsync copied a directory into one).

``rsync -a src dst/models`` without a trailing slash duplicates a model
directory inside another one: ``models/<class>/<idA>/<idB>/...``. Such copies
are unreachable for Snakemake but are walked by anything that scans the tree.
The link helper must skip them instead of failing.

A reported failure came from one of them: the copy held a
relative symlink that no longer resolved at its new depth, the helper planned a
link to it, and ``os.stat`` on the new link raised ``FileNotFoundError``,
aborting ``--apply`` halfway through its manifest.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LINK = REPO / "scripts" / "link_legacy_temporal_paths.py"

OUTER = (
    "DyRCNNx8:tsteps=14+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4+activityloss=0.0002"
    "+pattern=1+rctype=full+rctarget=output+skip=true+feedback=false+tidle=0_7000"
)
INNER = (
    "DyRCNNx8:tsteps=14+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4+energyloss=0.0002"
    "+pattern=1+rctype=full+rctarget=output+skip=true+feedback=false_7000"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    return _load("link_legacy_temporal_paths", LINK)


def _tree(root: Path, with_normal_copy: bool) -> Path:
    nested = root / "models" / "DyRCNNx8" / OUTER / INNER
    (nested / "imagenette").mkdir(parents=True)
    (nested / "imagenette" / "trained-best.pt").write_text("checkpoint")
    # the relative link an `rsync -a` copies verbatim, broken at the new depth
    os.symlink("../trained.pt", nested / "imagenette" / "last.pt")
    (root / ".snakemake").mkdir()
    if with_normal_copy:
        normal = root / "models" / "DyRCNNx8" / INNER / "imagenette"
        normal.mkdir(parents=True)
        (normal / "trained-best.pt").write_text("checkpoint")
    return nested


def _plan(helper, root):
    legacy = helper.LegacySpecs.from_configs([helper.DEFAULT_LEGACY_CONFIG])
    return helper.plan(root, legacy, False)


def test_link_helper_skips_a_nested_model_directory(helper, tmp_path):
    _tree(tmp_path, with_normal_copy=True)
    outcome = _plan(helper, tmp_path)
    assert [str(item[0]) for item in outcome["misplaced"]] == [
        f"models/DyRCNNx8/{OUTER}/{INNER}"
    ]
    # Nothing under the misplaced copy is linked; the model's own directory is.
    assert not [
        m
        for m in outcome["link"]
        if str(m.old).startswith(f"models/DyRCNNx8/{OUTER}/{INNER}")
    ]
    assert [str(m.old) for m in outcome["link"]] == [
        f"models/DyRCNNx8/{INNER}/imagenette/trained-best.pt"
    ]
    assert outcome["broken"] == []


def test_link_helper_does_not_crash_on_the_reported_case(helper, tmp_path):
    """The old code raised FileNotFoundError from mirror_mtime here."""
    _tree(tmp_path, with_normal_copy=False)
    outcome = _plan(helper, tmp_path)
    assert helper.apply(tmp_path, outcome) == 0
    assert not (
        tmp_path
        / "models"
        / "DyRCNNx8"
        / f"{OUTER.replace('tsteps=14', 'tsim=28')}"
        / INNER
    ).exists()


def test_a_source_that_does_not_resolve_is_not_linked(helper, tmp_path):
    """A broken symlink outside a misplaced directory is reported, not linked."""
    model = tmp_path / "models" / "DyRCNNx8" / OUTER / "imagenette"
    model.mkdir(parents=True)
    (model / "trained-best.pt").write_text("checkpoint")
    os.symlink("../missing.pt", model / "last.pt")
    (tmp_path / ".snakemake").mkdir()
    outcome = _plan(helper, tmp_path)
    assert [m.old.name for m in outcome["broken"]] == ["last.pt"]
    assert [m.old.name for m in outcome["link"]] == ["trained-best.pt"]
    assert helper.apply(tmp_path, outcome) == 1
