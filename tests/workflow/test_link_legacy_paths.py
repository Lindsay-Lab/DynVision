"""The link helper exposes legacy-named results under their ms names.

``scripts/link_legacy_temporal_paths.py`` must make Snakemake treat existing
models and test results as up to date under the new names. The claims pinned:

1. Names translate exactly: model identifiers token by token, hashed test
   directories by recovering, translating and re-hashing their spec. At dt=2
   the new hash equals the one the migrated config requests.
2. Links are file-level, relative, inside real directories, and carry their
   target's mtime; existing files are never moved or modified.
3. Wildcard-dt names, unknown hashes, incomplete outputs and conflicts are
   skipped and reported; runs are idempotent; --undo removes exactly the links.
4. End to end, Snakemake plans nothing for the new names of finished outputs.
5. ``test_spec_hash`` equals the workflow's original hashing pipeline.
"""

import base64
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from dynvision.workflow.snake_utils import args_product, test_spec_hash

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "link_legacy_temporal_paths.py"
CONFIGS = REPO / "dynvision" / "configs"

_spec = importlib.util.spec_from_file_location("link_legacy_temporal_paths", SCRIPT)
helper = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = helper  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(helper)

LEGACY_SPEC = ":dsteps=40+intro=1+stim=5+interval=4+idle=20"
OLD_TIME = 1_700_000_000  # a fixed past mtime for the "existing" results


def _model(args: str) -> str:
    return f"DyRCNNx8:{args}+tau=5_7000"


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (OLD_TIME, OLD_TIME))
    return path


@pytest.fixture
def root(tmp_path):
    legacy_hash = test_spec_hash("StimulusInterval", LEGACY_SPEC)
    for dt in ("2", "0.5"):
        model = _model(f"tsteps=20+dt={dt}")
        _write(tmp_path / "models" / "DyRCNNx8" / model / "imagenette" / "trained-best.pt")
        test_dir = tmp_path / "reports" / "interval" / model / "imagenette:all_trained-best" / legacy_hash
        _write(test_dir / "test_outputs.csv", f"outputs dt={dt}")
        _write(test_dir / "test_data.csv", f"data dt={dt}")
    group = tmp_path / "reports" / "interval" / _model("tsteps=20+dt=2") / "imagenette:all_trained-best"
    _write(group / "test_data.csv", "aggregate")
    _write(group / "deadbee" / "test_outputs.csv", "unknown hash")
    sweep = tmp_path / "reports" / "interval" / _model("tsteps=20+dt=*") / "imagenette:all_trained-best"
    _write(sweep / "test_data.csv", "dt sweep aggregate")
    return tmp_path


def _plan(root, include_aggregates=False):
    legacy = helper.LegacySpecs.from_configs([helper.DEFAULT_LEGACY_CONFIG])
    return helper.plan(root, legacy, include_aggregates)


def _new_interval_hash(dt_key: str) -> str:
    """Hash the migrated config requests for interval=4 steps at dt=2 (8 ms)."""
    entry = yaml.safe_load((CONFIGS / "config_experiments.yaml").read_text())["experiment_config"]["interval"]
    spec = next(s for s in args_product(entry["data_args"]) if "tinterval=8+" in s)
    return test_spec_hash(entry["data_loader"], spec)


class TestNameTranslation:
    def test_model_identifier(self):
        new, dt = helper.translate_model_identifier(_model("tsteps=20+dt=0.5"))
        assert (new, dt) == ("DyRCNNx8:tsim=10+dt=0.5+tau=5_7000", "0.5")

    def test_identifier_without_legacy_keys_is_unchanged(self):
        name = "DyRCNNx8:tsim=40+dt=2+tau=5_7000"
        assert helper.translate_model_identifier(name)[0] == name

    def test_wildcard_dt_is_unmappable(self):
        with pytest.raises(helper.Unmappable, match="dt='\\*'"):
            helper.translate_model_identifier(_model("tsteps=20+dt=*"))

    def test_hashed_test_identifier_at_dt_2_matches_the_migrated_config(self):
        legacy = helper.LegacySpecs.from_configs([helper.DEFAULT_LEGACY_CONFIG])
        old = test_spec_hash("StimulusInterval", LEGACY_SPEC)
        assert helper.translate_test_identifier(old, "2", legacy) == _new_interval_hash("2")

    def test_hashed_test_identifier_keeps_the_steps_at_other_dt(self):
        legacy = helper.LegacySpecs.from_configs([helper.DEFAULT_LEGACY_CONFIG])
        old = test_spec_hash("StimulusInterval", LEGACY_SPEC)
        expected = test_spec_hash(
            "StimulusInterval", ":tseq=20+tintro=0.5+tstim=2.5+tinterval=2+tidle=10"
        )
        assert helper.translate_test_identifier(old, "0.5", legacy) == expected

    def test_full_spec_test_identifier(self):
        legacy = helper.LegacySpecs()
        name = "StimulusDuration:dsteps=40+stim=20+tff=10"
        assert (
            helper.translate_test_identifier(name, "2", legacy)
            == "StimulusDuration:tseq=80+tstim=40+tff=10"
        )

    def test_spec_dt_overrides_model_dt(self):
        assert helper.translate_args("dsteps=40+dt=1", "2") == "tseq=40+dt=1"


class TestPlanAndApply:
    def test_dry_run_creates_nothing(self, root):
        before = sorted(p for p in root.rglob("*"))
        outcome = _plan(root)
        assert len(outcome["link"]) == 6  # 2 models + 2 x 2 per-test files
        assert sorted(p for p in root.rglob("*")) == before

    def test_skips_and_reports(self, root):
        outcome = _plan(root)
        assert len(outcome["aggregate"]) == 1
        reasons = sorted(reason for _, reason in outcome["unmappable"])
        assert any("deadbee" in reason for reason in reasons)
        assert any("dt='*'" in reason for reason in reasons)

    def test_links_are_relative_file_links_with_target_mtimes(self, root):
        outcome = _plan(root)
        assert helper.apply(root, outcome) == 6
        for mapping in outcome["link"]:
            link = root / mapping.new
            assert link.is_symlink() and not os.path.isabs(os.readlink(link))
            assert not link.parent.is_symlink()
            assert os.lstat(link).st_mtime == os.stat(link).st_mtime == OLD_TIME
            assert link.read_text() == (root / mapping.old).read_text()

    def test_dt_2_result_lands_where_the_migrated_config_looks(self, root):
        helper.apply(root, _plan(root))
        expected = (
            root / "reports" / "interval" / "DyRCNNx8:tsim=40+dt=2+tau=5_7000"
            / "imagenette:all_trained-best" / _new_interval_hash("2") / "test_outputs.csv"
        )
        assert expected.read_text() == "outputs dt=2"

    def test_originals_are_untouched(self, root):
        originals = {p: (p.read_text(), p.stat().st_ino) for p in root.rglob("*") if p.is_file()}
        helper.apply(root, _plan(root))
        for path, (text, inode) in originals.items():
            assert path.read_text() == text and path.stat().st_ino == inode

    def test_rerun_is_idempotent(self, root):
        helper.apply(root, _plan(root))
        outcome = _plan(root)
        assert len(outcome["link"]) == 0 and len(outcome["linked"]) == 6

    def test_existing_file_at_the_new_path_is_a_conflict(self, root):
        mapping = _plan(root)["link"][0]
        _write(root / mapping.new, "something else")
        outcome = _plan(root)
        assert mapping.new in {m.new for m in outcome["conflict"]}
        helper.apply(root, outcome)
        assert (root / mapping.new).read_text() == "something else"

    def test_include_aggregates(self, root):
        outcome = _plan(root, include_aggregates=True)
        assert len(outcome["link"]) == 7 and not outcome["aggregate"]

    def test_incomplete_outputs_are_skipped(self, root):
        mapping = _plan(root)["link"][0]
        marker = base64.urlsafe_b64encode(str(root / mapping.old).encode()).decode()
        _write(root / ".snakemake" / "incomplete" / marker, "")
        outcome = _plan(root)
        assert mapping.old in {m.old for m in outcome["incomplete"]}

    def test_undo_removes_exactly_the_links(self, root):
        before = sorted(str(p) for p in root.rglob("*"))
        helper.apply(root, _plan(root))
        removed, kept = helper.undo(root)
        assert (removed, kept) == (6, 0)
        assert sorted(str(p) for p in root.rglob("*")) == before


@pytest.mark.skipif(shutil.which("snakemake") is None and not Path(sys.executable).with_name("snakemake").exists(), reason="snakemake not installed")
def test_snakemake_plans_nothing_for_linked_results(tmp_path):
    """Run a toy model/test/process chain under legacy names, link, dry-run new names."""
    snakemake = str(Path(sys.executable).with_name("snakemake"))
    model = "models/DyRCNNx8/DyRCNNx8{args}_{seed}/imagenette/trained-best.pt"
    test_dir = "reports/interval/DyRCNNx8{args}_{seed}/imagenette:all_trained-best/{tid}"
    (tmp_path / "Snakefile").write_text(textwrap.dedent(f'''
        wildcard_constraints:
            args=r":[^/_]+", seed=r"[0-9]+", tid=r"[0-9a-f]+"
        rule train:
            output: "{model}"
            params: p=lambda w: w.args
            shell: "echo {{params.p}} > {{output}}"
        rule test:
            input: "{model}"
            output: "{test_dir}/test_outputs.csv"
            params: p=lambda w: w.args
            shell: "cat {{input}} > {{output}}; echo {{params.p}} >> {{output}}"
        rule process:
            input: "{test_dir}/test_outputs.csv"
            output: "{test_dir}/test_data.csv"
            shell: "cat {{input}} > {{output}}"
    '''))
    old_hash = test_spec_hash("StimulusInterval", LEGACY_SPEC)
    old_target = f"reports/interval/DyRCNNx8:tsteps=20+dt=2_7000/imagenette:all_trained-best/{old_hash}/test_data.csv"
    new_target = (
        f"reports/interval/DyRCNNx8:tsim=40+dt=2_7000/imagenette:all_trained-best/"
        f"{_new_interval_hash('2')}/test_data.csv"
    )
    run = lambda *args: subprocess.run(
        [snakemake, "-c1", *args], cwd=tmp_path, capture_output=True, text=True
    )
    assert run(old_target).returncode == 0
    os.utime(tmp_path, None)

    before = run("-n", new_target).stdout
    assert re.search(r"rule (train|test|process)", before), "control: new names need a run"

    # Outputs before inputs: with plain links (creation-time mtimes) this order
    # makes every input look updated, so only mirrored mtimes pass.
    outcome = _plan(tmp_path)
    outcome["link"].sort(key=lambda mapping: str(mapping.new), reverse=True)
    helper.apply(tmp_path, outcome)
    after = run("-n", new_target)
    assert "Nothing to be done" in after.stdout + after.stderr, after.stdout + after.stderr


def test_spec_hash_equals_the_workflow_pipeline():
    """The original .smk pipeline: compute_hash(loader, args).lstrip(':hash=')."""
    source = (REPO / "dynvision" / "workflow" / "snake_utils.smk").read_text()
    start = source.index("def compute_hash(")
    end = source.index("\ndef ", start + 1)
    namespace = {}
    exec(textwrap.dedent(source[start:end]), namespace)
    compute_hash = namespace["compute_hash"]

    specs = []
    for path in (CONFIGS / "config_experiments.yaml", CONFIGS / "legacy" / "config_experiments_steps.yaml"):
        experiments = yaml.safe_load(path.read_text())["experiment_config"]
        for entry in experiments.values():
            if isinstance(entry, dict) and "data_loader" in entry:
                specs += [(entry["data_loader"], s) for s in args_product(entry["data_args"])]
    assert len(specs) > 200
    for loader, spec in specs:
        assert test_spec_hash(loader, spec) == compute_hash(loader, spec).lstrip(":hash=")


def test_lstrip_quirk_is_kept():
    """A hash starting with 'a' loses it; existing directory names depend on that."""
    loader, spec = next(
        (loader, spec)
        for loader, spec in (("StimulusDuration", f":tstim={n}") for n in range(1000))
        if __import__("hashlib").md5(f"{loader}_{spec.lstrip(':')}".encode()).hexdigest().startswith("a")
    )
    assert len(test_spec_hash(loader, spec)) < 8
