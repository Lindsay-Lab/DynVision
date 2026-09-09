"""Regression tests for the lazy `project_paths` singleton.

`dynvision.project_paths` used to construct `project_paths_class()` eagerly at
import time, shelling out to `hostname` and mutating `os.environ["WANDB_DIR"]`
as a side effect of the import statement. It's now a lazy proxy over
`dynvision.path_layout.PathLayout`, built on first attribute access.

See: docs/development/planning/project-paths-seam.md
Issue: https://github.com/Lindsay-Lab/DynVision/issues/14
"""

import os
import sys

import pytest


class _FakeHostnameStream:
    def __init__(self, hostname):
        self._hostname = hostname

    def read(self):
        return self._hostname


def _fresh_import_project_paths(monkeypatch):
    """Reimport dynvision.project_paths from scratch for test isolation."""

    monkeypatch.delitem(sys.modules, "dynvision.project_paths", raising=False)
    import dynvision.project_paths as project_paths_module

    return project_paths_module


class TestImportHasNoSideEffects:
    def test_import_does_not_shell_out(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise AssertionError("os.popen was called at import time")

        monkeypatch.setattr(os, "popen", _raise)

        _fresh_import_project_paths(monkeypatch)

    def test_import_does_not_mutate_wandb_dir(self, monkeypatch):
        monkeypatch.delenv("WANDB_DIR", raising=False)

        _fresh_import_project_paths(monkeypatch)

        assert "WANDB_DIR" not in os.environ


class TestLazyConstruction:
    def test_constructs_exactly_once_across_multiple_attribute_accesses(
        self, monkeypatch
    ):
        calls = []

        def _fake_popen(cmd):
            calls.append(cmd)
            return _FakeHostnameStream("my-laptop")

        monkeypatch.setattr(os, "popen", _fake_popen)

        module = _fresh_import_project_paths(monkeypatch)

        _ = module.project_paths.data.raw
        _ = module.project_paths.models
        _ = module.project_paths.figures

        assert len(calls) == 1


class TestBackwardCompatibleAttributes:
    @pytest.fixture(autouse=True)
    def _stub_hostname(self, monkeypatch):
        monkeypatch.setattr(
            os, "popen", lambda cmd: _FakeHostnameStream("my-laptop")
        )

    def test_exposes_all_previously_public_attributes(self, monkeypatch):
        module = _fresh_import_project_paths(monkeypatch)
        pp = module.project_paths

        for attr in (
            "data",
            "models",
            "notebooks",
            "references",
            "reports",
            "figures",
            "logs",
            "large_logs",
            "benchmarks",
            "scripts",
            "project_name",
        ):
            assert hasattr(pp, attr), f"project_paths.{attr} missing"

        assert hasattr(pp.data, "raw")
        assert hasattr(pp.data, "interim")
        assert hasattr(pp.data, "processed")
        assert hasattr(pp.data, "external")
        assert hasattr(pp.scripts, "configs")
        assert callable(pp.iam_on_cluster)
        assert pp.iam_on_cluster() is False
