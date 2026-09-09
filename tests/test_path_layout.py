"""Tests for the project_paths seam (dynvision.path_layout).

See: docs/development/planning/project-paths-seam.md
Issue: https://github.com/Lindsay-Lab/DynVision/issues/14
"""

import os
import sys

import pytest


def _forbid_popen(monkeypatch):
    """Fail the test loudly if anything shells out to `hostname`."""

    def _raise(*args, **kwargs):
        raise AssertionError("os.popen was called; environment detection must be explicit")

    monkeypatch.setattr(os, "popen", _raise)


class TestImportHasNoSideEffects:
    def test_import_does_not_shell_out(self, monkeypatch):
        # Force a fresh import so any import-time os.popen call is caught.
        monkeypatch.delitem(sys.modules, "dynvision.path_layout", raising=False)
        _forbid_popen(monkeypatch)

        import dynvision.path_layout  # noqa: F401

    def test_import_does_not_mutate_wandb_dir(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "dynvision.path_layout", raising=False)
        monkeypatch.delenv("WANDB_DIR", raising=False)

        import dynvision.path_layout  # noqa: F401

        assert "WANDB_DIR" not in os.environ


class TestEnvironment:
    def test_local_is_not_cluster(self):
        from dynvision.path_layout import Environment

        env = Environment.local()
        assert env.is_cluster is False

    def test_cluster_is_cluster(self):
        from dynvision.path_layout import Environment

        env = Environment.cluster()
        assert env.is_cluster is True

    @pytest.mark.parametrize(
        "hostname",
        [
            "hpc-login01",
            "log-in-3",
            "greene.hpc.nyu.edu",
            "slurm-node-7",
            "compute-0-1",
            "node042",
            "cluster-master",
        ],
    )
    def test_detect_environment_classifies_known_cluster_hostnames(
        self, monkeypatch, hostname
    ):
        from dynvision import path_layout

        monkeypatch.setattr(os, "popen", lambda cmd: _FakeHostnameStream(hostname))

        env = path_layout.detect_environment()

        assert env.is_cluster is True
        assert env.hostname == hostname

    def test_detect_environment_classifies_local_hostname(self, monkeypatch):
        from dynvision import path_layout

        monkeypatch.setattr(
            os, "popen", lambda cmd: _FakeHostnameStream("my-laptop")
        )

        env = path_layout.detect_environment()

        assert env.is_cluster is False

    def test_detect_environment_calls_popen_exactly_once(self, monkeypatch):
        from dynvision import path_layout

        calls = []

        def _fake_popen(cmd):
            calls.append(cmd)
            return _FakeHostnameStream("my-laptop")

        monkeypatch.setattr(os, "popen", _fake_popen)

        path_layout.detect_environment()

        assert len(calls) == 1


class _FakeHostnameStream:
    def __init__(self, hostname):
        self._hostname = hostname

    def read(self):
        return self._hostname


class TestPathLayoutForEnvironment:
    def test_local_layout_resolves_under_working_dir(self, tmp_path):
        from dynvision.path_layout import Environment, PathLayout

        layout = PathLayout.for_environment(
            Environment.local(), working_dir=tmp_path, toolbox_dir=tmp_path
        )

        assert layout.data.raw == tmp_path / "data" / "raw"
        assert layout.data.interim == tmp_path / "data" / "interim"
        assert layout.data.processed == tmp_path / "data" / "processed"
        assert layout.data.external == tmp_path / "data" / "external"
        assert layout.models == tmp_path / "models"
        assert layout.notebooks == tmp_path / "notebooks"
        assert layout.references == tmp_path / "references"
        assert layout.reports == tmp_path / "reports"
        assert layout.figures == tmp_path / "figures"
        assert layout.logs == tmp_path / "logs"
        assert layout.large_logs == tmp_path / "logs"
        assert layout.benchmarks == tmp_path / "logs" / "benchmarks"
        assert layout.scripts.configs == tmp_path / "configs"

    def test_local_layout_no_filesystem_or_hostname_dependency(self, monkeypatch, tmp_path):
        from dynvision.path_layout import Environment, PathLayout

        _forbid_popen(monkeypatch)

        # Should not touch the real filesystem beyond Path arithmetic.
        layout = PathLayout.for_environment(
            Environment.local(), working_dir=tmp_path, toolbox_dir=tmp_path
        )

        assert layout.iam_on_cluster() is False

    def test_cluster_layout_overrides_large_dirs_to_scratch(self, tmp_path):
        from dynvision.path_layout import Environment, PathLayout

        layout = PathLayout.for_environment(
            Environment.cluster(), working_dir=tmp_path, toolbox_dir=tmp_path
        )

        assert str(layout.data.raw).startswith("/scratch")
        assert str(layout.data.interim).startswith("/scratch")
        assert str(layout.data.processed).startswith("/scratch")
        assert str(layout.data.external).startswith("/scratch")
        assert str(layout.models).startswith("/scratch")
        assert str(layout.reports).startswith("/scratch")
        assert str(layout.large_logs).startswith("/scratch")
        assert layout.iam_on_cluster() is True

    def test_cluster_layout_keeps_non_scratch_dirs_under_working_dir(self, tmp_path):
        from dynvision.path_layout import Environment, PathLayout

        layout = PathLayout.for_environment(
            Environment.cluster(), working_dir=tmp_path, toolbox_dir=tmp_path
        )

        assert layout.notebooks == tmp_path / "notebooks"
        assert layout.figures == tmp_path / "figures"


class TestSetWandbDir:
    def test_sets_wandb_dir_from_layout(self, monkeypatch, tmp_path):
        from dynvision.path_layout import Environment, PathLayout, set_wandb_dir

        monkeypatch.delenv("WANDB_DIR", raising=False)
        layout = PathLayout.for_environment(
            Environment.local(), working_dir=tmp_path, toolbox_dir=tmp_path
        )

        set_wandb_dir(layout)

        assert os.environ["WANDB_DIR"] == str(layout.large_logs.resolve())

    def test_does_not_shell_out(self, monkeypatch, tmp_path):
        from dynvision.path_layout import Environment, PathLayout, set_wandb_dir

        layout = PathLayout.for_environment(
            Environment.local(), working_dir=tmp_path, toolbox_dir=tmp_path
        )
        _forbid_popen(monkeypatch)

        set_wandb_dir(layout)
