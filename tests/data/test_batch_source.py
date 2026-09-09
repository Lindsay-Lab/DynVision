"""Tests for the BatchSource seam (dynvision.data.batch_source).

Regression coverage for Lindsay-Lab/DynVision#16: unguarded module-scope
``ffcv`` imports used to break ``import dynvision.params`` and
``import dynvision.data.datamodule`` in any environment without ffcv
installed. These tests simulate "ffcv not importable" via monkeypatching
``builtins.__import__`` so the assertions hold regardless of what is
actually installed in the environment running the suite.
"""

import builtins
import importlib
import sys

import pytest


def _block_ffcv_imports(monkeypatch):
    """Make every ``import ffcv...`` raise ModuleNotFoundError.

    Also purges any already-imported ``ffcv*`` / ``dynvision.data.batch_source``
    modules from ``sys.modules`` so the next import re-triggers the guarded
    (or unguarded) import path under test.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ffcv" or name.startswith("ffcv."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for mod_name in list(sys.modules):
        if mod_name == "ffcv" or mod_name.startswith("ffcv."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.delitem(sys.modules, "dynvision.data.batch_source", raising=False)


class TestBatchSourceImportWithoutFFCV:
    def test_module_imports_without_ffcv(self, monkeypatch):
        """dynvision.data.batch_source must import cleanly without ffcv."""
        _block_ffcv_imports(monkeypatch)

        module = importlib.import_module("dynvision.data.batch_source")
        assert hasattr(module, "BatchSource")
        assert hasattr(module, "TorchAdapter")
        assert hasattr(module, "FFCVAdapter")
        assert hasattr(module, "get_batch_source")

    def test_order_option_stand_in_available_without_ffcv(self, monkeypatch):
        _block_ffcv_imports(monkeypatch)

        module = importlib.import_module("dynvision.data.batch_source")
        assert module.OrderOption.RANDOM is not None
        assert module.OrderOption.SEQUENTIAL is not None
        assert module.OrderOption.QUASI_RANDOM is not None


class TestGetBatchSource:
    def test_use_ffcv_false_returns_torch_adapter(self):
        from dynvision.data.batch_source import TorchAdapter, get_batch_source

        adapter = get_batch_source(use_ffcv=False)
        assert isinstance(adapter, TorchAdapter)

    def test_torch_adapter_loader_class_resolves_without_ffcv(self, monkeypatch):
        """TorchAdapter must never touch ffcv, even if it's unavailable."""
        _block_ffcv_imports(monkeypatch)

        module = importlib.import_module("dynvision.data.batch_source")
        adapter = module.get_batch_source(use_ffcv=False)

        from dynvision.data.dataloader import get_data_loader

        assert adapter.loader_class is get_data_loader

    def test_use_ffcv_true_returns_ffcv_adapter_class(self):
        """FFCVAdapter selection succeeds even without instantiating it here."""
        from dynvision.data.batch_source import get_batch_source, FFCVAdapter

        try:
            adapter = get_batch_source(use_ffcv=True)
        except ImportError:
            pytest.skip("ffcv not installed in this environment")
        assert isinstance(adapter, FFCVAdapter)

    def test_ffcv_adapter_raises_clear_error_without_ffcv(self, monkeypatch):
        _block_ffcv_imports(monkeypatch)

        module = importlib.import_module("dynvision.data.batch_source")
        with pytest.raises(ImportError) as excinfo:
            module.get_batch_source(use_ffcv=True)

        message = str(excinfo.value)
        assert "use_ffcv" in message
        assert "ffcv" in message.lower()
