"""Regression test for Lindsay-Lab/DynVision#16.

Importing ``dynvision.data.datamodule`` used to require a fully working
``ffcv`` install (module-scope ``import ffcv`` and
``from dynvision.data.ffcv_dataloader import get_ffcv_dataloader`` at the top
of the file), even when ``use_ffcv=False``. This broke every runtime entry
point (``runtime/train_model.py``, ``runtime/init_model.py``,
``runtime/test_model.py``) in any environment without ffcv installed.
"""

import builtins
import importlib
import sys


def _block_ffcv_imports(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ffcv" or name.startswith("ffcv."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for mod_name in list(sys.modules):
        if mod_name == "ffcv" or mod_name.startswith("ffcv."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    for mod_name in (
        "dynvision.data.batch_source",
        "dynvision.data.datamodule",
    ):
        monkeypatch.delitem(sys.modules, mod_name, raising=False)


def test_datamodule_imports_without_ffcv(monkeypatch):
    _block_ffcv_imports(monkeypatch)

    module = importlib.import_module("dynvision.data.datamodule")
    assert hasattr(module, "DataModule")
    assert hasattr(module, "SimpleDataModule")
    assert hasattr(module, "TestingDataModule")


def test_params_imports_without_ffcv(monkeypatch):
    """dynvision.params must not require ffcv at import time either."""
    _block_ffcv_imports(monkeypatch)

    for mod_name in list(sys.modules):
        if mod_name.startswith("dynvision.params"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    module = importlib.import_module("dynvision.params")
    assert module is not None
