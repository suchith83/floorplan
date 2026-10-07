"""Unit tests run on the provisional interval model (D5), whatever fp/calibration.json the last benchmark wrote."""
import pytest

from fp import calibration as cal


@pytest.fixture(autouse=True)
def _provisional_intervals(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(cal, "PATH", tmp_path_factory.getbasetemp() / "no_calibration.json")
    cal.reset_cache()
    yield
    cal.reset_cache()
