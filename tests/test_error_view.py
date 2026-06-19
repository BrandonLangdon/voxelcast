"""Headless tests for the convergence/error plot widget."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets  # noqa: E402

from voxelcast.viewers.error_view import ErrorView  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_add_points_accumulate(qapp):
    v = ErrorView()
    v.reset()
    for i, loss in enumerate([1.0, 0.5, 0.25, 0.1], start=1):
        v.add_point(i, loss)
    assert v._xs == [1, 2, 3, 4]
    assert v._ys[-1] == 0.1
    v.set_complete()
    assert "final loss" in v.status.text()


def test_reset_clears(qapp):
    v = ErrorView()
    v.add_point(1, 0.9)
    v.reset()
    assert v._xs == [] and v._ys == []


def test_log_guard_on_nonpositive(qapp):
    v = ErrorView()
    v.reset()
    v.add_point(1, 0.0)        # non-positive -> log y must refuse
    v.log_y.setChecked(True)
    assert v.log_y.isChecked() is False
