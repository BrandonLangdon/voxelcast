"""Headless tests for the dedicated sinogram / projection viewer."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets  # noqa: E402

from voxelcast.viewers.sinogram_view import SinogramView  # noqa: E402
from voxelcast.model import Dataset  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _sino():
    # (nR, nTheta, nZ) = (20, 90, 6)
    return Dataset(np.random.default_rng(0).random((20, 90, 6)), "sino", "s")


def test_per_angle_frames(qapp):
    v = SinogramView()
    v.set_dataset(_sino())
    assert v._per_angle()
    assert v._n_frames() == 90              # nTheta
    assert v._frame(3).shape == (20, 6)     # (nR, nZ) projector image
    v.slider.setValue(45)
    assert "180.0°" in v.label.text()       # 45/90 * 360


def test_per_z_mode(qapp):
    v = SinogramView()
    v.set_dataset(_sino())
    v.mode.setCurrentText("Sinogram (per z)")
    assert v._n_frames() == 6               # nZ
    assert v._frame(2).shape == (20, 90)    # (nR, nTheta) classic sinogram


def test_play_toggle_runs(qapp):
    v = SinogramView()
    v.set_dataset(_sino())
    v.play_btn.setChecked(True)
    assert v._timer.isActive()
    v.play_btn.setChecked(False)
    assert not v._timer.isActive()


def test_2d_sinogram(qapp):
    v = SinogramView()
    v.set_dataset(Dataset(np.zeros((20, 90)), "sino", "s2d"))
    assert v._n_frames() == 1
    assert v._frame(0).shape == (20, 90)
