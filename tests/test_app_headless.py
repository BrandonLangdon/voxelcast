"""Headless GUI test for the main window logic.

Runs with the offscreen Qt platform and the 3D dock disabled (VTK's
QtInteractor segfaults under offscreen rendering). Verifies dataset routing /
dock-visibility logic without needing a display. Skipped if Qt can't init.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VOXELCAST_DISABLE_3D", "1")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from voxelcast.app import MainWindow  # noqa: E402
from voxelcast.model import Dataset  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_volume_shows_volume_dock(qapp):
    win = MainWindow()
    win.show()
    win.show_dataset(Dataset(np.zeros((16, 16, 16)), "recon", "vol"))
    qapp.processEvents()
    assert win.volume_dock.isVisible()        # 3D volume -> dock visible
    assert win.export_act.isEnabled()


def test_sinogram_hides_volume_dock(qapp):
    win = MainWindow()
    win.show()
    win.show_dataset(Dataset(np.zeros((16, 60, 4)), "sino", "sino"))
    qapp.processEvents()
    assert not win.volume_dock.isVisible()     # sinogram -> no 3D dock


def test_demo_loads(qapp):
    win = MainWindow()
    win.load_demo()
    qapp.processEvents()
    assert win._current is not None and win._current.is_volume
