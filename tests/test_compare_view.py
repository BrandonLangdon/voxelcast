"""Headless tests for the side-by-side comparison view."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets  # noqa: E402

from voxelcast.viewers.compare_view import CompareView  # noqa: E402
from voxelcast.model import Dataset  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_defaults_target_left_recon_right(qapp):
    v = CompareView()
    dsets = {
        "target": Dataset(np.zeros((8, 8, 5)), "target", "target"),
        "recon": Dataset(np.ones((8, 8, 5)), "recon", "recon"),
    }
    v.set_datasets(dsets)
    assert v.combos[0].currentText() == "target"
    assert v.combos[1].currentText() == "recon"
    assert v.slider.maximum() == 4  # nZ - 1


def test_shared_slider_clamps_per_side(qapp):
    v = CompareView()
    dsets = {
        "a": Dataset(np.zeros((8, 8, 10)), "recon", "a"),
        "b": Dataset(np.zeros((8, 8, 3)), "recon", "b"),
    }
    v.set_datasets(dsets)
    v.combos[0].setCurrentText("a")
    v.combos[1].setCurrentText("b")
    assert v.slider.maximum() == 9  # max depth of the two
    v.slider.setValue(9)            # side b (depth 3) must clamp internally, no error
    v._refresh()


def test_selection_preserved_on_new_dataset(qapp):
    v = CompareView()
    v.set_datasets({"target": Dataset(np.zeros((4, 4, 2)), "target", "target")})
    v.set_datasets({
        "target": Dataset(np.zeros((4, 4, 2)), "target", "target"),
        "recon": Dataset(np.zeros((4, 4, 2)), "recon", "recon"),
    })
    assert v.combos[0].currentText() == "target"
