"""3D volume viewer built on pyvistaqt (embedded VTK).

Reuses the same VTK pipeline VAMToolbox already drives via vedo/pyvista, but
embedded in a Qt widget so it lives inside the VoxelCast window instead of a
separate blocking render loop.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets

from voxelcast.model import Dataset


class VolumeView(QtWidgets.QWidget):
    """Volume rendering with an opacity transfer function + reset/clip controls."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        controls = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset camera")
        self.reset_btn.clicked.connect(self.plotter.reset_camera)
        controls.addWidget(self.reset_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

    def set_dataset(self, ds: Dataset) -> None:
        self.plotter.clear()
        arr = np.ascontiguousarray(np.squeeze(ds.array).astype(float))
        if arr.ndim != 3:
            # Nothing to volume-render; leave the view empty (but repaint so a
            # previous volume is cleared from the screen).
            self.plotter.render()
            return
        grid = pv.wrap(arr)  # numpy -> pyvista ImageData
        self.plotter.add_volume(grid, cmap="viridis", opacity="sigmoid")
        self.plotter.reset_camera()
        # Explicitly repaint: an already-shown QtInteractor does NOT auto-render
        # after the scene changes, so without this a dataset loaded *after* the
        # first paint (e.g. a finished reconstruction) shows nothing until the
        # user interacts with the view.
        self.plotter.render()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        # QtInteractor must release its VTK render window explicitly.
        self.plotter.close()
        super().closeEvent(event)
