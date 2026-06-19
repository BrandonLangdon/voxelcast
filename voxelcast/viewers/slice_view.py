"""2D slice / image viewer built on pyqtgraph.

Handles target/recon slices, plain 2D images, and sinograms. For 3D data
pyqtgraph's ImageView gives a built-in scrollbar to scrub through z (or angle,
for sinograms).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from voxelcast.model import Dataset


class SliceView(QtWidgets.QWidget):
    """Scrubbable 2D view. For volumes the slider walks z; for sinograms, angle
    is the in-plane axis and the slider walks z-layers (if present)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)

    def set_dataset(self, ds: Dataset) -> None:
        arr = np.asarray(ds.array)
        if ds.is_sinogram:
            # sinogram: show (angle, detector) per layer; scrub z if 3D
            disp = arr if arr.ndim == 2 else np.moveaxis(arr, 2, 0)
            self.image_view.setImage(
                np.ascontiguousarray(disp.astype(float)),
                autoLevels=True,
            )
            return

        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            self.image_view.setImage(squeezed.astype(float), autoLevels=True)
        else:
            # (nY, nX, nZ) -> (nZ, nY, nX) so the ImageView slider scrubs z.
            zfirst = np.moveaxis(squeezed.astype(float), 2, 0)
            self.image_view.setImage(np.ascontiguousarray(zfirst), autoLevels=True)
