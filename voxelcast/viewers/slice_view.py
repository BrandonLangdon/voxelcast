"""2D slice / image viewer built on pyqtgraph.

Handles target/recon slices, plain 2D images, and sinograms. For 3D data
pyqtgraph's ImageView gives a built-in scrollbar to scrub through z (or angle,
for sinograms).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from voxelcast.model import Dataset


class SliceView(QtWidgets.QWidget):
    """Scrubbable 2D view. For volumes the slider walks z; for sinograms, angle
    is the in-plane axis and the slider walks z-layers (if present).

    Emits `slice_changed(index, total)` whenever the scrubbed slice changes, so
    the 3D view can show where the current cross-section sits in the volume.
    """

    slice_changed = QtCore.Signal(int, int)  # (index, total)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.info = QtWidgets.QLabel("—")
        self.info.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self.info)

        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)

        self._total = 1
        self._axis = "z"
        self.image_view.sigTimeChanged.connect(self._on_time_changed)

    def set_dataset(self, ds: Dataset) -> None:
        arr = np.asarray(ds.array)
        if ds.is_sinogram:
            # sinogram: show (angle, detector) per layer; scrub z if 3D
            disp = arr if arr.ndim == 2 else np.moveaxis(arr, 2, 0)
            disp = np.ascontiguousarray(disp.astype(float))
            self.image_view.setImage(disp, autoLevels=True)
            self._axis = "z layer"
            if disp.ndim == 3:
                self._total = disp.shape[0]
                self._jump_to_busiest(disp)
            else:
                self._total = 1
                self._update_info(0)
            return

        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            self.image_view.setImage(squeezed.astype(float), autoLevels=True)
            self._total = 1
            self._axis = "z"
            self._update_info(0)
        else:
            # (nY, nX, nZ) -> (nZ, nY, nX) so the ImageView slider scrubs z.
            zfirst = np.ascontiguousarray(np.moveaxis(squeezed.astype(float), 2, 0))
            self.image_view.setImage(zfirst, autoLevels=True)
            self._total = zfirst.shape[0]
            self._axis = "z"
            self._jump_to_busiest(zfirst)

    def _jump_to_busiest(self, stack: np.ndarray) -> None:
        """Start on the slice with the most signal, not slice 0 (which is often
        an empty edge of the object, e.g. a blade tip -> a black-looking view)."""
        sums = stack.reshape(stack.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(sums)) if sums.size else 0
        self.image_view.setCurrentIndex(idx)
        self._update_info(idx)  # setCurrentIndex may not re-fire sigTimeChanged

    def _on_time_changed(self, index: int, _time: float) -> None:
        self._update_info(int(index))

    def _update_info(self, index: int) -> None:
        if self._total > 1:
            self.info.setText(f"{self._axis} slice {index} / {self._total - 1}")
        else:
            self.info.setText("single 2D slice")
        self.slice_changed.emit(int(index), int(self._total))
