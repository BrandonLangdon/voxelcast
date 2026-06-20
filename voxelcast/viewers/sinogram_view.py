"""Dedicated sinogram / projection viewer with an angle scrubber.

A VAM sinogram has shape (nR, nTheta, nZ): detector position x projection angle
x height. Two ways to look at it:

* **Projection (per angle)** — for a fixed angle theta, the image sino[:, theta, :]
  (nR x nZ) is exactly the pattern the DLP projector displays at that rotation.
  Scrubbing/playing the angle slider previews the whole projection sequence.
* **Sinogram (per z)** — for a fixed height z, sino[:, :, z] (nR x nTheta) is the
  classic sinogram for that slice.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from voxelcast.model import Dataset


class SinogramView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._arr: np.ndarray | None = None
        self._levels = (0.0, 1.0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(6, 2, 6, 2)
        controls.addWidget(QtWidgets.QLabel("View:"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Projection (per angle)", "Sinogram (per z)"])
        self.mode.currentTextChanged.connect(self._on_mode_changed)
        controls.addWidget(self.mode)

        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggled)
        controls.addWidget(self.play_btn)

        controls.addStretch(1)
        self.label = QtWidgets.QLabel("—")
        controls.addWidget(self.label)
        layout.addLayout(controls)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        layout.addWidget(self.glw)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(60)  # ~16 fps animation
        self._timer.timeout.connect(self._advance)

    # ----- public API ------------------------------------------------------
    def set_dataset(self, ds: Dataset) -> None:
        arr = np.ascontiguousarray(np.asarray(ds.array).astype(float))
        self._arr = arr
        if arr.size:
            self._levels = (float(arr.min()), float(arr.max()) or 1.0)
        self._is_3d = arr.ndim == 3
        self.mode.setEnabled(self._is_3d)
        self._configure_slider()
        self._show(self.slider.value())

    # ----- mode / slider ---------------------------------------------------
    def _n_frames(self) -> int:
        if self._arr is None:
            return 0
        if self._arr.ndim == 2:
            return 1
        # 3D: per-angle -> nTheta (axis 1); per-z -> nZ (axis 2)
        return self._arr.shape[1] if self._per_angle() else self._arr.shape[2]

    def _per_angle(self) -> bool:
        return self.mode.currentText().startswith("Projection")

    def _configure_slider(self) -> None:
        n = self._n_frames()
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, n - 1))
        self.slider.setValue(min(self.slider.value(), max(0, n - 1)))
        self.slider.blockSignals(False)
        self.slider.setEnabled(n > 1)

    def _frame(self, idx: int) -> np.ndarray:
        a = self._arr
        if a is None:
            return np.zeros((1, 1))
        if a.ndim == 2:
            return a
        if self._per_angle():
            return a[:, idx, :]      # (nR, nZ) projector image at this angle
        return a[:, :, idx]          # (nR, nTheta) classic sinogram at this z

    def _show(self, idx: int) -> None:
        if self._arr is None:
            return
        frame = self._frame(idx)
        # Per-angle frames are (nR, nZ); display transposed so the height/rotation
        # axis (Z) is vertical -- i.e. the blade stands upright, matching how the
        # DLP projector would display the pattern.
        disp = frame.T if (self._arr.ndim == 3 and self._per_angle()) else frame
        self.img.setImage(disp, levels=self._levels, autoLevels=False)
        self.vb.autoRange(padding=0.02)
        self._update_label(idx)

    def _update_label(self, idx: int) -> None:
        if self._arr is None:
            self.label.setText("—")
            return
        if self._arr.ndim == 2:
            self.label.setText("sinogram (nR × nTheta)")
        elif self._per_angle():
            n = self._arr.shape[1]
            deg = idx * 360.0 / n if n else 0.0
            self.label.setText(f"angle {idx} / {n - 1}   ({deg:.1f}°)")
        else:
            n = self._arr.shape[2]
            self.label.setText(f"z {idx} / {n - 1}")

    # ----- slots -----------------------------------------------------------
    def _on_slider(self, value: int) -> None:
        self._show(value)

    def _on_mode_changed(self, _text: str) -> None:
        self._configure_slider()
        self._show(self.slider.value())

    def _on_play_toggled(self, on: bool) -> None:
        self.play_btn.setText("⏸ Pause" if on else "▶ Play")
        if on and self._n_frames() > 1:
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        n = self._n_frames()
        if n <= 1:
            return
        self.slider.setValue((self.slider.value() + 1) % n)  # loop
