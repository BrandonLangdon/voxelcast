"""Side-by-side comparison view (e.g. target vs recon).

Two image panels, each with its own dataset picker, driven by a single shared
slice slider so you can step through both volumes in lock-step. Built on
pyqtgraph (raster-Qt), so it updates reliably.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from voxelcast.model import Dataset


class CompareView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: dict[str, Dataset] = {}
        self._arrs: list[np.ndarray | None] = [None, None]
        self._levels: list[tuple[float, float]] = [(0.0, 1.0), (0.0, 1.0)]
        # Until the user explicitly picks a side, keep auto-defaulting it so the
        # right panel switches to "recon" once a recon appears.
        self._user_set: list[bool] = [False, False]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        self.combos: list[QtWidgets.QComboBox] = []
        self.imgs: list[pg.ImageItem] = []
        self.vbs: list[pg.ViewBox] = []
        for side in range(2):
            col = QtWidgets.QVBoxLayout()
            combo = QtWidgets.QComboBox()
            combo.currentTextChanged.connect(lambda _t, s=side: self._on_combo(s))
            col.addWidget(combo)
            glw = pg.GraphicsLayoutWidget()
            glw.setBackground("w")
            vb = glw.addViewBox()
            vb.setAspectLocked(True)
            vb.invertY(True)
            img = pg.ImageItem(axisOrder="row-major")
            vb.addItem(img)
            col.addWidget(glw)
            row.addLayout(col)
            self.combos.append(combo)
            self.imgs.append(img)
            self.vbs.append(vb)
        layout.addLayout(row)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(6, 2, 6, 2)
        bottom.addWidget(QtWidgets.QLabel("slice"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(lambda _v: self._refresh())
        bottom.addWidget(self.slider)
        self.label = QtWidgets.QLabel("—")
        bottom.addWidget(self.label)
        layout.addLayout(bottom)

    # ----- public API ------------------------------------------------------
    def set_datasets(self, datasets: dict[str, Dataset]) -> None:
        """Refresh the dataset pickers (preserving current selections)."""
        self._datasets = dict(datasets)
        names = list(datasets.keys())
        for side, combo in enumerate(self.combos):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if self._user_set[side] and cur in names:
                combo.setCurrentText(cur)            # respect an explicit choice
            elif names:
                combo.setCurrentText(self._default_for_side(side, names))
            combo.blockSignals(False)
            self._load_side(side)
        self._update_slider_range()
        self._refresh()

    # ----- internals -------------------------------------------------------
    def _default_for_side(self, side: int, names: list[str]) -> str:
        pref = "target" if side == 0 else "recon"
        for n in names:
            if self._datasets[n].vol_type == pref:
                return n
        return names[min(side, len(names) - 1)]

    def _load_side(self, side: int) -> None:
        ds = self._datasets.get(self.combos[side].currentText())
        if ds is None:
            self._arrs[side] = None
            return
        arr = np.squeeze(np.asarray(ds.array).astype(float))
        self._arrs[side] = arr
        self._levels[side] = (
            (float(arr.min()), float(arr.max()) or 1.0) if arr.size else (0.0, 1.0)
        )

    def _on_combo(self, side: int) -> None:
        self._user_set[side] = True  # user made an explicit choice for this side
        self._load_side(side)
        self._update_slider_range()
        self._refresh()

    @staticmethod
    def _depth(arr: np.ndarray | None) -> int:
        if arr is None:
            return 1
        return arr.shape[2] if arr.ndim == 3 else 1

    def _update_slider_range(self) -> None:
        d = max(self._depth(self._arrs[0]), self._depth(self._arrs[1]), 1)
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, d - 1))
        self.slider.setValue(min(self.slider.value(), max(0, d - 1)))
        self.slider.blockSignals(False)
        self.slider.setEnabled(d > 1)

    def _refresh(self) -> None:
        idx = self.slider.value()
        for side in range(2):
            arr = self._arrs[side]
            if arr is None:
                self.imgs[side].clear()
                continue
            frame = arr if arr.ndim == 2 else arr[:, :, min(idx, arr.shape[2] - 1)]
            self.imgs[side].setImage(frame, levels=self._levels[side], autoLevels=False)
            self.vbs[side].autoRange(padding=0.02)
        self.label.setText(f"slice {idx}")
