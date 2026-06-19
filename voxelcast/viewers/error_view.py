"""Convergence / error plot built on pyqtgraph.

Plots the optimizer's per-iteration loss, filling in live as a reconstruction
runs (fed from the OptimizeWorker.progress signal). pyqtgraph is raster-Qt, so
this updates reliably (unlike the embedded VTK view).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets


class ErrorView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(6, 2, 6, 2)
        self.log_y = QtWidgets.QCheckBox("log y")
        self.log_y.setToolTip("Log scale on the loss axis")
        self.log_y.toggled.connect(self._on_log_toggled)
        controls.addWidget(self.log_y)
        controls.addStretch(1)
        self.status = QtWidgets.QLabel("no reconstruction yet")
        controls.addWidget(self.status)
        layout.addLayout(controls)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "iteration")
        self.plot.setLabel("left", "loss / error")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(
            [], [], pen=pg.mkPen("#1f77b4", width=2),
            symbol="o", symbolSize=5, symbolBrush="#1f77b4",
        )
        layout.addWidget(self.plot)

        self._xs: list[float] = []
        self._ys: list[float] = []

    def reset(self) -> None:
        """Clear the plot for a new reconstruction."""
        self._xs = []
        self._ys = []
        self.curve.setData([], [])
        self.status.setText("optimizing…")

    def add_point(self, iteration: int, loss: float) -> None:
        """Append one iteration's loss and redraw."""
        self._xs.append(float(iteration))
        self._ys.append(float(loss))
        self.curve.setData(self._xs, self._ys)
        self.status.setText(f"iter {iteration}   loss = {loss:.4g}")

    def set_complete(self) -> None:
        if self._ys:
            self.status.setText(
                f"done — {len(self._ys)} iters, final loss = {self._ys[-1]:.4g}"
            )
        else:
            # optimizer streamed no per-iteration loss (e.g. CAL)
            self.status.setText("done (no per-iteration loss reported)")

    def _on_log_toggled(self, on: bool) -> None:
        # Log scale needs strictly-positive values; guard against zeros.
        if on and self._ys and min(self._ys) <= 0:
            self.status.setText("log y unavailable (non-positive loss values)")
            self.log_y.blockSignals(True)
            self.log_y.setChecked(False)
            self.log_y.blockSignals(False)
            return
        self.plot.setLogMode(x=False, y=on)
