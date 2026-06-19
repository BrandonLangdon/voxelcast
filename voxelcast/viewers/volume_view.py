"""3D volume viewer built on pyvistaqt (embedded VTK).

Two render modes:

* **Surface** (default) — a solid threshold surface of the volume, colored by
  value. Crisp and *resolution-independent*: it shows the reconstructed shape
  clearly at any grid size. (Plain volume rendering gets progressively more
  translucent as resolution rises, so a fine recon can wash out.)
* **Volume** — translucent volume rendering of the full field (good for the
  dose distribution).

A threshold slider controls the surface isolevel (as a % of the max value).

macOS note: embedded VTK render windows can render blank or go black on
resize on some Macs. We mitigate with multi_samples=0 and explicit renders on
show/resize, and provide an "Open in window" button that pops a *native* VTK
window (which renders reliably) as a guaranteed fallback.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtWidgets

from voxelcast.model import Dataset

# MSAA/anti-aliasing is a common cause of blank embedded render windows on
# macOS; disabling it makes the embedded QtInteractor far more reliable.
pv.global_theme.multi_samples = 0


class VolumeView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid: pv.ImageData | None = None
        self._vmax: float = 1.0
        self._name: str = "volume"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        layout.addWidget(self.plotter.interactor)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Surface", "Volume"])
        self.mode.currentTextChanged.connect(self._on_control_changed)
        controls.addWidget(self.mode)

        controls.addWidget(QtWidgets.QLabel("Threshold:"))
        self.thr = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.thr.setRange(1, 99)
        self.thr.setValue(15)  # low default: a dose recon's structure is at low values
        self.thr.setToolTip("Surface isolevel as % of the max value")
        self.thr.valueChanged.connect(self._on_control_changed)
        controls.addWidget(self.thr, 1)

        self.reset_btn = QtWidgets.QPushButton("Reset camera")
        self.reset_btn.clicked.connect(self._reset_camera)
        controls.addWidget(self.reset_btn)

        self.popout_btn = QtWidgets.QPushButton("Open in window")
        self.popout_btn.setToolTip("Open this volume in a native VTK window "
                                   "(reliable if the embedded view stays blank)")
        self.popout_btn.clicked.connect(self.pop_out)
        controls.addWidget(self.popout_btn)
        layout.addLayout(controls)

    # ----- public API ------------------------------------------------------
    def set_dataset(self, ds: Dataset) -> None:
        arr = np.ascontiguousarray(np.squeeze(ds.array).astype(float))
        self._name = ds.name
        if arr.ndim != 3:
            self._grid = None
            self.plotter.clear()
            self._repaint()
            return
        self._grid = pv.wrap(arr)
        self._vmax = float(arr.max()) if arr.size else 1.0
        self._render_current(reset_camera=True)

    # ----- rendering -------------------------------------------------------
    def _on_control_changed(self, *args) -> None:
        self._render_current(reset_camera=False)

    def _build_actor(self, plotter: pv.Plotter) -> None:
        """Add the current dataset to `plotter` per the selected mode."""
        if self._grid is None:
            return
        if self.mode.currentText() == "Volume":
            plotter.add_volume(self._grid, cmap="viridis", opacity="sigmoid")
        else:
            level = (self.thr.value() / 100.0) * self._vmax
            try:
                body = self._grid.threshold(level)
            except Exception:
                body = None
            if body is not None and body.n_points > 0:
                plotter.add_mesh(body, cmap="viridis", clim=(0.0, self._vmax))

    def _render_current(self, reset_camera: bool) -> None:
        if self._grid is None:
            return
        self.thr.setEnabled(self.mode.currentText() == "Surface")
        self.plotter.clear()
        self._build_actor(self.plotter)
        if reset_camera:
            self.plotter.reset_camera()
        self._repaint()

    def _reset_camera(self) -> None:
        self.plotter.reset_camera()
        self._repaint()

    def _repaint(self) -> None:
        # An already-shown QtInteractor does not auto-render after the scene
        # changes; render now AND defer one more render to the next event-loop
        # tick to cover cases where the widget isn't laid out yet.
        self.plotter.render()
        QtCore.QTimer.singleShot(0, self.plotter.render)

    def pop_out(self) -> None:
        """Open the current volume in a native VTK window (reliable rendering)."""
        if self._grid is None:
            return
        plotter = pv.Plotter(title=f"VoxelCast — {self._name}")
        plotter.set_background("white")
        self._build_actor(plotter)
        plotter.show()  # blocking native window; closes on user request

    # ----- Qt events: keep the embedded render alive on show/resize --------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.plotter.render)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # macOS embedded VTK can go black on resize unless re-rendered.
        self.plotter.render()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.plotter.close()
        super().closeEvent(event)
