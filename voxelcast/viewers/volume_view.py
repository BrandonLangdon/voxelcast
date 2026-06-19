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
        self._slice_index: int | None = None
        self._slice_total: int = 1
        self._popouts: list = []  # keep native pop-out plotters alive (avoid GC crash)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.add_axes()  # orientation triad (X/Y/Z) for spatial context
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
        self._add_slice_plane_actor()  # clear() removed it; re-add
        if reset_camera:
            self.plotter.reset_camera()
        self._repaint()

    def _add_slice_plane_actor(self) -> None:
        """Draw a translucent plane at the current 2D-slice z position so the
        user can see where the cross-section sits in the volume."""
        if self._grid is None or self._slice_index is None:
            return
        b = self._grid.bounds
        total = max(self._slice_total, 1)
        frac = self._slice_index / (total - 1) if total > 1 else 0.0
        z = b[4] + frac * (b[5] - b[4])
        plane = pv.Plane(
            center=((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, z),
            direction=(0, 0, 1),
            i_size=(b[1] - b[0]) or 1.0,
            j_size=(b[3] - b[2]) or 1.0,
        )
        # name= replaces any previous plane actor (cheap update, no full clear).
        # lighting=False -> flat bright red regardless of viewing angle.
        self.plotter.add_mesh(
            plane, color="red", opacity=0.45, name="slice_plane",
            show_scalar_bar=False, lighting=False,
        )

    def set_slice_marker(self, index: int, total: int) -> None:
        """Update the slice-position plane (called when the 2D view scrubs)."""
        self._slice_index = index
        self._slice_total = total
        if self._grid is not None:
            self._add_slice_plane_actor()
            self._repaint()

    def _reset_camera(self) -> None:
        self.plotter.reset_camera()
        self._repaint()

    def _repaint(self) -> None:
        # An already-shown QtInteractor does not auto-render after the scene
        # changes. On macOS especially, a render() triggered from *another*
        # widget's signal (e.g. the 2D slider) renders to the back buffer but
        # never presents until the user interacts with the 3D view directly.
        # Force it: render, explicitly swap buffers (Frame), and do an
        # *immediate* synchronous Qt repaint (repaint(), not update()).
        self.plotter.render()
        rw = getattr(self.plotter, "render_window", None)
        if rw is not None:
            try:
                rw.Frame()  # force the OpenGL buffer swap (present)
            except Exception:
                pass
        try:
            self.plotter.repaint()  # immediate synchronous paintEvent
        except Exception:
            pass
        QtCore.QTimer.singleShot(0, self.plotter.render)

    def pop_out(self) -> None:
        """Open the current volume in a separate window that renders reliably.

        Uses pyvistaqt.BackgroundPlotter (a Qt window managed *within* this app)
        rather than a native pv.Plotter: a native window shares VTK/Qt state with
        the embedded interactor, so closing it crashes the whole app. The
        BackgroundPlotter is independent and closes cleanly.
        """
        if self._grid is None:
            return
        from pyvistaqt import BackgroundPlotter

        bp = BackgroundPlotter(title=f"VoxelCast — {self._name}")
        bp.set_background("white")
        bp.add_axes()
        self._build_actor(bp)
        if self._slice_index is not None:
            b = self._grid.bounds
            total = max(self._slice_total, 1)
            frac = self._slice_index / (total - 1) if total > 1 else 0.0
            z = b[4] + frac * (b[5] - b[4])
            plane = pv.Plane(
                center=((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, z),
                direction=(0, 0, 1),
                i_size=(b[1] - b[0]) or 1.0, j_size=(b[3] - b[2]) or 1.0,
            )
            bp.add_mesh(plane, color="red", opacity=0.45,
                        show_scalar_bar=False, lighting=False)
        bp.reset_camera()
        # Keep a reference so it isn't GC'd while open; drop it when closed.
        self._popouts.append(bp)
        bp.app_window.signal_close.connect(lambda: self._popouts.remove(bp)
                                           if bp in self._popouts else None)

    # ----- Qt events: keep the embedded render alive on show/resize --------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.plotter.render)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # macOS embedded VTK can go black on resize unless re-rendered.
        self.plotter.render()

    def closeEvent(self, event) -> None:  # noqa: N802
        for p in self._popouts:
            try:
                p.close()
            except Exception:
                pass
        self._popouts.clear()
        self.plotter.close()
        super().closeEvent(event)
