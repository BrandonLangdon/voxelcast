"""VoxelCast main window.

Open a VAMToolbox file (.target/.sino/.recon) or a neutral array (.npy/.tiff)
and visualize it: a 3D volume render (pyvistaqt) for volumes plus a scrubbable
2D slice/sinogram view (pyqtgraph), shown as dock panels.

The viewer backends are imported lazily so a missing optional dependency shows
a helpful placeholder instead of preventing the app from starting.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets

from voxelcast import __version__
from voxelcast.io import load, save_neutral, NATIVE_EXTS, NEUTRAL_EXTS, LoadError
from voxelcast.model import Dataset


def _placeholder(message: str) -> QtWidgets.QWidget:
    w = QtWidgets.QLabel(message)
    w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    w.setWordWrap(True)
    return w


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"VoxelCast {__version__}")
        self.resize(1100, 750)

        self._current: Dataset | None = None
        self._volume_view = None
        self._slice_view = None

        self._build_menu()
        self._build_docks()
        self.statusBar().showMessage("Open a file to begin  (File ▸ Open)")

    # ----- UI construction -------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_act = file_menu.addAction("&Open…")
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.open_file_dialog)

        self.export_act = file_menu.addAction("&Export (neutral)…")
        self.export_act.setShortcut("Ctrl+E")
        self.export_act.triggered.connect(self.export_dialog)
        self.export_act.setEnabled(False)

        file_menu.addSeparator()
        demo_act = file_menu.addAction("Load &demo volume")
        demo_act.triggered.connect(self.load_demo)

        file_menu.addSeparator()
        quit_act = file_menu.addAction("&Quit")
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)

    def _build_docks(self) -> None:
        # 3D volume dock. VTK's QtInteractor segfaults under the "offscreen" Qt
        # platform (headless/CI), so allow disabling it via env for testing.
        self.volume_dock = QtWidgets.QDockWidget("3D Volume", self)
        if os.environ.get("VOXELCAST_DISABLE_3D"):
            self.volume_dock.setWidget(
                _placeholder("3D view disabled (VOXELCAST_DISABLE_3D)")
            )
        else:
            try:
                from voxelcast.viewers.volume_view import VolumeView
                self._volume_view = VolumeView()
                self.volume_dock.setWidget(self._volume_view)
            except Exception as e:
                self.volume_dock.setWidget(
                    _placeholder(f"3D view unavailable:\n{e}\n\npip install pyvista pyvistaqt")
                )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.volume_dock)

        # 2D slice dock
        self.slice_dock = QtWidgets.QDockWidget("2D Slices", self)
        try:
            from voxelcast.viewers.slice_view import SliceView
            self._slice_view = SliceView()
            self.slice_dock.setWidget(self._slice_view)
        except Exception as e:
            self.slice_dock.setWidget(
                _placeholder(f"2D view unavailable:\n{e}\n\npip install pyqtgraph")
            )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.slice_dock)

    # ----- actions ---------------------------------------------------------
    def open_file_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in (*NATIVE_EXTS, *NEUTRAL_EXTS))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open volume / sinogram", "", f"VoxelCast data ({exts});;All files (*)"
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            ds = load(path)
        except LoadError as e:
            QtWidgets.QMessageBox.critical(self, "Could not open file", str(e))
            return
        self.show_dataset(ds)

    def show_dataset(self, ds: Dataset) -> None:
        self._current = ds
        self.export_act.setEnabled(True)
        self.setWindowTitle(f"VoxelCast {__version__} — {ds.name}")
        self.statusBar().showMessage(ds.summary())

        if self._slice_view is not None:
            self._slice_view.set_dataset(ds)
        if self._volume_view is not None:
            # only meaningful for true 3D volumes
            self._volume_view.set_dataset(ds)
        self.volume_dock.setVisible(ds.is_volume)

    def export_dialog(self) -> None:
        if self._current is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export (neutral)", self._current.name, "NumPy (*.npy);;TIFF (*.tif)"
        )
        if not path:
            return
        try:
            save_neutral(self._current, path)
            self.statusBar().showMessage(f"Exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def load_demo(self) -> None:
        """A synthetic volume so the GUI can be exercised without any files."""
        n = 64
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float)
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2)
        blob = np.clip(1.0 - r / (n * 0.45), 0, 1) ** 2
        blob = np.moveaxis(blob, 0, 2)  # (nY, nX, nZ)
        self.show_dataset(Dataset(array=blob, vol_type="recon", name="demo blob"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    win = MainWindow()
    win.show()

    # optional: open a file passed on the command line
    file_args = [a for a in argv[1:] if not a.startswith("-")]
    if file_args:
        win.open_path(file_args[0])
    elif "--demo" in argv:
        win.load_demo()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
