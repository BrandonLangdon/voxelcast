"""Dialog to choose an STL and reconstruction parameters."""
from __future__ import annotations

import os

from PySide6 import QtWidgets

from voxelcast.engine.reconstruct import ReconParams, OPTIMIZERS

FILTERS = ("hamming", "ram-lak", "shepp-logan", "cosine", "hanning")


class ReconstructDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, stl_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Reconstruct from STL")
        self.setMinimumWidth(440)

        form = QtWidgets.QFormLayout()

        # STL file picker
        self._path = QtWidgets.QLineEdit(stl_path)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._path, 1)
        row.addWidget(browse)
        path_w = QtWidgets.QWidget()
        path_w.setLayout(row)
        form.addRow("STL file:", path_w)

        self._resolution = QtWidgets.QSpinBox()
        self._resolution.setRange(8, 1000)
        self._resolution.setValue(80)
        self._resolution.setToolTip("Number of z layers (voxel grid resolution)")
        form.addRow("Resolution:", self._resolution)

        self._method = QtWidgets.QComboBox()
        self._method.addItems(OPTIMIZERS)
        self._method.setToolTip("OSMO/BCLP report iteration progress; CAL does not")
        form.addRow("Method:", self._method)

        self._n_iter = QtWidgets.QSpinBox()
        self._n_iter.setRange(1, 1000)
        self._n_iter.setValue(20)
        form.addRow("Iterations:", self._n_iter)

        self._num_angles = QtWidgets.QSpinBox()
        self._num_angles.setRange(2, 3600)
        self._num_angles.setValue(180)
        form.addRow("Projection angles:", self._num_angles)

        self._d_h = QtWidgets.QDoubleSpinBox()
        self._d_h.setRange(0.0, 1.0)
        self._d_h.setSingleStep(0.05)
        self._d_h.setValue(0.85)
        form.addRow("Dose high (d_h):", self._d_h)

        self._d_l = QtWidgets.QDoubleSpinBox()
        self._d_l.setRange(0.0, 1.0)
        self._d_l.setSingleStep(0.05)
        self._d_l.setValue(0.60)
        form.addRow("Dose low (d_l):", self._d_l)

        self._filter = QtWidgets.QComboBox()
        self._filter.addItems(FILTERS)
        form.addRow("Init filter:", self._filter)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose STL", os.path.dirname(self._path.text()), "STL meshes (*.stl)"
        )
        if path:
            self._path.setText(path)

    def accept(self) -> None:  # noqa: D102
        if not self._path.text().strip():
            QtWidgets.QMessageBox.warning(self, "No STL", "Please choose an STL file.")
            return
        super().accept()

    def params(self) -> ReconParams:
        return ReconParams(
            stl_path=self._path.text().strip(),
            resolution=self._resolution.value(),
            method=self._method.currentText(),
            n_iter=self._n_iter.value(),
            num_angles=self._num_angles.value(),
            d_h=self._d_h.value(),
            d_l=self._d_l.value(),
            filter=self._filter.currentText(),
        )
