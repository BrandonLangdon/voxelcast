# VoxelCast

A desktop viewer for [VAMToolbox](https://github.com/computed-axial-lithography/VAMToolbox)
2D/3D results — built on **PySide6** (Qt), **pyvistaqt** (embedded VTK) for 3D
volume rendering, and **pyqtgraph** for fast 2D slices and sinograms.

## Why
VAMToolbox produces volumes (target / reconstruction / dose) and sinograms, but
its built-in `.show()` methods spin up separate blocking matplotlib/vedo/pyglet
windows. VoxelCast consolidates these into one interactive desktop app and a
file browser, and reuses the same VTK pipeline VAMToolbox already depends on.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# To load native VAM files and run optimizations in-process:
pip install -e ".[engine]"   # requires vamtoolbox
```

## Run
```bash
voxelcast                       # empty window; File ▸ Open
voxelcast path/to/recon.recon   # open a file directly
voxelcast --demo                # synthetic volume, no data files needed
python -m voxelcast             # equivalent entry point
```

## Reconstruct from an STL
**Engine ▸ Reconstruct from STL…** (Ctrl+R) runs the full VAMToolbox pipeline
and shows the result — no scripting required:

1. **Voxelize** the STL into a target volume (VAMToolbox's OpenGL voxelizer,
   on the main thread — fast).
2. **Optimize** (OSMO/CAL/BCLP) on a background thread, with live iteration
   progress in the status bar (the window stays responsive).
3. The **target**, **sinogram**, and **recon** are added to the *Dataset*
   selector in the toolbar — switch between them; volumes get the 3D view.

Requires the `engine` extra (vamtoolbox). On macOS this runs entirely on CPU.

## Supported files
| Type | Extensions | Needs vamtoolbox? |
|------|------------|-------------------|
| Native VAMToolbox | `.target` `.sino` `.recon` | **Yes** — these are pickled `Volume` objects |
| Neutral / portable | `.npy` `.tif`/`.tiff` | No |

> **Note:** native VAM files are `dill` pickles of `vamtoolbox.geometry.Volume`,
> so opening them requires `vamtoolbox` importable. Use **File ▸ Export
> (neutral)** to write `.npy`/TIFF that can be viewed anywhere.

## Architecture
```
voxelcast/
├── model/       Dataset — thin, dependency-free array + metadata container
├── io/          load() / save_neutral() — files <-> Dataset (native + neutral)
├── engine/      OptimizeWorker — runs vamtoolbox.optimize off the UI thread
├── viewers/     VolumeView (pyvistaqt 3D), SliceView (pyqtgraph 2D)
└── app.py       MainWindow — menus, dock panels, wiring
```

Design rules:
- The UI depends only on `Dataset`, never on vamtoolbox internals — loaders do
  the conversion and drop the original object.
- Long optimizations run on a worker thread (`engine/`), never the GUI thread.
- Viewer backends import lazily; a missing optional dep shows a placeholder
  instead of crashing the app.

## Status
Early scaffold: file loading, 3D volume + 2D slice/sinogram views, neutral
export, and the worker-thread plumbing for in-process optimization. Roadmap:
sinogram angle scrubber, projector-sequence playback, convergence/error plot,
and an in-app "run optimization" panel.
