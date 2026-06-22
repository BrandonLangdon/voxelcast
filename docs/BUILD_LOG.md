# VoxelCast — Build Log

A running record of the **decisions** made in this repo (a PySide6 desktop GUI
that drives and visualizes the VAMToolbox tomography engine) — what was chosen,
why, and what was rejected — so the reasoning is recoverable without re-reading
every diff.

**Per-repo logs.** Each repo in this effort keeps its own `BUILD_LOG.md`:
- **voxelcast** (this file) — GUI
- **VAMToolbox** — engine
- **Blender3MFExporter** — Blender add-on writing role-tagged 3MF
- **VolumeFillingLattice / VFL** — Blender add-on generating beam lattices

**How to use this file.** Append a dated entry per meaningful decision. Keep the
format: *Context → Decision → Why → Alternatives considered → Status*. Record the
*why* and the roads not taken — the code already shows the *what*. Do not put
secrets here (tokens, credentials).

---

## 2026-06 — A desktop GUI for VAMToolbox

**Context.** Needed to *see* 2D/3D targets, sinograms, and reconstructions, and to
import STL files and run the engine interactively.

**Decision.** A separate PySide6 app (`voxelcast`) with **dual coupling**: it can
import `vamtoolbox` directly (in-process optimize on a `QThread`) *and* load saved
result files. 2D via pyqtgraph, 3D via pyvistaqt (VTK). A thin `Dataset` model
keeps the rest of the app from importing vamtoolbox internals.

**Why.** A standalone GUI keeps the engine library headless and reusable; dual
coupling supports both "run it now" and "inspect a prior run."

**macOS-specific decisions (VTK/OpenGL quirks).**
- 3D view goes blank/black on resize with MSAA → set `multi_samples=0`, and
  render on resize/show.
- Translucent volumes render empty at high res → added a **Surface threshold**
  mode alongside Volume.
- Embedded `QtInteractor` repaint is unreliable → "Open in window" uses
  `pyvistaqt.BackgroundPlotter`; switched the native plotter to BackgroundPlotter
  to fix a segfault when closing the pop-out + main window together.
- The slice plane only repaints on render churn embedded — accepted as-is.

**Status.** Done; works end-to-end. (The teardown segfault on close no longer
reproduces — see the Tomo-parity entry.)

---

## 2026-06-20 — End-to-end check with the Metal projector

**Context.** Verify the STL and 3MF chains work through the GUI now that
VAMToolbox has a Metal GPU projector (see VAMToolbox's log).

**Findings.**
- `metalcompute` was missing in voxelcast's venv — without it the UI silently used
  the CPU projector. Installed it; the UI now selects Metal.
- STL (TacticalBlade) → OpenGL voxelize → **Metal**, 4 ms/iter.
- 3MF lattice (no insert) → **Metal**, 47 ms/iter.
- 3MF multi-role (insert + zero_dose) → ~92 ms/iter once VAMToolbox's Metal
  occlusion landed (was 5.7 s/iter on the CPU attenuation projector).
- Test suite passed; GUI exits cleanly (the earlier teardown segfault is gone).

**Status.** Chain verified.

---

## 2026-06-20 — Tomo feature parity (guided four-stage workflow)

**Context.** Bring VoxelCast up to the bundled "Tomo" GUI's capability set: load
one or more STLs, GPU voxelize, optimize (OSMO/BCLP), then preview and export a
print-ready projection video — through a guided Prep → Voxelize → Optimize →
Preview interface, with live 3D previews, absorption/diffusion toggles, hardware
auto-tuning, and a z-slab memory mode for large parts.

**Decision — re-base the engine bridge on `VAMPipeline`/`PrintConfig`.** VoxelCast
had rolled its own low-level bridge (`engine/reconstruct.py`). VAMToolbox already
ships the high-level `VAMPipeline`/`PrintConfig`/`run_print` API that Tomo itself
uses — voxelize → optimize → rebin → MP4, with progress callbacks, hardware
tuning, z-slab, and absorption/diffusion built in. Driving that gives most of the
Tomo gaps "for free" and means VoxelCast runs the *same engine path as Tomo*.

**Decision — stage rail + reuse the existing viewers** (vs a full wizard rebuild or
a single pipeline dock). A left stage rail swaps the center control panel and
unlocks stages progressively; the existing 3D/slice/sinogram/convergence viewer
docks serve as the live previews. Lowest-risk: keeps the viewers and the legacy
"Reconstruct from STL/3MF" menu working.

**Decision — multi-STL by mesh merge.** `VAMPipeline.voxelize()` takes one
`stl_path`, and per-STL voxelization would misalign bounding boxes. So multiple
STLs are concatenated with `trimesh` into one temp mesh (shared bounding box) and
voxelized as a single aligned grid.

**Decision — threading split.** Voxelize uses OpenGL → main thread (macOS rule);
optimize/rebin/video run on a `PipelineWorker` QThread streaming
`progress(stage, fraction, message)` and supporting cancel.

**Why.** Reusing the engine's own high-level API avoids re-implementing rebin,
video export, hardware tuning, and slabbing in the GUI, and keeps behavior
identical to Tomo. On a Mac the optimize stage flows through the Metal projector
automatically.

**Alternatives considered.** Extending the low-level `reconstruct.py`
(rejected — would re-implement half the pipeline). Full wizard restructure
(rejected for now — rebuilds more of the app; the stage rail reaches parity with
less churn).

**Status.** Done on branch `tomo-parity`. `engine/pipeline_bridge.py`,
`widgets/stage_flow.py`, `app.py` wiring; 6 new tests; full suite 25 passed.
Verified headless end-to-end (2 STLs → voxelize → optimize → rebin → MP4) and via
the live GUI.

---

## 2026-06-21 — Guided-flow UI tweaks (first review round)

**Context.** First round of UI feedback after using the guided flow.

**Decisions.**
- **Stage navigation → top step bar.** Moved the Prep/Voxelize/Optimize/Preview
  rail from the left side to a horizontal `QTabBar` across the top; panels wrapped
  in `QScrollArea` so the window resizes freely without clipping (no fixed-size
  windows remain).
- **3MF in Prep.** "Add model(s)…" accepts `.stl` and `.3mf`; a single 3MF
  voxelizes with its roles (insert / zero-dose surfaced as datasets). Mixing 3MF
  with other models is blocked (it carries its own bodies/roles).
- **Per-model transform.** Each imported model has its own translate (X/Y/Z) and
  rotate (X/Y/Z); the old single global rotation is gone. STL meshes are
  transformed before merge (`merge_meshes`); 3MF uses its rotation.
- **Combine 2D + 3D into a tabbed viewer.** The 3D volume and 2D slice (and
  sinogram / compare) docks are tabified into one right-hand group — same dataset,
  so you swap tabs rather than view side by side. 3D is the default tab; selecting
  a dataset raises its canonical view.
- **Dropped the Prep import description** — the file picker + (future) help docs
  convey accepted formats.

**Status.** Done on `tomo-parity`. Full suite 26 passed.

---
```
Template for new entries:

## YYYY-MM-DD — Short title

**Context.** What problem/why now.
**Decision.** What we chose.
**Why.** The reasoning.
**Alternatives considered.** What we rejected and why.
**Status.** Done / in progress / branch / merged.
```
