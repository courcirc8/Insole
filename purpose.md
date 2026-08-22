## Insole: Therapeutic Insole Reconstruction & Parametric STL (Python)

> **Scope of this document.** This is the design/algorithm note. For installation,
> current commands and defaults, [README.md](README.md) is authoritative — some
> details below describe the original design intent and may lag the code.
> Read [DISCLAIMER.md](DISCLAIMER.md) before using any output: this is not a
> medical device.

### Purpose
Create a reproducible, open-source pipeline to turn a scanned therapeutic insole (captured on a reference plane) into a parametrizable, printable STL for flexible filaments (e.g., NinjaFlex/TPU).

### High-level workflow
1. View raw scan
   - `view_scan.py` displays point clouds/meshes (PLY/PCD/OBJ/STL), with GUI picker or `--path`.
2. Isolate insole
   - `remove_ground.py` segments the ground plane and isolates the insole. Default "safe" method: plane filter + DBSCAN largest cluster. Advanced `plane_surface` fits a smooth surface to remove residual non-insole points.
   - `isolate_insole.py` evaluates multiple methods (plane_only/adaptive/surface) to compare metrics.
3. Extract outline
   - `extract_outline.py` computes a 2D contour of the insole footprint and saves CSV (and preview PNG). Default `--method mask` (footprint-mask contour); `fast` (concave hull) and `alpha` (alpha shape) are also available.
4. Generate heightmap
   - `generate_heightmap.py` rasterizes Z within the outline using kNN regression; saves NPY (GX,GY,Z), PNG preview, and a gridded PLY.
5. Parametric modeling
   - `parametric_insole.py` composes base thickness and optional features (arch lift, heel posting, met pad) as smooth fields on the heightmap and exports STL.
6. Anatomical model
   - `insole_model.py` fits a scan to a named-zone anatomical model, or generates variants from a preset (`presets/`).

### Design principles
- Reproducibility: inputs and parameters are explicit; intermediate artefacts saved.
- Modularity: each stage is a small script (easy to swap/improve).
- Performance: downsampling, kNN/regression, DBSCAN; avoid brittle triangulation where possible.
- Printability: target watertight, thickness-controlled meshes suitable for TPU.

### Quick start
- Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- View a scan
```bash
python view_scan.py                     # GUI picker
python view_scan.py -p scans/scan1/scan1.ply  # direct
```
- Isolate insole
```bash
python remove_ground.py -p scans/scan1/scan1.ply --method safe --dist 0.5
# Output: scans/scan1/scan1_isolated.ply
```
- Extract outline
```bash
python extract_outline.py -p scans/scan1/scan1_isolated.ply \
  -o scans/scan1/scan1_isolated_outline.csv \
  --preview scans/scan1/scan1_isolated_outline.png
```
- Generate heightmap
```bash
python generate_heightmap.py \
  --pcd scans/scan1/scan1_isolated.ply \
  --outline scans/scan1/scan1_isolated_outline.csv \
  --res 1.5 --knn 12 --outbase scans/scan1/scan1_isolated
# Outputs: *_GX.npy, *_GY.npy, *_Z.npy, *_heightmap.png, *_grid.ply
```

### Algorithms & notes
- Ground isolation:
  - Plane RANSAC to detect ground; remove inliers.
  - Safe mode: DBSCAN to keep largest remaining cluster.
  - Advanced: fit a smooth surface (kNN regressor) in plane-aligned coordinates; reject residuals.
- Outline:
  - `mask` (default): occupancy-grid footprint mask, morphologically closed, largest contour traced and smoothed — covers 100% of points.
  - `alpha`: alpha shape with kNN-derived alpha heuristic; choose largest polygon.
  - `fast`: concave hull with a tunable concavity factor.
- Heightmap:
  - kNN (distance-weighted) regression over a uniform grid within the outline.

### Roadmap
Done: parametric thickness field, heightmap triangulation with shell/lip and
manifold export, YAML CLI configs, offscreen rendering (`render_stl.py`).

Open:
- Tests and CI: deterministic runs on a small synthetic fixture; mesh validation
  (watertightness, manifoldness, self-intersection) as an automated check.
- A small synthetic sample scan so the pipeline is runnable without private data.
- Consistent indentation across modules (see `.editorconfig`).

### Printing tips (TPU/95A)
See [DISCLAIMER.md](DISCLAIMER.md) — printed materials are not certified for
prolonged skin contact, and any orthotic should be validated by a clinician.

- Nozzle 0.4–0.6 mm; 3–5 perimeters; 0% infill or controlled sparse infill for feel.
- Layer height 0.2–0.3 mm; slow outer walls; minimal retraction.
- Brim and good bed adhesion; 220–240°C nozzle, 40–60°C bed (material-dependent).
