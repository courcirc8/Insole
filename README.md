# Insole — Therapeutic Insole Reconstruction Pipeline

Open-source Python pipeline to convert 3D-scanned therapeutic insoles into parametrizable, printable STL files for flexible filaments (TPU / NinjaFlex).

## Features

| Stage | Script | Description |
|-------|--------|-------------|
| 👁️ View | `view_scan.py` | Display point clouds & meshes (PLY/OBJ/STL) with GUI picker |
| ✂️ Isolate | `remove_ground.py` | Remove ground plane (RANSAC + DBSCAN) |
| 🔍 Compare | `isolate_insole.py` | Evaluate multiple isolation methods side-by-side |
| 🧹 Clean | `clean_artifacts.py` | Statistical / radius / conservative outlier removal |
| 📐 Outline | `extract_outline.py` | 2D alpha-shape contour of insole footprint |
| 🗺️ Heightmap | `generate_heightmap.py` | kNN-interpolated Z grid within outline |
| ⚙️ Parametric | `parametric_insole.py` | Arch support, heel posting, met pads → STL |
| 🌐 Web viewer | `ply_viewer_web.py` | Interactive 3D viewer with Z-filtering, angle adjustment & export |
| 🖥️ GUI viewer | `ply_viewer_gui.py` | Desktop Tkinter + matplotlib viewer |
| 🖼️ Render | `render_stl.py` | Render STL to PNG (headless) |
| 🔗 Pipeline | `process_insole.py` | Full pipeline orchestrator |
| 🔗 Pipeline v2 | `process_insole_clean.py` | Clean-first pipeline (filter → grid → STL) |

## Quick Start

```bash
# 1. Clone & setup
git clone <repo-url> && cd Insole
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. View a scan
python view_scan.py -p scans/scan1/scan1.ply

# 3. Run the pipeline
python remove_ground.py -p scans/scan1/scan1.ply --method safe --dist 0.5
python extract_outline.py -p outputs/scan1/scan1_isolated.ply -o outputs/scan1/outline.csv
python generate_heightmap.py --pcd outputs/scan1/scan1_isolated.ply --outline outputs/scan1/outline.csv --res 1.5

# 4. Interactive web viewer (with Z-filter + angle controls + save)
python ply_viewer_web.py --port 8055
# Open http://127.0.0.1:8055
```

## Web Viewer Features

The enhanced web viewer (`ply_viewer_web.py`) provides:
- **3D point cloud visualization** with Z-axis colormap
- **Z-axis filtering** — set min/max height thresholds interactively
- **Angle adjustment** — fine-tune X/Y rotation (±10°) to correct plane orientation
- **Save cleaned PLY** — export filtered & rotated point cloud with descriptive filename

## Project Structure

```
Insole/
├── scans/                  # Raw scan data (PLY/OBJ/MTL)
│   └── scan1/
├── outputs/                # Generated outputs (not tracked in git)
│   ├── plane_corrected/    # Plane-aligned point clouds
│   ├── height_analysis/    # Height-filtered results + STL
│   └── fine_tuned/         # GUI-exported cleaned PLY files
├── old/                    # Archived experimental scripts
├── view_scan.py            # Scan viewer
├── remove_ground.py        # Ground removal
├── isolate_insole.py       # Multi-method isolation
├── extract_outline.py      # Outline extraction
├── generate_heightmap.py   # Heightmap generation
├── parametric_insole.py    # Parametric STL generation
├── ply_viewer_web.py       # Web-based interactive viewer
├── ply_viewer_gui.py       # Desktop GUI viewer
├── render_stl.py           # STL → PNG rendering
├── process_insole.py       # Pipeline orchestrator
├── process_insole_clean.py # Clean-first pipeline
├── config_example.yaml     # Example parametric config
├── requirements.txt        # Python dependencies
├── purpose.md              # Detailed design doc & algorithms
└── README.md
```

## Pipeline Overview

```
Raw scan (PLY/OBJ)
  │
  ├─ 1. Ground removal (RANSAC + DBSCAN)
  ├─ 2. Plane correction (align lowest surface → Z=0)
  ├─ 3. Height-based filtering (Z min/max thresholds)
  ├─ 4. Statistical outlier removal
  ├─ 5. Outline extraction (alpha shape)
  ├─ 6. Heightmap generation (kNN interpolation)
  └─ 7. Parametric STL (thickness + arch + heel + met pad)
```

## Configuration

See `config_example.yaml` for parametric insole settings:
- Base/min/max thickness
- Arch support (position, height, width)
- Heel posting (varus angle, height)
- Metatarsal pad
- Edge lip & chamfer
- Smoothing

## Performance (500K points)

| Stage | Time |
|-------|------|
| Ground removal | ~5 s |
| Outline extraction | ~4 s |
| Heightmap generation | ~9 s |
| **Total** | **~18 s** |

## Printing Tips (TPU / 95A)

- Nozzle: 0.4–0.6 mm, 3–5 perimeters, 0% or sparse infill
- Layer height: 0.2–0.3 mm, slow outer walls, minimal retraction
- Brim for adhesion; 220–240 °C nozzle, 40–60 °C bed

## License

MIT
