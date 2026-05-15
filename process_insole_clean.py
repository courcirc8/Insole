#!/usr/bin/env python3
"""
process_insole_clean.py

Improved insole processing pipeline with artifact filtering BEFORE grid conversion.
Order: Ground removal → Artifact cleaning → Outline → Grid → STL
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path


def get_scan_name(scan_path: str) -> str:
    """Extract scan name from path (without extension)."""
    return Path(scan_path).stem


def ensure_output_dir(scan_name: str) -> str:
    """Create and return outputs/scan_name/ directory."""
    output_dir = f"outputs/{scan_name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_cmd(cmd: list, description: str):
    """Run command and report timing."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    print(result.stdout.strip())
    return result


def main():
    parser = argparse.ArgumentParser(description="Clean insole processing pipeline (artifacts filtered first).")
    parser.add_argument("--scan", "-s", required=True, help="Path to scan file (PLY/PCD/OBJ/STL).")
    parser.add_argument("--config", "-c", default="config_example.yaml", help="YAML config file.")
    
    # Ground removal
    parser.add_argument("--method", default="safe", choices=["safe", "plane_only"], help="Ground removal method.")
    parser.add_argument("--dist", type=float, default=0.5, help="Plane distance threshold.")
    
    # Artifact cleaning (NEW - before grid)
    parser.add_argument("--clean_method", default="statistical", choices=["statistical", "conservative", "radius"], help="Artifact cleaning method.")
    parser.add_argument("--std_ratio", type=float, default=2.5, help="Statistical outlier std ratio.")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier neighbors.")
    
    # Grid generation
    parser.add_argument("--res", type=float, default=1.5, help="Heightmap resolution.")
    parser.add_argument("--knn", type=int, default=12, help="kNN neighbors for grid interpolation.")
    
    # Output control
    parser.add_argument("--preview", action="store_true", help="Generate preview images.")
    args = parser.parse_args()
    
    scan_path = args.scan
    if not os.path.isfile(scan_path):
        print(f"Scan file not found: {scan_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 1
    
    scan_name = get_scan_name(scan_path)
    output_dir = ensure_output_dir(scan_name)
    print(f"Processing: {scan_path} → {output_dir}/")
    print(f"Pipeline: Ground → Clean Artifacts → Outline → Grid → STL")
    
    # Define paths
    isolated_path = f"{output_dir}/{scan_name}_isolated.ply"
    cleaned_path = f"{output_dir}/{scan_name}_cleaned.ply"
    outline_path = f"{output_dir}/{scan_name}_outline.csv"
    outline_preview = f"{output_dir}/{scan_name}_outline.png" if args.preview else None
    heightmap_base = f"{output_dir}/{scan_name}"
    thickness_preview = f"{output_dir}/{scan_name}_thickness.png" if args.preview else None
    final_stl = f"{output_dir}/{scan_name}_insole.stl"
    
    # Step 1: Remove ground
    cmd1 = [sys.executable, "remove_ground.py", "-p", scan_path, "--method", args.method, "--dist", str(args.dist), "-o", isolated_path]
    run_cmd(cmd1, "1. Ground Removal")
    
    # Step 2: Clean artifacts (NEW STEP - before grid)
    cmd2 = [
        sys.executable, "clean_artifacts.py",
        "-p", isolated_path,
        "-o", cleaned_path,
        "--method", args.clean_method,
        "--nb_neighbors", str(args.nb_neighbors),
        "--std_ratio", str(args.std_ratio),
        "--std_multiplier", str(args.std_ratio),
    ]
    run_cmd(cmd2, "2. Artifact Cleaning (PRE-GRID)")
    
    # Step 3: Extract outline from CLEAN points
    cmd3 = [sys.executable, "extract_outline.py", "-p", cleaned_path, "-o", outline_path]
    if outline_preview:
        cmd3.extend(["--preview", outline_preview])
    run_cmd(cmd3, "3. Outline Extraction (from clean data)")

    # Step 4: Generate heightmap from CLEAN points
    cmd4 = [sys.executable, "generate_heightmap.py", "--pcd", cleaned_path, "--outline", outline_path, "--res", str(args.res), "--knn", str(args.knn), "--outbase", heightmap_base]
    run_cmd(cmd4, "4. Heightmap Generation (clean grid)")

    # Step 5: Create parametric STL
    cmd5 = [sys.executable, "parametric_insole.py", "--heightmap", heightmap_base, "--outline", outline_path, "--config", args.config, "--output", final_stl]
    if thickness_preview:
        cmd5.extend(["--preview", thickness_preview])
    run_cmd(cmd5, "5. Parametric STL Generation")

    print(f"\n✅ CLEAN PIPELINE COMPLETE!")
    print(f"📁 Final STL: {final_stl}")

    # Optional: render final result
    if args.preview:
        render_path = f"{output_dir}/{scan_name}_render.png"
        cmd6 = [sys.executable, "render_stl.py", "--stl", final_stl, "--output", render_path]
        run_cmd(cmd6, "6. STL Rendering")
        print(f"📸 Render: {render_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


