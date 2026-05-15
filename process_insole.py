#!/usr/bin/env python3
"""
process_insole.py

Complete insole processing pipeline with organized outputs.
Automatically creates outputs/scan_name/ directory structure.
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
    parser = argparse.ArgumentParser(description="Complete insole processing pipeline.")
    parser.add_argument("--scan", "-s", required=True, help="Path to scan file (PLY/PCD/OBJ/STL).")
    parser.add_argument("--config", "-c", default="config_example.yaml", help="YAML config file.")
    parser.add_argument("--method", default="safe", choices=["safe", "plane_only"], help="Ground removal method.")
    parser.add_argument("--dist", type=float, default=0.5, help="Plane distance threshold.")
    parser.add_argument("--res", type=float, default=1.5, help="Heightmap resolution.")
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

    # Define output paths
    output = Path(output_dir)
    isolated_path = str(output / f"{scan_name}_isolated.ply")
    outline_path = str(output / f"{scan_name}_outline.csv")
    outline_preview = str(output / f"{scan_name}_outline.png") if args.preview else None
    heightmap_base = str(output / scan_name)
    thickness_preview = str(output / f"{scan_name}_thickness.png") if args.preview else None
    final_stl = str(output / f"{scan_name}_insole.stl")

    # Step 1: Remove ground
    cmd1 = [sys.executable, "remove_ground.py", "-p", scan_path, "--method", args.method, "--dist", str(args.dist), "-o", isolated_path]
    run_cmd(cmd1, "Ground Removal")

    # Step 2: Extract outline
    cmd2 = [sys.executable, "extract_outline.py", "-p", isolated_path, "-o", outline_path]
    if outline_preview:
        cmd2.extend(["--preview", outline_preview])
    run_cmd(cmd2, "Outline Extraction")

    # Step 3: Generate heightmap
    cmd3 = [sys.executable, "generate_heightmap.py", "--pcd", isolated_path, "--outline", outline_path, "--res", str(args.res), "--outbase", heightmap_base]
    run_cmd(cmd3, "Heightmap Generation")

    # Step 4: Create parametric STL
    cmd4 = [sys.executable, "parametric_insole.py", "--heightmap", heightmap_base, "--outline", outline_path, "--config", args.config, "--output", final_stl]
    if thickness_preview:
        cmd4.extend(["--preview", thickness_preview])
    run_cmd(cmd4, "Parametric STL Generation")

    print(f"\n✅ COMPLETE! Final STL: {final_stl}")

    # Optional: render final result
    if args.preview:
        render_path = str(output / f"{scan_name}_render.png")
        cmd5 = [sys.executable, "render_stl.py", "--stl", final_stl, "--output", render_path]
        run_cmd(cmd5, "STL Rendering")
        print(f"📸 Render saved: {render_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


