"""
clean_artifacts.py

Remove outlier/artifact points from an isolated point cloud before gridding.
Supports statistical, radius, conservative and insole (shape-prior) methods.
The "insole" method (default) exploits what an insole looks like — see
filter_insole.py — and is drastically more effective on scan ghosts/spikes.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

from io_utils import load_point_cloud
from filter_insole import InsolePriors, filter_insole_points, save_preview


def clean_insole(pcd: o3d.geometry.PointCloud, max_height: float, preview: str | None):
    """Shape-prior filtering: height + footprint + thin-shell priors."""
    points = np.asarray(pcd.points)
    priors = InsolePriors(max_height=max_height)
    kept, report, diag = filter_insole_points(points, priors)
    print(report.summary())
    if preview:
        pts_in = points.copy()
        pts_in[:, 2] -= diag.get("ground", 0.0)
        save_preview(preview, pts_in, kept, diag)
        print(f"Saved preview: {preview}")
    cleaned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kept))
    return cleaned, kept


def clean_statistical(pcd: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float):
    cleaned, inliers = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return cleaned, inliers


def clean_radius(pcd: o3d.geometry.PointCloud, nb_points: int, radius: float):
    cleaned, inliers = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cleaned, inliers


def clean_conservative(pcd: o3d.geometry.PointCloud, std_multiplier: float):
    """Conservative two-pass cleaning: tight statistical then a permissive radius pass."""
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30, std_ratio=std_multiplier
    )
    cleaned, inliers = cleaned.remove_radius_outlier(nb_points=8, radius=4.0)
    return cleaned, inliers


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean artifacts from an isolated point cloud.")
    parser.add_argument("--path", "-p", required=True, help="Input point cloud (PLY/PCD).")
    parser.add_argument("--output", "-o", required=True, help="Output cleaned PLY path.")
    parser.add_argument(
        "--method", default="insole",
        choices=["insole", "statistical", "radius", "conservative"],
        help="Cleaning method (default: insole shape-prior filtering).",
    )
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical neighbors.")
    parser.add_argument("--std_ratio", type=float, default=2.5, help="Statistical std ratio.")
    parser.add_argument("--nb_points", type=int, default=16, help="Radius min points.")
    parser.add_argument("--radius", type=float, default=3.0, help="Radius (units).")
    parser.add_argument("--std_multiplier", type=float, default=2.0, help="Conservative std multiplier.")
    parser.add_argument("--max_height", type=float, default=60.0, help="Insole method: max height above ground (mm).")
    parser.add_argument("--preview", default=None, help="Insole method: optional diagnostics PNG.")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    pcd = load_point_cloud(args.path)
    n_before = len(pcd.points)
    print(f"Before cleaning: {n_before:,} points")

    if args.method == "insole":
        cleaned, inliers = clean_insole(pcd, args.max_height, args.preview)
    elif args.method == "statistical":
        cleaned, inliers = clean_statistical(pcd, args.nb_neighbors, args.std_ratio)
    elif args.method == "radius":
        cleaned, inliers = clean_radius(pcd, args.nb_points, args.radius)
    else:
        cleaned, inliers = clean_conservative(pcd, args.std_multiplier)

    n_after = len(inliers)
    pct = (n_after / n_before * 100.0) if n_before else 0.0
    print(f"After {args.method} cleaning: {n_after:,} points ({pct:.1f}%)")

    pts = np.asarray(cleaned.points)
    if pts.size:
        print(f"Z range: [{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}]")

    if not o3d.io.write_point_cloud(args.output, cleaned):
        print(f"Failed to save: {args.output}", file=sys.stderr)
        return 1
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
