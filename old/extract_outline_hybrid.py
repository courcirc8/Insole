"""
extract_outline_hybrid.py

Hybrid outline extraction: use all points for boundary, high-Z points for refinement.
Solves the truncation issue while maintaining edge quality.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import shapely.geometry as geom
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from sklearn.neighbors import NearestNeighbors

try:
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False


def load_pcd(path: str) -> o3d.geometry.PointCloud:
	"""Load point cloud from file."""
	pcd = o3d.io.read_point_cloud(path)
	if pcd.is_empty():
		raise ValueError(f"Empty point cloud: {path}")
	return pcd


def extract_full_boundary(points: np.ndarray, downsample: int = 50000) -> np.ndarray:
	"""Extract boundary using ALL points for full coverage."""
	xy = points[:, :2]
	
	# Downsample for speed
	if len(xy) > downsample:
		idx = np.random.choice(len(xy), downsample, replace=False)
		xy_sample = xy[idx]
	else:
		xy_sample = xy
	
	# Use alpha shape on all points for full boundary
	import alphashape
	
	# Estimate good alpha
	nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(xy_sample)
	dists, _ = nbrs.kneighbors(xy_sample)
	alpha = np.mean(dists[:, 1]) * 2.0  # Conservative alpha
	
	try:
		alpha_shape = alphashape.alphashape([(p[0], p[1]) for p in xy_sample], alpha)
		if isinstance(alpha_shape, geom.MultiPolygon):
			alpha_shape = max(alpha_shape.geoms, key=lambda g: g.area)
		if isinstance(alpha_shape, geom.Polygon):
			return np.array(alpha_shape.exterior.coords[:-1])
	except:
		pass
	
	# Fallback: convex hull
	hull = ConvexHull(xy_sample)
	return xy_sample[hull.vertices]


def refine_with_high_z(boundary_points: np.ndarray, all_points: np.ndarray, z_percentile: float = 85.0) -> np.ndarray:
	"""Refine boundary using high-Z points for better edge detection."""
	z_threshold = np.percentile(all_points[:, 2], z_percentile)
	high_z_points = all_points[all_points[:, 2] >= z_threshold]
	
	if len(high_z_points) < 100:
		return boundary_points
	
	# For each boundary point, find nearest high-Z point and adjust
	from scipy.spatial.distance import cdist
	
	high_z_xy = high_z_points[:, :2]
	dists = cdist(boundary_points, high_z_xy)
	nearest_idx = np.argmin(dists, axis=1)
	
	# Blend boundary with high-Z positions (weighted by distance)
	refined_points = boundary_points.copy()
	
	for i, (boundary_pt, nearest_dist) in enumerate(zip(boundary_points, dists.min(axis=1))):
		if nearest_dist < 10.0:  # Only refine if high-Z point is nearby
			high_z_pt = high_z_xy[nearest_idx[i]]
			weight = max(0, 1 - nearest_dist / 10.0)  # Closer = more weight
			refined_points[i] = boundary_pt * (1 - weight) + high_z_pt * weight
	
	return refined_points


def smooth_outline(coords: np.ndarray, target_points: int = 1000, smoothing: float = 0.1) -> np.ndarray:
	"""Smooth and densify outline using spline interpolation."""
	if len(coords) < 4:
		return coords
	
	try:
		# Parametric spline fitting (periodic for closed curve)
		tck, u = splprep([coords[:, 0], coords[:, 1]], s=smoothing, per=True)
		
		# Evaluate at target_points
		u_new = np.linspace(0, 1, target_points, endpoint=False)
		smooth_coords = np.column_stack(splev(u_new, tck))
		
		return smooth_coords
		
	except Exception as e:
		print(f"Warning: Spline smoothing failed ({e}), using original")
		return coords


def save_outline(out_path: str, coords: np.ndarray):
	"""Save outline coordinates as CSV."""
	np.savetxt(out_path, coords, delimiter=",", header="x,y", comments="")


def save_preview(preview_path: str, all_points: np.ndarray, boundary_coords: np.ndarray, high_z_points: np.ndarray = None):
	"""Save preview with all points, boundary, and high-Z points."""
	if not _HAS_MPL:
		return
	
	plt.figure(figsize=(12, 8))
	
	# All points (light)
	xy_all = all_points[:, :2]
	if len(xy_all) > 100000:
		idx = np.random.choice(len(xy_all), 100000, replace=False)
		sample = xy_all[idx]
	else:
		sample = xy_all
	plt.scatter(sample[:, 0], sample[:, 1], s=0.1, c="lightgray", alpha=0.3, label="All points")
	
	# High-Z points if provided
	if high_z_points is not None:
		high_z_xy = high_z_points[:, :2]
		if len(high_z_xy) > 20000:
			idx = np.random.choice(len(high_z_xy), 20000, replace=False)
			sample_hz = high_z_xy[idx]
		else:
			sample_hz = high_z_xy
		plt.scatter(sample_hz[:, 0], sample_hz[:, 1], s=0.5, c="blue", alpha=0.6, label="High-Z points")
	
	# Boundary
	plt.plot(boundary_coords[:, 0], boundary_coords[:, 1], "-r", linewidth=2, label="Hybrid outline")
	
	plt.axis("equal")
	plt.legend()
	plt.title(f"Hybrid Outline: {len(boundary_coords)} points")
	plt.tight_layout()
	plt.savefig(preview_path, dpi=200)
	plt.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="Hybrid outline: full boundary + high-Z refinement.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to isolated point cloud.")
	parser.add_argument("--z_percentile", type=float, default=85.0, help="Z percentile for refinement.")
	parser.add_argument("--smooth_points", type=int, default=1000, help="Target points in smoothed outline.")
	parser.add_argument("--smoothing", type=float, default=0.1, help="Spline smoothing factor.")
	parser.add_argument("--out", "-o", type=str, default=None, help="Output CSV path.")
	parser.add_argument("--preview", type=str, default=None, help="Preview PNG path.")
	args = parser.parse_args()

	in_path = args.path
	if not os.path.isabs(in_path):
		in_path = os.path.join(os.getcwd(), in_path)
	if not os.path.isfile(in_path):
		print(f"File not found: {in_path}", file=sys.stderr)
		return 1

	# Load points
	pcd = load_pcd(in_path)
	points = np.asarray(pcd.points)
	
	print(f"Hybrid outline extraction from {len(points):,} points")
	
	# Step 1: Extract full boundary using all points
	full_boundary = extract_full_boundary(points)
	print(f"Full boundary: {len(full_boundary)} points")
	
	# Step 2: Refine with high-Z points
	z_threshold = np.percentile(points[:, 2], args.z_percentile)
	high_z_points = points[points[:, 2] >= z_threshold]
	print(f"High-Z points (>= {z_threshold:.1f}): {len(high_z_points):,}")
	
	refined_boundary = refine_with_high_z(full_boundary, points, args.z_percentile)
	print(f"Refined boundary: {len(refined_boundary)} points")
	
	# Step 3: Smooth and densify
	smooth_boundary = smooth_outline(refined_boundary, args.smooth_points, args.smoothing)
	print(f"Smooth boundary: {len(smooth_boundary)} points")
	
	# Save
	out_path = args.out or os.path.splitext(in_path)[0] + "_outline_hybrid.csv"
	save_outline(out_path, smooth_boundary)
	
	# Calculate final area
	poly = geom.Polygon(smooth_boundary)
	print(f"Final outline area: {poly.area:.0f}")
	print(f"Saved: {out_path}")
	
	if args.preview:
		save_preview(args.preview, points, smooth_boundary, high_z_points)
		print(f"Saved preview: {args.preview}")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())


