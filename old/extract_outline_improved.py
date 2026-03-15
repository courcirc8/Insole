"""
extract_outline_improved.py

Improved outline extraction using Z-aware edge detection.
Prioritizes high-Z points (insole edges) for accurate boundary detection.
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


def extract_edge_points(points: np.ndarray, z_percentile: float = 85.0, max_points: int = 50000) -> np.ndarray:
	"""
	Extract edge points using Z-elevation filtering.
	
	Args:
		points: 3D points (N, 3)
		z_percentile: Z percentile threshold for edge detection
		max_points: Max points to use for outline computation
	
	Returns:
		Edge points (M, 3) where M <= max_points
	"""
	z_threshold = np.percentile(points[:, 2], z_percentile)
	edge_mask = points[:, 2] >= z_threshold
	edge_points = points[edge_mask]
	
	print(f"Edge detection: Z >= {z_threshold:.1f} → {len(edge_points):,} points")
	
	# Downsample if too many
	if len(edge_points) > max_points:
		idx = np.random.choice(len(edge_points), max_points, replace=False)
		edge_points = edge_points[idx]
	
	return edge_points


def compute_z_aware_outline(edge_points: np.ndarray, concave_factor: float = 0.2) -> geom.Polygon:
	"""
	Compute outline from high-Z edge points using ConvexHull + refinement.
	
	Args:
		edge_points: High-Z edge points (N, 3)
		concave_factor: Concavity control (0=convex, 0.3=very concave)
	
	Returns:
		Shapely Polygon outline
	"""
	xy = edge_points[:, :2]
	z = edge_points[:, 2]
	
	if len(xy) < 3:
		raise ValueError("Need at least 3 edge points for outline")
	
	# Start with convex hull
	hull = ConvexHull(xy)
	hull_pts = xy[hull.vertices]
	hull_z = z[hull.vertices]
	
	if concave_factor <= 0:
		return geom.Polygon(hull_pts)
	
	# Refine using Z-weighted interior points
	refined_pts = []
	n_hull = len(hull_pts)
	
	# Estimate edge scale from kNN
	sample_size = min(5000, len(xy))
	sample_idx = np.random.choice(len(xy), sample_size, replace=False)
	nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(xy[sample_idx])
	dists, _ = nbrs.kneighbors(xy[sample_idx])
	typical_dist = np.mean(dists[:, 1])
	max_edge_len = typical_dist / concave_factor
	
	for i in range(n_hull):
		p1, p2 = hull_pts[i], hull_pts[(i + 1) % n_hull]
		z1, z2 = hull_z[i], hull_z[(i + 1) % n_hull]
		edge_len = np.linalg.norm(p2 - p1)
		
		refined_pts.append(p1)
		
		# Refine long edges with high-Z interior points
		if edge_len > max_edge_len:
			midpoint = (p1 + p2) / 2
			edge_vec = p2 - p1
			edge_normal = np.array([-edge_vec[1], edge_vec[0]])
			edge_normal = edge_normal / (np.linalg.norm(edge_normal) + 1e-12)
			
			# Find candidates near edge
			to_mid = xy - midpoint
			along_edge = np.dot(to_mid, edge_vec) / np.linalg.norm(edge_vec)
			across_edge = np.dot(to_mid, edge_normal)
			
			band_mask = (np.abs(along_edge) < edge_len * 0.6) & (np.abs(across_edge) < edge_len * 0.4)
			if np.any(band_mask):
				candidates_xy = xy[band_mask]
				candidates_z = z[band_mask]
				
				# Weight by Z elevation and inward distance
				for j, (cxy, cz) in enumerate(zip(candidates_xy, candidates_z)):
					inward_dist = np.dot(cxy - midpoint, edge_normal)
					z_weight = (cz - np.min([z1, z2])) / (np.max([z1, z2]) - np.min([z1, z2]) + 1e-6)
					
					# Prefer high-Z points that pull edge inward
					if inward_dist > typical_dist * 0.2 and z_weight > 0.5:
						refined_pts.append(cxy)
						break  # One point per edge
	
	if len(refined_pts) < 3:
		return geom.Polygon(hull_pts)
	
	return geom.Polygon(refined_pts)


def smooth_outline(poly: geom.Polygon, target_points: int = 1000, smoothing: float = 0.1) -> geom.Polygon:
	"""
	Smooth and densify outline using spline interpolation.
	
	Args:
		poly: Input polygon
		target_points: Target number of boundary points
		smoothing: Spline smoothing factor (0=interpolation, >0=smoothing)
	
	Returns:
		Smoothed polygon with target_points boundary points
	"""
	coords = np.array(poly.exterior.coords[:-1])  # Remove duplicate last point
	
	if len(coords) < 4:
		return poly
	
	# Parametric spline fitting
	try:
		# Periodic spline (closed curve)
		tck, u = splprep([coords[:, 0], coords[:, 1]], s=smoothing, per=True)
		
		# Evaluate at target_points
		u_new = np.linspace(0, 1, target_points, endpoint=False)
		smooth_coords = np.column_stack(splev(u_new, tck))
		
		return geom.Polygon(smooth_coords)
		
	except Exception as e:
		print(f"Warning: Spline smoothing failed ({e}), using original")
		return poly


def save_outline(out_path: str, poly: geom.Polygon):
	"""Save outline as CSV coordinates."""
	coords = np.asarray(poly.exterior.coords)
	np.savetxt(out_path, coords, delimiter=",", header="x,y", comments="")


def save_preview(preview_path: str, points_xy: np.ndarray, poly: geom.Polygon, edge_points: np.ndarray = None):
	"""Save preview image with points and outline."""
	if not _HAS_MPL:
		return
	
	plt.figure(figsize=(12, 8))
	
	# All points (light)
	if len(points_xy) > 100000:
		idx = np.random.choice(len(points_xy), 100000, replace=False)
		sample = points_xy[idx]
	else:
		sample = points_xy
	plt.scatter(sample[:, 0], sample[:, 1], s=0.1, c="lightblue", alpha=0.3, label="All points")
	
	# Edge points (darker)
	if edge_points is not None:
		edge_xy = edge_points[:, :2]
		if len(edge_xy) > 20000:
			idx = np.random.choice(len(edge_xy), 20000, replace=False)
			edge_sample = edge_xy[idx]
		else:
			edge_sample = edge_xy
		plt.scatter(edge_sample[:, 0], edge_sample[:, 1], s=1.0, c="blue", alpha=0.6, label="Edge points")
	
	# Outline
	coords = np.asarray(poly.exterior.coords)
	plt.plot(coords[:, 0], coords[:, 1], "-r", linewidth=2, label="Outline")
	
	plt.axis("equal")
	plt.legend()
	plt.title(f"Outline: {len(coords)} points, Area: {poly.area:.0f}")
	plt.tight_layout()
	plt.savefig(preview_path, dpi=200)
	plt.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="Extract improved insole outline using Z-aware edge detection.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to isolated point cloud (PLY/PCD).")
	parser.add_argument("--z_percentile", type=float, default=85.0, help="Z percentile for edge detection (default: 85).")
	parser.add_argument("--concave", type=float, default=0.2, help="Concave factor (0=convex, 0.3=very concave).")
	parser.add_argument("--smooth_points", type=int, default=1000, help="Target points in smoothed outline.")
	parser.add_argument("--smoothing", type=float, default=0.1, help="Spline smoothing factor.")
	parser.add_argument("--out", "-o", type=str, default=None, help="Output CSV path for outline coordinates.")
	parser.add_argument("--preview", type=str, default=None, help="Optional PNG path for preview.")
	args = parser.parse_args()

	in_path = args.path
	if not os.path.isabs(in_path):
		in_path = os.path.join(os.getcwd(), in_path)
	if not os.path.isfile(in_path):
		print(f"File not found: {in_path}", file=sys.stderr)
		return 1

	# Load and extract edge points
	pcd = load_pcd(in_path)
	points = np.asarray(pcd.points)
	edge_points = extract_edge_points(points, z_percentile=args.z_percentile)
	
	# Compute outline from edge points
	poly = compute_z_aware_outline(edge_points, concave_factor=args.concave)
	
	# Smooth and densify
	poly_smooth = smooth_outline(poly, target_points=args.smooth_points, smoothing=args.smoothing)
	
	# Save
	out_path = args.out or os.path.splitext(in_path)[0] + "_outline_improved.csv"
	save_outline(out_path, poly_smooth)
	
	print(f"Saved improved outline: {out_path}")
	print(f"  Original: {len(poly.exterior.coords)} points, Area: {poly.area:.0f}")
	print(f"  Smoothed: {len(poly_smooth.exterior.coords)} points, Area: {poly_smooth.area:.0f}")
	
	if args.preview:
		save_preview(args.preview, points[:, :2], poly_smooth, edge_points)
		print(f"Saved preview: {args.preview}")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())


