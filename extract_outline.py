"""
extract_outline.py

Extract 2D insole outline from isolated point cloud using fast ConvexHull + concave refinement.
Optimized to avoid alphashape performance bottlenecks on large point sets.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import shapely.geometry as geom
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

from io_utils import load_point_cloud as load_pcd

try:
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False


def project_to_xy(points: np.ndarray) -> np.ndarray:
	return points[:, :2]


def compute_fast_outline(points_xy: np.ndarray, concave_factor: float = 0.15, rng: np.random.Generator | None = None) -> geom.Polygon:
	"""
	Fast outline extraction using ConvexHull + concave refinement.

	Args:
		points_xy: 2D points (N, 2)
		concave_factor: Controls concavity (0=convex, 0.3=very concave)
		rng: Optional numpy Generator for reproducible sampling

	Returns:
		Shapely Polygon of the outline
	"""
	if points_xy.shape[0] < 3:
		raise ValueError("Need at least 3 points for outline")

	if rng is None:
		rng = np.random.default_rng()

	# Start with convex hull
	hull = ConvexHull(points_xy)
	hull_pts = points_xy[hull.vertices]

	if concave_factor <= 0:
		return geom.Polygon(hull_pts)

	# Refine edges that are too long by adding interior points
	refined_pts = []
	n_hull = len(hull_pts)

	# Estimate typical edge length from kNN
	sample_size = min(10000, points_xy.shape[0])
	sample_idx = rng.choice(points_xy.shape[0], sample_size, replace=False)
	sample_pts = points_xy[sample_idx]
	nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(sample_pts)
	dists, _ = nbrs.kneighbors(sample_pts)
	typical_dist = np.mean(dists[:, 1])  # Skip self-distance
	max_edge_len = typical_dist / concave_factor
	
	for i in range(n_hull):
		p1 = hull_pts[i]
		p2 = hull_pts[(i + 1) % n_hull]
		edge_len = np.linalg.norm(p2 - p1)
		
		refined_pts.append(p1)
		
		# If edge is too long, find interior points to make it concave
		if edge_len > max_edge_len:
			# Find points near the edge midpoint
			midpoint = (p1 + p2) / 2
			edge_vec = p2 - p1
			edge_normal = np.array([-edge_vec[1], edge_vec[0]])  # Perpendicular
			edge_normal = edge_normal / (np.linalg.norm(edge_normal) + 1e-12)
			
			# Look for points within a band around the edge
			band_width = edge_len * 0.3
			to_mid = points_xy - midpoint
			along_edge = np.dot(to_mid, edge_vec) / np.linalg.norm(edge_vec)
			across_edge = np.dot(to_mid, edge_normal)
			
			# Points roughly between p1 and p2, and not too far perpendicular
			mask = (np.abs(along_edge) < edge_len * 0.6) & (np.abs(across_edge) < band_width)
			if np.any(mask):
				candidates = points_xy[mask]
				# Pick the point that creates the most "inward" bulge
				best_pt = None
				best_inward = -1e9
				for pt in candidates:
					# Measure how much this point pulls the edge inward
					inward_dist = np.dot(pt - midpoint, edge_normal)
					if inward_dist > best_inward:
						best_inward = inward_dist
						best_pt = pt
				if best_pt is not None and best_inward > typical_dist * 0.1:
					refined_pts.append(best_pt)
	
	if len(refined_pts) < 3:
		return geom.Polygon(hull_pts)
	
	return geom.Polygon(refined_pts)


def save_outline(out_path: str, poly: geom.Polygon):
	# Save as simple CSV of boundary coordinates
	coords = np.asarray(poly.exterior.coords)
	np.savetxt(out_path, coords, delimiter=",", header="x,y", comments="")


def save_preview(preview_path: str, points_xy: np.ndarray, poly: geom.Polygon, max_scatter: int = 50000, rng: np.random.Generator | None = None):
	if not _HAS_MPL:
		return
	if rng is None:
		rng = np.random.default_rng()
	plt.figure(figsize=(8, 12))
	# light scatter of subsample
	n = points_xy.shape[0]
	if n > max_scatter:
		idx = rng.choice(n, max_scatter, replace=False)
		sample = points_xy[idx]
	else:
		sample = points_xy
	plt.scatter(sample[:, 0], sample[:, 1], s=0.1, c="#1f77b4", alpha=0.3)
	coords = np.asarray(poly.exterior.coords)
	plt.plot(coords[:, 0], coords[:, 1], "-r", linewidth=1.5)
	plt.axis("equal")
	plt.tight_layout()
	plt.savefig(preview_path, dpi=200)
	plt.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="Extract 2D insole outline via alpha shape from isolated PLY.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to isolated point cloud (PLY/PCD).")
	parser.add_argument("--alpha", type=float, default=None, help="Alpha parameter for alphashape; if omitted, uses fast ConvexHull method.")
	parser.add_argument("--concave", type=float, default=0.15, help="Concave factor for fast method (0=convex, 0.3=very concave).")
	parser.add_argument("--out", "-o", type=str, default=None, help="Output CSV path for outline coordinates.")
	parser.add_argument("--preview", type=str, default=None, help="Optional PNG path to save an outline preview overlay.")
	parser.add_argument("--max_points", type=int, default=100000, help="Max points used for outline computation.")
	parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
	args = parser.parse_args()

	in_path = args.path
	if not os.path.isabs(in_path):
		in_path = os.path.join(os.getcwd(), in_path)
	if not os.path.isfile(in_path):
		print(f"File not found: {in_path}", file=sys.stderr)
		return 1

	rng = np.random.default_rng(args.seed)
	pcd = load_pcd(in_path)
	pts = np.asarray(pcd.points)
	xy = project_to_xy(pts)
	# Downsample XY for speed
	n = xy.shape[0]
	if n > args.max_points:
		idx = rng.choice(n, args.max_points, replace=False)
		xy_ds = xy[idx]
	else:
		xy_ds = xy

	# Use fast method if alpha not specified, otherwise fall back to alphashape
	if args.alpha is None:
		poly = compute_fast_outline(xy_ds, concave_factor=args.concave, rng=rng)
	else:
		import alphashape
		pts = [tuple(p) for p in xy_ds]
		alpha_shape = alphashape.alphashape(pts, args.alpha)
		if isinstance(alpha_shape, geom.MultiPolygon):
			alpha_shape = max(alpha_shape.geoms, key=lambda g: g.area)
		if not isinstance(alpha_shape, geom.Polygon):
			raise ValueError("Alpha shape did not produce a Polygon.")
		poly = alpha_shape
	
	out_path = args.out or os.path.splitext(in_path)[0] + "_outline.csv"
	save_outline(out_path, poly)
	print(f"Saved outline: {out_path} (area={poly.area:.2f}, points={len(poly.exterior.coords)})")

	if args.preview:
		save_preview(args.preview, xy_ds, poly, rng=rng)
		print(f"Saved preview: {args.preview}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
