"""
generate_heightmap.py

Generate a heightmap from an isolated point cloud constrained within an outline.
Uses kNN regression to interpolate Z values over a uniform XY grid.
Outputs: NPY arrays (GX, GY, Z), PNG preview, and gridded PLY.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from shapely.geometry import Polygon, Point
try:
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False
from sklearn.neighbors import KNeighborsRegressor


def load_pcd(path: str) -> o3d.geometry.PointCloud:
	"""Load point cloud from file."""
	pcd = o3d.io.read_point_cloud(path)
	if pcd.is_empty():
		raise ValueError(f"Empty point cloud: {path}")
	return pcd


def load_outline_csv(path: str) -> Polygon:
	"""Load outline polygon from CSV coordinates."""
	coords = np.loadtxt(path, delimiter=",", skiprows=1)
	return Polygon(coords)


def mask_points_in_polygon(xy: np.ndarray, poly: Polygon) -> np.ndarray:
	"""Return boolean mask for points inside polygon using prepared geometry."""
	from shapely.prepared import prep
	pp = prep(poly)
	mask = np.fromiter((pp.contains(Point(p[0], p[1])) for p in xy), dtype=bool, count=xy.shape[0])
	return mask


def generate_grid(poly: Polygon, resolution: float) -> tuple:
	"""Generate uniform XY grid covering polygon bounds, pre-masked to outline."""
	minx, miny, maxx, maxy = poly.bounds
	nx = max(2, int((maxx - minx) / resolution))
	ny = max(2, int((maxy - miny) / resolution))
	xv = np.linspace(minx, maxx, nx)
	yv = np.linspace(miny, maxy, ny)
	GX, GY = np.meshgrid(xv, yv)
	
	# Pre-mask grid points outside polygon
	from shapely.prepared import prep
	pp = prep(poly)
	grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
	inside_mask = np.array([pp.contains(Point(p[0], p[1])) for p in grid_pts])
	
	# Set outside points to NaN in coordinate grids
	GX_masked = GX.copy()
	GY_masked = GY.copy()
	outside_2d = ~inside_mask.reshape(GX.shape)
	GX_masked[outside_2d] = np.nan
	GY_masked[outside_2d] = np.nan
	
	return GX_masked, GY_masked


def rasterize_heightmap(points: np.ndarray, poly: Polygon, resolution: float, knn: int = 10) -> tuple:
	"""
	Rasterize point cloud Z values to a uniform grid within polygon outline.
	
	Args:
		points: 3D points (N, 3)
		poly: Shapely polygon outline
		resolution: Grid spacing
		knn: Number of neighbors for regression
	
	Returns:
		(GX, GY, Z_mask) where Z_mask has NaN outside polygon
	"""
	xy = points[:, :2]
	z = points[:, 2]
	# Fit KNN on all points inside a buffered polygon to avoid edge issues
	buffer_poly = poly.buffer(resolution * 2.0)
	mask = mask_points_in_polygon(xy, buffer_poly)
	xy_fit = xy[mask]
	z_fit = z[mask]
	if xy_fit.shape[0] < 1000:
		raise ValueError("Not enough points inside outline to fit heightmap.")
	knn_model = KNeighborsRegressor(n_neighbors=knn, weights="distance", algorithm="kd_tree")
	knn_model.fit(xy_fit, z_fit)
	GX, GY = generate_grid(poly, resolution)
	
	# Only predict for valid (non-NaN) grid points
	valid_mask = np.isfinite(GX) & np.isfinite(GY)
	if not np.any(valid_mask):
		raise ValueError("No valid grid points inside polygon")
	
	valid_indices = np.where(valid_mask)
	valid_grid_pts = np.column_stack([GX[valid_indices], GY[valid_indices]])
	
	# Predict Z only for valid points
	Z = np.full_like(GX, np.nan)
	Z[valid_indices] = knn_model.predict(valid_grid_pts)
	
	return GX, GY, Z


def save_heightmap(base: str, GX: np.ndarray, GY: np.ndarray, Z: np.ndarray):
	"""Save heightmap as NPY arrays and PNG preview."""
	np.save(base + "_GX.npy", GX)
	np.save(base + "_GY.npy", GY)
	np.save(base + "_Z.npy", Z)
	if _HAS_MPL:
		plt.figure(figsize=(6, 10))
		plt.imshow(np.flipud(Z), cmap="viridis", interpolation="nearest")
		plt.colorbar(label="Z")
		plt.title("Heightmap")
		plt.tight_layout()
		plt.savefig(base + "_heightmap.png", dpi=200)
		plt.close()


def save_gridded_ply(path: str, GX: np.ndarray, GY: np.ndarray, Z: np.ndarray):
	"""Save finite grid points as PLY point cloud."""
	mask = np.isfinite(Z)
	pts = np.column_stack([GX[mask], GY[mask], Z[mask]])
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
	o3d.io.write_point_cloud(path, pcd)


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate heightmap from isolated point cloud within outline.")
	parser.add_argument("--pcd", required=True, help="Path to isolated point cloud (PLY/PCD).")
	parser.add_argument("--outline", required=True, help="Path to outline CSV.")
	parser.add_argument("--res", type=float, default=1.5, help="Grid resolution (units).")
	parser.add_argument("--knn", type=int, default=12, help="KNN neighbors for height regression.")
	parser.add_argument("--outbase", type=str, default=None, help="Base path for outputs (without extension).")
	args = parser.parse_args()

	pcd = load_pcd(args.pcd)
	poly = load_outline_csv(args.outline)
	points = np.asarray(pcd.points)

	GX, GY, Z = rasterize_heightmap(points, poly, resolution=args.res, knn=args.knn)

	base = args.outbase or os.path.splitext(args.pcd)[0]
	save_heightmap(base, GX, GY, Z)
	save_gridded_ply(base + "_grid.ply", GX, GY, Z)
	print(f"Saved heightmap NPY/PNG and gridded PLY with base: {base}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
