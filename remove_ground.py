"""
remove_ground.py

Isolate an insole from a scan by removing the ground plane and optional residuals.
Methods:
- safe (default): plane filter + DBSCAN largest cluster
- plane_only: plane filter only
- plane_surface: plane filter + kNN surface residual filter (thread-limited)
"""
import argparse
import os
import sys
from typing import Tuple

import numpy as np
import open3d as o3d
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN


POINT_CLOUD_EXTS = {
	".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"
}
MESH_EXTS = {
	".stl", ".obj", ".off", ".gltf", ".glb", ".fbx", ".dae"
}


def splitext_with_suffix(path: str, suffix: str) -> str:
	"""Return path with suffix inserted before extension."""
	root, ext = os.path.splitext(path)
	return f"{root}{suffix}{ext}"


def load_scan(path: str):
	"""Load scan as Open3D point cloud or mesh based on extension."""
	ext = os.path.splitext(path)[1].lower()
	if ext in POINT_CLOUD_EXTS:
		pcd = o3d.io.read_point_cloud(path)
		if pcd.is_empty():
			raise ValueError(f"Loaded point cloud is empty: {path}")
		return ("pcd", pcd)
	elif ext in MESH_EXTS:
		mesh = o3d.io.read_triangle_mesh(path)
		if mesh.is_empty():
			raise ValueError(f"Loaded mesh is empty: {path}")
		return ("mesh", mesh)
	else:
		raise ValueError(
			f"Unsupported file extension '{ext}'.\n"
			f"Supported point clouds: {sorted(POINT_CLOUD_EXTS)}\n"
			f"Supported meshes: {sorted(MESH_EXTS)}"
		)


def segment_ground_from_pcd(
	pcd: o3d.geometry.PointCloud,
	distance_threshold: float,
	ransac_n: int,
	num_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return plane coefficients (a,b,c,d) and inlier indices for the dominant plane."""
	plane_model, inliers = pcd.segment_plane(
		distance_threshold=distance_threshold,
		ransac_n=ransac_n,
		num_iterations=num_iterations,
	)
	return np.asarray(plane_model, dtype=float), np.asarray(inliers, dtype=int)


def remove_ground_from_mesh(
	mesh: o3d.geometry.TriangleMesh,
	plane_abcd: np.ndarray,
	distance_threshold: float,
) -> o3d.geometry.TriangleMesh:
	"""Remove triangles whose vertices lie within distance_threshold of the plane."""
	vertices = np.asarray(mesh.vertices)
	triangles = np.asarray(mesh.triangles)
	a, b, c, d = plane_abcd
	norm = np.sqrt(a * a + b * b + c * c)
	if norm == 0:
		raise ValueError("Invalid plane normal (zero length)")
	dists = np.abs((a * vertices[:, 0] + b * vertices[:, 1] + c * vertices[:, 2] + d) / norm)
	tri_dists = dists[triangles]  # (T, 3)
	keep_mask = (tri_dists > distance_threshold).all(axis=1)
	triangles_kept = triangles[keep_mask]
	if triangles_kept.size == 0:
		raise ValueError("All triangles were removed by ground filtering; check threshold.")
	# Reindex vertices to remove unused ones
	used_verts = np.unique(triangles_kept.reshape(-1))
	index_map = -np.ones(vertices.shape[0], dtype=np.int64)
	index_map[used_verts] = np.arange(used_verts.shape[0])
	new_vertices = vertices[used_verts]
	new_triangles = index_map[triangles_kept]
	isolated = o3d.geometry.TriangleMesh(
		vertices=o3d.utility.Vector3dVector(new_vertices),
		triangles=o3d.utility.Vector3iVector(new_triangles),
	)
	isolated.compute_vertex_normals()
	return isolated


def distance_to_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
	"""Absolute point-to-plane distance for each point (Nx3)."""
	a, b, c, d = plane
	norm = np.sqrt(a * a + b * b + c * c) + 1e-12
	return np.abs((a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm)


def rotation_to_align_plane_normal(plane: np.ndarray) -> np.ndarray:
	"""Rotation matrix to align plane normal to +Z using scipy."""
	from scipy.spatial.transform import Rotation as R
	n = plane[:3]
	n = n / (np.linalg.norm(n) + 1e-12)
	target = np.array([0.0, 0.0, 1.0])
	
	# If already aligned, return identity
	if np.abs(np.dot(n, target) - 1.0) < 1e-6:
		return np.eye(3)
	
	# Use scipy for robust rotation
	v = np.cross(n, target)
	s = np.linalg.norm(v)
	if s < 1e-8:  # Parallel or anti-parallel
		if np.dot(n, target) > 0:
			return np.eye(3)
		else:
			# 180 degree rotation around any perpendicular axis
			return R.from_rotvec([np.pi, 0, 0]).as_matrix()
	
	# Rotation around cross product axis
	angle = np.arcsin(s)
	if np.dot(n, target) < 0:
		angle = np.pi - angle
	
	axis = v / s
	return R.from_rotvec(axis * angle).as_matrix()


def plane_only(points: np.ndarray, plane: np.ndarray, dist: float) -> np.ndarray:
	"""Keep points farther than dist from the plane (i.e., not ground)."""
	d = distance_to_plane(points, plane)
	return points[d > dist]


def dbscan_largest_cluster(points: np.ndarray, eps: float = 3.0, min_samples: int = 500) -> np.ndarray:
	"""Return points from the largest DBSCAN cluster (non-noise)."""
	labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)
	mask = labels >= 0
	if not np.any(mask):
		return points
	labs, counts = np.unique(labels[mask], return_counts=True)
	keep_label = labs[np.argmax(counts)]
	return points[labels == keep_label]


def plane_surface(points: np.ndarray, plane: np.ndarray, base_dist: float, grid_res: float, residual_thresh: float) -> np.ndarray:
	"""Plane removal then kNN surface regression in plane-aligned frame; keep low-residual points."""
	pts = plane_only(points, plane, base_dist)
	if pts.shape[0] < 1000:
		return pts
	Rmat = rotation_to_align_plane_normal(plane)
	pts_local = (Rmat @ pts[:, :3].T).T
	xy = pts_local[:, :2]
	z = pts_local[:, 2]
	max_train = 200000
	if xy.shape[0] > max_train:
		idx = np.random.choice(xy.shape[0], max_train, replace=False)
		xy_train = xy[idx]
		z_train = z[idx]
	else:
		xy_train = xy
		z_train = z
	# Limit threads to reduce segfault risk on macOS BLAS/OpenMP stacks
	os.environ.setdefault("OMP_NUM_THREADS", "1")
	os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
	os.environ.setdefault("MKL_NUM_THREADS", "1")
	knn = KNeighborsRegressor(n_neighbors=20, weights="distance", algorithm="kd_tree")
	knn.fit(xy_train, z_train)
	z_pred = knn.predict(xy)
	res = np.abs(z - z_pred)
	mask = res <= residual_thresh
	kept_local = pts_local[mask]
	kept = (Rmat.T @ kept_local.T).T
	return kept


def main() -> int:
	"""CLI entrypoint."""
	parser = argparse.ArgumentParser(description="Remove ground plane from a scan (point cloud or mesh).")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to input scan file.")
	parser.add_argument("--method", type=str, default="safe", choices=["safe", "plane_only", "plane_surface"], help="Isolation method (default: safe).")
	parser.add_argument("--dist", type=float, default=0.5, help="Base plane distance threshold (default: 0.5).")
	parser.add_argument("--grid_res", type=float, default=2.0, help="(Unused in KNN mode) kept for CLI compatibility.")
	parser.add_argument("--residual_thresh", type=float, default=5.0, help="Residual tolerance (units) for surface fit.")
	parser.add_argument("--ransac_n", type=int, default=3, help="RANSAC n (default: 3).")
	parser.add_argument("--iters", type=int, default=5000, help="RANSAC iterations (default: 5000).")
	parser.add_argument("--output", "-o", type=str, default=None, help="Optional output path; defaults to <name>_isolated<ext>.")
	args = parser.parse_args()

	in_path = args.path
	if not os.path.isabs(in_path):
		in_path = os.path.join(os.getcwd(), in_path)
	if not os.path.isfile(in_path):
		print(f"File not found: {in_path}", file=sys.stderr)
		return 1

	kind, geometry = load_scan(in_path)
	print(f"Loaded {kind}: {in_path}")

	# Build a point cloud for plane detection
	if kind == "pcd":
		pcd = geometry
	else:
		pcd = geometry.sample_points_poisson_disk(number_of_points=max(200000, len(geometry.vertices)))

	plane_abcd, inliers = segment_ground_from_pcd(
		pcd,
		distance_threshold=args.dist,
		ransac_n=args.ransac_n,
		num_iterations=args.iters,
	)
	print(f"Plane model: a={plane_abcd[0]:.6f}, b={plane_abcd[1]:.6f}, c={plane_abcd[2]:.6f}, d={plane_abcd[3]:.6f}")
	print(f"Inliers (ground) count: {len(inliers)} / {np.asarray(pcd.points).shape[0]}")

	out_path = args.output or splitext_with_suffix(in_path, "_isolated")

	if kind == "pcd":
		pts = np.asarray(pcd.points)
		if args.method == "plane_only":
			kept = plane_only(pts, plane_abcd, args.dist)
		elif args.method == "plane_surface":
			kept = plane_surface(pts, plane_abcd, base_dist=args.dist, grid_res=args.grid_res, residual_thresh=args.residual_thresh)
		else:  # safe
			kept = plane_only(pts, plane_abcd, args.dist)
			kept = dbscan_largest_cluster(kept, eps=3.0, min_samples=500)
		if kept.shape[0] == 0:
			raise ValueError("Isolated point cloud is empty; adjust thresholds.")
		
		# Simple Z alignment: translate so minimum Z becomes 0
		z_min = kept[:, 2].min()
		kept[:, 2] -= z_min
		
		print(f"Aligned insole Z range: [{kept[:, 2].min():.1f}, {kept[:, 2].max():.1f}]")
		
		isolated_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kept[:, :3]))
		print(f"Saving isolated point cloud: {out_path}")
		ok = o3d.io.write_point_cloud(out_path, isolated_pcd)
		if not ok:
			raise RuntimeError(f"Failed to save: {out_path}")
	else:
		# For meshes, apply triangle plane distance removal as a baseline
		isolated_mesh = remove_ground_from_mesh(geometry, plane_abcd, distance_threshold=args.dist)
		print(f"Saving isolated mesh: {out_path}")
		ok = o3d.io.write_triangle_mesh(out_path, isolated_mesh)
		if not ok:
			raise RuntimeError(f"Failed to save: {out_path}")

	print("Done.")
	return 0


if __name__ == "__main__":
	sys.exit(main()) 