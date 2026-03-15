import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata


POINT_CLOUD_EXTS = {
	".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"
}
MESH_EXTS = {
	".stl", ".obj", ".off", ".gltf", ".glb", ".fbx", ".dae"
}


@dataclass
class Metrics:
	method: str
	points: int
	ground_residual_fraction: float
	largest_cluster_ratio: float
	avg_surface_residual: float


def ensure_pcd(path: str):
	ext = os.path.splitext(path)[1].lower()
	if ext in POINT_CLOUD_EXTS:
		pcd = o3d.io.read_point_cloud(path)
		if pcd.is_empty():
			raise ValueError(f"Empty point cloud: {path}")
		return pcd
	elif ext in MESH_EXTS:
		mesh = o3d.io.read_triangle_mesh(path)
		if mesh.is_empty():
			raise ValueError(f"Empty mesh: {path}")
		# Sample sufficient points
		pcd = mesh.sample_points_poisson_disk(number_of_points=max(400000, len(mesh.vertices)))
		return pcd
	else:
		raise ValueError(f"Unsupported extension: {ext}")


def segment_plane(pcd: o3d.geometry.PointCloud, dist=0.5, ransac_n=3, iters=5000) -> Tuple[np.ndarray, np.ndarray]:
	plane_model, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=ransac_n, num_iterations=iters)
	return np.asarray(plane_model), np.asarray(inliers, dtype=int)


def distance_to_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
	a, b, c, d = plane
	norm = np.sqrt(a * a + b * b + c * c)
	return np.abs((a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / (norm + 1e-12))


def rotation_to_align_plane_normal(plane: np.ndarray) -> np.ndarray:
	# Rotate plane normal to +Z
	n = plane[:3]
	n = n / (np.linalg.norm(n) + 1e-12)
	target = np.array([0.0, 0.0, 1.0])
	v = np.cross(n, target)
	s = np.linalg.norm(v)
	c = np.dot(n, target)
	if s < 1e-8:
		return np.eye(3)
	vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	Rmat = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
	return Rmat


def dbscan_largest_cluster(points: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, float]:
	labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)
	if (labels >= 0).sum() == 0:
		return np.array([], dtype=int), 0.0
	# Largest non-negative label
	unique, counts = np.unique(labels[labels >= 0], return_counts=True)
	largest_label = unique[np.argmax(counts)]
	idx = np.nonzero(labels == largest_label)[0]
	return idx, counts.max() / points.shape[0]


def compute_normals(pcd: o3d.geometry.PointCloud, radius: float = 3.0, max_nn: int = 50):
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
	pcd.normalize_normals()


def plane_only(points: np.ndarray, plane: np.ndarray, dist: float) -> np.ndarray:
	d = distance_to_plane(points, plane)
	return points[d > dist]


def plane_adaptive(points: np.ndarray, plane: np.ndarray, base_dist: float, seed_eps: float = 5.0, seed_min: int = 500):
	# Initial filter
	kept = plane_only(points, plane, base_dist / 2.0)
	# Seed via DBSCAN on kept points (above plane)
	seed_idx, _ = dbscan_largest_cluster(kept, eps=seed_eps, min_samples=seed_min)
	if seed_idx.size == 0:
		return plane_only(points, plane, base_dist)  # fallback
	seed = kept[seed_idx]
	# Nearest distance to seed cluster
	nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(seed[:, :3])
	dist_to_seed, _ = nbrs.kneighbors(points[:, :3])
	dist_to_seed = dist_to_seed.reshape(-1)
	# Adaptive threshold
	maxd = np.percentile(dist_to_seed, 95)
	th = base_dist + 0.5 * base_dist * (dist_to_seed / (maxd + 1e-6))
	dplane = distance_to_plane(points, plane)
	mask = dplane > th
	return points[mask]


def plane_surface(points: np.ndarray, plane: np.ndarray, base_dist: float, grid_res: float = 2.0, residual_thresh: float = 5.0):
	# Remove ground first with base_dist
	pts = plane_only(points, plane, base_dist)
	if pts.shape[0] < 1000:
		return pts
	# Transform so plane ~ XY
	Rmat = rotation_to_align_plane_normal(plane)
	pts_local = (Rmat @ pts[:, :3].T).T
	# Downsample for gridding
	if pts_local.shape[0] > 300000:
		idx = np.random.choice(pts_local.shape[0], 300000, replace=False)
		pts_ds = pts_local[idx]
	else:
		pts_ds = pts_local
	xy = pts_ds[:, :2]
	z = pts_ds[:, 2]
	xmin, ymin = xy.min(axis=0)
	xmax, ymax = xy.max(axis=0)
	nx = max(10, int((xmax - xmin) / grid_res))
	ny = max(10, int((ymax - ymin) / grid_res))
	gx = np.linspace(xmin, xmax, nx)
	gy = np.linspace(ymin, ymax, ny)
	GX, GY = np.meshgrid(gx, gy)
	GZ = griddata(xy, z, (GX, GY), method="linear")
	# Fill NaNs with nearest
	if np.isnan(GZ).any():
		GZ_nn = griddata(xy, z, (GX, GY), method="nearest")
		GZ = np.where(np.isnan(GZ), GZ_nn, GZ)
	# Evaluate residuals of all kept points
	xy_all = pts_local[:, :2]
	z_all = pts_local[:, 2]
	ix = np.clip(((xy_all[:, 0] - xmin) / (xmax - xmin + 1e-9) * (nx - 1)).astype(int), 0, nx - 1)
	iy = np.clip(((xy_all[:, 1] - ymin) / (ymax - ymin + 1e-9) * (ny - 1)).astype(int), 0, ny - 1)
	z_pred = GZ[iy, ix]
	res = np.abs(z_all - z_pred)
	mask = res <= residual_thresh
	return pts[mask]


def dbscan_post(points: np.ndarray, eps: float = 3.0, min_samples: int = 500) -> Tuple[np.ndarray, float]:
	idx, ratio = dbscan_largest_cluster(points, eps=eps, min_samples=min_samples)
	if idx.size == 0:
		return points, 0.0
	return points[idx], ratio


def compute_metrics(points: np.ndarray, plane: np.ndarray, eval_residuals: bool = True) -> Metrics:
	residual_fraction = float((distance_to_plane(points, plane) <= 0.5).mean())
	idx, ratio = dbscan_largest_cluster(points, eps=3.0, min_samples=500)
	avg_res = -1.0
	if eval_residuals and points.shape[0] > 2000:
		# Approximate local surface residual via kNN smoothing in XY after plane align
		Rmat = rotation_to_align_plane_normal(plane)
		p_local = (Rmat @ points[:, :3].T).T
		xy = p_local[:, :2]
		z = p_local[:, 2]
		nbrs = NearestNeighbors(n_neighbors=10, algorithm="kd_tree").fit(xy)
		idxs = nbrs.kneighbors(xy, return_distance=False)
		z_smooth = z[idxs].mean(axis=1)
		avg_res = float(np.mean(np.abs(z - z_smooth)))
	return Metrics(method="", points=int(points.shape[0]), ground_residual_fraction=residual_fraction, largest_cluster_ratio=float(ratio), avg_surface_residual=avg_res)


def save_pcd(path: str, points: np.ndarray):
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
	o3d.io.write_point_cloud(path, pcd)


def run_methods(in_path: str, out_dir: str, base_dist: float) -> Dict[str, Metrics]:
	pcd = ensure_pcd(in_path)
	pts = np.asarray(pcd.points)
	plane, inliers = segment_plane(pcd, dist=base_dist)
	metrics: Dict[str, Metrics] = {}
	os.makedirs(out_dir, exist_ok=True)

	# Method 1: plane_only
	m1_pts = plane_only(pts, plane, dist=base_dist)
	m1_path = os.path.join(out_dir, "insole_iso_plane_only.ply")
	save_pcd(m1_path, m1_pts)
	m = compute_metrics(m1_pts, plane)
	m.method = "plane_only"
	metrics[m.method] = m

	# Method 2: plane_adaptive
	m2_pts = plane_adaptive(pts, plane, base_dist=base_dist, seed_eps=5.0, seed_min=500)
	m2_pts, _ = dbscan_post(m2_pts, eps=3.0, min_samples=500)
	m2_path = os.path.join(out_dir, "insole_iso_plane_adaptive.ply")
	save_pcd(m2_path, m2_pts)
	m = compute_metrics(m2_pts, plane)
	m.method = "plane_adaptive"
	metrics[m.method] = m

	# Method 3: plane_surface
	m3_pts = plane_surface(pts, plane, base_dist=base_dist, grid_res=2.0, residual_thresh=5.0)
	m3_pts, _ = dbscan_post(m3_pts, eps=3.0, min_samples=500)
	m3_path = os.path.join(out_dir, "insole_iso_plane_surface.ply")
	save_pcd(m3_path, m3_pts)
	m = compute_metrics(m3_pts, plane)
	m.method = "plane_surface"
	metrics[m.method] = m

	# Method 4: plane_only + dbscan_post
	m4_pts = m1_pts
	m4_pts, _ = dbscan_post(m4_pts, eps=3.0, min_samples=500)
	m4_path = os.path.join(out_dir, "insole_iso_dbscan_post.ply")
	save_pcd(m4_path, m4_pts)
	m = compute_metrics(m4_pts, plane)
	m.method = "plane_dbscan_post"
	metrics[m.method] = m

	return metrics


def main() -> int:
	parser = argparse.ArgumentParser(description="Isolate insole using multiple strategies and evaluate.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Input scan path (point cloud or mesh).")
	parser.add_argument("--out", "-o", type=str, default="outputs", help="Output directory for isolated results.")
	parser.add_argument("--dist", type=float, default=0.5, help="Base plane distance threshold (units of model).")
	args = parser.parse_args()

	in_path = args.path
	if not os.path.isabs(in_path):
		in_path = os.path.join(os.getcwd(), in_path)
	if not os.path.isfile(in_path):
		print(f"File not found: {in_path}", file=sys.stderr)
		return 1

	metrics = run_methods(in_path, args.out, base_dist=args.dist)
	print("\nMethod evaluation:")
	for name, m in metrics.items():
		print(f"- {name}: points={m.points}, ground_residual_fraction={m.ground_residual_fraction:.4f}, "
			f"largest_cluster_ratio={m.largest_cluster_ratio:.3f}, avg_surface_residual={m.avg_surface_residual:.3f}")

	# Pick best: minimize ground residual, maximize largest cluster ratio, maximize points; simple score
	best_name = None
	best_score = -1e18
	for name, m in metrics.items():
		score = -5.0 * m.ground_residual_fraction + 2.0 * m.largest_cluster_ratio + 1e-6 * m.points - 0.1 * max(0.0, m.avg_surface_residual)
		if score > best_score:
			best_score = score
			best_name = name
	print(f"\nBest method: {best_name} (score {best_score:.3f})")
	print(f"Outputs saved in: {os.path.abspath(args.out)}")
	return 0


if __name__ == "__main__":
	sys.exit(main()) 