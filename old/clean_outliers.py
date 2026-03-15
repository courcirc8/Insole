"""
clean_outliers.py

Remove outlier points far from the main insole body using clustering and statistical methods.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
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


def remove_statistical_outliers(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0):
	"""Remove statistical outliers using Open3D."""
	clean_pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
	return clean_pcd, len(inliers)


def remove_distant_clusters(points: np.ndarray, eps: float = 5.0, min_samples: int = 1000, keep_largest: bool = True):
	"""
	Remove distant point clusters using DBSCAN.
	
	Args:
		points: 3D points (N, 3)
		eps: DBSCAN epsilon (distance threshold)
		min_samples: Minimum samples per cluster
		keep_largest: If True, keep only largest cluster; if False, keep all clusters above min_samples
	
	Returns:
		Cleaned points array
	"""
	# Use XY for clustering (ignore Z variations within insole)
	xy = points[:, :2]
	
	# DBSCAN clustering
	labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(xy)
	
	# Analyze clusters
	unique_labels = np.unique(labels)
	cluster_info = []
	
	for label in unique_labels:
		if label == -1:  # Noise
			count = np.sum(labels == label)
			cluster_info.append((label, count, "noise"))
		else:
			count = np.sum(labels == label)
			cluster_info.append((label, count, "cluster"))
	
	print(f"DBSCAN found {len([c for c in cluster_info if c[2] == 'cluster'])} clusters:")
	for label, count, ctype in sorted(cluster_info, key=lambda x: x[1], reverse=True):
		if ctype == "cluster":
			print(f"  Cluster {label}: {count:,} points")
		else:
			print(f"  Noise: {count:,} points")
	
	if keep_largest:
		# Keep only the largest cluster
		cluster_sizes = [(label, count) for label, count, ctype in cluster_info if ctype == "cluster"]
		if not cluster_sizes:
			raise ValueError("No valid clusters found")
		
		largest_label = max(cluster_sizes, key=lambda x: x[1])[0]
		mask = labels == largest_label
		print(f"Keeping largest cluster {largest_label}: {np.sum(mask):,} points")
	else:
		# Keep all clusters (remove only noise)
		mask = labels >= 0
		print(f"Keeping all clusters: {np.sum(mask):,} points")
	
	return points[mask]


def remove_radius_outliers(pcd: o3d.geometry.PointCloud, nb_points: int = 16, radius: float = 3.0):
	"""Remove radius outliers using Open3D."""
	clean_pcd, inliers = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
	return clean_pcd, len(inliers)


def save_preview(preview_path: str, original_points: np.ndarray, cleaned_points: np.ndarray):
	"""Save before/after comparison."""
	if not _HAS_MPL:
		return
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
	
	# Original
	if len(original_points) > 50000:
		idx = np.random.choice(len(original_points), 50000, replace=False)
		sample_orig = original_points[idx]
	else:
		sample_orig = original_points
	
	ax1.scatter(sample_orig[:, 0], sample_orig[:, 1], s=0.1, c=sample_orig[:, 2], cmap="viridis", alpha=0.6)
	ax1.set_title(f"Original ({len(original_points):,} points)")
	ax1.set_xlabel("X")
	ax1.set_ylabel("Y")
	ax1.axis("equal")
	
	# Cleaned
	if len(cleaned_points) > 50000:
		idx = np.random.choice(len(cleaned_points), 50000, replace=False)
		sample_clean = cleaned_points[idx]
	else:
		sample_clean = cleaned_points
	
	ax2.scatter(sample_clean[:, 0], sample_clean[:, 1], s=0.1, c=sample_clean[:, 2], cmap="viridis", alpha=0.6)
	ax2.set_title(f"Cleaned ({len(cleaned_points):,} points)")
	ax2.set_xlabel("X")
	ax2.set_ylabel("Y")
	ax2.axis("equal")
	
	plt.tight_layout()
	plt.savefig(preview_path, dpi=150)
	plt.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="Remove outlier points from isolated point cloud.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to point cloud (PLY/PCD).")
	parser.add_argument("--method", default="dbscan", choices=["dbscan", "statistical", "radius", "combined"], help="Outlier removal method.")
	parser.add_argument("--eps", type=float, default=5.0, help="DBSCAN epsilon (distance threshold).")
	parser.add_argument("--min_samples", type=int, default=1000, help="DBSCAN minimum samples per cluster.")
	parser.add_argument("--nb_neighbors", type=int, default=20, help="Statistical outlier neighbors.")
	parser.add_argument("--std_ratio", type=float, default=2.0, help="Statistical outlier std ratio.")
	parser.add_argument("--radius", type=float, default=3.0, help="Radius outlier search radius.")
	parser.add_argument("--nb_points", type=int, default=16, help="Radius outlier min neighbors.")
	parser.add_argument("--output", "-o", type=str, default=None, help="Output path (default: input_cleaned.ply).")
	parser.add_argument("--preview", type=str, default=None, help="Save before/after comparison PNG.")
	args = parser.parse_args()

	in_path = args.path
	if not os.path.isabs(in_path):
		in_path = os.path.join(os.getcwd(), in_path)
	if not os.path.isfile(in_path):
		print(f"File not found: {in_path}", file=sys.stderr)
		return 1

	# Load point cloud
	pcd = load_pcd(in_path)
	original_points = np.asarray(pcd.points)
	print(f"Original: {len(original_points):,} points")

	# Apply cleaning method
	if args.method == "dbscan":
		cleaned_points = remove_distant_clusters(
			original_points, 
			eps=args.eps, 
			min_samples=args.min_samples, 
			keep_largest=True
		)
		cleaned_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cleaned_points))
		
	elif args.method == "statistical":
		cleaned_pcd, n_inliers = remove_statistical_outliers(pcd, args.nb_neighbors, args.std_ratio)
		cleaned_points = np.asarray(cleaned_pcd.points)
		print(f"Statistical outlier removal: kept {n_inliers:,} / {len(original_points):,} points")
		
	elif args.method == "radius":
		cleaned_pcd, n_inliers = remove_radius_outliers(pcd, args.nb_points, args.radius)
		cleaned_points = np.asarray(cleaned_pcd.points)
		print(f"Radius outlier removal: kept {n_inliers:,} / {len(original_points):,} points")
		
	elif args.method == "combined":
		# Multi-stage cleaning
		print("Stage 1: DBSCAN clustering...")
		stage1_points = remove_distant_clusters(original_points, eps=args.eps, min_samples=args.min_samples)
		stage1_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stage1_points))
		
		print("Stage 2: Statistical outlier removal...")
		cleaned_pcd, n_inliers = remove_statistical_outliers(stage1_pcd, args.nb_neighbors, args.std_ratio)
		cleaned_points = np.asarray(cleaned_pcd.points)
		print(f"Combined cleaning: {len(original_points):,} → {len(stage1_points):,} → {len(cleaned_points):,} points")

	# Save cleaned point cloud
	out_path = args.output or os.path.splitext(in_path)[0] + "_cleaned.ply"
	o3d.io.write_point_cloud(out_path, cleaned_pcd)
	print(f"Saved cleaned point cloud: {out_path}")
	print(f"Reduction: {len(original_points):,} → {len(cleaned_points):,} points ({len(cleaned_points)/len(original_points)*100:.1f}%)")

	# Save preview
	if args.preview:
		save_preview(args.preview, original_points, cleaned_points)
		print(f"Saved preview: {args.preview}")

	return 0


if __name__ == "__main__":
	sys.exit(main())


