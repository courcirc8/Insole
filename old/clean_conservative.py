"""
clean_conservative.py

Conservative cleaning that preserves insole structure while removing obvious artifacts.
Uses multiple gentle filters instead of aggressive surface fitting.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
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


def remove_z_floor_artifacts(points: np.ndarray, z_min_threshold: float = 2.0) -> np.ndarray:
	"""Remove obvious Z=0 floor artifacts."""
	mask = points[:, 2] >= z_min_threshold
	removed = len(points) - np.sum(mask)
	print(f"  Z-floor filter: removed {removed:,} points below Z={z_min_threshold}")
	return points[mask]


def remove_isolated_outliers(points: np.ndarray, radius: float = 5.0, min_neighbors: int = 10) -> np.ndarray:
	"""Remove isolated points with few neighbors."""
	xy = points[:, :2]
	tree = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(xy)
	neighbor_counts = tree.radius_neighbors(xy, return_distance=False)
	
	# Count neighbors for each point
	neighbor_counts = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])  # -1 to exclude self
	
	# Keep points with enough neighbors
	mask = neighbor_counts >= min_neighbors
	removed = len(points) - np.sum(mask)
	print(f"  Isolation filter: removed {removed:,} isolated points")
	return points[mask]


def remove_thickness_outliers(points: np.ndarray, thickness_percentile: float = 99.0) -> np.ndarray:
	"""Remove points with unrealistic thickness relative to local area."""
	xy = points[:, :2]
	z = points[:, 2]
	
	# For each point, estimate local surface height
	nbrs = NearestNeighbors(n_neighbors=15, algorithm='kd_tree').fit(xy)
	distances, indices = nbrs.kneighbors(xy)
	
	# Local max Z (approximate top surface)
	local_top_z = np.array([z[idx].max() for idx in indices])
	
	# Thickness from local top
	thickness = local_top_z - z
	
	# Remove extreme thickness outliers
	thickness_threshold = np.percentile(thickness, thickness_percentile)
	mask = thickness <= thickness_threshold
	removed = len(points) - np.sum(mask)
	print(f"  Thickness filter: removed {removed:,} points with thickness > {thickness_threshold:.1f}mm")
	return points[mask]


def method_conservative_cleaning(points: np.ndarray, 
                               z_min_threshold: float = 2.0,
                               isolation_radius: float = 5.0, 
                               min_neighbors: int = 10,
                               thickness_percentile: float = 99.0) -> np.ndarray:
	"""
	Conservative multi-stage cleaning that preserves insole structure.
	
	Args:
		points: 3D points (N, 3)
		z_min_threshold: Remove points below this Z
		isolation_radius: Radius for isolation detection
		min_neighbors: Minimum neighbors to not be considered isolated
		thickness_percentile: Percentile threshold for thickness outliers
	
	Returns:
		Conservatively cleaned points
	"""
	print(f"Conservative cleaning with {len(points):,} input points")
	
	# Stage 1: Remove obvious floor artifacts
	stage1 = remove_z_floor_artifacts(points, z_min_threshold)
	
	# Stage 2: Remove isolated outliers
	stage2 = remove_isolated_outliers(stage1, isolation_radius, min_neighbors)
	
	# Stage 3: Remove extreme thickness outliers
	stage3 = remove_thickness_outliers(stage2, thickness_percentile)
	
	reduction = (len(points) - len(stage3)) / len(points) * 100
	print(f"Total reduction: {len(points):,} → {len(stage3):,} ({reduction:.1f}% removed)")
	
	return stage3


def method_gentle_statistical(points: np.ndarray, std_multiplier: float = 3.0, nb_neighbors: int = 20) -> np.ndarray:
	"""Gentle statistical outlier removal."""
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	cleaned_pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_multiplier)
	cleaned_points = np.asarray(cleaned_pcd.points)
	removed = len(points) - len(cleaned_points)
	print(f"Gentle statistical: removed {removed:,} points ({removed/len(points)*100:.1f}%)")
	return cleaned_points


def save_comparison_plot(path: str, original: np.ndarray, methods_results: dict):
	"""Save before/after comparison with Z histograms."""
	if not _HAS_MPL:
		return
	
	n_methods = len(methods_results)
	fig, axes = plt.subplots(3, n_methods + 1, figsize=(5 * (n_methods + 1), 12))
	
	def plot_method(col, name, pts, color='blue'):
		# Z histogram
		axes[0, col].hist(pts[:, 2], bins=50, alpha=0.7, color=color, edgecolor='black')
		axes[0, col].set_title(f'{name}\n{len(pts):,} points')
		axes[0, col].set_xlabel('Z Height')
		axes[0, col].set_ylabel('Count')
		axes[0, col].grid(True, alpha=0.3)
		
		# XY scatter colored by Z
		if len(pts) > 40000:
			idx = np.random.choice(len(pts), 40000, replace=False)
			sample = pts[idx]
		else:
			sample = pts
		
		scatter = axes[1, col].scatter(sample[:, 0], sample[:, 1], s=0.3, c=sample[:, 2], cmap='viridis', alpha=0.8)
		axes[1, col].set_title(f'{name} (XY view)')
		axes[1, col].axis('equal')
		
		# Z vs XY distance from center
		center = np.mean(sample[:, :2], axis=0)
		distances = np.linalg.norm(sample[:, :2] - center, axis=1)
		axes[2, col].scatter(distances, sample[:, 2], s=0.3, alpha=0.6, color=color)
		axes[2, col].set_xlabel('Distance from center')
		axes[2, col].set_ylabel('Z Height')
		axes[2, col].set_title(f'{name} (Z vs distance)')
		axes[2, col].grid(True, alpha=0.3)
	
	# Original
	plot_method(0, 'Original', original, 'gray')
	
	# Methods
	colors = ['blue', 'red', 'green', 'orange']
	for i, (name, cleaned) in enumerate(methods_results.items()):
		plot_method(i + 1, name, cleaned, colors[i % len(colors)])
	
	plt.tight_layout()
	plt.savefig(path, dpi=150, bbox_inches='tight')
	plt.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="Conservative thickness artifact cleaning.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to point cloud.")
	parser.add_argument("--z_min", type=float, default=2.0, help="Minimum Z threshold (remove floor artifacts).")
	parser.add_argument("--isolation_radius", type=float, default=5.0, help="Radius for isolation detection.")
	parser.add_argument("--min_neighbors", type=int, default=10, help="Min neighbors to not be isolated.")
	parser.add_argument("--thickness_percentile", type=float, default=99.0, help="Thickness outlier percentile.")
	parser.add_argument("--std_multiplier", type=float, default=3.0, help="Statistical outlier std multiplier.")
	parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
	parser.add_argument("--analysis", type=str, default=None, help="Save analysis plot.")
	args = parser.parse_args()

	in_path = args.path
	if not os.path.isabs(in_path):
		in_path = os.path.join(os.getcwd(), in_path)
	if not os.path.isfile(in_path):
		print(f"File not found: {in_path}", file=sys.stderr)
		return 1

	# Load points
	pcd = load_pcd(in_path)
	original_points = np.asarray(pcd.points)
	print(f"Original: {len(original_points):,} points, Z=[{original_points[:,2].min():.1f}, {original_points[:,2].max():.1f}]")

	# Test conservative methods
	methods_results = {}
	
	print(f"\n=== Conservative Multi-Stage ===")
	cleaned_conservative = method_conservative_cleaning(
		original_points,
		z_min_threshold=args.z_min,
		isolation_radius=args.isolation_radius,
		min_neighbors=args.min_neighbors,
		thickness_percentile=args.thickness_percentile
	)
	methods_results["Conservative"] = cleaned_conservative
	
	print(f"\n=== Gentle Statistical ===")
	cleaned_statistical = method_gentle_statistical(original_points, std_multiplier=args.std_multiplier)
	methods_results["Statistical"] = cleaned_statistical
	
	# Save results
	output_dir = args.output_dir or os.path.dirname(in_path)
	base_name = os.path.splitext(os.path.basename(in_path))[0]
	
	best_method = None
	best_score = -1
	
	for method_name, cleaned_points in methods_results.items():
		safe_name = method_name.lower().replace(" ", "_")
		out_path = os.path.join(output_dir, f"{base_name}_{safe_name}.ply")
		cleaned_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cleaned_points))
		o3d.io.write_point_cloud(out_path, cleaned_pcd)
		
		# Score: balance artifact removal with data preservation
		z_min = cleaned_points[:, 2].min()
		z_span = cleaned_points[:, 2].max() - z_min
		point_ratio = len(cleaned_points) / len(original_points)
		
		# Prefer: Z_min > 1 (removes artifacts), high point ratio (preserves data), good Z span
		artifact_score = max(0, z_min - 1.0) * 5  # Bonus for removing Z=0 artifacts
		preservation_score = point_ratio * 10  # Bonus for keeping data
		span_score = min(z_span / 40.0, 1.0) * 2  # Bonus for good Z span (up to 40mm)
		
		score = artifact_score + preservation_score + span_score
		
		print(f"Saved {method_name}: {out_path}")
		print(f"  Points: {len(cleaned_points):,} ({point_ratio:.1%})")
		print(f"  Z range: [{z_min:.1f}, {cleaned_points[:,2].max():.1f}] (span: {z_span:.1f})")
		print(f"  Score: {score:.1f} (artifact={artifact_score:.1f}, preserve={preservation_score:.1f}, span={span_score:.1f})")
		
		if score > best_score:
			best_score = score
			best_method = method_name
	
	print(f"\n🏆 Recommended: {best_method} (score: {best_score:.1f})")
	
	# Analysis plot
	if args.analysis:
		save_comparison_plot(args.analysis, original_points, methods_results)
		print(f"Saved analysis: {args.analysis}")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())


