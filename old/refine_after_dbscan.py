#!/usr/bin/env python3
"""
refine_after_dbscan.py

Refinement algorithms to apply after DBSCAN main cluster extraction.
3 targeted approaches to clean remaining outliers while preserving insole structure.
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


def evaluate_cleanness(points: np.ndarray) -> dict:
	"""Evaluate point cloud cleanness."""
	xy = points[:, :2]
	z = points[:, 2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	
	# Count different types of outliers
	z_artifacts = np.sum(z < 1.0)
	distance_outliers = np.sum(distances > np.percentile(distances, 99))
	z_extreme_high = np.sum(z > np.percentile(z, 99))
	z_extreme_low = np.sum(z < np.percentile(z, 1))
	
	total_outliers = z_artifacts + distance_outliers + z_extreme_high + z_extreme_low
	quality = max(0, 100 - (total_outliers / len(points)) * 100)
	
	return {
		'points': len(points),
		'quality': quality,
		'z_artifacts': z_artifacts,
		'distance_outliers': distance_outliers,
		'z_extreme_high': z_extreme_high,
		'z_extreme_low': z_extreme_low,
		'z_min': float(z.min()),
		'z_max': float(z.max())
	}


def method_1_distance_percentile(points: np.ndarray, keep_percentile: float = 98.5) -> np.ndarray:
	"""Method 1: Remove far outliers by distance percentile."""
	xy = points[:, :2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	
	threshold = np.percentile(distances, keep_percentile)
	keep_mask = distances <= threshold
	result = points[keep_mask]
	removed = len(points) - len(result)
	
	print(f"  Distance percentile ({keep_percentile}%): removed {removed:,} points beyond {threshold:.1f}mm")
	return result


def method_2_local_density_filter(points: np.ndarray, density_percentile: float = 10.0, radius: float = 3.0) -> np.ndarray:
	"""Method 2: Remove points in sparse areas (low local density)."""
	xy = points[:, :2]
	
	# Calculate local density using radius neighbors
	nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(xy)
	neighbor_counts = nbrs.radius_neighbors(xy, return_distance=False)
	local_density = np.array([len(neighbors) - 1 for neighbors in neighbor_counts])  # -1 for self
	
	# Remove points with low density
	density_threshold = np.percentile(local_density, density_percentile)
	keep_mask = local_density >= density_threshold
	result = points[keep_mask]
	removed = len(points) - len(result)
	
	print(f"  Local density filter ({density_percentile}%): removed {removed:,} sparse points (density < {density_threshold:.1f})")
	return result


def method_3_z_range_plus_statistical(points: np.ndarray, z_trim_percent: float = 2.0, std_ratio: float = 2.5) -> np.ndarray:
	"""Method 3: Trim Z extremes + statistical outlier removal."""
	z = points[:, 2]
	
	# Stage 1: Remove Z extremes (top and bottom percentiles)
	z_low = np.percentile(z, z_trim_percent)
	z_high = np.percentile(z, 100 - z_trim_percent)
	z_mask = (z >= z_low) & (z <= z_high)
	stage1 = points[z_mask]
	z_removed = len(points) - len(stage1)
	
	# Stage 2: Statistical outlier removal
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stage1))
	cleaned_pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
	stage2 = np.asarray(cleaned_pcd.points)
	stat_removed = len(stage1) - len(stage2)
	
	print(f"  Z-range trim ({z_trim_percent}%): removed {z_removed:,} extreme Z points")
	print(f"  Statistical (std={std_ratio}): removed {stat_removed:,} outliers")
	
	return stage2


def save_results_and_analysis(output_dir: str, scan_name: str, original: np.ndarray, methods: dict):
	"""Save results and create analysis plot."""
	
	# Save PLY files
	for method_name, points in methods.items():
		safe_name = method_name.lower().replace(" ", "_").replace("+", "_plus").replace("-", "_")
		out_path = os.path.join(output_dir, f"{scan_name}_{safe_name}.ply")
		pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
		o3d.io.write_point_cloud(out_path, pcd)
		print(f"💾 {method_name}: {out_path}")
	
	# Create analysis plot
	if not _HAS_MPL:
		return
	
	fig, axes = plt.subplots(2, len(methods) + 1, figsize=(5 * (len(methods) + 1), 8))
	
	# Original
	axes[0, 0].scatter(original[:, 0], original[:, 1], s=0.1, c=original[:, 2], cmap='viridis', alpha=0.6)
	axes[0, 0].set_title(f'Original\n{len(original):,} points')
	axes[0, 0].axis('equal')
	
	axes[1, 0].hist(original[:, 2], bins=50, alpha=0.7, color='gray')
	axes[1, 0].set_title('Z Distribution')
	
	# Methods
	colors = ['blue', 'red', 'green']
	for i, (method_name, points) in enumerate(methods.items()):
		col = i + 1
		color = colors[i % len(colors)]
		
		# Sample for plotting
		if len(points) > 50000:
			idx = np.random.choice(len(points), 50000, replace=False)
			sample = points[idx]
		else:
			sample = points
		
		axes[0, col].scatter(sample[:, 0], sample[:, 1], s=0.1, c=sample[:, 2], cmap='viridis', alpha=0.6)
		axes[0, col].set_title(f'{method_name}\n{len(points):,} points')
		axes[0, col].axis('equal')
		
		axes[1, col].hist(points[:, 2], bins=50, alpha=0.7, color=color)
		axes[1, col].set_title('Z Distribution')
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f"{scan_name}_refinement_analysis.png"), dpi=150)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Refine DBSCAN result with 3 targeted algorithms.")
	parser.add_argument("--scan", "-s", required=True, help="Path to DBSCAN-cleaned PLY.")
	parser.add_argument("--output_dir", "-o", default="outputs/refinement_test", help="Output directory.")
	args = parser.parse_args()
	
	if not os.path.isfile(args.scan):
		print(f"Scan file not found: {args.scan}")
		return 1
	
	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Load DBSCAN result
	pcd = load_pcd(args.scan)
	original_points = np.asarray(pcd.points)
	scan_name = os.path.splitext(os.path.basename(args.scan))[0]
	
	print(f"🔬 REFINEMENT AFTER DBSCAN")
	print(f"Input: {args.scan} ({len(original_points):,} points)")
	
	# Evaluate original quality
	orig_metrics = evaluate_cleanness(original_points)
	print(f"Starting quality: {orig_metrics['quality']:.1f}%")
	
	# Test 3 refinement methods
	methods = {}
	
	print(f"\n=== Refinement Methods ===")
	
	print(f"\n1️⃣ Distance Percentile Filter")
	method1 = method_1_distance_percentile(original_points, keep_percentile=98.5)
	methods["Distance-Percentile"] = method1
	m1 = evaluate_cleanness(method1)
	print(f"  Result: {m1['points']:,} points, quality={m1['quality']:.1f}%")
	
	print(f"\n2️⃣ Local Density Filter")
	method2 = method_2_local_density_filter(original_points, density_percentile=10.0, radius=3.0)
	methods["Local-Density"] = method2
	m2 = evaluate_cleanness(method2)
	print(f"  Result: {m2['points']:,} points, quality={m2['quality']:.1f}%")
	
	print(f"\n3️⃣ Z-Range + Statistical")
	method3 = method_3_z_range_plus_statistical(original_points, z_trim_percent=2.0, std_ratio=2.5)
	methods["Z-Range+Statistical"] = method3
	m3 = evaluate_cleanness(method3)
	print(f"  Result: {m3['points']:,} points, quality={m3['quality']:.1f}%")
	
	# Find best improvement
	qualities = [m1['quality'], m2['quality'], m3['quality']]
	method_names = list(methods.keys())
	best_idx = np.argmax(qualities)
	best_method = method_names[best_idx]
	best_quality = qualities[best_idx]
	improvement = best_quality - orig_metrics['quality']
	
	print(f"\n📊 REFINEMENT RESULTS:")
	print(f"Original: {orig_metrics['quality']:.1f}%")
	for name, quality in zip(method_names, qualities):
		improvement_str = f"(+{quality - orig_metrics['quality']:.1f}%)" if quality > orig_metrics['quality'] else f"({quality - orig_metrics['quality']:.1f}%)"
		print(f"{name}: {quality:.1f}% {improvement_str}")
	
	print(f"\n🏆 BEST REFINEMENT: {best_method} ({best_quality:.1f}%)")
	
	if best_quality >= 99.5:
		print(f"✅ EXCELLENT! Near-perfect quality achieved")
		best_points = methods[best_method]
		final_path = os.path.join(args.output_dir, f"{scan_name}_REFINED_BEST.ply")
		final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_points))
		o3d.io.write_point_cloud(final_path, final_pcd)
		print(f"💎 Saved best: {final_path}")
	
	# Save all results
	save_results_and_analysis(args.output_dir, scan_name, original_points, methods)
	
	print(f"\n📁 Results in: {args.output_dir}")
	print(f"🔬 Review in GUI to see refinement effects")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())


