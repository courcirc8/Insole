#!/usr/bin/env python3
"""
filter_optimizer.py

Test and optimize 3 filtering algorithms until 100% clean insole data is achieved.
Deletes previous outputs and focuses solely on perfecting the filtering stage.
"""
import argparse
import os
import sys
import shutil
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

try:
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False


def cleanup_outputs(output_dir: str):
	"""Delete all previous outputs."""
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir, exist_ok=True)
	print(f"🗑️  Cleaned output directory: {output_dir}")


def load_pcd(path: str) -> o3d.geometry.PointCloud:
	"""Load point cloud from file."""
	pcd = o3d.io.read_point_cloud(path)
	if pcd.is_empty():
		raise ValueError(f"Empty point cloud: {path}")
	return pcd


def evaluate_cleanness(points: np.ndarray, method_name: str) -> Dict[str, float]:
	"""
	Comprehensive cleanness evaluation.
	Returns metrics dict with scores 0-100 (100 = perfect).
	"""
	xy = points[:, :2]
	z = points[:, 2]
	
	# 1. Z-artifact detection (floor/ceiling artifacts)
	z_artifacts = np.sum(z < 2.0)  # Points stuck at ground level
	z_ceiling_artifacts = np.sum(z > np.percentile(z, 99) + 10)  # Extreme high points
	z_artifact_score = max(0, 100 - (z_artifacts + z_ceiling_artifacts) / len(points) * 1000)
	
	# 2. Spatial outlier detection
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	d95 = np.percentile(distances, 95)
	far_outliers = np.sum(distances > d95 + 15)  # Points far from main body
	spatial_score = max(0, 100 - far_outliers / len(points) * 1000)
	
	# 3. Z-distribution quality (should be smooth, not spiky)
	z_hist, _ = np.histogram(z, bins=50)
	z_std = np.std(z_hist)
	z_mean = np.mean(z_hist)
	z_cv = z_std / (z_mean + 1e-6)  # Coefficient of variation
	z_distribution_score = max(0, 100 - z_cv * 20)  # Lower CV = better
	
	# 4. Density consistency (uniform point distribution)
	# Grid-based density analysis
	x_bins = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 20)
	y_bins = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 20)
	density_grid = np.zeros((19, 19))
	
	for i in range(19):
		for j in range(19):
			mask = ((xy[:, 0] >= x_bins[i]) & (xy[:, 0] < x_bins[i+1]) & 
					(xy[:, 1] >= y_bins[j]) & (xy[:, 1] < y_bins[j+1]))
			density_grid[i, j] = np.sum(mask)
	
	density_cv = np.std(density_grid[density_grid > 0]) / (np.mean(density_grid[density_grid > 0]) + 1e-6)
	density_score = max(0, 100 - density_cv * 10)
	
	# 5. Overall quality score
	overall_score = (z_artifact_score * 0.4 + spatial_score * 0.3 + 
					z_distribution_score * 0.2 + density_score * 0.1)
	
	metrics = {
		'method': method_name,
		'points': len(points),
		'z_artifacts': z_artifacts,
		'far_outliers': far_outliers,
		'z_artifact_score': z_artifact_score,
		'spatial_score': spatial_score,
		'z_distribution_score': z_distribution_score,
		'density_score': density_score,
		'overall_score': overall_score,
		'z_min': float(z.min()),
		'z_max': float(z.max()),
		'z_span': float(z.max() - z.min())
	}
	
	return metrics


def method_1_distance_filter(points: np.ndarray, percentile: float = 98.0) -> np.ndarray:
	"""Method 1: Remove far outliers based on distance from center."""
	xy = points[:, :2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	
	threshold = np.percentile(distances, percentile)
	keep_mask = distances <= threshold
	removed = len(points) - np.sum(keep_mask)
	
	print(f"  Distance filter ({percentile}%): removed {removed:,} points beyond {threshold:.1f}mm")
	return points[keep_mask]


def method_2_dbscan_main_cluster(points: np.ndarray, eps: float = 8.0, min_samples: int = 1000) -> np.ndarray:
	"""Method 2: DBSCAN to keep only main insole cluster."""
	xy = points[:, :2]
	
	# DBSCAN clustering in XY space
	labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(xy)
	
	# Find largest cluster
	unique_labels = np.unique(labels[labels >= 0])
	if len(unique_labels) == 0:
		print(f"  DBSCAN: No clusters found, keeping all points")
		return points
	
	cluster_sizes = []
	for label in unique_labels:
		size = np.sum(labels == label)
		cluster_sizes.append((label, size))
	
	# Keep largest cluster
	largest_label = max(cluster_sizes, key=lambda x: x[1])[0]
	keep_mask = labels == largest_label
	removed = len(points) - np.sum(keep_mask)
	
	print(f"  DBSCAN (eps={eps}): found {len(unique_labels)} clusters, kept largest ({np.sum(keep_mask):,} pts)")
	print(f"  Removed {removed:,} outlier points from {len(unique_labels)-1} small clusters + noise")
	
	return points[keep_mask]


def method_3_conservative_statistical(points: np.ndarray, nb_neighbors: int = 25, std_ratio: float = 3.5) -> np.ndarray:
	"""Method 3: Very conservative statistical outlier removal."""
	# Remove obvious Z artifacts first
	z_mask = points[:, 2] >= 1.0
	stage1 = points[z_mask]
	z_removed = len(points) - len(stage1)
	
	# Conservative statistical cleaning
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(stage1))
	cleaned_pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
	stage2 = np.asarray(cleaned_pcd.points)
	stat_removed = len(stage1) - len(stage2)
	
	print(f"  Z-filter: removed {z_removed:,} points < 1.0mm")
	print(f"  Statistical (nb={nb_neighbors}, std={std_ratio}): removed {stat_removed:,} outliers")
	
	return stage2


def save_results(output_dir: str, scan_name: str, methods_results: Dict[str, np.ndarray], metrics: Dict[str, Dict]):
	"""Save filtered point clouds and analysis."""
	
	# Save point clouds
	for method_name, points in methods_results.items():
		safe_name = method_name.lower().replace(" ", "_").replace("-", "_")
		out_path = os.path.join(output_dir, f"{scan_name}_{safe_name}.ply")
		pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
		o3d.io.write_point_cloud(out_path, pcd)
		print(f"💾 Saved: {out_path}")
	
	# Save analysis plot
	if _HAS_MPL:
		save_analysis_plot(os.path.join(output_dir, f"{scan_name}_filter_analysis.png"), methods_results, metrics)


def save_analysis_plot(path: str, methods_results: Dict[str, np.ndarray], metrics: Dict[str, Dict]):
	"""Save comprehensive analysis plot."""
	n_methods = len(methods_results)
	fig, axes = plt.subplots(3, n_methods, figsize=(5 * n_methods, 12))
	
	if n_methods == 1:
		axes = axes.reshape(-1, 1)
	
	colors = ['blue', 'red', 'green', 'orange']
	
	for i, (method_name, points) in enumerate(methods_results.items()):
		color = colors[i % len(colors)]
		m = metrics[method_name]
		
		# Z histogram
		axes[0, i].hist(points[:, 2], bins=50, alpha=0.7, color=color, edgecolor='black')
		axes[0, i].set_title(f'{method_name}\nScore: {m["overall_score"]:.1f}/100')
		axes[0, i].set_xlabel('Z Height')
		axes[0, i].set_ylabel('Count')
		axes[0, i].grid(True, alpha=0.3)
		
		# XY scatter
		if len(points) > 30000:
			idx = np.random.choice(len(points), 30000, replace=False)
			sample = points[idx]
		else:
			sample = points
		
		scatter = axes[1, i].scatter(sample[:, 0], sample[:, 1], s=0.3, c=sample[:, 2], cmap='viridis', alpha=0.8)
		axes[1, i].set_title(f'Spatial ({len(points):,} pts)')
		axes[1, i].axis('equal')
		
		# Quality metrics bar chart
		scores = [m['z_artifact_score'], m['spatial_score'], m['z_distribution_score'], m['density_score']]
		labels = ['Z-Artifacts', 'Spatial', 'Z-Distrib', 'Density']
		bars = axes[2, i].bar(labels, scores, color=color, alpha=0.7)
		axes[2, i].set_ylim(0, 100)
		axes[2, i].set_ylabel('Score')
		axes[2, i].set_title(f'Quality Breakdown')
		axes[2, i].tick_params(axis='x', rotation=45)
		
		# Add score text on bars
		for bar, score in zip(bars, scores):
			axes[2, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
							f'{score:.0f}', ha='center', va='bottom', fontsize=8)
	
	plt.tight_layout()
	plt.savefig(path, dpi=150, bbox_inches='tight')
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Optimize filtering until 100% clean insole data.")
	parser.add_argument("--scan", "-s", required=True, help="Path to isolated insole PLY (after ground removal).")
	parser.add_argument("--output_dir", "-o", default="outputs/filter_test", help="Output directory.")
	parser.add_argument("--target_score", type=float, default=95.0, help="Target overall quality score.")
	args = parser.parse_args()
	
	if not os.path.isfile(args.scan):
		print(f"Scan file not found: {args.scan}")
		return 1
	
	# Cleanup previous outputs
	cleanup_outputs(args.output_dir)
	
	# Load input
	pcd = load_pcd(args.scan)
	original_points = np.asarray(pcd.points)
	scan_name = os.path.splitext(os.path.basename(args.scan))[0]
	
	print(f"🔬 FILTERING OPTIMIZATION")
	print(f"Input: {args.scan} ({len(original_points):,} points)")
	print(f"Target: {args.target_score:.1f}% quality score")
	print(f"Output: {args.output_dir}")
	
	# Test 3 filtering methods
	methods_results = {}
	metrics = {}
	
	print(f"\n=== Testing Filtering Methods ===")
	
	# Method 1: Distance filter (remove far outliers)
	print(f"\n1️⃣ Distance Filter (Far Outlier Removal)")
	filtered_1 = method_1_distance_filter(original_points, percentile=98.0)
	methods_results["Distance-Filter"] = filtered_1
	metrics["Distance-Filter"] = evaluate_cleanness(filtered_1, "Distance-Filter")
	
	# Method 2: DBSCAN main cluster  
	print(f"\n2️⃣ DBSCAN Main Cluster")
	filtered_2 = method_2_dbscan_main_cluster(original_points, eps=8.0, min_samples=1000)
	methods_results["DBSCAN-Cluster"] = filtered_2
	metrics["DBSCAN-Cluster"] = evaluate_cleanness(filtered_2, "DBSCAN-Cluster")
	
	# Method 3: Conservative statistical
	print(f"\n3️⃣ Conservative Statistical")
	filtered_3 = method_3_conservative_statistical(original_points, nb_neighbors=25, std_ratio=3.5)
	methods_results["Conservative-Statistical"] = filtered_3
	metrics["Conservative-Statistical"] = evaluate_cleanness(filtered_3, "Conservative-Statistical")
	
	# Print results
	print(f"\n📊 FILTERING RESULTS:")
	print(f"{'Method':<12} {'Points':<8} {'Z-Art':<6} {'Spatial':<7} {'Overall':<7} {'Quality'}")
	print("-" * 60)
	
	best_method = None
	best_score = 0
	
	for method_name in ["Distance-Filter", "DBSCAN-Cluster", "Conservative-Statistical"]:
		m = metrics[method_name]
		points = methods_results[method_name]
		
		quality_status = "✅ PERFECT" if m['overall_score'] >= args.target_score else "⚠️  NEEDS WORK"
		
		print(f"{method_name:<12} {m['points']:<8,} {m['z_artifacts']:<6} {m['far_outliers']:<7} {m['overall_score']:<7.1f} {quality_status}")
		
		if m['overall_score'] > best_score:
			best_score = m['overall_score']
			best_method = method_name
	
	print(f"\n🏆 BEST METHOD: {best_method} (Score: {best_score:.1f}/100)")
	
	# Check if target achieved
	if best_score >= args.target_score:
		print(f"✅ TARGET ACHIEVED! {best_method} reaches {best_score:.1f}% quality")
		best_points = methods_results[best_method]
		
		# Save best result as final
		final_path = os.path.join(args.output_dir, f"{scan_name}_FINAL_CLEAN.ply")
		final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_points))
		o3d.io.write_point_cloud(final_path, final_pcd)
		print(f"💎 Saved final clean data: {final_path}")
		
	else:
		print(f"❌ TARGET NOT REACHED. Best: {best_score:.1f}% < {args.target_score:.1f}%")
		print(f"💡 Try adjusting parameters or combining methods")
	
	# Save all results for comparison
	save_results(args.output_dir, scan_name, methods_results, metrics)
	
	print(f"\n📁 All results saved in: {args.output_dir}")
	print(f"🔬 Review in PLY viewer to select best method")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())
