#!/usr/bin/env python3
"""
filter_refined.py

Refined filtering: DBSCAN first step + additional filters to reach 100% clean.
Focus on removing far outliers while preserving insole structure.
"""
import argparse
import os
import sys
import shutil

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

try:
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False


def cleanup_outputs():
	"""Delete all PLY files in outputs directory."""
	import glob
	ply_files = glob.glob("outputs/**/*.ply", recursive=True)
	for f in ply_files:
		os.remove(f)
	print(f"🗑️ Deleted {len(ply_files)} PLY files")


def load_pcd(path: str) -> o3d.geometry.PointCloud:
	"""Load point cloud from file."""
	pcd = o3d.io.read_point_cloud(path)
	if pcd.is_empty():
		raise ValueError(f"Empty point cloud: {path}")
	return pcd


def evaluate_quality(points: np.ndarray) -> float:
	"""Simple quality score: 100 - % outliers."""
	xy = points[:, :2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	
	# Count artifacts
	z_artifacts = np.sum(points[:, 2] < 1.0)
	far_outliers = np.sum(distances > np.percentile(distances, 99))
	
	total_artifacts = z_artifacts + far_outliers
	quality = max(0, 100 - (total_artifacts / len(points)) * 100)
	
	return quality, z_artifacts, far_outliers


def step1_dbscan_main_cluster(points: np.ndarray, eps: float = 8.0, min_samples: int = 1000) -> np.ndarray:
	"""Step 1: DBSCAN to isolate main insole cluster."""
	xy = points[:, :2]
	labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(xy)
	
	# Keep largest cluster
	unique_labels = np.unique(labels[labels >= 0])
	if len(unique_labels) == 0:
		return points
	
	cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
	largest_label = max(cluster_sizes, key=lambda x: x[1])[0]
	keep_mask = labels == largest_label
	
	result = points[keep_mask]
	print(f"  DBSCAN: {len(unique_labels)} clusters → kept largest ({len(result):,} pts)")
	return result


def step2_distance_refinement(points: np.ndarray, percentile: float = 99.0) -> np.ndarray:
	"""Step 2: Remove remaining far outliers by distance percentile."""
	xy = points[:, :2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	
	threshold = np.percentile(distances, percentile)
	keep_mask = distances <= threshold
	result = points[keep_mask]
	removed = len(points) - len(result)
	
	print(f"  Distance refine: removed {removed:,} points beyond {threshold:.1f}mm")
	return result


def step3_statistical_polish(points: np.ndarray, std_ratio: float = 3.0) -> np.ndarray:
	"""Step 3: Final statistical polish to remove remaining outliers."""
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	cleaned_pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
	result = np.asarray(cleaned_pcd.points)
	removed = len(points) - len(result)
	
	print(f"  Statistical polish: removed {removed:,} outliers")
	return result


def method_1_dbscan_only(points: np.ndarray) -> np.ndarray:
	"""Method 1: DBSCAN only (baseline)."""
	return step1_dbscan_main_cluster(points, eps=8.0, min_samples=1000)


def method_2_dbscan_plus_distance(points: np.ndarray) -> np.ndarray:
	"""Method 2: DBSCAN + distance refinement."""
	stage1 = step1_dbscan_main_cluster(points, eps=8.0, min_samples=1000)
	stage2 = step2_distance_refinement(stage1, percentile=99.0)
	return stage2


def method_3_dbscan_plus_full_refine(points: np.ndarray) -> np.ndarray:
	"""Method 3: DBSCAN + distance + statistical polish."""
	stage1 = step1_dbscan_main_cluster(points, eps=8.0, min_samples=1000)
	stage2 = step2_distance_refinement(stage1, percentile=99.0)
	stage3 = step3_statistical_polish(stage2, std_ratio=3.0)
	return stage3


def save_comparison(output_dir: str, scan_name: str, original: np.ndarray, methods: dict):
	"""Save comparison plot and results."""
	
	# Save PLY files
	for method_name, points in methods.items():
		safe_name = method_name.lower().replace(" ", "_").replace("+", "_plus")
		out_path = os.path.join(output_dir, f"{scan_name}_{safe_name}.ply")
		pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
		o3d.io.write_point_cloud(out_path, pcd)
		print(f"💾 {method_name}: {out_path}")
	
	# Save analysis plot
	if _HAS_MPL:
		fig, axes = plt.subplots(2, len(methods) + 1, figsize=(5 * (len(methods) + 1), 8))
		
		# Original
		quality_orig, z_art_orig, far_orig = evaluate_quality(original)
		
		if len(original) > 50000:
			idx = np.random.choice(len(original), 50000, replace=False)
			sample = original[idx]
		else:
			sample = original
		
		axes[0, 0].scatter(sample[:, 0], sample[:, 1], s=0.2, c=sample[:, 2], cmap='viridis', alpha=0.6)
		axes[0, 0].set_title(f'Original\n{len(original):,} pts, {quality_orig:.1f}%')
		axes[0, 0].axis('equal')
		
		axes[1, 0].hist(original[:, 2], bins=50, alpha=0.7, color='gray')
		axes[1, 0].set_title('Z Distribution')
		axes[1, 0].set_xlabel('Z Height')
		
		# Methods
		for i, (method_name, points) in enumerate(methods.items()):
			col = i + 1
			quality, z_art, far = evaluate_quality(points)
			
			if len(points) > 50000:
				idx = np.random.choice(len(points), 50000, replace=False)
				sample = points[idx]
			else:
				sample = points
			
			axes[0, col].scatter(sample[:, 0], sample[:, 1], s=0.2, c=sample[:, 2], cmap='viridis', alpha=0.6)
			axes[0, col].set_title(f'{method_name}\n{len(points):,} pts, {quality:.1f}%')
			axes[0, col].axis('equal')
			
			axes[1, col].hist(points[:, 2], bins=50, alpha=0.7)
			axes[1, col].set_title(f'Z Distribution')
			axes[1, col].set_xlabel('Z Height')
		
		plt.tight_layout()
		plt.savefig(os.path.join(output_dir, f"{scan_name}_refined_analysis.png"), dpi=150)
		plt.close()


def main():
	parser = argparse.ArgumentParser(description="Refined filtering: DBSCAN + additional steps.")
	parser.add_argument("--scan", "-s", required=True, help="Path to isolated insole PLY.")
	parser.add_argument("--output_dir", "-o", default="outputs/refined_filter", help="Output directory.")
	args = parser.parse_args()
	
	if not os.path.isfile(args.scan):
		print(f"Scan file not found: {args.scan}")
		return 1
	
	# Step 0: Cleanup (commented out to preserve input file)
	# cleanup_outputs()
	
	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Load input
	pcd = load_pcd(args.scan)
	original_points = np.asarray(pcd.points)
	scan_name = os.path.splitext(os.path.basename(args.scan))[0]
	
	print(f"🎯 REFINED FILTERING")
	print(f"Input: {args.scan} ({len(original_points):,} points)")
	
	# Test 3 progressive refinement methods
	methods = {}
	
	print(f"\n=== Method Testing ===")
	
	print(f"\n1️⃣ DBSCAN Only (Baseline)")
	method1 = method_1_dbscan_only(original_points)
	methods["DBSCAN-Only"] = method1
	q1, z1, f1 = evaluate_quality(method1)
	print(f"  Result: {len(method1):,} points, quality={q1:.1f}% (z_art={z1}, far={f1})")
	
	print(f"\n2️⃣ DBSCAN + Distance Refine")
	method2 = method_2_dbscan_plus_distance(original_points)
	methods["DBSCAN+Distance"] = method2
	q2, z2, f2 = evaluate_quality(method2)
	print(f"  Result: {len(method2):,} points, quality={q2:.1f}% (z_art={z2}, far={f2})")
	
	print(f"\n3️⃣ DBSCAN + Distance + Statistical")
	method3 = method_3_dbscan_plus_full_refine(original_points)
	methods["DBSCAN+Distance+Statistical"] = method3
	q3, z3, f3 = evaluate_quality(method3)
	print(f"  Result: {len(method3):,} points, quality={q3:.1f}% (z_art={z3}, far={f3})")
	
	# Find best
	qualities = [q1, q2, q3]
	method_names = ["DBSCAN-Only", "DBSCAN+Distance", "DBSCAN+Distance+Statistical"]
	best_idx = np.argmax(qualities)
	best_method = method_names[best_idx]
	best_quality = qualities[best_idx]
	
	print(f"\n🏆 BEST: {best_method} ({best_quality:.1f}% quality)")
	
	if best_quality >= 99.0:
		print(f"✅ EXCELLENT QUALITY ACHIEVED!")
		best_points = list(methods.values())[best_idx]
		final_path = os.path.join(args.output_dir, f"{scan_name}_PERFECT_CLEAN.ply")
		final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_points))
		o3d.io.write_point_cloud(final_path, final_pcd)
		print(f"💎 Saved: {final_path}")
	else:
		print(f"⚠️ Quality {best_quality:.1f}% < 99%. May need parameter tuning.")
	
	# Save all results
	save_comparison(args.output_dir, scan_name, original_points, methods)
	
	print(f"\n📁 Results in: {args.output_dir}")
	print(f"🔬 Review in GUI: http://127.0.0.1:8051")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())
