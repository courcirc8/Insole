#!/usr/bin/env python3
"""
simple_align_filter.py

Simplified approach: translate insole so lowest points are at Z=0, then filter by thickness.
"""
import argparse
import os
import sys
import glob

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def cleanup_outputs():
	"""Delete output PLY files."""
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


def simple_z_align(points: np.ndarray, bottom_percentile: float = 2.0) -> np.ndarray:
	"""Simple Z alignment: translate so bottom points are at Z=0."""
	z_bottom = np.percentile(points[:, 2], bottom_percentile)
	aligned = points.copy()
	aligned[:, 2] -= z_bottom
	
	print(f"Z-alignment: bottom {bottom_percentile}% ({z_bottom:.1f}) → Z=0")
	print(f"New Z range: [{aligned[:,2].min():.1f}, {aligned[:,2].max():.1f}]")
	
	return aligned


def method_1_thickness_only(points: np.ndarray, max_thickness: float = 12.0) -> np.ndarray:
	"""Method 1: Simple thickness filter."""
	keep_mask = points[:, 2] <= max_thickness
	result = points[keep_mask]
	removed = len(points) - len(result)
	print(f"  Thickness filter (<={max_thickness}mm): kept {len(result):,}, removed {removed:,}")
	return result


def method_2_dbscan_then_thickness(points: np.ndarray, max_thickness: float = 12.0) -> np.ndarray:
	"""Method 2: DBSCAN main cluster + thickness filter."""
	# DBSCAN first
	xy = points[:, :2]
	labels = DBSCAN(eps=8.0, min_samples=1000, n_jobs=-1).fit_predict(xy)
	
	unique_labels = np.unique(labels[labels >= 0])
	if len(unique_labels) > 0:
		cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
		largest_label = max(cluster_sizes, key=lambda x: x[1])[0]
		cluster_mask = labels == largest_label
		stage1 = points[cluster_mask]
		print(f"  DBSCAN: kept main cluster {len(stage1):,} / {len(points):,}")
	else:
		stage1 = points
	
	# Then thickness filter
	thickness_mask = stage1[:, 2] <= max_thickness
	result = stage1[thickness_mask]
	removed = len(stage1) - len(result)
	print(f"  Thickness: kept {len(result):,}, removed {removed:,}")
	
	return result


def method_3_statistical_then_thickness(points: np.ndarray, max_thickness: float = 12.0) -> np.ndarray:
	"""Method 3: Statistical outlier removal + thickness filter."""
	# Statistical first
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	cleaned_pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
	stage1 = np.asarray(cleaned_pcd.points)
	print(f"  Statistical: kept {len(stage1):,} / {len(points):,}")
	
	# Then thickness filter
	thickness_mask = stage1[:, 2] <= max_thickness
	result = stage1[thickness_mask]
	removed = len(stage1) - len(result)
	print(f"  Thickness: kept {len(result):,}, removed {removed:,}")
	
	return result


def evaluate_simple_quality(points: np.ndarray) -> dict:
	"""Simple quality metrics."""
	xy = points[:, :2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	
	z_negative = np.sum(points[:, 2] < 0)
	z_too_thick = np.sum(points[:, 2] > 20)
	far_outliers = np.sum(distances > np.percentile(distances, 99))
	
	total_issues = z_negative + z_too_thick + far_outliers
	quality = max(0, 100 - (total_issues / len(points)) * 100)
	
	return {
		'points': len(points),
		'quality': quality,
		'z_negative': z_negative,
		'z_too_thick': z_too_thick,
		'far_outliers': far_outliers,
		'z_min': float(points[:, 2].min()),
		'z_max': float(points[:, 2].max())
	}


def main():
	parser = argparse.ArgumentParser(description="Simple Z-align and filter pipeline.")
	parser.add_argument("--scan", "-s", required=True, help="Path to ground-removed scan.")
	parser.add_argument("--output_dir", "-o", default="outputs/simple_aligned", help="Output directory.")
	parser.add_argument("--max_thickness", type=float, default=12.0, help="Maximum insole thickness.")
	args = parser.parse_args()
	
	# Step 1: Cleanup (but preserve input)
	if os.path.exists(args.scan):
		cleanup_outputs()
	
	if not os.path.isfile(args.scan):
		print(f"Scan file not found: {args.scan}")
		return 1
	
	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Load and align
	pcd = load_pcd(args.scan)
	original_points = np.asarray(pcd.points)
	scan_name = os.path.splitext(os.path.basename(args.scan))[0]
	
	print(f"🎯 SIMPLE ALIGN & FILTER")
	print(f"Input: {len(original_points):,} points")
	print(f"Original Z: [{original_points[:,2].min():.1f}, {original_points[:,2].max():.1f}]")
	
	# Align to Z=0
	aligned_points = simple_z_align(original_points, bottom_percentile=2.0)
	
	# Save aligned version
	aligned_path = os.path.join(args.output_dir, f"{scan_name}_aligned.ply")
	aligned_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned_points))
	o3d.io.write_point_cloud(aligned_path, aligned_pcd)
	print(f"💾 Aligned: {aligned_path}")
	
	# Test 3 filtering methods
	methods = {}
	
	print(f"\n=== Filtering Methods ===")
	
	print(f"\n1️⃣ Thickness Only")
	method1 = method_1_thickness_only(aligned_points, args.max_thickness)
	methods["Thickness-Only"] = method1
	
	print(f"\n2️⃣ DBSCAN + Thickness")
	method2 = method_2_dbscan_then_thickness(aligned_points, args.max_thickness)
	methods["DBSCAN+Thickness"] = method2
	
	print(f"\n3️⃣ Statistical + Thickness")
	method3 = method_3_statistical_then_thickness(aligned_points, args.max_thickness)
	methods["Statistical+Thickness"] = method3
	
	# Evaluate and save
	print(f"\n📊 RESULTS:")
	best_method = None
	best_quality = 0
	
	for method_name, points in methods.items():
		metrics = evaluate_simple_quality(points)
		safe_name = method_name.lower().replace("+", "_plus")
		out_path = os.path.join(args.output_dir, f"{scan_name}_{safe_name}.ply")
		
		pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
		o3d.io.write_point_cloud(out_path, pcd)
		
		print(f"{method_name}: {metrics['points']:,} pts, {metrics['quality']:.1f}% quality")
		print(f"  Z=[{metrics['z_min']:.1f}, {metrics['z_max']:.1f}], issues: neg={metrics['z_negative']}, thick={metrics['z_too_thick']}, far={metrics['far_outliers']}")
		print(f"  Saved: {out_path}")
		
		if metrics['quality'] > best_quality:
			best_quality = metrics['quality']
			best_method = method_name
	
	print(f"\n🏆 BEST: {best_method} ({best_quality:.1f}% quality)")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())


