#!/usr/bin/env python3
"""
align_and_filter.py

1. Erase all output files
2. Compute orientation of the lower plane (touching lowest points)
3. Re-align PLY so lower plane is at Z=0
4. Apply filtering with proper Z-alignment
"""
import argparse
import os
import sys
import shutil
import glob

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN

try:
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False


def cleanup_all_outputs(preserve_file: str = None):
	"""Delete ALL output files except the one being processed."""
	patterns = ["outputs/**/*.ply", "outputs/**/*.csv", "outputs/**/*.png", "outputs/**/*.npy", "outputs/**/*.stl"]
	total_deleted = 0
	
	for pattern in patterns:
		files = glob.glob(pattern, recursive=True)
		for f in files:
			# Don't delete the file we're about to process
			if preserve_file and os.path.abspath(f) == os.path.abspath(preserve_file):
				continue
			os.remove(f)
			total_deleted += 1
	
	print(f"🗑️ Deleted {total_deleted} output files (preserved input)")


def load_pcd(path: str) -> o3d.geometry.PointCloud:
	"""Load point cloud from file."""
	pcd = o3d.io.read_point_cloud(path)
	if pcd.is_empty():
		raise ValueError(f"Empty point cloud: {path}")
	return pcd


def compute_lower_plane(points: np.ndarray, bottom_percentile: float = 5.0) -> tuple:
	"""
	Compute the orientation of the lower plane touching the lowest points.
	
	Args:
		points: 3D points (N, 3)
		bottom_percentile: Percentile of lowest points to use for plane fitting
	
	Returns:
		(plane_normal, plane_point, plane_coeffs)
	"""
	z = points[:, 2]
	z_threshold = np.percentile(z, bottom_percentile)
	
	# Get bottom points
	bottom_mask = z <= z_threshold
	bottom_points = points[bottom_mask]
	
	print(f"Lower plane fitting: using {len(bottom_points):,} bottom points (Z <= {z_threshold:.1f})")
	
	if len(bottom_points) < 100:
		print("Warning: Very few bottom points, using more...")
		z_threshold = np.percentile(z, 10.0)
		bottom_mask = z <= z_threshold
		bottom_points = points[bottom_mask]
	
	# Fit plane to bottom points using PCA
	center = np.mean(bottom_points, axis=0)
	centered = bottom_points - center
	
	# PCA to find plane normal (smallest eigenvector)
	cov_matrix = np.cov(centered.T)
	eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
	
	# Normal is eigenvector with smallest eigenvalue
	normal = eigenvecs[:, 0]
	
	# Ensure normal points upward
	if normal[2] < 0:
		normal = -normal
	
	# Plane equation: ax + by + cz + d = 0
	a, b, c = normal
	d = -np.dot(normal, center)
	
	print(f"Lower plane: normal=({a:.3f}, {b:.3f}, {c:.3f}), d={d:.3f}")
	
	return normal, center, np.array([a, b, c, d])


def align_to_z_plane(points: np.ndarray, plane_normal: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
	"""
	Align points so the lower plane becomes the Z=0 plane.
	
	Args:
		points: 3D points (N, 3)
		plane_normal: Normal vector of lower plane
		plane_point: Point on the lower plane
	
	Returns:
		Aligned points with lower plane at Z=0
	"""
	# Step 1: Translate so plane point is at origin
	translated = points - plane_point
	
	# Step 2: Rotate so plane normal aligns with Z-axis
	target_normal = np.array([0.0, 0.0, 1.0])
	
	# Handle edge cases
	if np.allclose(plane_normal, target_normal):
		rotation_matrix = np.eye(3)
	elif np.allclose(plane_normal, -target_normal):
		# 180 degree rotation around X-axis
		rotation_matrix = R.from_rotvec([np.pi, 0, 0]).as_matrix()
	else:
		# General rotation using scipy
		# Find rotation that maps plane_normal to target_normal
		cross = np.cross(plane_normal, target_normal)
		sin_angle = np.linalg.norm(cross)
		cos_angle = np.dot(plane_normal, target_normal)
		
		if sin_angle < 1e-8:
			rotation_matrix = np.eye(3)
		else:
			axis = cross / sin_angle
			angle = np.arctan2(sin_angle, cos_angle)
			rotation_matrix = R.from_rotvec(axis * angle).as_matrix()
	
	# Apply rotation
	aligned = (rotation_matrix @ translated.T).T
	
	# Step 3: Translate so minimum Z becomes 0
	z_min = aligned[:, 2].min()
	aligned[:, 2] -= z_min
	
	print(f"Alignment: Z range [{aligned[:,2].min():.1f}, {aligned[:,2].max():.1f}]")
	
	return aligned


def method_1_z_thickness_filter(points: np.ndarray, max_thickness: float = 15.0) -> np.ndarray:
	"""Method 1: Filter by realistic insole thickness from Z=0."""
	z_mask = points[:, 2] <= max_thickness
	result = points[z_mask]
	removed = len(points) - len(result)
	print(f"  Thickness filter (<={max_thickness}mm): removed {removed:,} points")
	return result


def method_2_z_layers_filter(points: np.ndarray, layer_gap: float = 3.0, min_layer_size: int = 1000) -> np.ndarray:
	"""Method 2: Remove isolated Z layers (likely artifacts)."""
	z = points[:, 2]
	
	# Cluster by Z height
	z_reshaped = z.reshape(-1, 1)
	labels = DBSCAN(eps=layer_gap, min_samples=min_layer_size).fit_predict(z_reshaped)
	
	# Keep largest Z cluster
	unique_labels = np.unique(labels[labels >= 0])
	if len(unique_labels) == 0:
		return points
	
	cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
	largest_label = max(cluster_sizes, key=lambda x: x[1])[0]
	keep_mask = labels == largest_label
	
	result = points[keep_mask]
	removed = len(points) - len(result)
	print(f"  Z-layer filter: found {len(unique_labels)} layers, kept largest, removed {removed:,} points")
	
	return result


def method_3_combined_clean(points: np.ndarray, max_thickness: float = 15.0, distance_percentile: float = 99.0) -> np.ndarray:
	"""Method 3: Combined thickness + distance filtering."""
	# Stage 1: Thickness filter
	z_mask = points[:, 2] <= max_thickness
	stage1 = points[z_mask]
	z_removed = len(points) - len(stage1)
	
	# Stage 2: Distance filter
	xy = stage1[:, :2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	d_threshold = np.percentile(distances, distance_percentile)
	d_mask = distances <= d_threshold
	stage2 = stage1[d_mask]
	d_removed = len(stage1) - len(stage2)
	
	print(f"  Combined: thickness removed {z_removed:,}, distance removed {d_removed:,}")
	
	return stage2


def evaluate_quality(points: np.ndarray) -> float:
	"""Simple quality evaluation."""
	xy = points[:, :2]
	center = np.mean(xy, axis=0)
	distances = np.linalg.norm(xy - center, axis=1)
	
	z_artifacts = np.sum(points[:, 2] < 0.5)  # Should be none with proper alignment
	far_outliers = np.sum(distances > np.percentile(distances, 99))
	
	quality = max(0, 100 - (z_artifacts + far_outliers) / len(points) * 100)
	return quality


def save_results(output_dir: str, scan_name: str, aligned_points: np.ndarray, methods: dict):
	"""Save aligned points and filtered results."""
	
	# Save aligned points
	aligned_path = os.path.join(output_dir, f"{scan_name}_aligned.ply")
	aligned_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned_points))
	o3d.io.write_point_cloud(aligned_path, aligned_pcd)
	print(f"💾 Aligned: {aligned_path}")
	
	# Save filtered results
	for method_name, points in methods.items():
		safe_name = method_name.lower().replace(" ", "_").replace("+", "_plus")
		out_path = os.path.join(output_dir, f"{scan_name}_{safe_name}.ply")
		pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
		o3d.io.write_point_cloud(out_path, pcd)
		quality = evaluate_quality(points)
		print(f"💾 {method_name}: {out_path} ({len(points):,} pts, {quality:.1f}% quality)")


def main():
	parser = argparse.ArgumentParser(description="Align insole to Z=0 base plane and filter.")
	parser.add_argument("--scan", "-s", required=True, help="Path to ground-removed scan.")
	parser.add_argument("--output_dir", "-o", default="outputs/aligned_filter", help="Output directory.")
	parser.add_argument("--bottom_percentile", type=float, default=5.0, help="Percentile for bottom plane fitting.")
	parser.add_argument("--max_thickness", type=float, default=15.0, help="Maximum insole thickness (mm).")
	args = parser.parse_args()
	
	if not os.path.isfile(args.scan):
		print(f"Scan file not found: {args.scan}")
		return 1
	
	# Step 1: Cleanup all outputs (preserve input file)
	cleanup_all_outputs(preserve_file=args.scan)
	
	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Load scan
	pcd = load_pcd(args.scan)
	original_points = np.asarray(pcd.points)
	scan_name = os.path.splitext(os.path.basename(args.scan))[0]
	
	print(f"🎯 ALIGN AND FILTER PIPELINE")
	print(f"Input: {args.scan} ({len(original_points):,} points)")
	print(f"Original Z range: [{original_points[:,2].min():.1f}, {original_points[:,2].max():.1f}]")
	
	# Step 2: Compute lower plane orientation
	print(f"\n=== Computing Lower Plane ===")
	plane_normal, plane_point, plane_coeffs = compute_lower_plane(original_points, args.bottom_percentile)
	
	# Step 3: Align to Z=0 plane
	print(f"\n=== Aligning to Z=0 ===")
	aligned_points = align_to_z_plane(original_points, plane_normal, plane_point)
	
	# Step 4: Apply 3 filtering methods with proper Z-alignment
	methods = {}
	
	print(f"\n=== Filtering with Z-Alignment ===")
	
	print(f"\n1️⃣ Z-Thickness Filter")
	method1 = method_1_z_thickness_filter(aligned_points, max_thickness=args.max_thickness)
	methods["Z-Thickness"] = method1
	
	print(f"\n2️⃣ Z-Layers Filter")
	method2 = method_2_z_layers_filter(aligned_points, layer_gap=3.0, min_layer_size=1000)
	methods["Z-Layers"] = method2
	
	print(f"\n3️⃣ Combined Clean")
	method3 = method_3_combined_clean(aligned_points, max_thickness=args.max_thickness, distance_percentile=99.0)
	methods["Combined"] = method3
	
	# Evaluate and find best
	print(f"\n📊 RESULTS:")
	best_method = None
	best_quality = 0
	
	for method_name, points in methods.items():
		quality = evaluate_quality(points)
		if quality > best_quality:
			best_quality = quality
			best_method = method_name
		print(f"{method_name}: {len(points):,} points, {quality:.1f}% quality")
	
	print(f"\n🏆 BEST: {best_method} ({best_quality:.1f}% quality)")
	
	# Save results
	save_results(args.output_dir, scan_name, aligned_points, methods)
	
	# Save best as final
	if best_quality >= 98.0:
		best_points = methods[best_method]
		final_path = os.path.join(args.output_dir, f"{scan_name}_FINAL_ALIGNED_CLEAN.ply")
		final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_points))
		o3d.io.write_point_cloud(final_path, final_pcd)
		print(f"💎 Saved final: {final_path}")
	
	print(f"\n📁 Results in: {args.output_dir}")
	print(f"🔬 Review in GUI: http://127.0.0.1:8051")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())
