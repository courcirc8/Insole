"""
clean_thickness_smart.py

Smart thickness-based cleaning using top-down gap detection with local surface estimation.
Improved implementation of the gap detection method.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

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


def estimate_local_top_surface(points: np.ndarray, grid_size: float = 3.0, percentile: float = 95.0) -> np.ndarray:
	"""
	Estimate local top surface height for each XY location.
	
	Args:
		points: 3D points (N, 3)
		grid_size: Local neighborhood size
		percentile: Percentile for top surface (95% = robust top)
	
	Returns:
		top_surface_z for each point
	"""
	xy = points[:, :2]
	z = points[:, 2]
	
	# Build spatial index
	tree = cKDTree(xy)
	
	# For each point, find local neighborhood and estimate top surface
	top_surface_z = np.zeros(len(points))
	
	for i in range(len(points)):
		# Find neighbors within grid_size
		neighbors = tree.query_ball_point(xy[i], grid_size)
		
		if len(neighbors) < 3:
			top_surface_z[i] = z[i]  # Fallback
		else:
			neighbor_z = z[neighbors]
			top_surface_z[i] = np.percentile(neighbor_z, percentile)
	
	return top_surface_z


def method_smart_gap_detection(points: np.ndarray, max_thickness: float = 12.0, gap_threshold: float = 4.0, grid_size: float = 3.0) -> np.ndarray:
	"""
	Smart gap detection: estimate local top surface, remove points beyond max_thickness or large gaps.
	
	Args:
		points: 3D points (N, 3)
		max_thickness: Maximum realistic insole thickness
		gap_threshold: Minimum gap to consider as artifact boundary
		grid_size: Local neighborhood size for surface estimation
	
	Returns:
		Cleaned points
	"""
	print(f"Smart gap detection: max_thickness={max_thickness}, gap_threshold={gap_threshold}")
	
	# Estimate local top surface
	top_surface_z = estimate_local_top_surface(points, grid_size=grid_size)
	
	# Calculate thickness from local top surface
	thickness = top_surface_z - points[:, 2]
	
	# Remove points beyond max thickness
	thickness_mask = thickness <= max_thickness
	
	# Additional gap detection: remove isolated low points
	z_vals = points[:, 2]
	xy = points[:, :2]
	
	# For each point, check if there's a large gap to nearby higher points
	tree = cKDTree(xy)
	gap_mask = np.ones(len(points), dtype=bool)
	
	for i in range(len(points)):
		if not thickness_mask[i]:  # Already filtered by thickness
			gap_mask[i] = False
			continue
			
		# Find nearby points
		neighbors = tree.query_ball_point(xy[i], grid_size)
		
		if len(neighbors) < 3:
			continue
			
		neighbor_z = z_vals[neighbors]
		current_z = z_vals[i]
		
		# Check if current point is isolated below a gap
		higher_neighbors = neighbor_z[neighbor_z > current_z]
		if len(higher_neighbors) > 0:
			min_gap_to_higher = np.min(higher_neighbors - current_z)
			if min_gap_to_higher > gap_threshold:
				gap_mask[i] = False
	
	# Combine masks
	final_mask = thickness_mask & gap_mask
	
	print(f"  Thickness filter: kept {np.sum(thickness_mask):,} / {len(points):,}")
	print(f"  Gap filter: kept {np.sum(gap_mask):,} / {len(points):,}")
	print(f"  Combined: kept {np.sum(final_mask):,} / {len(points):,} ({np.sum(final_mask)/len(points)*100:.1f}%)")
	
	return points[final_mask]


def method_surface_distance(points: np.ndarray, surface_percentile: float = 90.0, max_distance: float = 8.0) -> np.ndarray:
	"""
	Alternative: fit a surface to high-Z points, remove points too far below.
	"""
	z_vals = points[:, 2]
	xy = points[:, :2]
	
	# Define top surface from high-Z points
	top_threshold = np.percentile(z_vals, surface_percentile)
	top_mask = z_vals >= top_threshold
	
	if np.sum(top_mask) < 1000:
		print(f"Warning: only {np.sum(top_mask)} top surface points")
		return points
	
	# Fit kNN regressor to top surface
	from sklearn.neighbors import KNeighborsRegressor
	top_xy = xy[top_mask]
	top_z = z_vals[top_mask]
	
	knn = KNeighborsRegressor(n_neighbors=20, weights="distance", algorithm="kd_tree")
	knn.fit(top_xy, top_z)
	
	# Predict surface height for all points
	predicted_top_z = knn.predict(xy)
	
	# Keep points within max_distance of predicted surface
	distance_to_surface = predicted_top_z - z_vals
	keep_mask = distance_to_surface <= max_distance
	
	print(f"Surface distance: kept {np.sum(keep_mask):,} / {len(points):,} ({np.sum(keep_mask)/len(points)*100:.1f}%)")
	print(f"  Distance range: [{distance_to_surface.min():.1f}, {distance_to_surface.max():.1f}]")
	
	return points[keep_mask]


def save_comparison_plot(path: str, original: np.ndarray, methods_results: dict):
	"""Save Z distribution comparison."""
	if not _HAS_MPL:
		return
	
	n_methods = len(methods_results)
	fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
	
	# Original
	axes[0, 0].hist(original[:, 2], bins=50, alpha=0.7, color='gray', edgecolor='black')
	axes[0, 0].set_title(f'Original\n{len(original):,} points')
	axes[0, 0].set_xlabel('Z Height')
	axes[0, 0].set_ylabel('Count')
	axes[0, 0].grid(True, alpha=0.3)
	
	# XY scatter colored by Z
	if len(original) > 30000:
		idx = np.random.choice(len(original), 30000, replace=False)
		sample = original[idx]
	else:
		sample = original
	
	scatter = axes[1, 0].scatter(sample[:, 0], sample[:, 1], s=0.5, c=sample[:, 2], cmap='viridis', alpha=0.7)
	axes[1, 0].set_title('Original (XY view)')
	axes[1, 0].axis('equal')
	
	# Methods
	for i, (name, cleaned) in enumerate(methods_results.items()):
		col = i + 1
		
		# Z histogram
		axes[0, col].hist(cleaned[:, 2], bins=50, alpha=0.7, edgecolor='black')
		axes[0, col].set_title(f'{name}\n{len(cleaned):,} points')
		axes[0, col].set_xlabel('Z Height')
		axes[0, col].set_ylabel('Count')
		axes[0, col].grid(True, alpha=0.3)
		
		# XY scatter
		if len(cleaned) > 30000:
			idx = np.random.choice(len(cleaned), 30000, replace=False)
			sample = cleaned[idx]
		else:
			sample = cleaned
		
		axes[1, col].scatter(sample[:, 0], sample[:, 1], s=0.5, c=sample[:, 2], cmap='viridis', alpha=0.7)
		axes[1, col].set_title(f'{name} (XY view)')
		axes[1, col].axis('equal')
	
	plt.tight_layout()
	plt.savefig(path, dpi=150, bbox_inches='tight')
	plt.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="Smart thickness artifact removal.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to point cloud.")
	parser.add_argument("--max_thickness", type=float, default=12.0, help="Max insole thickness (mm).")
	parser.add_argument("--gap_threshold", type=float, default=4.0, help="Gap threshold (mm).")
	parser.add_argument("--grid_size", type=float, default=3.0, help="Local neighborhood size (mm).")
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

	# Test methods
	methods_results = {}
	
	print(f"\n=== Smart Gap Detection ===")
	cleaned_smart = method_smart_gap_detection(
		original_points, 
		max_thickness=args.max_thickness,
		gap_threshold=args.gap_threshold,
		grid_size=args.grid_size
	)
	methods_results["Smart Gap"] = cleaned_smart
	
	print(f"\n=== Surface Distance ===")
	cleaned_surface = method_surface_distance(original_points, surface_percentile=90.0, max_distance=args.max_thickness)
	methods_results["Surface Distance"] = cleaned_surface
	
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
		
		# Score: prefer higher Z_min (removes artifacts) and reasonable point count
		z_min = cleaned_points[:, 2].min()
		z_span = cleaned_points[:, 2].max() - z_min
		point_ratio = len(cleaned_points) / len(original_points)
		score = z_min + 0.1 * z_span + 10 * point_ratio  # Weighted score
		
		print(f"Saved {method_name}: {out_path}")
		print(f"  Score: {score:.1f} (Z_min={z_min:.1f}, span={z_span:.1f}, ratio={point_ratio:.3f})")
		
		if score > best_score:
			best_score = score
			best_method = method_name
	
	print(f"\n🏆 Best method: {best_method} (score: {best_score:.1f})")
	
	# Analysis plot
	if args.analysis:
		save_comparison_plot(args.analysis, original_points, methods_results)
		print(f"Saved analysis: {args.analysis}")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())


