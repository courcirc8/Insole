"""
clean_thickness_artifacts.py

Remove thickness artifacts using top-down gap detection.
Estimates insole thickness from top surface and removes points separated by large gaps.
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


def method_gap_detection(points: np.ndarray, gap_threshold: float = 5.0, top_percentile: float = 75.0) -> np.ndarray:
	"""
	Your proposed method: detect thickness from top, remove points below large gaps.
	
	Args:
		points: 3D points (N, 3)
		gap_threshold: Minimum gap size to consider as artifact boundary
		top_percentile: Percentile to define "top surface"
	
	Returns:
		Cleaned points
	"""
	z_vals = points[:, 2]
	z_top_threshold = np.percentile(z_vals, top_percentile)
	
	# For each XY location, find the top surface and detect gaps
	xy = points[:, :2]
	
	# Grid the XY space for local thickness analysis
	x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
	y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
	
	# Create grid cells
	grid_size = 5.0  # mm
	nx = max(10, int((x_max - x_min) / grid_size))
	ny = max(10, int((y_max - y_min) / grid_size))
	
	keep_mask = np.ones(len(points), dtype=bool)
	
	for i in range(nx):
		for j in range(ny):
			x_start = x_min + i * (x_max - x_min) / nx
			x_end = x_min + (i + 1) * (x_max - x_min) / nx
			y_start = y_min + j * (y_max - y_min) / ny
			y_end = y_min + (j + 1) * (y_max - y_min) / ny
			
			# Points in this grid cell
			in_cell = ((xy[:, 0] >= x_start) & (xy[:, 0] < x_end) & 
					  (xy[:, 1] >= y_start) & (xy[:, 1] < y_end))
			
			if not np.any(in_cell):
				continue
			
			cell_z = z_vals[in_cell]
			cell_indices = np.where(in_cell)[0]
			
			if len(cell_z) < 3:
				continue
			
			# Sort by Z (top to bottom)
			z_sorted_idx = np.argsort(cell_z)[::-1]
			cell_z_sorted = cell_z[z_sorted_idx]
			
			# Find first large gap from top
			z_diffs = np.diff(cell_z_sorted)  # Negative (going down)
			large_gap_idx = np.where(np.abs(z_diffs) > gap_threshold)[0]
			
			if len(large_gap_idx) > 0:
				# Keep only points above first large gap
				first_gap = large_gap_idx[0]
				keep_in_cell = z_sorted_idx[:first_gap + 1]
				discard_in_cell = z_sorted_idx[first_gap + 1:]
				
				# Mark discarded points
				discard_global = cell_indices[discard_in_cell]
				keep_mask[discard_global] = False
	
	return points[keep_mask]


def method_z_clustering(points: np.ndarray, z_gap: float = 3.0, min_cluster_size: int = 1000) -> np.ndarray:
	"""
	Alternative: cluster points by Z height, keep largest cluster.
	"""
	from sklearn.cluster import DBSCAN
	
	z_vals = points[:, 2].reshape(-1, 1)
	
	# Cluster in Z dimension only
	clustering = DBSCAN(eps=z_gap, min_samples=min_cluster_size).fit(z_vals)
	labels = clustering.labels_
	
	# Find largest cluster
	unique_labels = np.unique(labels)
	cluster_sizes = []
	
	for label in unique_labels:
		if label == -1:  # Noise
			continue
		size = np.sum(labels == label)
		z_range = z_vals[labels == label]
		cluster_sizes.append((label, size, z_range.min(), z_range.max()))
	
	if not cluster_sizes:
		return points
	
	# Keep largest cluster
	largest_label = max(cluster_sizes, key=lambda x: x[1])[0]
	keep_mask = labels == largest_label
	
	print(f"Z-clustering: kept cluster {largest_label} with {np.sum(keep_mask):,} points")
	for label, size, z_min, z_max in sorted(cluster_sizes, key=lambda x: x[1], reverse=True):
		print(f"  Cluster {label}: {size:,} points, Z=[{z_min:.1f}, {z_max:.1f}]")
	
	return points[keep_mask]


def method_thickness_filter(points: np.ndarray, max_thickness: float = 15.0) -> np.ndarray:
	"""
	Alternative: estimate local top surface, remove points beyond max_thickness below.
	"""
	xy = points[:, :2]
	z = points[:, 2]
	
	# For each point, find local top surface using kNN
	nbrs = NearestNeighbors(n_neighbors=20, algorithm="kd_tree").fit(xy)
	distances, indices = nbrs.kneighbors(xy)
	
	# Local top surface = max Z in neighborhood
	local_top_z = np.array([z[idx].max() for idx in indices])
	
	# Keep points within max_thickness of local top
	thickness = local_top_z - z
	keep_mask = thickness <= max_thickness
	
	print(f"Thickness filter: kept {np.sum(keep_mask):,} / {len(points):,} points")
	print(f"  Thickness range: [{thickness.min():.1f}, {thickness.max():.1f}]")
	
	return points[keep_mask]


def save_analysis_plot(path: str, original: np.ndarray, methods_results: dict):
	"""Save comparison plot of different methods."""
	if not _HAS_MPL:
		return
	
	fig, axes = plt.subplots(2, len(methods_results) + 1, figsize=(4 * (len(methods_results) + 1), 8))
	
	# Original
	ax = axes[0, 0] if len(methods_results) > 0 else axes[0]
	ax.hist(original[:, 2], bins=50, alpha=0.7, color='gray')
	ax.set_title(f'Original\n{len(original):,} points')
	ax.set_xlabel('Z')
	ax.set_ylabel('Count')
	
	ax = axes[1, 0] if len(methods_results) > 0 else axes[1]
	if len(original) > 50000:
		idx = np.random.choice(len(original), 50000, replace=False)
		sample = original[idx]
	else:
		sample = original
	ax.scatter(sample[:, 0], sample[:, 1], s=0.1, c=sample[:, 2], cmap='viridis', alpha=0.6)
	ax.set_title('Original (XY view)')
	ax.axis('equal')
	
	# Methods
	for i, (name, cleaned) in enumerate(methods_results.items()):
		ax = axes[0, i + 1]
		ax.hist(cleaned[:, 2], bins=50, alpha=0.7)
		ax.set_title(f'{name}\n{len(cleaned):,} points')
		ax.set_xlabel('Z')
		ax.set_ylabel('Count')
		
		ax = axes[1, i + 1]
		if len(cleaned) > 50000:
			idx = np.random.choice(len(cleaned), 50000, replace=False)
			sample = cleaned[idx]
		else:
			sample = cleaned
		ax.scatter(sample[:, 0], sample[:, 1], s=0.1, c=sample[:, 2], cmap='viridis', alpha=0.6)
		ax.set_title(f'{name} (XY view)')
		ax.axis('equal')
	
	plt.tight_layout()
	plt.savefig(path, dpi=150)
	plt.close()


def main() -> int:
	parser = argparse.ArgumentParser(description="Clean thickness artifacts using gap detection methods.")
	parser.add_argument("--path", "-p", type=str, required=True, help="Path to cleaned point cloud.")
	parser.add_argument("--gap_threshold", type=float, default=5.0, help="Gap threshold for top-down method (mm).")
	parser.add_argument("--max_thickness", type=float, default=15.0, help="Max insole thickness (mm).")
	parser.add_argument("--z_gap", type=float, default=3.0, help="Z clustering gap (mm).")
	parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results.")
	parser.add_argument("--analysis", type=str, default=None, help="Save analysis plot PNG.")
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
	print(f"Analyzing: {in_path}")
	print(f"Original: {len(original_points):,} points, Z=[{original_points[:,2].min():.1f}, {original_points[:,2].max():.1f}]")

	# Test different methods
	methods_results = {}
	
	print(f"\n=== Method 1: Gap Detection (Your Proposal) ===")
	cleaned_gap = method_gap_detection(original_points, gap_threshold=args.gap_threshold)
	methods_results["Gap Detection"] = cleaned_gap
	print(f"Result: {len(cleaned_gap):,} points ({len(cleaned_gap)/len(original_points)*100:.1f}%)")
	
	print(f"\n=== Method 2: Z-Clustering ===")
	cleaned_z_cluster = method_z_clustering(original_points, z_gap=args.z_gap)
	methods_results["Z-Clustering"] = cleaned_z_cluster
	print(f"Result: {len(cleaned_z_cluster):,} points ({len(cleaned_z_cluster)/len(original_points)*100:.1f}%)")
	
	print(f"\n=== Method 3: Thickness Filter ===")
	cleaned_thickness = method_thickness_filter(original_points, max_thickness=args.max_thickness)
	methods_results["Thickness Filter"] = cleaned_thickness
	print(f"Result: {len(cleaned_thickness):,} points ({len(cleaned_thickness)/len(original_points)*100:.1f}%)")
	
	# Save results
	output_dir = args.output_dir or os.path.dirname(in_path)
	base_name = os.path.splitext(os.path.basename(in_path))[0]
	
	for method_name, cleaned_points in methods_results.items():
		safe_name = method_name.lower().replace(" ", "_").replace("-", "_")
		out_path = os.path.join(output_dir, f"{base_name}_{safe_name}.ply")
		cleaned_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cleaned_points))
		o3d.io.write_point_cloud(out_path, cleaned_pcd)
		print(f"Saved {method_name}: {out_path}")
	
	# Analysis plot
	if args.analysis:
		save_analysis_plot(args.analysis, original_points, methods_results)
		print(f"Saved analysis: {args.analysis}")
	
	# Recommendation
	print(f"\n=== Recommendations ===")
	for name, cleaned in methods_results.items():
		z_span = cleaned[:, 2].max() - cleaned[:, 2].min()
		z_min = cleaned[:, 2].min()
		print(f"{name}: {len(cleaned):,} points, Z_min={z_min:.1f}, Z_span={z_span:.1f}")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())


