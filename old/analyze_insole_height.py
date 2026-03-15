#!/usr/bin/env python3
"""
Analyze insole height distribution and construct valid cleaned object.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import os

def analyze_height_distribution(ply_file):
    """Analyze the height distribution of the point cloud."""
    print(f"\n=== ANALYZING HEIGHT DISTRIBUTION ===")
    print(f"File: {ply_file}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")
    
    points = np.asarray(pcd.points)
    print(f"Total points: {len(points):,}")
    
    # Height analysis
    z_values = points[:, 2]
    print(f"\nHeight Statistics:")
    print(f"  Min Z: {z_values.min():.2f}mm")
    print(f"  Max Z: {z_values.max():.2f}mm")
    print(f"  Mean Z: {z_values.mean():.2f}mm")
    print(f"  Std Z: {z_values.std():.2f}mm")
    print(f"  Range: {z_values.max() - z_values.min():.2f}mm")
    
    # Height histogram analysis
    bins = np.arange(z_values.min(), z_values.max() + 0.5, 0.5)
    counts, bin_edges = np.histogram(z_values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print(f"\nHeight Distribution Analysis:")
    print(f"  Bin size: 0.5mm")
    print(f"  Number of bins: {len(counts)}")
    
    # Find peaks and valleys
    peak_indices = []
    valley_indices = []
    
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i-1] and counts[i] > counts[i+1]:
            peak_indices.append(i)
        elif counts[i] < counts[i-1] and counts[i] < counts[i+1]:
            valley_indices.append(i)
    
    print(f"\nPeaks found: {len(peak_indices)}")
    for i, peak_idx in enumerate(peak_indices[:5]):  # Top 5 peaks
        height = bin_centers[peak_idx]
        count = counts[peak_idx]
        print(f"  Peak {i+1}: {height:.1f}mm ({count:,} points)")
    
    print(f"\nValleys found: {len(valley_indices)}")
    for i, valley_idx in enumerate(valley_indices[:5]):  # Top 5 valleys
        height = bin_centers[valley_idx]
        count = counts[valley_idx]
        print(f"  Valley {i+1}: {height:.1f}mm ({count:,} points)")
    
    # Gradient analysis for steep drops
    gradient = np.gradient(counts)
    steep_drops = []
    steep_rises = []
    
    for i in range(len(gradient)):
        if gradient[i] < -1000:  # Steep drop threshold
            steep_drops.append((bin_centers[i], gradient[i], counts[i]))
        elif gradient[i] > 1000:  # Steep rise threshold
            steep_rises.append((bin_centers[i], gradient[i], counts[i]))
    
    print(f"\nSteep Drops (gradient < -1000):")
    for height, grad, count in steep_drops[:5]:
        print(f"  {height:.1f}mm: gradient={grad:.0f}, count={count:,}")
    
    print(f"\nSteep Rises (gradient > 1000):")
    for height, grad, count in steep_rises[:5]:
        print(f"  {height:.1f}mm: gradient={grad:.0f}, count={count:,}")
    
    return {
        'points': points,
        'z_values': z_values,
        'bin_centers': bin_centers,
        'counts': counts,
        'gradient': gradient,
        'peaks': [(bin_centers[i], counts[i]) for i in peak_indices],
        'valleys': [(bin_centers[i], counts[i]) for i in valley_indices],
        'steep_drops': steep_drops,
        'steep_rises': steep_rises
    }

def identify_insole_boundaries(analysis):
    """Identify the optimal height boundaries for the insole."""
    print(f"\n=== IDENTIFYING INSOLE BOUNDARIES ===")
    
    z_values = analysis['z_values']
    steep_drops = analysis['steep_drops']
    steep_rises = analysis['steep_rises']
    
    # Strategy 1: Use steep drops to find support-insole boundary
    if steep_drops:
        support_boundary = steep_drops[0][0]  # First steep drop
        print(f"Support-Insole boundary (steep drop): {support_boundary:.2f}mm")
    else:
        # Fallback: use 5th percentile
        support_boundary = np.percentile(z_values, 5)
        print(f"Support-Insole boundary (5th percentile): {support_boundary:.2f}mm")
    
    # Strategy 2: Find insole top boundary
    # Look for the main insole body (should be a significant peak)
    peaks = analysis['peaks']
    if peaks:
        # Find the highest significant peak
        significant_peaks = [(h, c) for h, c in peaks if c > len(z_values) * 0.01]  # At least 1% of points
        if significant_peaks:
            insole_top = max(significant_peaks, key=lambda x: x[0])[0]
            print(f"Insole top (highest significant peak): {insole_top:.2f}mm")
        else:
            insole_top = np.percentile(z_values, 95)
            print(f"Insole top (95th percentile): {insole_top:.2f}mm")
    else:
        insole_top = np.percentile(z_values, 95)
        print(f"Insole top (95th percentile): {insole_top:.2f}mm")
    
    # Strategy 3: Conservative boundaries
    conservative_min = max(0.0, support_boundary - 1.0)  # 1mm below support boundary
    conservative_max = min(insole_top + 2.0, np.percentile(z_values, 98))  # 2mm above top or 98th percentile
    
    print(f"\nRecommended Boundaries:")
    print(f"  Conservative Min: {conservative_min:.2f}mm")
    print(f"  Conservative Max: {conservative_max:.2f}mm")
    print(f"  Thickness: {conservative_max - conservative_min:.2f}mm")
    
    return {
        'support_boundary': support_boundary,
        'insole_top': insole_top,
        'conservative_min': conservative_min,
        'conservative_max': conservative_max
    }

def create_cleaned_insole(ply_file, boundaries, output_dir):
    """Create cleaned insole using height boundaries."""
    print(f"\n=== CREATING CLEANED INSOLE ===")
    
    # Load original point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    
    # Apply height filter
    z_min = boundaries['conservative_min']
    z_max = boundaries['conservative_max']
    
    height_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    filtered_points = points[height_mask]
    
    print(f"Height filtering ({z_min:.2f}mm to {z_max:.2f}mm):")
    print(f"  Original: {len(points):,} points")
    print(f"  Filtered: {len(filtered_points):,} points")
    print(f"  Kept: {len(filtered_points)/len(points)*100:.1f}%")
    
    # Additional outlier removal (conservative)
    if len(filtered_points) > 1000:
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Statistical outlier removal (very conservative)
        pcd_clean, inliers = pcd_filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        final_points = np.asarray(pcd_clean.points)
        
        print(f"Statistical cleaning (nb=20, std=3.0):")
        print(f"  Before: {len(filtered_points):,} points")
        print(f"  After: {len(final_points):,} points")
        print(f"  Kept: {len(final_points)/len(filtered_points)*100:.1f}%")
    else:
        final_points = filtered_points
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(final_points)
    
    # Save cleaned insole
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'scan1_height_cleaned.ply')
    o3d.io.write_point_cloud(output_file, pcd_clean)
    
    print(f"\nSaved cleaned insole: {output_file}")
    print(f"Final statistics:")
    print(f"  Points: {len(final_points):,}")
    print(f"  Z range: [{final_points[:, 2].min():.2f}, {final_points[:, 2].max():.2f}]mm")
    print(f"  Thickness: {final_points[:, 2].max() - final_points[:, 2].min():.2f}mm")
    
    return output_file, final_points

def visualize_height_analysis(analysis, boundaries, output_dir):
    """Create visualization plots of the height analysis."""
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Height histogram
    ax1.bar(analysis['bin_centers'], analysis['counts'], width=0.4, alpha=0.7, color='skyblue')
    ax1.axvline(boundaries['support_boundary'], color='red', linestyle='--', 
                label=f"Support boundary: {boundaries['support_boundary']:.2f}mm")
    ax1.axvline(boundaries['insole_top'], color='green', linestyle='--', 
                label=f"Insole top: {boundaries['insole_top']:.2f}mm")
    ax1.axvline(boundaries['conservative_min'], color='orange', linestyle='-', linewidth=2,
                label=f"Filter min: {boundaries['conservative_min']:.2f}mm")
    ax1.axvline(boundaries['conservative_max'], color='purple', linestyle='-', linewidth=2,
                label=f"Filter max: {boundaries['conservative_max']:.2f}mm")
    
    ax1.set_xlabel('Height (mm)')
    ax1.set_ylabel('Point Count')
    ax1.set_title('Height Distribution Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient analysis
    ax2.plot(analysis['bin_centers'], analysis['gradient'], color='darkblue', linewidth=1)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(-1000, color='red', linestyle=':', alpha=0.7, label='Steep drop threshold')
    ax2.axhline(1000, color='green', linestyle=':', alpha=0.7, label='Steep rise threshold')
    
    # Mark steep drops and rises
    for height, grad, count in analysis['steep_drops'][:3]:
        ax2.plot(height, grad, 'ro', markersize=8)
        ax2.annotate(f'{height:.1f}mm', (height, grad), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Height (mm)')
    ax2.set_ylabel('Gradient (points/mm)')
    ax2.set_title('Height Distribution Gradient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'height_analysis.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved analysis plot: {plot_file}")
    return plot_file

def main():
    # Input file (plane corrected)
    input_file = 'outputs/plane_corrected/scan1_plane_corrected.ply'
    output_dir = 'outputs/height_analysis'
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    try:
        # Step 1: Analyze height distribution
        analysis = analyze_height_distribution(input_file)
        
        # Step 2: Identify optimal boundaries
        boundaries = identify_insole_boundaries(analysis)
        
        # Step 3: Create cleaned insole
        cleaned_file, final_points = create_cleaned_insole(input_file, boundaries, output_dir)
        
        # Step 4: Create visualizations
        plot_file = visualize_height_analysis(analysis, boundaries, output_dir)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"✅ Cleaned insole: {cleaned_file}")
        print(f"✅ Analysis plot: {plot_file}")
        print(f"✅ Ready for STL generation!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
