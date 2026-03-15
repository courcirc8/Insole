#!/usr/bin/env python3
"""
Fixed elevation analysis script that properly handles negative points.
First elevates all points so minimum Z = 0, then analyzes elevation distribution.

Usage:
    python analyze_elevation_fixed.py <input_ply> [--max_height 10] [--resolution 0.1] [--plot]
"""

import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path


def elevate_points_to_zero(points):
    """
    Elevate all points so that the minimum Z becomes 0.
    
    Args:
        points: Point cloud array (N, 3)
        
    Returns:
        elevated_points: Points with minimum Z = 0
        elevation_offset: Amount added to all Z values
    """
    z_min = points[:, 2].min()
    elevation_offset = -z_min  # Amount to add to make min = 0
    
    elevated_points = points.copy()
    elevated_points[:, 2] = points[:, 2] + elevation_offset
    
    return elevated_points, elevation_offset


def analyze_elevation_distribution_fixed(points, max_height=10.0, resolution=0.1):
    """
    Analyze elevation distribution after fixing negative points.
    
    Args:
        points: Point cloud array (N, 3) with min Z = 0
        max_height: Maximum height to analyze (mm)
        resolution: Height resolution for analysis (mm)
        
    Returns:
        heights: Array of height values
        counts_per_bin: Points per height bin
        cumulative_counts: Cumulative point counts
        gradients: First derivative of cumulative
        second_derivative: Second derivative for steep changes
        steep_points: Indices of steep change points
    """
    z_values = points[:, 2]
    
    # Create height bins from 0 to max_height
    heights = np.arange(0, max_height + resolution, resolution)
    counts_per_bin = np.zeros(len(heights))
    cumulative_counts = np.zeros(len(heights))
    
    # Count points in each height bin and cumulative
    for i, height in enumerate(heights):
        # Points up to this height (cumulative)
        cumulative_mask = z_values <= height
        cumulative_counts[i] = np.sum(cumulative_mask)
        
        # Points in this specific bin
        if i == 0:
            bin_mask = z_values <= height
        else:
            prev_height = heights[i-1]
            bin_mask = (z_values > prev_height) & (z_values <= height)
        counts_per_bin[i] = np.sum(bin_mask)
    
    # Calculate gradients (rate of change in cumulative)
    gradients = np.gradient(cumulative_counts)
    
    # Find steep changes (second derivative)
    second_derivative = np.gradient(gradients)
    
    # Find points where gradient changes significantly
    if len(second_derivative) > 1:
        steep_threshold = np.std(second_derivative) * 1.5
        steep_points = np.where(np.abs(second_derivative) > steep_threshold)[0]
    else:
        steep_points = np.array([])
    
    return heights, counts_per_bin, cumulative_counts, gradients, second_derivative, steep_points


def find_optimal_cutoffs(heights, counts_per_bin, cumulative_counts, gradients, steep_points, total_points):
    """
    Find multiple optimal cutoff points using different methods.
    """
    cutoffs = {}
    
    # Method 1: Steep drop analysis
    if len(steep_points) > 0:
        # Look for significant negative changes in second derivative
        negative_steep = []
        for idx in steep_points:
            if idx > 0 and idx < len(gradients):
                if gradients[idx] < gradients[max(0, idx-1)] * 0.7:  # 30% drop
                    negative_steep.append(idx)
        
        if negative_steep:
            cutoffs['steep_drop'] = heights[negative_steep[0]]
        else:
            cutoffs['steep_drop'] = heights[steep_points[0]] if len(steep_points) > 0 else 2.0
    else:
        cutoffs['steep_drop'] = 2.0
    
    # Method 2: Gradient threshold (where gradient drops to 20% of peak)
    if len(gradients) > 5:
        peak_gradient = np.max(gradients[1:len(gradients)//2])  # Peak in first half
        threshold_gradient = peak_gradient * 0.2
        threshold_indices = np.where(gradients < threshold_gradient)[0]
        threshold_indices = threshold_indices[threshold_indices > 5]  # Skip very early points
        
        if len(threshold_indices) > 0:
            cutoffs['gradient_20pct'] = heights[threshold_indices[0]]
        else:
            cutoffs['gradient_20pct'] = 5.0
    else:
        cutoffs['gradient_20pct'] = 5.0
    
    # Method 3: Density drop (where bin density drops significantly)
    if len(counts_per_bin) > 10:
        # Find peak density (excluding first few bins which might be sparse)
        peak_density = np.max(counts_per_bin[5:min(50, len(counts_per_bin))])
        density_threshold = peak_density * 0.3
        
        # Find first point after peak where density drops below threshold
        peak_idx = np.argmax(counts_per_bin[5:min(50, len(counts_per_bin))]) + 5
        drop_indices = np.where(counts_per_bin[peak_idx:] < density_threshold)[0]
        
        if len(drop_indices) > 0:
            cutoffs['density_drop'] = heights[peak_idx + drop_indices[0]]
        else:
            cutoffs['density_drop'] = 6.0
    else:
        cutoffs['density_drop'] = 6.0
    
    # Method 4: Percentage-based cutoffs
    cutoffs['50pct'] = heights[np.where(cumulative_counts >= total_points * 0.5)[0][0]] if np.any(cumulative_counts >= total_points * 0.5) else 5.0
    cutoffs['80pct'] = heights[np.where(cumulative_counts >= total_points * 0.8)[0][0]] if np.any(cumulative_counts >= total_points * 0.8) else 7.0
    cutoffs['90pct'] = heights[np.where(cumulative_counts >= total_points * 0.9)[0][0]] if np.any(cumulative_counts >= total_points * 0.9) else 8.0
    
    return cutoffs


def plot_fixed_analysis(heights, counts_per_bin, cumulative_counts, gradients, second_derivative, 
                       steep_points, cutoffs, elevation_offset, output_path=None):
    """
    Plot the fixed elevation analysis results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Point counts per height bin
    ax1.bar(heights, counts_per_bin, width=0.08, alpha=0.7, color='lightblue', edgecolor='navy')
    ax1.set_xlabel('Height above lowest point (mm)')
    ax1.set_ylabel('Points per bin')
    ax1.set_title(f'Point Distribution by Height (elevated by {elevation_offset:.1f}mm)')
    ax1.grid(True, alpha=0.3)
    
    # Add cutoff lines
    colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
    for i, (method, cutoff) in enumerate(cutoffs.items()):
        if i < len(colors):
            ax1.axvline(cutoff, color=colors[i], linestyle='--', alpha=0.7, 
                       label=f'{method}: {cutoff:.1f}mm')
    ax1.legend()
    
    # Plot 2: Cumulative counts
    ax2.plot(heights, cumulative_counts, linewidth=2, color='green')
    ax2.set_xlabel('Height above lowest point (mm)')
    ax2.set_ylabel('Cumulative points')
    ax2.set_title('Cumulative Point Count')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage lines
    total = cumulative_counts[-1]
    for pct in [50, 80, 90]:
        pct_line = total * (pct/100)
        ax2.axhline(pct_line, color='gray', linestyle=':', alpha=0.5, label=f'{pct}%')
    ax2.legend()
    
    # Plot 3: First derivative (gradient)
    ax3.plot(heights, gradients, linewidth=2, color='orange')
    ax3.set_xlabel('Height above lowest point (mm)')
    ax3.set_ylabel('Gradient (points/mm)')
    ax3.set_title('Rate of Change (First Derivative)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Second derivative with steep points
    ax4.plot(heights, second_derivative, linewidth=2, color='purple')
    if len(steep_points) > 0:
        ax4.scatter(heights[steep_points], second_derivative[steep_points], 
                   color='red', s=50, zorder=5, label=f'{len(steep_points)} steep changes')
        ax4.legend()
    ax4.set_xlabel('Height above lowest point (mm)')
    ax4.set_ylabel('Second Derivative')
    ax4.set_title('Curvature Analysis (Second Derivative)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Fixed elevation analysis with proper Z=0 alignment')
    parser.add_argument('input_ply', help='Input PLY file')
    parser.add_argument('--max_height', type=float, default=15.0, 
                       help='Maximum height to analyze (mm)')
    parser.add_argument('--resolution', type=float, default=0.1,
                       help='Height resolution for analysis (mm)')
    parser.add_argument('--plot', action='store_true',
                       help='Show elevation analysis plots')
    parser.add_argument('--save_plot', type=str,
                       help='Save plot to file (e.g., elevation_fixed.png)')
    
    args = parser.parse_args()
    
    # Load point cloud
    print(f"📂 Loading: {args.input_ply}")
    pcd = o3d.io.read_point_cloud(args.input_ply)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print("❌ Error: No points loaded from file")
        return
    
    print(f"📊 Original data: {len(points):,} points")
    print(f"📏 Original Z range: [{points[:,2].min():.1f}, {points[:,2].max():.1f}] mm")
    
    # Fix negative points by elevating to Z=0
    print(f"\n🔧 FIXING NEGATIVE POINTS:")
    elevated_points, elevation_offset = elevate_points_to_zero(points)
    print(f"Elevation offset applied: +{elevation_offset:.1f} mm")
    print(f"Fixed Z range: [{elevated_points[:,2].min():.1f}, {elevated_points[:,2].max():.1f}] mm")
    
    # Analyze elevation distribution
    print(f"\n🔍 Analyzing elevation distribution (0-{args.max_height}mm, resolution={args.resolution}mm)")
    
    heights, counts_per_bin, cumulative_counts, gradients, second_derivative, steep_points = \
        analyze_elevation_distribution_fixed(elevated_points, args.max_height, args.resolution)
    
    # Find optimal cutoffs
    cutoffs = find_optimal_cutoffs(heights, counts_per_bin, cumulative_counts, gradients, 
                                 steep_points, len(elevated_points))
    
    # Print results
    print(f"\n📈 FIXED ELEVATION ANALYSIS RESULTS:")
    print(f"Total points: {len(elevated_points):,}")
    print(f"Height bins analyzed: {len(heights)}")
    print(f"Steep change points found: {len(steep_points)}")
    
    print(f"\n🎯 OPTIMAL CUTOFF RECOMMENDATIONS:")
    for method, cutoff in cutoffs.items():
        mask = elevated_points[:,2] <= cutoff
        points_below = np.sum(mask)
        percentage = (points_below / len(elevated_points)) * 100
        print(f"  {method:15s}: {cutoff:5.1f}mm → {points_below:7,} points ({percentage:5.1f}%)")
    
    # Print elevation statistics
    print(f"\n📊 ELEVATION STATISTICS (after fixing negatives):")
    for height_threshold in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        if height_threshold <= args.max_height:
            mask = elevated_points[:,2] <= height_threshold
            count = np.sum(mask)
            percentage = (count / len(elevated_points)) * 100
            print(f"≤ {height_threshold:4.1f}mm: {count:7,} points ({percentage:6.1f}%)")
    
    # Show steep change details
    if len(steep_points) > 0:
        print(f"\n⚡ STEEP CHANGES DETECTED:")
        for i, idx in enumerate(steep_points[:8]):  # Show first 8
            height = heights[idx]
            gradient_change = second_derivative[idx]
            cumulative_at_point = cumulative_counts[idx]
            pct_at_point = (cumulative_at_point / len(elevated_points)) * 100
            print(f"  {i+1}. {height:5.1f}mm: gradient change = {gradient_change:8.1f}, "
                  f"cumulative = {cumulative_at_point:6,.0f} ({pct_at_point:4.1f}%)")
    
    # Create plots if requested
    if args.plot or args.save_plot:
        plot_fixed_analysis(heights, counts_per_bin, cumulative_counts, gradients, 
                          second_derivative, steep_points, cutoffs, elevation_offset,
                          args.save_plot)
    
    # Save the elevation-fixed point cloud
    output_dir = Path(args.input_ply).parent / "elevation_fixed"
    output_dir.mkdir(exist_ok=True)
    
    fixed_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(elevated_points))
    fixed_path = output_dir / f"{Path(args.input_ply).stem}_elevation_fixed.ply"
    o3d.io.write_point_cloud(str(fixed_path), fixed_pcd)
    print(f"\n💾 Saved elevation-fixed points: {fixed_path}")
    
    print(f"\n✅ Fixed elevation analysis complete!")


if __name__ == "__main__":
    main()


