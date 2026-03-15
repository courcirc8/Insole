#!/usr/bin/env python3
"""
Analyze elevation distribution to find steep changes in point density curve.
This helps determine optimal height cutoff for insole isolation.

Usage:
    python analyze_elevation_curve.py <input_ply> [--max_height 10] [--resolution 0.1] [--plot]
"""

import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_elevation_distribution(points, max_height=10.0, resolution=0.1):
    """
    Analyze elevation distribution and find steep changes in the curve.
    
    Args:
        points: Point cloud array (N, 3)
        max_height: Maximum height to analyze (mm)
        resolution: Height resolution for analysis (mm)
        
    Returns:
        heights: Array of height values
        counts: Array of point counts at each height
        gradients: Array of gradient changes
        steep_points: Indices of steep change points
    """
    z_values = points[:, 2]
    
    # Create height bins
    heights = np.arange(0, max_height + resolution, resolution)
    counts = np.zeros(len(heights))
    
    # Count points at each elevation
    for i, height in enumerate(heights):
        if i == 0:
            # Count points from 0 to first height
            mask = (z_values >= 0) & (z_values <= height)
        else:
            # Count points in this height bin
            prev_height = heights[i-1]
            mask = (z_values > prev_height) & (z_values <= height)
        counts[i] = np.sum(mask)
    
    # Calculate cumulative counts (total points below each height)
    cumulative_counts = np.cumsum(counts)
    
    # Calculate gradients (rate of change)
    gradients = np.gradient(cumulative_counts)
    
    # Find steep changes (second derivative)
    second_derivative = np.gradient(gradients)
    
    # Find points where gradient changes significantly
    steep_threshold = np.std(second_derivative) * 2
    steep_points = np.where(np.abs(second_derivative) > steep_threshold)[0]
    
    return heights, counts, cumulative_counts, gradients, second_derivative, steep_points


def find_optimal_cutoff(heights, cumulative_counts, gradients, steep_points):
    """
    Find optimal height cutoff based on steep changes in the curve.
    
    Args:
        heights: Height values
        cumulative_counts: Cumulative point counts
        gradients: First derivative of cumulative counts
        steep_points: Indices of steep change points
        
    Returns:
        optimal_height: Suggested optimal cutoff height
        analysis: Dictionary with analysis details
    """
    analysis = {}
    
    # Find the steepest drop in gradient (indicates transition from dense to sparse)
    if len(steep_points) > 0:
        # Look for negative steep changes (drops in gradient)
        negative_steep = []
        for idx in steep_points:
            if idx > 0 and gradients[idx] < gradients[idx-1] * 0.5:  # 50% drop
                negative_steep.append(idx)
        
        if negative_steep:
            # Choose the first significant drop
            optimal_idx = negative_steep[0]
            optimal_height = heights[optimal_idx]
            analysis['method'] = 'steep_drop'
            analysis['steep_points'] = len(negative_steep)
        else:
            # Fallback: find where gradient drops below 10% of maximum
            max_gradient = np.max(gradients[:len(gradients)//2])  # Look in first half
            threshold_gradient = max_gradient * 0.1
            threshold_idx = np.where(gradients < threshold_gradient)[0]
            if len(threshold_idx) > 0:
                optimal_idx = threshold_idx[0]
                optimal_height = heights[optimal_idx]
                analysis['method'] = 'gradient_threshold'
            else:
                optimal_height = 5.0  # Default fallback
                analysis['method'] = 'default'
    else:
        # Fallback: find elbow point
        # Use ratio of cumulative counts to total
        total_points = cumulative_counts[-1]
        ratios = cumulative_counts / total_points
        
        # Find where we have captured 80-90% of points
        target_ratio = 0.85
        target_idx = np.where(ratios >= target_ratio)[0]
        if len(target_idx) > 0:
            optimal_idx = target_idx[0]
            optimal_height = heights[optimal_idx]
            analysis['method'] = 'ratio_based'
        else:
            optimal_height = 5.0
            analysis['method'] = 'default'
    
    analysis['optimal_height'] = optimal_height
    analysis['points_below_cutoff'] = cumulative_counts[optimal_idx] if 'optimal_idx' in locals() else 0
    
    return optimal_height, analysis


def plot_elevation_analysis(heights, counts, cumulative_counts, gradients, second_derivative, 
                          steep_points, optimal_height, output_path=None):
    """
    Plot elevation analysis results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Point counts per height bin
    ax1.bar(heights, counts, width=0.08, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Height (mm)')
    ax1.set_ylabel('Points per bin')
    ax1.set_title('Point Distribution by Height')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(optimal_height, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal cutoff: {optimal_height:.1f}mm')
    ax1.legend()
    
    # Plot 2: Cumulative counts
    ax2.plot(heights, cumulative_counts, linewidth=2, color='green')
    ax2.set_xlabel('Height (mm)')
    ax2.set_ylabel('Cumulative points')
    ax2.set_title('Cumulative Point Count')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(optimal_height, color='red', linestyle='--', linewidth=2,
                label=f'Optimal cutoff: {optimal_height:.1f}mm')
    ax2.legend()
    
    # Plot 3: First derivative (gradient)
    ax3.plot(heights, gradients, linewidth=2, color='orange')
    ax3.set_xlabel('Height (mm)')
    ax3.set_ylabel('Gradient (points/mm)')
    ax3.set_title('Rate of Change (First Derivative)')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(optimal_height, color='red', linestyle='--', linewidth=2)
    
    # Plot 4: Second derivative with steep points
    ax4.plot(heights, second_derivative, linewidth=2, color='purple')
    if len(steep_points) > 0:
        ax4.scatter(heights[steep_points], second_derivative[steep_points], 
                   color='red', s=50, zorder=5, label='Steep changes')
    ax4.set_xlabel('Height (mm)')
    ax4.set_ylabel('Second Derivative')
    ax4.set_title('Curvature Analysis (Second Derivative)')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(optimal_height, color='red', linestyle='--', linewidth=2)
    if len(steep_points) > 0:
        ax4.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze elevation distribution for optimal cutoff')
    parser.add_argument('input_ply', help='Input PLY file (plane-corrected)')
    parser.add_argument('--max_height', type=float, default=10.0, 
                       help='Maximum height to analyze (mm)')
    parser.add_argument('--resolution', type=float, default=0.1,
                       help='Height resolution for analysis (mm)')
    parser.add_argument('--plot', action='store_true',
                       help='Show elevation analysis plots')
    parser.add_argument('--save_plot', type=str,
                       help='Save plot to file (e.g., elevation_analysis.png)')
    
    args = parser.parse_args()
    
    # Load point cloud
    print(f"📂 Loading: {args.input_ply}")
    pcd = o3d.io.read_point_cloud(args.input_ply)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print("❌ Error: No points loaded from file")
        return
    
    print(f"📊 Loaded {len(points):,} points")
    print(f"📏 Z range: [{points[:,2].min():.1f}, {points[:,2].max():.1f}] mm")
    
    # Analyze elevation distribution
    print(f"\n🔍 Analyzing elevation distribution (0-{args.max_height}mm, resolution={args.resolution}mm)")
    
    heights, counts, cumulative_counts, gradients, second_derivative, steep_points = \
        analyze_elevation_distribution(points, args.max_height, args.resolution)
    
    # Find optimal cutoff
    optimal_height, analysis = find_optimal_cutoff(heights, cumulative_counts, gradients, steep_points)
    
    # Print results
    print(f"\n📈 ELEVATION ANALYSIS RESULTS:")
    print(f"Total points analyzed: {len(points):,}")
    print(f"Height bins: {len(heights)}")
    print(f"Steep change points found: {len(steep_points)}")
    
    print(f"\n🎯 OPTIMAL CUTOFF ANALYSIS:")
    print(f"Recommended cutoff height: {optimal_height:.1f} mm")
    print(f"Detection method: {analysis['method']}")
    if 'points_below_cutoff' in analysis:
        retention_rate = (analysis['points_below_cutoff'] / len(points)) * 100
        print(f"Points below cutoff: {analysis['points_below_cutoff']:,} ({retention_rate:.1f}%)")
    
    # Print elevation statistics
    print(f"\n📊 ELEVATION STATISTICS:")
    for height_threshold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if height_threshold <= args.max_height:
            mask = points[:,2] <= height_threshold
            count = np.sum(mask)
            percentage = (count / len(points)) * 100
            print(f"≤ {height_threshold:2d}mm: {count:6,} points ({percentage:5.1f}%)")
    
    # Show steep change locations
    if len(steep_points) > 0:
        print(f"\n⚡ STEEP CHANGES DETECTED:")
        for i, idx in enumerate(steep_points[:5]):  # Show first 5
            height = heights[idx]
            gradient_change = second_derivative[idx]
            print(f"  {i+1}. Height {height:.1f}mm: gradient change = {gradient_change:.1f}")
    
    # Create plots if requested
    if args.plot or args.save_plot:
        plot_elevation_analysis(heights, counts, cumulative_counts, gradients, 
                              second_derivative, steep_points, optimal_height,
                              args.save_plot)
    
    print(f"\n✅ Analysis complete! Recommended cutoff: {optimal_height:.1f}mm")


if __name__ == "__main__":
    main()


