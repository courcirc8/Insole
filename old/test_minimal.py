#!/usr/bin/env python3
"""
Minimal test to identify the segfault issue.
"""

import numpy as np
import open3d as o3d

def test_minimal():
    print("Testing minimal operations...")
    
    # Test 1: Load point cloud
    try:
        print("Loading point cloud...")
        pcd = o3d.io.read_point_cloud('outputs/height_analysis/scan1_height_cleaned.ply')
        print(f"Loaded {len(pcd.points):,} points")
        
        # Test 2: Convert to numpy
        print("Converting to numpy...")
        points = np.asarray(pcd.points)
        print(f"Numpy array shape: {points.shape}")
        
        # Test 3: Basic statistics
        print("Computing statistics...")
        print(f"X range: {points[:, 0].min():.1f} to {points[:, 0].max():.1f}")
        print(f"Y range: {points[:, 1].min():.1f} to {points[:, 1].max():.1f}")
        print(f"Z range: {points[:, 2].min():.1f} to {points[:, 2].max():.1f}")
        
        # Test 4: Small subset processing
        print("Testing subset...")
        subset_size = min(1000, len(points))
        subset = points[:subset_size]
        print(f"Subset size: {len(subset)}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_minimal()
