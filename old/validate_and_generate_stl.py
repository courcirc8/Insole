#!/usr/bin/env python3
"""
Validate the cleaned insole and generate final STL.
"""

import numpy as np
import open3d as o3d
import os
from pathlib import Path

def validate_cleaned_insole(ply_file):
    """Validate the quality of the cleaned insole."""
    print(f"\n=== VALIDATING CLEANED INSOLE ===")
    print(f"File: {ply_file}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")
    
    points = np.asarray(pcd.points)
    print(f"Total points: {len(points):,}")
    
    # Basic statistics
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    
    print(f"\nDimensions:")
    print(f"  X range: {x_range:.1f}mm")
    print(f"  Y range: {y_range:.1f}mm")
    print(f"  Z range: {z_range:.1f}mm (thickness)")
    
    # Density analysis
    total_area = x_range * y_range
    point_density = len(points) / total_area
    print(f"\nQuality metrics:")
    print(f"  Footprint area: {total_area:.0f} mm²")
    print(f"  Point density: {point_density:.1f} points/mm²")
    
    # Check for outliers (points far from main body)
    xy_center = np.mean(points[:, :2], axis=0)
    xy_distances = np.linalg.norm(points[:, :2] - xy_center, axis=1)
    outlier_threshold = np.percentile(xy_distances, 99)
    outliers = np.sum(xy_distances > outlier_threshold)
    
    print(f"  Center: ({xy_center[0]:.1f}, {xy_center[1]:.1f})")
    print(f"  99% of points within: {outlier_threshold:.1f}mm of center")
    print(f"  Potential outliers: {outliers} ({outliers/len(points)*100:.2f}%)")
    
    # Height distribution check
    z_values = points[:, 2]
    z_percentiles = np.percentile(z_values, [1, 5, 25, 50, 75, 95, 99])
    print(f"\nHeight percentiles:")
    for i, p in enumerate([1, 5, 25, 50, 75, 95, 99]):
        print(f"  {p}%: {z_percentiles[i]:.2f}mm")
    
    # Quality assessment
    quality_score = 100.0
    issues = []
    
    if point_density < 50:
        quality_score -= 20
        issues.append(f"Low point density ({point_density:.1f} < 50 pts/mm²)")
    
    if outliers > len(points) * 0.01:
        quality_score -= 15
        issues.append(f"Too many outliers ({outliers/len(points)*100:.2f}% > 1%)")
    
    if z_range < 2 or z_range > 15:
        quality_score -= 10
        issues.append(f"Unusual thickness ({z_range:.1f}mm)")
    
    if len(points) < 50000:
        quality_score -= 10
        issues.append(f"Low point count ({len(points):,} < 50,000)")
    
    print(f"\n=== QUALITY ASSESSMENT ===")
    print(f"Quality Score: {quality_score:.1f}%")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ No significant issues found!")
    
    return {
        'points': points,
        'quality_score': quality_score,
        'issues': issues,
        'dimensions': (x_range, y_range, z_range),
        'point_density': point_density
    }

def generate_outline_and_stl(points, output_dir):
    """Generate outline and STL from cleaned points."""
    print(f"\n=== GENERATING OUTLINE AND STL ===")
    
    # Step 1: Generate Z-aware outline
    print("Generating Z-aware outline...")
    
    # Use points at different height levels for better outline
    height_levels = [0.2, 1.0, 2.0, 3.0]  # mm
    all_outline_points = []
    
    for level in height_levels:
        # Find points near this height level
        tolerance = 0.5  # mm
        level_mask = np.abs(points[:, 2] - level) < tolerance
        level_points = points[level_mask]
        
        if len(level_points) < 100:
            continue
            
        print(f"  Level {level}mm: {len(level_points):,} points")
        
        # Find convex hull at this level using scipy
        xy_points = level_points[:, :2]
        
        # Use scipy for 2D convex hull
        from scipy.spatial import ConvexHull
        
        try:
            hull_2d = ConvexHull(xy_points)
            hull_points = xy_points[hull_2d.vertices]
        except Exception as e:
            print(f"    ConvexHull failed for level {level}mm: {e}")
            # Fallback: use boundary points
            x_coords, y_coords = xy_points[:, 0], xy_points[:, 1]
            boundary_indices = []
            
            # Find extreme points
            boundary_indices.extend([
                np.argmin(x_coords), np.argmax(x_coords),
                np.argmin(y_coords), np.argmax(y_coords)
            ])
            
            hull_points = xy_points[boundary_indices]
        
        all_outline_points.append(hull_points)
    
    # Combine all outline points and find overall convex hull
    if all_outline_points:
        combined_outline = np.vstack(all_outline_points)
        
        # Final 2D convex hull using scipy
        try:
            final_hull = ConvexHull(combined_outline)
            outline_points = combined_outline[final_hull.vertices]
            print(f"Final outline: {len(outline_points)} points")
        except Exception as e:
            print(f"Final ConvexHull failed: {e}")
            # Use first level outline as fallback
            outline_points = all_outline_points[0]
            print(f"Using first level outline: {len(outline_points)} points")
    else:
        # Fallback: simple 2D convex hull of all XY points
        xy_all = points[:, :2]
        
        try:
            hull_all = ConvexHull(xy_all)
            outline_points = xy_all[hull_all.vertices]
            print(f"Fallback outline: {len(outline_points)} points")
        except Exception as e:
            print(f"Fallback ConvexHull failed: {e}")
            # Ultimate fallback: bounding box
            x_min, x_max = xy_all[:, 0].min(), xy_all[:, 0].max()
            y_min, y_max = xy_all[:, 1].min(), xy_all[:, 1].max()
            outline_points = np.array([
                [x_min, y_min], [x_max, y_min], 
                [x_max, y_max], [x_min, y_max]
            ])
            print(f"Using bounding box outline: {len(outline_points)} points")
    
    # Save outline
    outline_file = os.path.join(output_dir, 'scan1_height_cleaned_outline.csv')
    np.savetxt(outline_file, outline_points, delimiter=',', 
               header='x,y', comments='', fmt='%.6f')
    print(f"Saved outline: {outline_file}")
    
    # Step 2: Generate heightmap
    print("\nGenerating heightmap...")
    
    # Define grid parameters
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Grid resolution (adjust based on point density)
    grid_size = 0.5  # mm
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    print(f"Grid parameters:")
    print(f"  Resolution: {grid_size}mm")
    print(f"  Grid size: {x_bins} x {y_bins}")
    
    # Create grid
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    
    # Assign points to grid cells and compute max height
    heightmap = np.full((y_bins, x_bins), np.nan)
    
    for i in range(len(points)):
        x, y, z = points[i]
        
        # Find grid cell
        x_idx = np.searchsorted(x_edges, x) - 1
        y_idx = np.searchsorted(y_edges, y) - 1
        
        # Clamp to valid range
        x_idx = max(0, min(x_bins - 1, x_idx))
        y_idx = max(0, min(y_bins - 1, y_idx))
        
        # Update heightmap with maximum height
        if np.isnan(heightmap[y_idx, x_idx]) or z > heightmap[y_idx, x_idx]:
            heightmap[y_idx, x_idx] = z
    
    # Fill holes with interpolation
    mask = ~np.isnan(heightmap)
    filled_cells = np.sum(mask)
    total_cells = heightmap.size
    coverage = filled_cells / total_cells * 100
    
    print(f"Heightmap coverage: {coverage:.1f}% ({filled_cells}/{total_cells} cells)")
    
    # Simple hole filling
    from scipy.ndimage import binary_dilation
    from scipy.interpolate import griddata
    
    # Find holes
    holes = ~mask
    if np.any(holes):
        # Dilate the mask to include nearby cells for interpolation
        dilated_mask = binary_dilation(mask, iterations=3)
        
        # Get coordinates of valid and hole cells
        y_coords, x_coords = np.meshgrid(range(y_bins), range(x_bins), indexing='ij')
        
        valid_points = np.column_stack([
            x_coords[mask].ravel(),
            y_coords[mask].ravel()
        ])
        valid_values = heightmap[mask].ravel()
        
        hole_points = np.column_stack([
            x_coords[holes & dilated_mask].ravel(),
            y_coords[holes & dilated_mask].ravel()
        ])
        
        if len(hole_points) > 0 and len(valid_points) > 3:
            interpolated = griddata(valid_points, valid_values, hole_points, method='linear')
            heightmap[holes & dilated_mask] = interpolated
            print(f"Filled {len(hole_points)} holes with interpolation")
    
    # Save heightmap
    heightmap_file = os.path.join(output_dir, 'scan1_height_cleaned_heightmap.npy')
    np.save(heightmap_file, heightmap)
    print(f"Saved heightmap: {heightmap_file}")
    
    # Step 3: Generate STL
    print("\nGenerating STL...")
    
    # Create mesh from heightmap
    vertices = []
    faces = []
    vertex_idx = 0
    
    # Generate vertices
    for j in range(y_bins):
        for i in range(x_bins):
            if not np.isnan(heightmap[j, i]):
                x = x_min + i * grid_size
                y = y_min + j * grid_size
                z = heightmap[j, i]
                vertices.append([x, y, z])
            else:
                vertices.append([0, 0, 0])  # Placeholder
    
    # Generate faces (triangles)
    for j in range(y_bins - 1):
        for i in range(x_bins - 1):
            # Check if all 4 corners have valid heights
            corners = [
                (j, i), (j, i+1), (j+1, i), (j+1, i+1)
            ]
            
            valid_corners = []
            for row, col in corners:
                if not np.isnan(heightmap[row, col]):
                    valid_corners.append(row * x_bins + col)
            
            if len(valid_corners) >= 3:
                # Create triangles
                if len(valid_corners) == 4:
                    # Two triangles for a quad
                    v0, v1, v2, v3 = valid_corners
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
                elif len(valid_corners) == 3:
                    # One triangle
                    faces.append(valid_corners)
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.faces = o3d.utility.Vector3iVector(faces)
    
    # Clean up mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"Mesh statistics:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    
    # Save STL
    stl_file = os.path.join(output_dir, 'scan1_height_cleaned.stl')
    o3d.io.write_triangle_mesh(stl_file, mesh)
    print(f"Saved STL: {stl_file}")
    
    return outline_file, heightmap_file, stl_file

def main():
    # Input file
    input_file = 'outputs/height_analysis/scan1_height_cleaned.ply'
    output_dir = 'outputs/height_analysis'
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    try:
        # Step 1: Validate cleaned insole
        validation = validate_cleaned_insole(input_file)
        
        if validation['quality_score'] < 70:
            print(f"\n⚠️  WARNING: Quality score is low ({validation['quality_score']:.1f}%)")
            print("Consider reviewing the cleaning parameters.")
        
        # Step 2: Generate outline and STL
        outline_file, heightmap_file, stl_file = generate_outline_and_stl(
            validation['points'], output_dir
        )
        
        print(f"\n=== GENERATION COMPLETE ===")
        print(f"✅ Validated insole quality: {validation['quality_score']:.1f}%")
        print(f"✅ Generated outline: {outline_file}")
        print(f"✅ Generated heightmap: {heightmap_file}")
        print(f"✅ Generated STL: {stl_file}")
        print(f"\n🎯 Valid insole object constructed successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
