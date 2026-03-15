#!/usr/bin/env python3
"""
Simple and robust insole construction from height-cleaned data.
"""

import numpy as np
import open3d as o3d
import os

def construct_simple_insole(ply_file, output_dir):
    """Construct insole with simple but robust methods."""
    print(f"\n=== CONSTRUCTING INSOLE ===")
    print(f"Input: {ply_file}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    print(f"Points: {len(points):,}")
    
    # Basic statistics
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    
    print(f"Dimensions: {x_range:.1f} x {y_range:.1f} x {z_range:.1f} mm")
    
    # Step 1: Create simple outline (alpha shape approach)
    print("\nGenerating outline...")
    
    # Get XY points
    xy_points = points[:, :2]
    
    # Simple boundary extraction: use percentile-based approach
    x_coords, y_coords = xy_points[:, 0], xy_points[:, 1]
    
    # Create a dense boundary by finding edge points
    n_sectors = 64  # Number of angular sectors
    center_x, center_y = np.mean(x_coords), np.mean(y_coords)
    
    # Convert to polar coordinates
    dx = x_coords - center_x
    dy = y_coords - center_y
    angles = np.arctan2(dy, dx)
    distances = np.sqrt(dx**2 + dy**2)
    
    # Find boundary points for each angular sector
    boundary_points = []
    for i in range(n_sectors):
        angle_start = -np.pi + i * 2 * np.pi / n_sectors
        angle_end = -np.pi + (i + 1) * 2 * np.pi / n_sectors
        
        # Find points in this angular sector
        if i == n_sectors - 1:  # Handle wrap-around
            mask = (angles >= angle_start) | (angles < angle_end)
        else:
            mask = (angles >= angle_start) & (angles < angle_end)
        
        if np.any(mask):
            # Find the furthest point in this sector
            sector_distances = distances[mask]
            sector_points = xy_points[mask]
            
            # Use 95th percentile distance to avoid extreme outliers
            target_distance = np.percentile(sector_distances, 95)
            closest_idx = np.argmin(np.abs(sector_distances - target_distance))
            
            boundary_points.append(sector_points[closest_idx])
    
    outline_points = np.array(boundary_points)
    print(f"Generated outline with {len(outline_points)} points")
    
    # Save outline
    os.makedirs(output_dir, exist_ok=True)
    outline_file = os.path.join(output_dir, 'scan1_simple_outline.csv')
    np.savetxt(outline_file, outline_points, delimiter=',', 
               header='x,y', comments='', fmt='%.6f')
    print(f"Saved: {outline_file}")
    
    # Step 2: Create heightmap
    print("\nGenerating heightmap...")
    
    # Grid parameters
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Adaptive grid size based on point density
    total_area = (x_max - x_min) * (y_max - y_min)
    point_density = len(points) / total_area
    
    if point_density > 10:
        grid_size = 0.5  # High density: fine grid
    elif point_density > 5:
        grid_size = 1.0  # Medium density: medium grid
    else:
        grid_size = 2.0  # Low density: coarse grid
    
    print(f"Point density: {point_density:.1f} pts/mm²")
    print(f"Grid size: {grid_size}mm")
    
    # Create grid
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    print(f"Grid dimensions: {x_bins} x {y_bins} = {x_bins * y_bins:,} cells")
    
    # Initialize heightmap
    heightmap = np.full((y_bins, x_bins), np.nan)
    point_counts = np.zeros((y_bins, x_bins), dtype=int)
    
    # Populate heightmap
    for i in range(len(points)):
        x, y, z = points[i]
        
        # Find grid cell
        x_idx = int((x - x_min) / grid_size)
        y_idx = int((y - y_min) / grid_size)
        
        # Clamp to valid range
        x_idx = max(0, min(x_bins - 1, x_idx))
        y_idx = max(0, min(y_bins - 1, y_idx))
        
        # Update heightmap with maximum height
        if np.isnan(heightmap[y_idx, x_idx]) or z > heightmap[y_idx, x_idx]:
            heightmap[y_idx, x_idx] = z
        
        point_counts[y_idx, x_idx] += 1
    
    # Calculate coverage
    filled_cells = np.sum(~np.isnan(heightmap))
    coverage = filled_cells / (x_bins * y_bins) * 100
    print(f"Coverage: {coverage:.1f}% ({filled_cells:,}/{x_bins * y_bins:,} cells)")
    
    # Simple hole filling using nearest neighbor
    if coverage < 80:  # Only if significant holes
        print("Filling holes...")
        
        # Find holes
        holes = np.isnan(heightmap)
        
        # For each hole, find nearest valid cell
        for j in range(y_bins):
            for i in range(x_bins):
                if holes[j, i]:
                    # Search in expanding squares
                    found = False
                    for radius in range(1, min(x_bins, y_bins) // 2):
                        for dj in range(-radius, radius + 1):
                            for di in range(-radius, radius + 1):
                                ni, nj = i + di, j + dj
                                if (0 <= ni < x_bins and 0 <= nj < y_bins and 
                                    not np.isnan(heightmap[nj, ni])):
                                    heightmap[j, i] = heightmap[nj, ni]
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
        
        new_coverage = np.sum(~np.isnan(heightmap)) / (x_bins * y_bins) * 100
        print(f"After filling: {new_coverage:.1f}% coverage")
    
    # Save heightmap
    heightmap_file = os.path.join(output_dir, 'scan1_simple_heightmap.npy')
    np.save(heightmap_file, heightmap)
    print(f"Saved: {heightmap_file}")
    
    # Step 3: Generate STL mesh
    print("\nGenerating STL...")
    
    # Create mesh vertices
    vertices = []
    vertex_map = np.full((y_bins, x_bins), -1, dtype=int)
    
    for j in range(y_bins):
        for i in range(x_bins):
            if not np.isnan(heightmap[j, i]):
                x = x_min + i * grid_size
                y = y_min + j * grid_size
                z = heightmap[j, i]
                
                vertex_map[j, i] = len(vertices)
                vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    print(f"Vertices: {len(vertices):,}")
    
    # Create faces
    faces = []
    
    for j in range(y_bins - 1):
        for i in range(x_bins - 1):
            # Get vertex indices for this cell
            v00 = vertex_map[j, i]
            v10 = vertex_map[j, i + 1]
            v01 = vertex_map[j + 1, i]
            v11 = vertex_map[j + 1, i + 1]
            
            # Only create faces if we have valid vertices
            valid_vertices = [v for v in [v00, v10, v01, v11] if v >= 0]
            
            if len(valid_vertices) >= 3:
                if len(valid_vertices) == 4:
                    # Create two triangles
                    faces.append([v00, v10, v01])
                    faces.append([v10, v11, v01])
                elif len(valid_vertices) == 3:
                    # Create one triangle
                    faces.append(valid_vertices)
    
    faces = np.array(faces)
    print(f"Faces: {len(faces):,}")
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.faces = o3d.utility.Vector3iVector(faces)
    
    # Clean mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"Clean mesh - Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.faces):,}")
    
    # Calculate volume
    if mesh.is_watertight():
        volume = mesh.get_volume()
        print(f"Volume: {volume:.2f} mm³ (watertight)")
    else:
        print("Mesh is not watertight")
        volume = 0
    
    # Save STL
    stl_file = os.path.join(output_dir, 'scan1_simple_insole.stl')
    success = o3d.io.write_triangle_mesh(stl_file, mesh)
    
    if success:
        print(f"✅ Saved STL: {stl_file}")
    else:
        print(f"❌ Failed to save STL: {stl_file}")
    
    return outline_file, heightmap_file, stl_file, {
        'vertices': len(vertices),
        'faces': len(faces),
        'volume': volume,
        'coverage': coverage,
        'dimensions': (x_range, y_range, z_range)
    }

def main():
    input_file = 'outputs/height_analysis/scan1_height_cleaned.ply'
    output_dir = 'outputs/simple_construction'
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    try:
        outline_file, heightmap_file, stl_file, stats = construct_simple_insole(
            input_file, output_dir
        )
        
        print(f"\n=== CONSTRUCTION COMPLETE ===")
        print(f"✅ Outline: {outline_file}")
        print(f"✅ Heightmap: {heightmap_file}")
        print(f"✅ STL: {stl_file}")
        print(f"\nStatistics:")
        print(f"  Dimensions: {stats['dimensions'][0]:.1f} x {stats['dimensions'][1]:.1f} x {stats['dimensions'][2]:.1f} mm")
        print(f"  Vertices: {stats['vertices']:,}")
        print(f"  Faces: {stats['faces']:,}")
        print(f"  Coverage: {stats['coverage']:.1f}%")
        if stats['volume'] > 0:
            print(f"  Volume: {stats['volume']:.2f} mm³")
        
        print(f"\n🎯 Simple insole construction successful!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
