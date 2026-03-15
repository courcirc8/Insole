#!/usr/bin/env python3
"""
Memory-safe insole construction avoiding segfaults.
"""

import numpy as np
import open3d as o3d
import os

def safe_construct_insole(ply_file, output_dir):
    """Construct insole with memory-safe operations."""
    print(f"\n=== SAFE INSOLE CONSTRUCTION ===")
    print(f"Input: {ply_file}")
    
    # Load and validate
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    print(f"Points: {len(points):,}")
    
    # Basic stats
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    print(f"Dimensions: {x_range:.1f} x {y_range:.1f} x {z_range:.1f} mm")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Simple rectangular outline (most robust)
    print("\nGenerating simple outline...")
    
    # Use percentiles for robust boundary
    x_bounds = np.percentile(points[:, 0], [2, 98])
    y_bounds = np.percentile(points[:, 1], [2, 98])
    
    outline_points = np.array([
        [x_bounds[0], y_bounds[0]],  # bottom-left
        [x_bounds[1], y_bounds[0]],  # bottom-right
        [x_bounds[1], y_bounds[1]],  # top-right
        [x_bounds[0], y_bounds[1]]   # top-left
    ])
    
    print(f"Rectangular outline: {len(outline_points)} points")
    
    # Save outline
    outline_file = os.path.join(output_dir, 'scan1_safe_outline.csv')
    np.savetxt(outline_file, outline_points, delimiter=',', 
               header='x,y', comments='', fmt='%.6f')
    print(f"Saved: {outline_file}")
    
    # Step 2: Create coarse heightmap (avoid memory issues)
    print("\nGenerating coarse heightmap...")
    
    # Conservative grid size
    grid_size = 2.0  # mm - larger cells to reduce memory
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    print(f"Grid: {x_bins} x {y_bins} = {x_bins * y_bins:,} cells")
    
    # Process in smaller chunks to avoid memory issues
    chunk_size = 50000  # Process 50k points at a time
    heightmap = np.full((y_bins, x_bins), np.nan)
    
    for chunk_start in range(0, len(points), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(points))
        chunk_points = points[chunk_start:chunk_end]
        
        print(f"  Processing chunk {chunk_start//chunk_size + 1}/{(len(points)-1)//chunk_size + 1}")
        
        for i in range(len(chunk_points)):
            x, y, z = chunk_points[i]
            
            # Find grid cell
            x_idx = int((x - x_min) / grid_size)
            y_idx = int((y - y_min) / grid_size)
            
            # Clamp to valid range
            x_idx = max(0, min(x_bins - 1, x_idx))
            y_idx = max(0, min(y_bins - 1, y_idx))
            
            # Update with max height
            if np.isnan(heightmap[y_idx, x_idx]) or z > heightmap[y_idx, x_idx]:
                heightmap[y_idx, x_idx] = z
    
    # Calculate coverage
    filled_cells = np.sum(~np.isnan(heightmap))
    coverage = filled_cells / (x_bins * y_bins) * 100
    print(f"Coverage: {coverage:.1f}% ({filled_cells:,}/{x_bins * y_bins:,} cells)")
    
    # Simple hole filling (only if needed)
    if coverage < 70:
        print("Filling major holes...")
        for j in range(y_bins):
            for i in range(x_bins):
                if np.isnan(heightmap[j, i]):
                    # Find nearest valid neighbor
                    for radius in range(1, 5):  # Limited search
                        found = False
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
    
    # Save heightmap
    heightmap_file = os.path.join(output_dir, 'scan1_safe_heightmap.npy')
    np.save(heightmap_file, heightmap)
    print(f"Saved: {heightmap_file}")
    
    # Step 3: Generate simple STL
    print("\nGenerating simple STL...")
    
    # Create vertices (only for valid cells)
    vertices = []
    vertex_indices = {}
    
    for j in range(y_bins):
        for i in range(x_bins):
            if not np.isnan(heightmap[j, i]):
                x = x_min + i * grid_size
                y = y_min + j * grid_size
                z = heightmap[j, i]
                
                vertex_indices[(j, i)] = len(vertices)
                vertices.append([x, y, z])
    
    print(f"Generated {len(vertices):,} vertices")
    
    # Create faces (conservative approach)
    faces = []
    
    for j in range(y_bins - 1):
        for i in range(x_bins - 1):
            # Check if we have a 2x2 block of valid cells
            corners = [(j, i), (j, i+1), (j+1, i), (j+1, i+1)]
            valid_corners = [c for c in corners if c in vertex_indices]
            
            if len(valid_corners) == 4:
                # Create two triangles
                v00 = vertex_indices[(j, i)]
                v10 = vertex_indices[(j, i+1)]
                v01 = vertex_indices[(j+1, i)]
                v11 = vertex_indices[(j+1, i+1)]
                
                faces.append([v00, v10, v01])
                faces.append([v10, v11, v01])
    
    print(f"Generated {len(faces):,} faces")
    
    # Create mesh
    if len(vertices) > 0 and len(faces) > 0:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        mesh.faces = o3d.utility.Vector3iVector(np.array(faces))
        
        # Basic cleanup
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        
        print(f"Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
        
        # Save STL
        stl_file = os.path.join(output_dir, 'scan1_safe_insole.stl')
        success = o3d.io.write_triangle_mesh(stl_file, mesh)
        
        if success:
            print(f"✅ Saved STL: {stl_file}")
        else:
            print(f"❌ Failed to save STL")
            stl_file = None
    else:
        print("❌ No valid mesh generated")
        stl_file = None
    
    return outline_file, heightmap_file, stl_file

def main():
    input_file = 'outputs/height_analysis/scan1_height_cleaned.ply'
    output_dir = 'outputs/safe_construction'
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    try:
        outline_file, heightmap_file, stl_file = safe_construct_insole(
            input_file, output_dir
        )
        
        print(f"\n=== SAFE CONSTRUCTION COMPLETE ===")
        print(f"✅ Outline: {outline_file}")
        print(f"✅ Heightmap: {heightmap_file}")
        if stl_file:
            print(f"✅ STL: {stl_file}")
        else:
            print(f"❌ STL generation failed")
        
        print(f"\n🎯 Safe insole construction finished!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
