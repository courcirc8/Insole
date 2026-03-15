#!/usr/bin/env python3
"""
Generate STL from existing heightmap (avoiding segfault issues).
"""

import numpy as np
import os

def create_stl_from_heightmap(heightmap_file, output_file):
    """Create STL file from heightmap using simple text format."""
    print(f"Loading heightmap: {heightmap_file}")
    
    # Load heightmap
    heightmap = np.load(heightmap_file)
    y_bins, x_bins = heightmap.shape
    print(f"Heightmap shape: {y_bins} x {x_bins}")
    
    # Calculate grid parameters (assuming from earlier analysis)
    x_range = 333.4  # mm (from validation output)
    y_range = 361.1  # mm
    x_min = -143.5   # mm
    y_min = -87.0    # mm
    
    grid_size = x_range / x_bins
    print(f"Grid size: {grid_size:.2f}mm")
    
    # Count valid cells
    valid_mask = ~np.isnan(heightmap)
    valid_cells = np.sum(valid_mask)
    coverage = valid_cells / (x_bins * y_bins) * 100
    print(f"Valid cells: {valid_cells:,} ({coverage:.1f}%)")
    
    # Create vertex mapping
    vertex_map = np.full((y_bins, x_bins), -1, dtype=int)
    vertices = []
    
    vertex_count = 0
    for j in range(y_bins):
        for i in range(x_bins):
            if valid_mask[j, i]:
                x = x_min + i * grid_size
                y = y_min + j * grid_size
                z = heightmap[j, i]
                
                vertex_map[j, i] = vertex_count
                vertices.append((x, y, z))
                vertex_count += 1
    
    print(f"Generated {len(vertices):,} vertices")
    
    # Create faces
    triangles = []
    
    for j in range(y_bins - 1):
        for i in range(x_bins - 1):
            # Get vertex indices for 2x2 cell
            v00 = vertex_map[j, i]
            v10 = vertex_map[j, i+1]
            v01 = vertex_map[j+1, i]
            v11 = vertex_map[j+1, i+1]
            
            # Only create triangles if all 4 vertices exist
            if all(v >= 0 for v in [v00, v10, v01, v11]):
                # Two triangles per cell
                triangles.append((v00, v10, v01))
                triangles.append((v10, v11, v01))
    
    print(f"Generated {len(triangles):,} triangles")
    
    # Write STL file in ASCII format (more reliable)
    print(f"Writing STL: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("solid insole\n")
        
        for tri in triangles:
            v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            
            # Calculate normal vector
            edge1 = np.array(v2) - np.array(v1)
            edge2 = np.array(v3) - np.array(v1)
            normal = np.cross(edge1, edge2)
            normal_len = np.linalg.norm(normal)
            
            if normal_len > 0:
                normal = normal / normal_len
            else:
                normal = np.array([0, 0, 1])  # Default upward normal
            
            f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid insole\n")
    
    print(f"✅ STL file created successfully!")
    
    # Calculate approximate volume
    total_volume = 0
    for j in range(y_bins):
        for i in range(x_bins):
            if valid_mask[j, i]:
                cell_volume = grid_size * grid_size * heightmap[j, i]
                total_volume += cell_volume
    
    print(f"Approximate volume: {total_volume:.2f} mm³")
    
    return {
        'vertices': len(vertices),
        'triangles': len(triangles),
        'volume': total_volume,
        'coverage': coverage
    }

def main():
    heightmap_file = 'outputs/height_analysis/scan1_height_cleaned_heightmap.npy'
    output_file = 'outputs/height_analysis/scan1_height_cleaned.stl'
    
    if not os.path.exists(heightmap_file):
        print(f"ERROR: Heightmap file not found: {heightmap_file}")
        return
    
    try:
        stats = create_stl_from_heightmap(heightmap_file, output_file)
        
        print(f"\n=== STL GENERATION COMPLETE ===")
        print(f"✅ STL file: {output_file}")
        print(f"📊 Statistics:")
        print(f"   Vertices: {stats['vertices']:,}")
        print(f"   Triangles: {stats['triangles']:,}")
        print(f"   Coverage: {stats['coverage']:.1f}%")
        print(f"   Volume: {stats['volume']:.2f} mm³")
        
        # Verify file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"   File size: {file_size:,} bytes")
            print(f"\n🎯 Valid insole STL object constructed successfully!")
        else:
            print(f"❌ STL file was not created")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
