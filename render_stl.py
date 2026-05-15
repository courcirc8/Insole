"""
render_stl.py

Render STL files to PNG for inspection when GUI display is not available.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend; safe when no display is available
import matplotlib.pyplot as plt
import trimesh
# Registering 'mpl_toolkits.mplot3d.Axes3D' is implicit via projection='3d'.


def render_mesh_to_png(mesh_path: str, output_path: str):
    """Render mesh to PNG using matplotlib."""
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Could not load a triangle mesh from: {mesh_path}")
    
    fig = plt.figure(figsize=(12, 8))
    
    # Multiple views
    views = [
        (45, 45, "Perspective"),
        (0, 0, "Front"), 
        (90, 0, "Side"),
        (0, 90, "Top")
    ]
    
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_trisurf(
            mesh.vertices[:, 0], 
            mesh.vertices[:, 1], 
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            alpha=0.8,
            cmap='viridis'
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Mesh info:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Volume: {mesh.volume:.2f}")
    print(f"  Bounds: X=[{mesh.bounds[0,0]:.1f}, {mesh.bounds[1,0]:.1f}], Y=[{mesh.bounds[0,1]:.1f}, {mesh.bounds[1,1]:.1f}], Z=[{mesh.bounds[0,2]:.1f}, {mesh.bounds[1,2]:.1f}]")
    print(f"  Is watertight: {mesh.is_watertight}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render STL to PNG for inspection.")
    parser.add_argument("--stl", required=True, help="Path to STL file.")
    parser.add_argument("--output", "-o", default="render.png", help="Output PNG path.")
    args = parser.parse_args()

    try:
        render_mesh_to_png(args.stl, args.output)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Saved render: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


