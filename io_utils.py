"""
io_utils.py

Shared I/O helpers for 3D scan files.
"""
import os
from typing import Tuple, Union

import open3d as o3d


POINT_CLOUD_EXTS = {
    ".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts",
}
MESH_EXTS = {
    ".stl", ".obj", ".off", ".gltf", ".glb", ".fbx", ".dae",
}


Geometry = Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]


def load_geometry(path: str, compute_mesh_normals: bool = False) -> Tuple[str, Geometry]:
    """Load a scan file as ('pcd', PointCloud) or ('mesh', TriangleMesh)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in POINT_CLOUD_EXTS:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Loaded point cloud is empty: {path}")
        return "pcd", pcd
    if ext in MESH_EXTS:
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise ValueError(f"Loaded mesh is empty: {path}")
        if compute_mesh_normals and not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return "mesh", mesh
    raise ValueError(
        f"Unsupported file extension '{ext}'.\n"
        f"Supported point clouds: {sorted(POINT_CLOUD_EXTS)}\n"
        f"Supported meshes: {sorted(MESH_EXTS)}"
    )


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud file; raises ValueError if empty or unsupported."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in POINT_CLOUD_EXTS:
        raise ValueError(
            f"Expected a point cloud extension, got '{ext}'. "
            f"Supported: {sorted(POINT_CLOUD_EXTS)}"
        )
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    return pcd


def splitext_with_suffix(path: str, suffix: str) -> str:
    """Insert a suffix before the file extension: foo.ply + _x -> foo_x.ply."""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"
