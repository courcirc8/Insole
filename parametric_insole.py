"""
parametric_insole.py

Compose parametric thickness fields on a heightmap to create a customizable insole.
Features: base thickness, arch support, heel posting, met pads, edge control.
Outputs: modified heightmap and watertight STL mesh.
"""
import argparse
import dataclasses
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import trimesh
from shapely.geometry import Polygon, Point


@dataclass
class InsoleParams:
    """Parametric insole configuration."""
    # Base
    min_thickness: float = 2.0  # mm
    max_thickness: float = 8.0  # mm
    base_thickness: float = 3.0  # mm
    
    # Arch support
    arch_enabled: bool = True
    arch_height: float = 4.0  # mm additional height
    arch_position: tuple = (0.25, 0.5)  # (longitudinal %, medial-lateral %)
    arch_width: float = 0.3  # fraction of foot width
    arch_length: float = 0.2  # fraction of foot length
    
    # Heel posting/wedge
    heel_enabled: bool = False
    heel_varus_angle: float = 0.0  # degrees (positive = varus)
    heel_height: float = 2.0  # mm additional at heel
    heel_transition: float = 0.4  # fraction of length for transition
    
    # Met pad
    met_pad_enabled: bool = False
    met_pad_position: tuple = (0.7, 0.5)  # (longitudinal %, medial-lateral %)
    met_pad_height: float = 2.0  # mm
    met_pad_radius: float = 15.0  # mm
    
    # Edge control
    edge_lip_height: float = 1.0  # mm
    edge_chamfer: float = 2.0  # mm
    
    # Smoothing
    smooth_sigma: float = 2.0  # Gaussian blur sigma


def load_heightmap(base_path: str) -> tuple:
    """Load GX, GY, Z arrays from NPY files."""
    GX = np.load(base_path + "_GX.npy")
    GY = np.load(base_path + "_GY.npy")
    Z = np.load(base_path + "_Z.npy")
    return GX, GY, Z


def load_outline_csv(path: str) -> Polygon:
    """Load outline polygon from CSV coordinates."""
    coords = np.loadtxt(path, delimiter=",", skiprows=1)
    return Polygon(coords)


def normalize_coords(GX: np.ndarray, GY: np.ndarray) -> tuple:
    """Normalize grid coordinates to [0,1] for parametric positioning."""
    x_min, x_max = np.nanmin(GX), np.nanmax(GX)
    y_min, y_max = np.nanmin(GY), np.nanmax(GY)
    GX_norm = (GX - x_min) / (x_max - x_min + 1e-12)
    GY_norm = (GY - y_min) / (y_max - y_min + 1e-12)
    return GX_norm, GY_norm, (x_min, x_max, y_min, y_max)


def create_arch_field(GX_norm: np.ndarray, GY_norm: np.ndarray, params: InsoleParams) -> np.ndarray:
    """Create arch support field as a 2D Gaussian."""
    if not params.arch_enabled:
        return np.zeros_like(GX_norm)
    
    # Center at arch position
    cx, cy = params.arch_position
    dx = GX_norm - cx
    dy = GY_norm - cy
    
    # Gaussian with anisotropic scaling
    sigma_x = params.arch_length / 3  # 3-sigma coverage
    sigma_y = params.arch_width / 3
    
    field = np.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2)))
    return field * params.arch_height


def create_heel_field(GX_norm: np.ndarray, GY_norm: np.ndarray, params: InsoleParams) -> np.ndarray:
    """Create heel posting field with varus/valgus angle."""
    if not params.heel_enabled:
        return np.zeros_like(GX_norm)
    
    # Heel is typically at low longitudinal values (0.0 = heel, 1.0 = toe)
    heel_mask = GX_norm < params.heel_transition
    field = np.zeros_like(GX_norm)
    
    if params.heel_varus_angle != 0:
        # Medial-lateral gradient for posting
        angle_rad = np.radians(params.heel_varus_angle)
        # Positive varus = higher on medial side (lower GY_norm values)
        ml_gradient = np.tan(angle_rad) * (0.5 - GY_norm)  # Center at 0.5
        field[heel_mask] = ml_gradient[heel_mask] * params.heel_height
    else:
        # Uniform heel lift
        transition = 1 - GX_norm / params.heel_transition
        transition = np.clip(transition, 0, 1)
        field = transition * params.heel_height
    
    return field


def create_met_pad_field(GX_norm: np.ndarray, GY_norm: np.ndarray, params: InsoleParams, bounds: tuple) -> np.ndarray:
    """Create metatarsal pad field as a localized bump."""
    if not params.met_pad_enabled:
        return np.zeros_like(GX_norm)
    
    cx, cy = params.met_pad_position
    x_min, x_max, y_min, y_max = bounds
    # Convert back to world units for radius
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    
    dx_world = (GX_norm - cx) * x_scale
    dy_world = (GY_norm - cy) * y_scale
    dist = np.sqrt(dx_world**2 + dy_world**2)
    
    # Smooth circular bump
    field = np.exp(-(dist / params.met_pad_radius)**2)
    return field * params.met_pad_height


def apply_thickness_constraints(Z: np.ndarray, params: InsoleParams) -> np.ndarray:
    """Enforce min/max thickness constraints."""
    return np.clip(Z, params.min_thickness, params.max_thickness)


def create_parametric_insole(
    GX: np.ndarray, 
    GY: np.ndarray, 
    Z_base: np.ndarray, 
    params: InsoleParams
) -> np.ndarray:
    """
    Compose parametric thickness field from base heightmap and feature parameters.
    
    Args:
        GX, GY: Grid coordinates
        Z_base: Base heightmap from scan
        params: Insole parameters
    
    Returns:
        Z_composed: Final thickness field
    """
    # Normalize coordinates for parametric positioning
    GX_norm, GY_norm, bounds = normalize_coords(GX, GY)
    
    # Start with base thickness
    Z_composed = np.full_like(Z_base, params.base_thickness)
    
    # Add scan-derived variations (where available)
    mask_valid = np.isfinite(Z_base)
    if np.any(mask_valid):
        # Blend scan data with base thickness
        Z_scan_offset = Z_base - np.nanmean(Z_base[mask_valid])
        Z_composed[mask_valid] += Z_scan_offset[mask_valid] * 0.5  # 50% scan influence
    
    # Add parametric features
    arch_field = create_arch_field(GX_norm, GY_norm, params)
    heel_field = create_heel_field(GX_norm, GY_norm, params)
    met_field = create_met_pad_field(GX_norm, GY_norm, params, bounds)
    
    Z_composed += arch_field + heel_field + met_field
    
    # Smooth and constrain
    if params.smooth_sigma > 0:
        Z_composed = gaussian_filter(Z_composed, sigma=params.smooth_sigma)
    
    Z_composed = apply_thickness_constraints(Z_composed, params)
    
    return Z_composed


def heightmap_to_mesh(GX: np.ndarray, GY: np.ndarray, Z: np.ndarray, outline: Polygon, shell_thickness: float = 2.0) -> trimesh.Trimesh:
    """
    Convert heightmap to watertight STL mesh following the insole outline.
    
    Args:
        GX, GY, Z: Grid arrays (Z has NaN outside outline)
        outline: Shapely polygon for boundary
        shell_thickness: Bottom shell thickness
    
    Returns:
        Watertight trimesh
    """
    mask = np.isfinite(Z)
    if not np.any(mask):
        raise ValueError("No finite Z values in heightmap")
    
    # Extract valid points and create top surface
    valid_indices = np.where(mask)
    top_vertices = np.column_stack([GX[valid_indices], GY[valid_indices], Z[valid_indices]])
    
    # Triangulate top surface using 2D Delaunay
    xy_points = top_vertices[:, :2]
    tri = Delaunay(xy_points)
    
    # Create bottom vertices
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] -= shell_thickness
    
    # Combine vertices
    all_vertices = np.vstack([top_vertices, bottom_vertices])
    n_top = len(top_vertices)
    
    faces = []
    
    # Top surface faces
    faces.extend(tri.simplices.tolist())
    
    # Bottom surface faces (reversed winding)
    bottom_faces = tri.simplices + n_top
    faces.extend(bottom_faces[:, ::-1].tolist())
    
    # Create side walls along the outline boundary
    outline_coords = np.array(outline.exterior.coords[:-1])  # Remove duplicate last point
    
    # Find closest vertices to outline points
    dists = cdist(outline_coords, xy_points)
    boundary_indices = np.argmin(dists, axis=1)
    
    # Create side faces connecting boundary vertices
    n_boundary = len(boundary_indices)
    for i in range(n_boundary):
        v_top_curr = boundary_indices[i]
        v_top_next = boundary_indices[(i + 1) % n_boundary]
        v_bot_curr = v_top_curr + n_top
        v_bot_next = v_top_next + n_top
        
        # Two triangles per side edge
        faces.append([v_top_curr, v_top_next, v_bot_curr])
        faces.append([v_top_next, v_bot_next, v_bot_curr])
    
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces)
    
    # Clean up
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    
    # Fix winding if volume is negative
    if mesh.volume < 0:
        mesh.invert()
    
    return mesh


def main() -> int:
    """CLI entrypoint for parametric insole generation."""
    parser = argparse.ArgumentParser(description="Generate parametric insole STL from heightmap.")
    parser.add_argument("--heightmap", required=True, help="Base path for heightmap NPY files (without _GX.npy suffix).")
    parser.add_argument("--outline", required=True, help="Path to outline CSV.")
    parser.add_argument("--config", type=str, default=None, help="YAML config file for parameters.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output STL path.")
    parser.add_argument("--preview", type=str, default=None, help="Save thickness field preview PNG.")
    
    # Quick parameter overrides
    parser.add_argument("--arch_height", type=float, default=None, help="Override arch height (mm).")
    parser.add_argument("--base_thickness", type=float, default=None, help="Override base thickness (mm).")
    parser.add_argument("--heel_angle", type=float, default=None, help="Override heel varus angle (degrees).")
    
    args = parser.parse_args()
    
    # Load heightmap
    GX, GY, Z_base = load_heightmap(args.heightmap)
    outline = load_outline_csv(args.outline)
    
    # Load parameters
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        if not isinstance(config_dict, dict):
            print(f"Invalid YAML config (expected a mapping at root): {args.config}", file=sys.stderr)
            return 1
        valid_keys = {f.name for f in dataclasses.fields(InsoleParams)}
        unknown = set(config_dict) - valid_keys
        if unknown:
            print(f"Unknown config keys: {sorted(unknown)}. Valid keys: {sorted(valid_keys)}", file=sys.stderr)
            return 1
        params = InsoleParams(**config_dict)
    else:
        params = InsoleParams()
    
    # Apply CLI overrides
    if args.arch_height is not None:
        params.arch_height = args.arch_height
    if args.base_thickness is not None:
        params.base_thickness = args.base_thickness
    if args.heel_angle is not None:
        params.heel_varus_angle = args.heel_angle
    
    # Generate parametric thickness field
    Z_composed = create_parametric_insole(GX, GY, Z_base, params)
    
    # Save preview if requested
    if args.preview:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 10))
            plt.imshow(np.flipud(Z_composed), cmap="plasma", interpolation="nearest")
            plt.colorbar(label="Thickness (mm)")
            plt.title("Parametric Insole Thickness")
            plt.tight_layout()
            plt.savefig(args.preview, dpi=200)
            plt.close()
            print(f"Saved thickness preview: {args.preview}")
        except Exception:
            print("Warning: Could not save preview (matplotlib unavailable)")
    
    # Generate mesh
    try:
        mesh = heightmap_to_mesh(GX, GY, Z_composed, outline, shell_thickness=2.0)
        
        # Save STL
        out_path = args.output or args.heightmap + "_parametric.stl"
        mesh.export(out_path)
        print(f"Saved parametric STL: {out_path}")
        print(f"Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Volume: {mesh.volume:.2f} cubic units")
        
    except Exception as e:
        print(f"Mesh generation failed: {e}")
        print("Saving heightmap arrays instead...")
        np.save(args.heightmap + "_parametric_Z.npy", Z_composed)
        print(f"Saved: {args.heightmap}_parametric_Z.npy")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
