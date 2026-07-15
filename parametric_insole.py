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

    # Underside: "flat" = solid block from z=0 up to the thickness field
    # (sits flat in the shoe / on the print bed); "shell" = constant-thickness
    # shell draped under the top surface (legacy behaviour).
    bottom_mode: str = "flat"
    shell_thickness: float = 2.0  # mm, used by bottom_mode="shell"


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

    # Add parametric features. GX_norm/GY_norm are NaN outside the outline,
    # so the fields are NaN there too — zero them instead of letting NaN
    # contaminate Z_composed (values outside the footprint are discarded by
    # the final mask anyway).
    arch_field = create_arch_field(GX_norm, GY_norm, params)
    heel_field = create_heel_field(GX_norm, GY_norm, params)
    met_field = create_met_pad_field(GX_norm, GY_norm, params, bounds)

    Z_composed += np.nan_to_num(arch_field + heel_field + met_field)

    # Smooth within the footprint only. A plain gaussian_filter would mix
    # in out-of-footprint values and, with any NaN present, eat ~3*sigma
    # cells off the entire rim.
    if params.smooth_sigma > 0:
        weight = mask_valid.astype(float)
        num = gaussian_filter(np.where(mask_valid, Z_composed, 0.0), sigma=params.smooth_sigma)
        den = gaussian_filter(weight, sigma=params.smooth_sigma)
        with np.errstate(invalid="ignore"):
            Z_smoothed = num / den
        Z_composed = np.where(den > 1e-6, Z_smoothed, Z_composed)

    Z_composed = apply_thickness_constraints(Z_composed, params)

    # The footprint is defined by the scan heightmap: NaN outside it so the
    # mesher keeps exactly the outline-masked region.
    Z_composed = np.where(mask_valid, Z_composed, np.nan)

    return Z_composed


def heightmap_to_mesh(GX: np.ndarray, GY: np.ndarray, Z: np.ndarray, outline: Polygon,
                      shell_thickness: float = 2.0, bottom_mode: str = "flat") -> trimesh.Trimesh:
    """
    Convert heightmap to watertight STL mesh following the insole outline.

    Args:
        GX, GY, Z: Grid arrays (Z has NaN outside outline)
        outline: Shapely polygon for boundary
        shell_thickness: Shell thickness (bottom_mode="shell" only)
        bottom_mode: "flat" = solid from z=0 up to Z (flat underside);
                     "shell" = underside drapes Z - shell_thickness

    Returns:
        Watertight trimesh
    """
    mask = np.isfinite(Z)
    if not np.any(mask):
        raise ValueError("No finite Z values in heightmap")

    # Extract valid points and create top surface
    valid_indices = np.where(mask)
    top_vertices = np.column_stack([GX[valid_indices], GY[valid_indices], Z[valid_indices]])

    # Triangulate top surface using 2D Delaunay. Delaunay fills the CONVEX
    # HULL of the points, so triangles bridging concave parts of the outline
    # (e.g. the medial arch waist) must be discarded: keep only triangles
    # whose centroid lies inside the outline polygon.
    xy_points = top_vertices[:, :2]
    tri = Delaunay(xy_points)
    simplices_all = tri.simplices

    def _signed_area2(s):
        p0, p1, p2 = (xy_points[s[:, k]] for k in range(3))
        return (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
               (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])

    def _directed_boundary(kept):
        """Directed edges belonging to exactly one triangle, with the
        triangle's own winding."""
        edges = np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]])
        _, first_idx, counts = np.unique(
            np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True)
        return edges[first_idx[counts == 1]]

    centroids = xy_points[simplices_all].mean(axis=1)
    import shapely
    inside = shapely.contains_xy(outline.buffer(1e-9), centroids[:, 0], centroids[:, 1])
    area2_all = _signed_area2(simplices_all)
    nondegenerate = np.abs(area2_all) > 1e-12
    keep = inside & nondegenerate
    if not keep.any():
        raise ValueError("No triangles remain inside the outline")

    # Pinch repair: where the kept region touches itself at a single vertex
    # (a boundary staircase artifact), that vertex ends up on 4 boundary
    # edges and the vertical wall edge would be shared by 4 faces — not
    # watertight. Re-adding the excluded triangle(s) incident to the pinch
    # vertex merges the regions and removes the pinch. Only grid-scale
    # triangles qualify: re-adding a long triangle that bridges a concave
    # stretch of the outline would seal it off and add a tunnel (genus).
    tri_pts = xy_points[simplices_all]
    edge_len = np.linalg.norm(tri_pts - np.roll(tri_pts, -1, axis=1), axis=2)
    grid_scale = np.median(edge_len[keep])
    small = edge_len.max(axis=1) <= 2.5 * grid_scale
    for _ in range(50):
        kept_idx = np.flatnonzero(keep)
        kept_tris = simplices_all[kept_idx]
        boundary = _directed_boundary(kept_tris)
        verts, vcounts = np.unique(boundary.ravel(), return_counts=True)
        pinch = verts[vcounts > 2]
        if pinch.size == 0:
            break
        # Preferred repair: re-add a grid-scale excluded triangle at the
        # pinch (fills the notch without changing the outline).
        incident = (~keep) & nondegenerate & small & \
            np.isin(simplices_all, pinch).any(axis=1)
        if incident.any():
            keep |= incident
            continue
        # Fallback (pinch along a concave stretch, where the only excluded
        # neighbours are long outline-bridging triangles): drop every fan of
        # kept triangles at the pinch vertex except the largest. The loss is
        # about one grid cell.
        changed = False
        for p in pinch:
            t_local = np.flatnonzero((kept_tris == p).any(axis=1))
            if t_local.size < 2:
                continue
            # Group the triangles at p into fans connected via edges at p.
            parent = {int(t): int(t) for t in t_local}

            def _find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            edge_owner = {}
            for t in t_local:
                tri_v = kept_tris[t]
                for other in tri_v[tri_v != p]:
                    o = int(other)
                    if o in edge_owner:
                        ra, rb = _find(edge_owner[o]), _find(int(t))
                        if ra != rb:
                            parent[ra] = rb
                    else:
                        edge_owner[o] = int(t)
            fans = {}
            for t in t_local:
                fans.setdefault(_find(int(t)), []).append(int(t))
            if len(fans) < 2:
                continue
            for fan in sorted(fans.values(), key=len)[:-1]:
                keep[kept_idx[fan]] = False
                changed = True
        if not changed:
            break

    simplices = simplices_all[keep]

    # Enforce CCW winding (Qhull does not guarantee orientation); the wall
    # construction below relies on consistent top-face winding.
    signed2 = _signed_area2(simplices)
    simplices[signed2 < 0] = simplices[signed2 < 0][:, ::-1]

    # Drop vertices not referenced by any kept triangle and reindex
    used = np.unique(simplices)
    remap = -np.ones(len(top_vertices), dtype=np.int64)
    remap[used] = np.arange(used.size)
    simplices = remap[simplices]
    top_vertices = top_vertices[used]

    # Create bottom vertices
    bottom_vertices = top_vertices.copy()
    if bottom_mode == "flat":
        bottom_vertices[:, 2] = 0.0
    elif bottom_mode == "shell":
        bottom_vertices[:, 2] -= shell_thickness
    else:
        raise ValueError(f"Unknown bottom_mode: {bottom_mode!r} (use 'flat' or 'shell')")

    all_vertices = np.vstack([top_vertices, bottom_vertices])
    n_top = len(top_vertices)

    faces = []
    faces.extend(simplices.tolist())                    # top (CCW, normals up)
    faces.extend((simplices + n_top)[:, ::-1].tolist())  # bottom (reversed)

    # Side walls on the TRUE boundary of the kept triangulation: an edge is
    # a boundary edge iff it belongs to exactly one triangle. Walking each
    # directed edge with its triangle's winding gives consistently oriented
    # walls, so every edge of the closed shell is shared by exactly two
    # faces — watertight by construction.
    # A directed boundary edge (u, v) is traversed u->v by its top triangle;
    # the wall must traverse it v->u so every shared edge is walked in
    # opposite directions by its two faces (consistent outward winding).
    boundary_directed = _directed_boundary(simplices)
    for u, v in boundary_directed:
        faces.append([int(v), int(u), int(u) + n_top])
        faces.append([int(v), int(u) + n_top, int(v) + n_top])

    mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces, process=False)

    # Clean up
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

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
    parser.add_argument("--bottom", type=str, default=None, choices=["flat", "shell"], help="Override underside mode (flat = solid, shell = draped).")
    
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
    if args.bottom is not None:
        params.bottom_mode = args.bottom
    
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
        mesh = heightmap_to_mesh(GX, GY, Z_composed, outline,
                                 shell_thickness=params.shell_thickness,
                                 bottom_mode=params.bottom_mode)
        
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
