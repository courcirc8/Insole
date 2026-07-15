"""
filter_insole.py

Shape-prior ("insole-aware") point filtering.

Generic outlier removal (statistical / radius) treats every point cloud the
same. This module instead exploits what we KNOW a scanned therapeutic insole
looks like:

1. Height prior      — the insole sits on the reference plane; everything
                       lives within [0, max_height] mm above it.
2. Footprint prior   — seen from above it is ONE connected, elongated,
                       foot-shaped region (length ~150-350 mm, width
                       ~45-130 mm, aspect ratio ~1.8-4.5, mostly solid).
                       Occupancy-grid morphology + prior-scored component
                       selection removes wisps, ghosts and unrelated blobs.
3. Thin-shell prior  — seen from above the surface is single-valued:
                       z = f(x, y). A robust per-cell top surface is fitted
                       and points far BELOW it (scanner see-through ghost
                       layers) or far ABOVE it (spikes) are rejected. The
                       tolerance widens with local slope so heel-cup walls
                       and arch flanks are preserved.
4. Connectivity prior — the insole is ONE connected shell in 3D: only the
                       largest voxel-connected component is kept, which
                       removes floating ghost patches that locally mimic a
                       surface (and thus survive the thin-shell prior).
5. Local-support prior — a real surface patch is locally dense; a light
                       statistical pass polishes remaining stragglers.

Every stage is vectorised on a raster grid, so the full filter runs in
seconds on 500k-point scans.

CLI:
    python filter_insole.py -p isolated.ply -o cleaned.ply --preview diag.png

Library:
    from filter_insole import filter_insole_points, InsolePriors
    kept, report = filter_insole_points(points, InsolePriors())
"""
import argparse
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

@dataclass
class InsolePriors:
    """What a scanned therapeutic insole looks like (units: mm)."""

    # Height prior (above the reference/ground plane)
    max_height: float = 60.0        # nothing on an insole is taller than this
    ground_slack: float = 1.5       # keep points slightly below robust ground

    # Footprint prior
    cell: float = 1.5               # occupancy-grid cell size
    min_length: float = 150.0       # plausible insole length range
    max_length: float = 350.0
    min_width: float = 45.0         # plausible insole width range
    max_width: float = 130.0
    min_aspect: float = 1.8         # length / width
    max_aspect: float = 4.5
    closing_mm: float = 4.0         # bridge gaps up to this size
    opening_mm: float = 2.0         # shave wisps thinner than this

    # Thin-shell (single-valued surface) prior
    surf_cell: float = 2.0          # cell size for the top-surface raster
    top_percentile: float = 90.0    # robust per-cell top estimate
    above_tol: float = 2.0          # allowed residual above the top surface
    below_tol: float = 3.5          # allowed residual below it (flat areas)
    slope_factor: float = 1.5       # tolerance growth per unit surface slope
    surf_iterations: int = 2        # robust re-fit iterations

    # 3D connectivity prior
    connect_voxel: float = 1.5      # voxel size; gaps > 2*voxel disconnect

    # Local-support polish
    stat_neighbors: int = 24
    stat_std_ratio: float = 2.8

    # Component plausibility below this triggers a warning
    plausibility_floor: float = 0.05


@dataclass
class FilterReport:
    """Per-stage bookkeeping returned alongside the filtered points."""
    n_input: int = 0
    n_after_height: int = 0
    n_after_footprint: int = 0
    n_after_surface: int = 0
    n_after_connect: int = 0
    n_after_polish: int = 0
    footprint_length: float = 0.0
    footprint_width: float = 0.0
    footprint_plausibility: float = 0.0
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        def pct(n):
            return f"{n:,} ({n / max(self.n_input, 1) * 100:.1f}%)"
        lines = [
            f"Input points:            {self.n_input:,}",
            f"After height prior:      {pct(self.n_after_height)}",
            f"After footprint prior:   {pct(self.n_after_footprint)}",
            f"After thin-shell prior:  {pct(self.n_after_surface)}",
            f"After connectivity:      {pct(self.n_after_connect)}",
            f"After local polish:      {pct(self.n_after_polish)}",
            f"Footprint: {self.footprint_length:.0f} x {self.footprint_width:.0f} mm "
            f"(insole plausibility {self.footprint_plausibility:.2f})",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def _grid_shape(xy: np.ndarray, cell: float):
    """Return (origin, (ny, nx)) covering the XY extent with `cell` spacing."""
    origin = xy.min(axis=0)
    span = xy.max(axis=0) - origin
    nx = int(span[0] / cell) + 1
    ny = int(span[1] / cell) + 1
    return origin, (ny, nx)


def _cell_ids(xy: np.ndarray, origin: np.ndarray, cell: float, shape):
    """Flat raster index of every point (row = y, col = x)."""
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, shape[1] - 1)  # x -> col
    ij[:, 1] = np.clip(ij[:, 1], 0, shape[0] - 1)  # y -> row
    return ij[:, 1] * shape[1] + ij[:, 0]


def _per_cell_percentile(flat_ids: np.ndarray, values: np.ndarray, n_cells: int,
                         q: float, min_count: int = 3):
    """Vectorised per-cell percentile. Cells with < min_count points get NaN."""
    order = np.lexsort((values, flat_ids))
    fi = flat_ids[order]
    vs = values[order]
    starts = np.flatnonzero(np.r_[True, fi[1:] != fi[:-1]])
    counts = np.diff(np.r_[starts, fi.size])
    take = starts + np.round((q / 100.0) * (counts - 1)).astype(np.int64)
    out = np.full(n_cells, np.nan)
    valid = counts >= min_count
    out[fi[starts[valid]]] = vs[take[valid]]
    return out


def _nan_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs (normalised convolution)."""
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    num = ndimage.gaussian_filter(filled, sigma)
    den = ndimage.gaussian_filter(finite.astype(float), sigma)
    with np.errstate(invalid="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out


def _nan_median3(arr: np.ndarray) -> np.ndarray:
    """3x3 median filter that ignores NaNs (kills single-cell spikes)."""
    padded = np.pad(arr, 1, constant_values=np.nan)
    stack = [padded[1 + dy:padded.shape[0] - 1 + dy,
                    1 + dx:padded.shape[1] - 1 + dx]
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.stack(stack), axis=0)


def _fill_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaN cells with the value of the nearest finite cell."""
    invalid = ~np.isfinite(arr)
    if not invalid.any():
        return arr
    if invalid.all():
        return np.zeros_like(arr)
    idx = ndimage.distance_transform_edt(invalid, return_distances=False,
                                         return_indices=True)
    return arr[tuple(idx)]


# ---------------------------------------------------------------------------
# Stage 1 — height prior
# ---------------------------------------------------------------------------

def apply_height_prior(points: np.ndarray, priors: InsolePriors):
    """Clip to [robust_ground - slack, robust_ground + max_height].

    The ground level is re-estimated with a low percentile instead of the raw
    minimum, so a single deep artifact cannot shift the whole height window.
    """
    z = points[:, 2]
    ground = np.percentile(z, 0.5)
    mask = (z >= ground - priors.ground_slack) & (z <= ground + priors.max_height)
    return mask, ground


# ---------------------------------------------------------------------------
# Stage 2 — footprint prior
# ---------------------------------------------------------------------------

def footprint_plausibility(length: float, width: float, priors: InsolePriors) -> float:
    """Soft [0, 1] score for how insole-like a footprint's dimensions are."""
    def band(v, lo, hi, soft):
        if v < lo:
            return float(np.exp(-((lo - v) / soft) ** 2))
        if v > hi:
            return float(np.exp(-((v - hi) / soft) ** 2))
        return 1.0

    aspect = length / max(width, 1e-9)
    return (band(length, priors.min_length, priors.max_length, 60.0)
            * band(width, priors.min_width, priors.max_width, 30.0)
            * band(aspect, priors.min_aspect, priors.max_aspect, 1.0))


def _component_dims(rows: np.ndarray, cols: np.ndarray, cell: float):
    """PCA extents (length, width) of a set of raster cells, in mm."""
    pts = np.column_stack([cols, rows]).astype(float) * cell
    pts -= pts.mean(axis=0)
    if len(pts) < 3:
        return 0.0, 0.0
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    proj = pts @ evecs
    ext = proj.max(axis=0) - proj.min(axis=0)
    length, width = float(max(ext)), float(min(ext))
    return length, width


def apply_footprint_prior(points: np.ndarray, priors: InsolePriors,
                          report: FilterReport):
    """Occupancy-grid morphology + insole-scored connected component.

    Returns (mask over points, footprint boolean grid, grid origin).
    """
    xy = points[:, :2]
    origin, shape = _grid_shape(xy, priors.cell)
    flat = _cell_ids(xy, origin, priors.cell, shape)
    counts = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)

    # Density-adaptive occupancy: sparse ghost wisps occupy cells with only a
    # couple of hits while the real surface is dense.
    occupied = counts[counts > 0]
    thresh = max(2, int(0.08 * np.median(occupied))) if occupied.size else 1
    mask = counts >= thresh

    # Morphology: bridge small scan gaps, fill interior holes, shave wisps.
    # Pad first: scipy treats beyond-border as empty (border_value=0), so an
    # unpadded erosion would bite into regions touching the array edge —
    # exactly the toe/heel extremes, since the grid spans the data extent.
    it_close = max(1, int(round(priors.closing_mm / priors.cell)))
    it_open = max(1, int(round(priors.opening_mm / priors.cell)))
    pad = it_close + it_open + 1
    mask = np.pad(mask, pad)
    mask = ndimage.binary_closing(mask, iterations=it_close)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=it_open)
    mask = mask[pad:-pad, pad:-pad]

    # Score every connected component against the insole shape prior instead
    # of blindly keeping the largest one.
    labels, n_comp = ndimage.label(mask)
    if n_comp == 0:
        report.warnings.append("footprint prior found no dense region; stage skipped")
        return np.ones(len(points), dtype=bool), mask, origin

    best_label, best_score, best_dims, best_plaus = 0, -1.0, (0.0, 0.0), 0.0
    for lab in range(1, n_comp + 1):
        rows, cols = np.nonzero(labels == lab)
        area = rows.size * priors.cell ** 2
        length, width = _component_dims(rows, cols, priors.cell)
        plaus = footprint_plausibility(length, width, priors)
        score = area * max(plaus, 1e-3)  # area breaks ties among implausible blobs
        if score > best_score:
            best_label, best_score = lab, score
            best_dims, best_plaus = (length, width), plaus

    report.footprint_length, report.footprint_width = best_dims
    report.footprint_plausibility = best_plaus
    if best_plaus < priors.plausibility_floor:
        report.warnings.append(
            f"best footprint ({best_dims[0]:.0f} x {best_dims[1]:.0f} mm) does not "
            "look like an insole; check units (mm expected) or isolation step")

    keep_grid = labels == best_label
    # One-cell dilation so legitimate boundary points are not clipped.
    keep_grid = ndimage.binary_dilation(keep_grid, iterations=1)
    point_mask = keep_grid.ravel()[flat]
    return point_mask, keep_grid, origin


def footprint_polygon(points: np.ndarray, priors: InsolePriors | None = None,
                      smooth_mm: float = 3.0):
    """Insole outline as a shapely Polygon traced from the occupied footprint.

    Expects an ALREADY-FILTERED cloud (noise removed). Every occupied raster
    cell counts — no density threshold, no opening — so thin but genuine
    boundary regions (toe tip, heel rim) are preserved; small scan gaps are
    bridged by closing/hole-filling and the boundary staircase is smoothed
    with a closing/opening buffer pass. Unlike concave-hull heuristics this
    can never cut wedges into the footprint: the polygon covers exactly the
    region that actually contains points.
    """
    import shapely
    from shapely.ops import unary_union

    if priors is None:
        priors = InsolePriors()
    xy = np.asarray(points, dtype=float)[:, :2]
    origin, shape = _grid_shape(xy, priors.cell)
    flat = _cell_ids(xy, origin, priors.cell, shape)
    counts = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)
    mask = counts > 0
    # Pad before closing: scipy's border_value=0 would otherwise erode the
    # regions touching the array edge (toe/heel extremes).
    it_close = max(1, int(round(priors.closing_mm / priors.cell)))
    pad = it_close + 1
    mask = np.pad(mask, pad)
    mask = ndimage.binary_closing(mask, iterations=it_close)
    mask = ndimage.binary_fill_holes(mask)
    mask = mask[pad:-pad, pad:-pad]
    labels, n_comp = ndimage.label(mask)
    if n_comp == 0:
        raise ValueError("Footprint mask is empty; cannot trace an outline.")
    if n_comp > 1:
        sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n_comp + 1))
        mask = labels == (1 + int(np.argmax(sizes)))

    rows, cols = np.nonzero(mask)
    c = priors.cell
    x0 = origin[0] + cols * c
    y0 = origin[1] + rows * c
    poly = unary_union(shapely.box(x0, y0, x0 + c, y0 + c))
    # Morphological closing (dilate then erode) smooths the raster staircase
    # and NEVER shrinks below the occupied region — an opening would bite
    # into high-curvature toe/heel tips. A half-cell outward margin then
    # guarantees boundary points fall inside the polygon.
    s = max(smooth_mm, c)
    poly = poly.buffer(s).buffer(-s).buffer(0.5 * c)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return shapely.geometry.Polygon(poly.exterior).simplify(0.5 * c)


# ---------------------------------------------------------------------------
# Stage 3 — thin-shell (single-valued surface) prior
# ---------------------------------------------------------------------------

def apply_surface_prior(points: np.ndarray, priors: InsolePriors):
    """Reject points far below/above a robust, slope-aware top surface.

    Scanner see-through produces ghost layers UNDER the true top surface,
    and specular reflections produce spikes ABOVE it. Both are far from the
    single-valued surface z = f(x, y) that a thin insole shell presents from
    above. Near steep walls (heel cup, arch flank) the tolerance widens with
    the local surface gradient so genuine wall points survive.

    Returns (mask, diagnostics dict).
    """
    xy = points[:, :2]
    z = points[:, 2]
    origin, shape = _grid_shape(xy, priors.surf_cell)
    n_cells = shape[0] * shape[1]
    flat = _cell_ids(xy, origin, priors.surf_cell, shape)

    keep = np.ones(len(points), dtype=bool)
    zref_grid = None
    tol_below_grid = None

    for it in range(max(1, priors.surf_iterations)):
        # Robust per-cell top estimate from currently kept points. The first
        # pass uses a high percentile (top surface despite ghosts below);
        # later passes tighten toward the median of survivors.
        q = priors.top_percentile if it == 0 else 75.0
        ztop = _per_cell_percentile(flat[keep], z[keep], n_cells, q)
        grid = ztop.reshape(shape)

        # Denoise the raster surface: median kills single-cell spikes,
        # Gaussian gives a smooth reference; nearest-fill covers gaps.
        grid = _nan_median3(grid)
        grid = _nan_gaussian(grid, sigma=1.0)
        grid = _fill_nearest(grid)

        # Slope-aware tolerance (mm): steep cells may legitimately contain
        # points spanning a large z range within one cell.
        gy, gx = np.gradient(grid, priors.surf_cell)
        slope = np.hypot(gx, gy)
        tol_below = priors.below_tol + priors.slope_factor * slope * priors.surf_cell
        tol_above = priors.above_tol + 0.5 * priors.slope_factor * slope * priors.surf_cell

        residual = z - grid.ravel()[flat]
        keep = (residual >= -tol_below.ravel()[flat]) & \
               (residual <= tol_above.ravel()[flat])
        zref_grid, tol_below_grid = grid, tol_below

    diag = {"zref": zref_grid, "tol_below": tol_below_grid,
            "origin": origin, "cell": priors.surf_cell}
    return keep, diag


# ---------------------------------------------------------------------------
# Stage 4 — 3D connectivity prior
# ---------------------------------------------------------------------------

def apply_connectivity_prior(points: np.ndarray, priors: InsolePriors,
                             report: FilterReport) -> np.ndarray:
    """Keep only the largest 3D voxel-connected component.

    A real insole is ONE connected shell. Floating ghost patches can locally
    mimic a smooth surface (they define their own per-cell "top"), so the
    thin-shell prior alone cannot reject them — but they hang detached in
    space. With voxel size v and 26-connectivity, any gap wider than 2*v
    disconnects, so a patch >= 2*connect_voxel away from the shell is dropped.
    """
    v = priors.connect_voxel
    mins = points.min(axis=0)
    ijk = np.floor((points - mins) / v).astype(np.int64)
    grid = np.zeros(ijk.max(axis=0) + 1, dtype=bool)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    labels, n_comp = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    if n_comp <= 1:
        return np.ones(len(points), dtype=bool)
    point_labels = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    counts = np.bincount(point_labels.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    mask = point_labels == keep
    if mask.sum() < 0.5 * len(points):
        # A shell this fragmented means the voxel size does not match the
        # sampling density; refuse to guess and keep everything.
        report.warnings.append(
            "connectivity prior skipped: largest 3D component holds <50% of "
            "points (increase connect_voxel for sparse scans)")
        return np.ones(len(points), dtype=bool)
    return mask


# ---------------------------------------------------------------------------
# Stage 5 — local-support polish
# ---------------------------------------------------------------------------

def apply_local_polish(points: np.ndarray, priors: InsolePriors) -> np.ndarray:
    """Light statistical pass to drop the few remaining stragglers."""
    try:
        import open3d as o3d
    except Exception:
        return np.ones(len(points), dtype=bool)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, inliers = pcd.remove_statistical_outlier(
        nb_neighbors=priors.stat_neighbors, std_ratio=priors.stat_std_ratio)
    mask = np.zeros(len(points), dtype=bool)
    mask[np.asarray(inliers, dtype=np.int64)] = True
    return mask


# ---------------------------------------------------------------------------
# Full filter
# ---------------------------------------------------------------------------

def filter_insole_points(points: np.ndarray, priors: InsolePriors | None = None,
                         polish: bool = True):
    """Run the full insole-aware filter.

    Args:
        points: (N, 3) array, ground plane already removed and roughly
                aligned so that +Z is "up" (output of remove_ground.py).
        priors: shape priors; defaults to InsolePriors() (units: mm).
        polish: run the final statistical pass (requires open3d).

    Returns:
        (kept_points (M, 3), FilterReport, diagnostics dict)
    """
    if priors is None:
        priors = InsolePriors()
    points = np.asarray(points, dtype=float)
    report = FilterReport(n_input=len(points))
    diag = {"input": points}

    # 1 — height prior
    mask, ground = apply_height_prior(points, priors)
    pts = points[mask]
    pts[:, 2] -= ground  # re-level so the raster stages see z=0 at ground
    report.n_after_height = len(pts)
    diag["ground"] = ground

    # 2 — footprint prior
    if len(pts):
        mask, footprint_grid, fp_origin = apply_footprint_prior(pts, priors, report)
        pts = pts[mask]
        diag["footprint_grid"] = footprint_grid
        diag["footprint_origin"] = fp_origin
    report.n_after_footprint = len(pts)

    # 3 — thin-shell prior
    if len(pts):
        mask, surf_diag = apply_surface_prior(pts, priors)
        diag.update(surf_diag)
        pts = pts[mask]
    report.n_after_surface = len(pts)

    # 4 — 3D connectivity prior
    if len(pts):
        mask = apply_connectivity_prior(pts, priors, report)
        pts = pts[mask]
    report.n_after_connect = len(pts)

    # 5 — polish
    if polish and len(pts):
        mask = apply_local_polish(pts, priors)
        pts = pts[mask]
    report.n_after_polish = len(pts)

    if report.n_input and report.n_after_polish < 0.2 * report.n_input:
        report.warnings.append(
            "less than 20% of points survived; priors may not match the scan "
            "(check units and the isolation step)")

    diag["kept"] = pts
    return pts, report, diag


# ---------------------------------------------------------------------------
# Diagnostics preview
# ---------------------------------------------------------------------------

def save_preview(path: str, points_in: np.ndarray, points_out: np.ndarray,
                 diag: dict, max_scatter: int = 60000, seed: int = 0):
    """2x3 panel: before / after / reference surface + z histograms."""
    if not _HAS_MPL:
        return
    rng = np.random.default_rng(seed)

    def sample(arr):
        if len(arr) > max_scatter:
            return arr[rng.choice(len(arr), max_scatter, replace=False)]
        return arr

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    s_in, s_out = sample(points_in), sample(points_out)
    vmin = min(s_in[:, 2].min(), s_out[:, 2].min()) if len(s_out) else s_in[:, 2].min()
    vmax = max(s_in[:, 2].max(), s_out[:, 2].max()) if len(s_out) else s_in[:, 2].max()

    ax = axes[0, 0]
    ax.scatter(s_in[:, 0], s_in[:, 1], s=0.3, c=s_in[:, 2], cmap="viridis",
               vmin=vmin, vmax=vmax)
    ax.set_title(f"Before — {len(points_in):,} pts")
    ax.axis("equal")

    ax = axes[0, 1]
    if len(s_out):
        ax.scatter(s_out[:, 0], s_out[:, 1], s=0.3, c=s_out[:, 2], cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_title(f"After — {len(points_out):,} pts")
    ax.axis("equal")

    ax = axes[0, 2]
    zref = diag.get("zref")
    if zref is not None:
        im = ax.imshow(zref, origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Fitted top surface (mm)")

    ax = axes[1, 0]
    ax.hist(points_in[:, 2], bins=80, color="gray", alpha=0.8)
    ax.set_title("Z histogram — before")
    ax.set_xlabel("Z (mm)")

    ax = axes[1, 1]
    if len(points_out):
        ax.hist(points_out[:, 2], bins=80, color="#2a9d8f", alpha=0.8)
    ax.set_title("Z histogram — after")
    ax.set_xlabel("Z (mm)")

    ax = axes[1, 2]
    fp = diag.get("footprint_grid")
    if fp is not None:
        ax.imshow(fp, origin="lower", cmap="gray_r")
    ax.set_title("Footprint mask")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insole-aware point filtering (height + footprint + "
                    "thin-shell shape priors).")
    parser.add_argument("--path", "-p", required=True, help="Input PLY/PCD (ground already removed).")
    parser.add_argument("--output", "-o", required=True, help="Output cleaned PLY.")
    parser.add_argument("--preview", default=None, help="Optional diagnostics PNG.")
    parser.add_argument("--max_height", type=float, default=60.0, help="Max height above ground (mm).")
    parser.add_argument("--cell", type=float, default=1.5, help="Footprint grid cell (mm).")
    parser.add_argument("--surf_cell", type=float, default=2.0, help="Surface grid cell (mm).")
    parser.add_argument("--above_tol", type=float, default=2.0, help="Allowed residual above top surface (mm).")
    parser.add_argument("--below_tol", type=float, default=3.5, help="Allowed residual below top surface (mm).")
    parser.add_argument("--slope_factor", type=float, default=1.5, help="Tolerance growth with local slope.")
    parser.add_argument("--no_polish", action="store_true", help="Skip the final statistical pass.")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    import open3d as o3d
    from io_utils import load_point_cloud

    pcd = load_point_cloud(args.path)
    points = np.asarray(pcd.points)

    priors = InsolePriors(
        max_height=args.max_height, cell=args.cell, surf_cell=args.surf_cell,
        above_tol=args.above_tol, below_tol=args.below_tol,
        slope_factor=args.slope_factor)

    kept, report, diag = filter_insole_points(points, priors,
                                              polish=not args.no_polish)
    print(report.summary())

    if kept.size == 0:
        print("No points survived filtering; aborting save.", file=sys.stderr)
        return 1

    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kept))
    if not o3d.io.write_point_cloud(args.output, out):
        print(f"Failed to save: {args.output}", file=sys.stderr)
        return 1
    print(f"Saved: {args.output}")

    if args.preview:
        pts_in = points.copy()
        pts_in[:, 2] -= diag.get("ground", 0.0)
        save_preview(args.preview, pts_in, kept, diag)
        print(f"Saved preview: {args.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
