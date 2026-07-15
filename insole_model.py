"""
insole_model.py

Anatomical parametric insole model.

Instead of a generic "thickness field", the insole is described by named
anatomical zones in normalized foot coordinates, so one YAML file can
reproduce a scanned insole and generate many variants (sizes, pathologies):

    s in [0, 1]  : longitudinal station, 0 = back of heel, 1 = toe tip
    t in [-1, 1] : transverse position, -1 = medial edge, +1 = lateral edge
    d (mm)       : distance to the footprint boundary

Height field (mm above ground) = sum of:
    base profile B(s)      — longitudinal platform (heel cushion -> thin toes)
    heel cup               — rim ridge along the boundary around the heel,
                             the cup recess comes from B(s) being lower than
                             the rim
    arch dome              — anisotropic bump on the medial side
    retro-capital pad      — transverse dome JUST BEHIND the metatarsal
                             heads (barre retro-capitale)
    met-head relief        — shallow recess UNDER the metatarsal heads
    edge skirt             — smooth descent to the ground near the boundary,
                             closing the outer walls the scanner cannot see
                             (fixes the "truncated heel" look of raw scans)

The footprint is either generated parametrically (length + 4 widths) or
loaded from an outline CSV (to reproduce a scanned insole).

CLI:
    python insole_model.py generate --config presets/griffes_orteils.yaml \
        --output out.stl --preview out.png
    python insole_model.py fit --heightmap outputs/scan1/scan1 \
        --config-out presets/scan1_replica.yaml --output replica.stl
"""
import argparse
import os
import sys

import numpy as np
import yaml
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


DEFAULT_CONFIG = {
    "foot": {
        # Footprint. Either parametric (these values) or from an outline CSV
        # via generate --outline; side controls which edge is medial.
        "length": 268.0,
        "heel_width": 64.0,
        "waist_width": 60.0,   # narrowest part, under the arch
        "ball_width": 92.0,    # metatarsal heads, widest part
        "toe_width": 66.0,
        "side": "left",        # "left": medial edge = -t; mirrored for "right"
    },
    # Longitudinal platform at the centerline: [s, height_mm]
    "base_profile": [
        [0.00, 19.0],
        [0.15, 20.0],
        [0.36, 22.0],
        [0.62, 9.0],
        [0.80, 4.0],
        [1.00, 3.0],
    ],
    "heel_cup": {
        "rim_height": 11.0,    # ridge above the local base
        "rim_offset": 5.0,     # ridge crest distance from the boundary (mm)
        "rim_width": 5.0,      # ridge gaussian width (mm)
        "s_end": 0.32,         # rim fades out by this station
    },
    "arch": {
        "height": 16.0,        # dome above the local base
        "s": 0.36, "t": -0.55, # centre (medial side)
        "sigma_s": 0.11, "sigma_t": 0.38,
    },
    "met_pad": {               # retro-capital pad/bar (0 height = disabled)
        "height": 0.0,
        "s": 0.62, "t": -0.05,
        "sigma_s": 0.045, "sigma_t": 0.45,
    },
    "met_relief": {            # recess under the metatarsal heads (0 = off)
        "depth": 0.0,
        "s": 0.72, "t": -0.05,
        "sigma_s": 0.05, "sigma_t": 0.5,
    },
    "edge": {
        "skirt_width": 5.0,    # walls blend to ground over this distance
        "min_thickness": 1.6,  # printable floor at the very edge
    },
    "smoothing_mm": 2.5,
    "resolution": 1.5,
}


def _merge(base: dict, override: dict) -> dict:
    out = {}
    for k, v in base.items():
        if k in override and isinstance(v, dict) and isinstance(override[k], dict):
            out[k] = _merge(v, override[k])
        elif k in override:
            out[k] = override[k]
        else:
            out[k] = v
    for k in override:
        if k not in out:
            raise ValueError(f"Unknown config key: {k!r}")
    return out


def load_config(path: str | None) -> dict:
    if path is None:
        return DEFAULT_CONFIG
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return _merge(DEFAULT_CONFIG, user)


# ---------------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------------

def parametric_halfwidths(cfg_foot: dict):
    """Return (half_width(s), length). Smooth PCHIP through anatomical widths
    with elliptical shrink at the heel and toe caps."""
    L = float(cfg_foot["length"])
    ctrl_s = np.array([0.00, 0.12, 0.40, 0.70, 0.92, 1.00])
    ctrl_w = 0.5 * np.array([
        0.2 * cfg_foot["heel_width"],   # cap start
        cfg_foot["heel_width"],
        cfg_foot["waist_width"],
        cfg_foot["ball_width"],
        cfg_foot["toe_width"],
        0.15 * cfg_foot["toe_width"],   # cap end
    ])
    hw = PchipInterpolator(ctrl_s, ctrl_w)

    def half_width(s):
        s = np.clip(s, 0.0, 1.0)
        w = hw(s)
        # Elliptical rounding of the extreme caps
        cap = np.ones_like(w)
        r0, r1 = 0.045, 0.055
        m0 = s < r0
        cap = np.where(m0, np.sqrt(np.clip(1 - ((r0 - s) / r0) ** 2, 0, 1)), cap)
        m1 = s > 1 - r1
        cap = np.where(m1, np.sqrt(np.clip(1 - ((s - (1 - r1)) / r1) ** 2, 0, 1)), cap)
        return w * cap

    return half_width, L


def build_grid(cfg: dict, outline_poly=None):
    """Uniform XY grid with foot coordinates.

    Returns dict with GX, GY (mm), mask (inside), S, T, D (distance to
    boundary, mm) and the boundary polygon.
    """
    import shapely
    import shapely.geometry as geom

    res = float(cfg["resolution"])
    if outline_poly is None:
        half_width, L = parametric_halfwidths(cfg["foot"])
        ss = np.linspace(0, 1, 400)
        hw = half_width(ss)
        xs = np.r_[hw, -hw[::-1]]
        ys = np.r_[ss, ss[::-1]] * L
        outline_poly = geom.Polygon(np.column_stack([xs, ys])).buffer(0)

    minx, miny, maxx, maxy = outline_poly.bounds
    xv = np.arange(minx - res, maxx + 2 * res, res)
    yv = np.arange(miny - res, maxy + 2 * res, res)
    GX, GY = np.meshgrid(xv, yv)
    inside = shapely.contains_xy(outline_poly, GX.ravel(), GY.ravel()).reshape(GX.shape)

    # Foot coordinates. s along +Y from the heel end; t normalized by the
    # local footprint extent so t = -1 / +1 always lands on the edges.
    L = maxy - miny
    S = (GY - miny) / L
    T = np.full_like(GX, np.nan)
    for i in range(GX.shape[0]):
        m = inside[i]
        if m.sum() < 2:
            continue
        x_lo, x_hi = GX[i][m].min(), GX[i][m].max()
        c, h = 0.5 * (x_lo + x_hi), max(0.5 * (x_hi - x_lo), 1e-6)
        T[i] = (GX[i] - c) / h
    if str(cfg["foot"].get("side", "left")).lower() == "right":
        T = -T  # medial edge stays at t = -1

    D = ndimage.distance_transform_edt(inside) * res
    return {"GX": GX, "GY": GY, "mask": inside, "S": S, "T": T, "D": D,
            "poly": outline_poly, "res": res}


# ---------------------------------------------------------------------------
# Height field
# ---------------------------------------------------------------------------

def _gauss2(S, T, s0, t0, sig_s, sig_t):
    return np.exp(-0.5 * (((S - s0) / sig_s) ** 2 + ((T - t0) / sig_t) ** 2))


def compose_height(cfg: dict, grid: dict) -> np.ndarray:
    """Evaluate the anatomical height field on the grid (NaN outside)."""
    S, T, D, mask = grid["S"], grid["T"], grid["D"], grid["mask"]

    prof = np.array(cfg["base_profile"], dtype=float)
    base = PchipInterpolator(prof[:, 0], prof[:, 1])(np.clip(S, 0, 1))

    Z = base.copy()

    hc = cfg["heel_cup"]
    if hc["rim_height"] > 0:
        ridge = np.exp(-0.5 * ((D - hc["rim_offset"]) / hc["rim_width"]) ** 2)
        fade = np.clip((hc["s_end"] - S) / max(hc["s_end"], 1e-6), 0, 1)
        fade = fade * fade * (3 - 2 * fade)  # smoothstep
        Z += hc["rim_height"] * ridge * fade

    a = cfg["arch"]
    if a["height"] > 0:
        Z += a["height"] * _gauss2(S, T, a["s"], a["t"], a["sigma_s"], a["sigma_t"])

    p = cfg["met_pad"]
    if p["height"] > 0:
        Z += p["height"] * _gauss2(S, T, p["s"], p["t"], p["sigma_s"], p["sigma_t"])

    r = cfg["met_relief"]
    if r["depth"] > 0:
        Z -= r["depth"] * _gauss2(S, T, r["s"], r["t"], r["sigma_s"], r["sigma_t"])

    # Edge skirt: smooth descent to the ground at the boundary. This closes
    # the outer walls everywhere (incl. the back of the heel, which top-down
    # scans cannot see) instead of ending in a vertical cliff.
    e = cfg["edge"]
    w = max(float(e["skirt_width"]), grid["res"])
    fall = np.clip(D / w, 0, 1)
    Z = Z * (fall * fall * (3 - 2 * fall))
    Z = np.maximum(Z, float(e["min_thickness"]))

    if cfg["smoothing_mm"] > 0:
        sigma = cfg["smoothing_mm"] / grid["res"]
        weight = mask.astype(float)
        num = ndimage.gaussian_filter(np.where(mask, Z, 0.0), sigma)
        den = ndimage.gaussian_filter(weight, sigma)
        with np.errstate(invalid="ignore"):
            Z = np.where(den > 1e-6, num / den, Z)
        Z = np.maximum(Z, float(e["min_thickness"]))

    return np.where(mask, Z, np.nan)


# ---------------------------------------------------------------------------
# Fit a config to a scanned heightmap (reproduce an existing insole)
# ---------------------------------------------------------------------------

def fit_to_heightmap(GXs, GYs, Zs, cfg: dict) -> tuple[dict, dict]:
    """Extract anatomical parameters from a scanned heightmap.

    Direct, robust extraction (no fragile optimizer): per-station medians for
    the base profile, boundary-band maxima for the heel rim, and the medial
    dome for the arch. Returns (fitted config, diagnostics).
    """
    finite = np.isfinite(Zs)
    ys = np.where(finite.any(axis=1), np.nanmean(np.where(finite, GYs, np.nan), axis=1), np.nan)
    y0, y1 = np.nanmin(GYs), np.nanmax(GYs)
    L = y1 - y0

    res = float(np.nanmedian(np.diff(GXs[np.isfinite(GXs).any(axis=1)][0])))
    D = ndimage.distance_transform_edt(finite) * res

    S = (GYs - y0) / L
    T = np.full_like(GXs, np.nan)
    for i in range(GXs.shape[0]):
        m = finite[i]
        if m.sum() < 2:
            continue
        x_lo, x_hi = GXs[i][m].min(), GXs[i][m].max()
        c, h = 0.5 * (x_lo + x_hi), max(0.5 * (x_hi - x_lo), 1e-6)
        T[i] = (GXs[i] - c) / h

    # Medial side = side of the global maximum in the midfoot
    mid = finite & (S > 0.25) & (S < 0.55)
    t_at_max = T[mid][np.nanargmax(Zs[mid])]
    if t_at_max > 0:  # flip so the arch sits at negative t
        T = -T
        side = "right"
    else:
        side = "left"

    # Base profile: central-band median away from rim and arch
    central = finite & (np.abs(T) < 0.35) & (D > 12)
    ctrl_s = [0.02, 0.15, 0.36, 0.62, 0.80, 0.97]
    prof = []
    for s0 in ctrl_s:
        band = central & (np.abs(S - s0) < 0.05)
        if band.sum() < 10:
            band = finite & (np.abs(S - s0) < 0.05)
        prof.append([round(s0, 2), round(float(np.nanmedian(Zs[band])), 1)])

    base_interp = PchipInterpolator(np.array(prof)[:, 0], np.array(prof)[:, 1])
    base_grid = base_interp(np.clip(S, 0, 1))
    resid = Zs - base_grid

    # Heel rim: boundary band in the heel region
    heel_band = finite & (S < 0.25) & (D < 12)
    rim_height = float(np.nanpercentile(resid[heel_band], 95))
    rim_d = float(np.nanmedian(D[heel_band & (resid > 0.8 * rim_height)]))

    # Arch dome: medial midfoot residual peak; widths from the half-max area
    arch_zone = finite & (S > 0.22) & (S < 0.55) & (T < -0.1)
    k = np.nanargmax(np.where(arch_zone, resid, -np.inf))
    arch_h = float(resid.ravel()[k])
    arch_s = float(S.ravel()[k]); arch_t = float(T.ravel()[k])
    half = arch_zone & (resid > arch_h / 2)
    sig_s = max(float(np.nanstd(S[half])), 0.04)
    sig_t = max(float(np.nanstd(T[half])), 0.15)

    widths = {}
    for name, s0 in [("heel_width", 0.12), ("waist_width", 0.40),
                     ("ball_width", 0.70), ("toe_width", 0.92)]:
        rows = np.abs((ys - y0) / L - s0) < 0.03
        w = [GXs[i][finite[i]].max() - GXs[i][finite[i]].min()
             for i in np.flatnonzero(rows) if finite[i].sum() > 2]
        widths[name] = round(float(np.median(w)), 1) if w else None

    fitted = _merge(cfg, {
        "foot": {"length": round(float(L), 1), "side": side, **widths},
        "base_profile": prof,
        "heel_cup": {"rim_height": round(rim_height, 1),
                     "rim_offset": round(rim_d, 1)},
        "arch": {"height": round(arch_h, 1), "s": round(arch_s, 2),
                 "t": round(arch_t, 2), "sigma_s": round(sig_s, 3),
                 "sigma_t": round(sig_t, 3)},
    })
    diag = {"S": S, "T": T, "D": D, "finite": finite}
    return fitted, diag


def residual_report(Zs, Zm, finite) -> str:
    r = (Zm - Zs)[finite & np.isfinite(Zm)]
    return (f"residual model-scan: median {np.median(np.abs(r)):.2f} mm, "
            f"p90 {np.percentile(np.abs(r), 90):.2f} mm, "
            f"max {np.abs(r).max():.2f} mm")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_stl(cfg: dict, grid: dict, Z: np.ndarray, out_path: str):
    from parametric_insole import heightmap_to_mesh
    GXm = np.where(grid["mask"], grid["GX"], np.nan)
    GYm = np.where(grid["mask"], grid["GY"], np.nan)
    mesh = heightmap_to_mesh(GXm, GYm, Z, grid["poly"], bottom_mode="flat")
    mesh.export(out_path)
    b = mesh.bounds
    print(f"Saved STL: {out_path}")
    print(f"  {b[1][0]-b[0][0]:.1f} x {b[1][1]-b[0][1]:.1f} x {b[1][2]-b[0][2]:.1f} mm, "
          f"volume {mesh.volume/1000:.1f} cm3, watertight={mesh.is_watertight}")
    return mesh


def save_preview(path: str, Z: np.ndarray, grid: dict, title: str,
                 Zs=None):
    if not _HAS_MPL:
        return
    n = 3 if Zs is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 9), squeeze=False)
    ext = [np.nanmin(grid["GX"]), np.nanmax(grid["GX"]),
           np.nanmin(grid["GY"]), np.nanmax(grid["GY"])]
    im = axes[0][0].imshow(Z, origin="lower", cmap="viridis", extent=ext)
    fig.colorbar(im, ax=axes[0][0], shrink=0.7, label="z (mm)")
    axes[0][0].set_title(title)
    if Zs is not None:
        im = axes[0][1].imshow(Zs, origin="lower", cmap="viridis", extent=ext)
        fig.colorbar(im, ax=axes[0][1], shrink=0.7)
        axes[0][1].set_title("Scan")
        r = Z - Zs
        im = axes[0][2].imshow(r, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5, extent=ext)
        fig.colorbar(im, ax=axes[0][2], shrink=0.7)
        axes[0][2].set_title("Résidu modèle - scan (mm)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved preview: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_generate(args) -> int:
    cfg = load_config(args.config)
    outline_poly = None
    if args.outline:
        import shapely.geometry as geom
        coords = np.loadtxt(args.outline, delimiter=",", skiprows=1)
        outline_poly = geom.Polygon(coords)
    grid = build_grid(cfg, outline_poly)
    Z = compose_height(cfg, grid)
    zmax = np.nanmax(Z)
    print(f"Height field: {np.nanmin(Z):.1f}..{zmax:.1f} mm over "
          f"{int(np.isfinite(Z).sum())} cells")
    if args.output:
        export_stl(cfg, grid, Z, args.output)
    if args.preview:
        save_preview(args.preview, Z, grid, "Modèle paramétrique")
    return 0


def cmd_fit(args) -> int:
    GXs = np.load(args.heightmap + "_GX.npy")
    GYs = np.load(args.heightmap + "_GY.npy")
    Zs = np.load(args.heightmap + "_Z.npy")
    cfg = load_config(args.config)
    fitted, diag = fit_to_heightmap(GXs, GYs, Zs, cfg)

    if args.config_out:
        os.makedirs(os.path.dirname(args.config_out) or ".", exist_ok=True)
        with open(args.config_out, "w") as f:
            yaml.safe_dump(fitted, f, sort_keys=False)
        print(f"Saved fitted config: {args.config_out}")

    # Rebuild the model on the SCAN outline for a fair residual comparison
    from filter_insole import footprint_polygon
    pts = np.column_stack([GXs[diag["finite"]], GYs[diag["finite"]],
                           Zs[diag["finite"]]])
    outline_poly = footprint_polygon(pts)
    grid = build_grid(fitted, outline_poly)
    Z = compose_height(fitted, grid)

    # Interpolate scan onto the model grid for residuals/preview. The
    # heightmap axes are uniform linspaces over the bounds (see
    # generate_heightmap.py), so they can be reconstructed exactly.
    from scipy.interpolate import RegularGridInterpolator
    xs = np.linspace(np.nanmin(GXs), np.nanmax(GXs), GXs.shape[1])
    ys = np.linspace(np.nanmin(GYs), np.nanmax(GYs), GYs.shape[0])
    interp = RegularGridInterpolator((ys, xs), Zs, bounds_error=False,
                                     fill_value=np.nan)
    Zs_on = interp(np.column_stack([grid["GY"].ravel(), grid["GX"].ravel()]))
    Zs_on = Zs_on.reshape(grid["GX"].shape)
    print(residual_report(Zs_on, Z, np.isfinite(Zs_on)))

    if args.output:
        export_stl(fitted, grid, Z, args.output)
    if args.preview:
        save_preview(args.preview, Z, grid, "Modèle ajusté", Zs=Zs_on)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Anatomical parametric insole model.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate an insole STL from a config.")
    g.add_argument("--config", "-c", default=None, help="YAML config (defaults built in).")
    g.add_argument("--outline", default=None, help="Optional outline CSV instead of the parametric footprint.")
    g.add_argument("--output", "-o", default=None, help="Output STL path.")
    g.add_argument("--preview", default=None, help="Preview PNG path.")
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("fit", help="Fit the model to a scanned heightmap (reproduce an insole).")
    f.add_argument("--heightmap", required=True, help="Heightmap NPY base path (from generate_heightmap.py).")
    f.add_argument("--config", "-c", default=None, help="Base YAML config to start from.")
    f.add_argument("--config-out", default=None, help="Where to save the fitted YAML.")
    f.add_argument("--output", "-o", default=None, help="Optional STL of the fitted model.")
    f.add_argument("--preview", default=None, help="Preview PNG (model vs scan vs residual).")
    f.set_defaults(func=cmd_fit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
