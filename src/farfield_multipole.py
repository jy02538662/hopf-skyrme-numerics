"""Far-field multipole pre-check for a converged Hopf-Skyrme configuration.

This is the decisive test referenced in the preprint section 6.1:
does the localized soliton's far-field deformation decay as 1/r (monopole,
Newtonian) or 1/r^2 (dipole)?  It also reports the angular power split
across l=0,1,2 so you can see which multipole dominates the tail.

Input: an nfield .npy of shape [N, N, N, 3], vacuum n0 = (0,0,1),
grid x = linspace(-length, length, N).

Method (numpy, CPU is fine -- this is not the compute bottleneck):
  1. Build a scalar deformation proxy phi(x) = deviation from vacuum.
  2. For a set of radii r, sample many points on each sphere (Fibonacci),
     trilinearly interpolate phi.
  3. Decompose the shell values into real spherical harmonics up to l=2:
        c0  (monopole),  dipole rms,  quadrupole rms.
  4. Log-log fit amplitude(r) ~ r^{-alpha} for the dominant multipole.

Interpretation:
  alpha ~ 1  -> monopole/long-range: assumption B supported.
  alpha ~ 2  -> dipole dominated: assumption B rejected, Newtonian analogy fails.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_field(path):
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected [N,N,N,3], got {arr.shape}")
    return arr.astype(np.float64)


def deformation_proxy(nfield, kind):
    """Scalar field that -> 0 at vacuum n0=(0,0,1)."""
    nx, ny, nz = nfield[..., 0], nfield[..., 1], nfield[..., 2]
    if kind == "one_minus_nz":
        return 1.0 - nz
    if kind == "transverse":
        return np.sqrt(np.clip(nx * nx + ny * ny, 0.0, None))
    if kind == "dev_norm":
        return np.sqrt(nx * nx + ny * ny + (nz - 1.0) ** 2)
    raise ValueError(f"unknown proxy kind: {kind}")


def fibonacci_sphere(num):
    """Return [num, 3] unit vectors roughly evenly spread on S^2."""
    idx = np.arange(num, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / num)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * idx
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def trilinear_sample(vol, pts_xyz, length):
    """Sample scalar volume `vol` [N,N,N] at physical points pts_xyz [M,3].

    Grid coordinate along each axis: linspace(-length, length, N).
    Points outside the grid return 0.
    """
    n = vol.shape[0]
    h = 2.0 * length / (n - 1)
    grid = (pts_xyz + length) / h  # -> index space [0, N-1]
    out = np.zeros(pts_xyz.shape[0], dtype=np.float64)
    i0 = np.floor(grid).astype(np.int64)
    frac = grid - i0
    inside = np.all((i0 >= 0) & (i0 + 1 <= n - 1), axis=1)
    ii = i0[inside]
    ff = frac[inside]
    vals = np.zeros(ii.shape[0], dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (ff[:, 0] if dx else 1 - ff[:, 0]) * \
                    (ff[:, 1] if dy else 1 - ff[:, 1]) * \
                    (ff[:, 2] if dz else 1 - ff[:, 2])
                vals += w * vol[ii[:, 0] + dx, ii[:, 1] + dy, ii[:, 2] + dz]
    out[inside] = vals
    return out


def real_sph_harm_l012(dirs):
    """Return unnormalized real spherical harmonic basis columns for l=0,1,2.

    dirs: [M,3] unit vectors. Returns dict of basis arrays.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    Y0 = np.ones_like(x)
    # l=1
    Y1 = np.stack([x, y, z], axis=1)
    # l=2 (real, traceless combinations)
    Y2 = np.stack([
        x * y,
        y * z,
        z * x,
        x * x - y * y,
        3.0 * z * z - 1.0,
    ], axis=1)
    return Y0, Y1, Y2


def shell_multipoles(shell_vals, dirs):
    """Least-squares project shell values onto l=0,1,2 basis; return power per l."""
    Y0, Y1, Y2 = real_sph_harm_l012(dirs)
    basis = np.concatenate([Y0[:, None], Y1, Y2], axis=1)  # [M, 9]
    coeff, *_ = np.linalg.lstsq(basis, shell_vals, rcond=None)
    c0 = coeff[0]
    c1 = coeff[1:4]
    c2 = coeff[4:9]
    p0 = abs(c0)
    p1 = float(np.sqrt(np.sum(c1 * c1)))
    p2 = float(np.sqrt(np.sum(c2 * c2)))
    return p0, p1, p2


def powerlaw_fit(r, amp):
    """Fit amp ~ A r^{-alpha} via log-log least squares. Returns (alpha, A, resid)."""
    mask = (amp > 0) & (r > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    lr = np.log(r[mask])
    la = np.log(amp[mask])
    slope, intercept = np.polyfit(lr, la, 1)
    resid = float(np.std(la - (slope * lr + intercept)))
    return -slope, math.exp(intercept), resid


def run(args):
    nfield = load_field(args.field)
    n = nfield.shape[0]
    length = args.length
    phi = deformation_proxy(nfield, args.proxy)

    r0 = args.r_min if args.r_min > 0 else 0.25 * length
    r1 = args.r_max if args.r_max > 0 else 0.85 * length
    radii = np.linspace(r0, r1, args.n_shells)
    dirs = fibonacci_sphere(args.n_points)

    rows = []
    amp_dom = []
    amp_mono = []
    for r in radii:
        pts = dirs * r
        vals = trilinear_sample(phi, pts, length)
        p0, p1, p2 = shell_multipoles(vals, dirs)
        dom = max(p0, p1, p2)
        rows.append({"r": float(r), "p0_monopole": float(p0),
                     "p1_dipole": float(p1), "p2_quad": float(p2),
                     "rms": float(np.sqrt(np.mean(vals ** 2)))})
        amp_dom.append(dom)
        amp_mono.append(p0)

    radii_arr = np.array(radii)
    alpha_mono, A_mono, res_mono = powerlaw_fit(radii_arr, np.array(amp_mono))
    alpha_rms, A_rms, res_rms = powerlaw_fit(radii_arr, np.array([r["rms"] for r in rows]))

    mean_p0 = np.mean([r["p0_monopole"] for r in rows])
    mean_p1 = np.mean([r["p1_dipole"] for r in rows])
    mean_p2 = np.mean([r["p2_quad"] for r in rows])
    dominant = ["monopole", "dipole", "quadrupole"][int(np.argmax([mean_p0, mean_p1, mean_p2]))]

    if dominant == "monopole" and abs(alpha_mono - 1.0) < 0.3:
        verdict = "MONOPOLE / 1-over-r : assumption B SUPPORTED (Newtonian analogy alive)"
    elif dominant == "dipole" or abs(alpha_mono - 2.0) < 0.3:
        verdict = "DIPOLE / 1-over-r^2 : assumption B REJECTED (Newtonian analogy fails)"
    else:
        verdict = "INCONCLUSIVE : refine grid/box or fit window"

    summary = {
        "field": args.field,
        "N": int(n),
        "length": length,
        "proxy": args.proxy,
        "fit_window": [float(r0), float(r1)],
        "n_shells": args.n_shells,
        "n_points": args.n_points,
        "alpha_monopole": alpha_mono,
        "alpha_rms": alpha_rms,
        "resid_monopole": res_mono,
        "mean_power": {"monopole": float(mean_p0), "dipole": float(mean_p1), "quadrupole": float(mean_p2)},
        "dominant_multipole": dominant,
        "verdict": verdict,
    }

    out_dir = Path(args.out) if args.out else Path(args.field).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "farfield_multipole.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "shells": rows}, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nper-shell (r, monopole, dipole, quad, rms):")
    for row in rows:
        print(f"  r={row['r']:7.3f}  p0={row['p0_monopole']:.4e}  "
              f"p1={row['p1_dipole']:.4e}  p2={row['p2_quad']:.4e}  rms={row['rms']:.4e}")
    print(f"\nVERDICT: {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Far-field monopole-vs-dipole pre-check (preprint sec 6.1)")
    parser.add_argument("--field", required=True, help="nfield .npy [N,N,N,3]")
    parser.add_argument("--length", type=float, required=True, help="box half-length (grid = linspace(-L,L,N))")
    parser.add_argument("--proxy", default="dev_norm",
                        choices=["one_minus_nz", "transverse", "dev_norm"],
                        help="scalar deformation proxy")
    parser.add_argument("--r-min", type=float, default=0.0, help="fit window inner radius (default 0.25 L)")
    parser.add_argument("--r-max", type=float, default=0.0, help="fit window outer radius (default 0.85 L)")
    parser.add_argument("--n-shells", type=int, default=24)
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
