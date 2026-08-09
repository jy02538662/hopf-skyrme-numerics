"""D2 symmetry probe for Q=2 Hopf-Skyrme far-field (signed-component version).

Purpose (preprint section 5.7.4 / Assumption 5.A; qeff_definition_v1.md v1.1):
    For the Q=2 double-ring orthogonal configuration with D2 symmetry, verify
    numerically that on a far-field shell the C2(z) operation acts as the
    reflection R_z(pi) on physical space, which translates into the angular
    parity rule:

        delta_n_perp(r, theta, phi+pi)  =  - delta_n_perp(r, theta, phi)   (*)

    This is the signed-component form of Assumption 5.A.  Equation (*) holds
    iff the even-m coefficient block of delta_n_x and delta_n_y is identically
    zero; that block is exactly what sh_decompose_components.py audits.

Inputs:
    --field : nfield_q1_torch.npy  (shape [N,N,N,3], vacuum n0=(0,0,1))
    --length : half box length (grid linspace(-L,L,N))
    --q-shells : list of radii in [r_min, r_max]
    --out : directory for JSON / numpy artifacts

Outputs (in --out):
    d2_symmetry_probe.json         : summary + per-shell signed antisymmetry
    d2_antisym_vs_r.npy            : per-shell array
                                     (mean |dn_perp|, mean |residual|,
                                      mean |dn_perp| decomposed into
                                       even-m vs odd-m style)
    shell_delta_n_original.npy     : sample shell snapshot

Diagnostic levels:
    1. asym_index_signed = |<dn_perp(R_z(-pi) x) + dn_perp(x)>| / <|dn_perp(x)|>
       -> ~0 means rule (*) holds.
       -> ~2 means rule (+) holds instead (symmetric, not antisymmetric).
       -> ~1 means mixed (e.g. some components antisymmetric, some symmetric).

    2. If rule (*) holds, the only mechanism that can produce the small residual
       is numerical interpolation error / finite-box cutoff.  Pass criterion:
          asym_index_signed < 1e-2.
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


def fibonacci_sphere(num):
    idx = np.arange(num, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / num)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * idx
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def trilinear_sample_vec3(vol, pts_xyz, length):
    """Sample a vector field vol [N,N,N,3] at physical points pts_xyz [M,3]."""
    n = vol.shape[0]
    h = 2.0 * length / (n - 1)
    grid = (pts_xyz + length) / h
    m = pts_xyz.shape[0]
    out = np.zeros((m, 3), dtype=np.float64)
    i0 = np.floor(grid).astype(np.int64)
    frac = grid - i0
    inside = np.all((i0 >= 0) & (i0 + 1 <= n - 1), axis=1)
    ii = i0[inside]
    ff = frac[inside]
    acc = np.zeros((ii.shape[0], 3), dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (ff[:, 0] if dx else 1 - ff[:, 0])
                    * (ff[:, 1] if dy else 1 - ff[:, 1])
                    * (ff[:, 2] if dz else 1 - ff[:, 2])
                )
                sample = vol[ii[:, 0] + dx, ii[:, 1] + dy, ii[:, 2] + dz, :]
                acc += w[:, None] * sample
    out[inside] = acc
    return out


def rotate_pts_c2z(pts):
    """C2(z): (x,y,z) -> (-x,-y, z)."""
    out = pts.copy()
    out[:, 0] = -pts[:, 0]
    out[:, 1] = -pts[:, 1]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="D2 antisymmetry probe, signed-component form (sec 5.7.4)")
    parser.add_argument("--field", required=True, help="nfield .npy [N,N,N,3]")
    parser.add_argument("--length", type=float, required=True, help="box half-length")
    parser.add_argument("--q-shells", type=float, nargs="+",
                        default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    parser.add_argument("--n-points", type=int, default=6000)
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nfield = load_field(args.field)
    n = nfield.shape[0]
    print(f"[d2_probe] loaded {args.field}, shape={nfield.shape}, dtype={nfield.dtype}")

    dirs = fibonacci_sphere(args.n_points)

    per_shell = []
    for r in args.q_shells:
        pts = dirs * r
        # Original delta_n_perp on the shell
        nv_orig = trilinear_sample_vec3(nfield, pts, args.length)
        dn_orig = nv_orig[:, :2]
        # Sample the rotated field: at physical point x we want the value
        # the original field has at R_z(-pi) x = (-x, -y, z).
        pts_rot_input = rotate_pts_c2z(pts)
        nv_rot = trilinear_sample_vec3(nfield, pts_rot_input, args.length)
        dn_rot = nv_rot[:, :2]
        # Rule (*): dn_rot = -dn_orig.
        residual = dn_rot + dn_orig
        abs_orig = np.linalg.norm(dn_orig, axis=1)
        abs_resid = np.linalg.norm(residual, axis=1)

        mean_abs_orig = float(np.mean(abs_orig))
        mean_abs_resid = float(np.mean(abs_resid))
        max_abs_resid = float(np.max(abs_resid))
        if mean_abs_orig > 1e-12:
            asym_signed = mean_abs_resid / mean_abs_orig
        else:
            asym_signed = float("nan")

        # Also decompose dn_orig into phi-even and phi-odd halves as a
        # cheap proxy of the even-m vs odd-m block.  phi-even means
        # f(phi+pi) = +f(phi) and phi-odd means f(phi+pi) = -f(phi).
        # Build the even/odd split by averaging:
        #   dn_even = (dn(x) + dn(R_z(-pi) x)) / 2
        #   dn_odd  = (dn(x) - dn(R_z(-pi) x)) / 2
        # Rule (*) says dn_even ~ 0, dn_odd ~ dn_orig.
        dn_even = 0.5 * (dn_orig + dn_rot)  # = residual / 2
        dn_odd = 0.5 * (dn_orig - dn_rot)
        mean_abs_even = float(np.mean(np.linalg.norm(dn_even, axis=1)))
        mean_abs_odd = float(np.mean(np.linalg.norm(dn_odd, axis=1)))
        if mean_abs_orig > 1e-12:
            even_fraction = mean_abs_even / mean_abs_orig
            odd_fraction = mean_abs_odd / mean_abs_orig
        else:
            even_fraction = float("nan")
            odd_fraction = float("nan")

        per_shell.append({
            "r": float(r),
            "mean_abs_delta_n_perp": mean_abs_orig,
            "mean_abs_residual_signed": mean_abs_resid,
            "max_abs_residual_signed": max_abs_resid,
            "antisymmetry_index_signed": asym_signed,
            "phi_even_rms": mean_abs_even,
            "phi_odd_rms": mean_abs_odd,
            "even_fraction": even_fraction,
            "odd_fraction": odd_fraction,
        })
        print(f"  r={r:6.2f}  <|dn|>={mean_abs_orig:.4e}  "
              f"<|dn_rot+dn|>={mean_abs_resid:.4e}  asym_signed={asym_signed:.4e}  "
              f"even_frac={even_fraction:.4e}  odd_frac={odd_fraction:.4e}")

    np.save(out_dir / "d2_antisym_vs_r.npy",
            np.array([[s["r"], s["mean_abs_delta_n_perp"], s["mean_abs_residual_signed"],
                       s["phi_even_rms"], s["phi_odd_rms"]]
                      for s in per_shell]))

    if args.q_shells:
        snap_r = args.q_shells[len(args.q_shells) // 2]
        snap_pts = dirs * snap_r
        snap_nv = trilinear_sample_vec3(nfield, snap_pts, args.length)
        np.save(out_dir / "shell_delta_n_original.npy", snap_nv)

    # Verdict: median over shells of antisymmetry_index_signed.
    asym_vals = [s["antisymmetry_index_signed"] for s in per_shell
                 if not math.isnan(s["antisymmetry_index_signed"])]
    even_frac_vals = [s["even_fraction"] for s in per_shell
                      if not math.isnan(s["even_fraction"])]
    median_asym = float(np.median(asym_vals)) if asym_vals else float("nan")
    median_even_frac = float(np.median(even_frac_vals)) if even_frac_vals else float("nan")

    if median_asym < 1e-2 and median_even_frac < 1e-2:
        verdict = ("C2(z) ACTS AS -I ON FAR-FIELD delta_n_perp  "
                   "(Assumption 5.A SUPPORTED, even-m block null)")
    elif median_asym < 1e-1:
        verdict = ("C2(z) MOSTLY ACTS AS -I  "
                   "(Assumption 5.A PARTIAL, residual numerical noise)")
    elif median_asym > 1.5:
        verdict = ("C2(z) ACTS AS +I ON FAR-FIELD delta_n_perP  "
                   "(Assumption 5.A REJECTED, symmetric, not antisymmetric)")
    else:
        verdict = ("C2(z) ACTS AS MIXED  "
                   "(Assumption 5.A REJECTED, partial antisymmetry)")

    summary = {
        "field": args.field,
        "N": int(n),
        "length": args.length,
        "n_points": int(args.n_points),
        "q_shells": list(args.q_shells),
        "per_shell": per_shell,
        "median_antisymmetry_index_signed": median_asym,
        "median_even_fraction": median_even_frac,
        "verdict": verdict,
    }
    with open(out_dir / "d2_symmetry_probe.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nVERDICT: {verdict}")
    print(f"median signed antisymmetry index: {median_asym:.4e}")
    print(f"median even_fraction (phi-even block / total): {median_even_frac:.4e}")
    print(f"artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()