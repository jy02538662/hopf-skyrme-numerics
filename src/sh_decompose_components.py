"""Per-component spherical-harmonic decomposition of the far-field delta_n.

Purpose (preprint sections 5.7.3, 5.7.4, gap 2; qeff_definition_v1.md v1.1):
    Decompose delta_n_x and delta_n_y separately into real spherical harmonics
    up to lmax, distinguish even-m and odd-m coefficient blocks (Assumption 5.A
    states D2 forces m-odd only).

    Per shell radius r we report:

      1. l=0 coefficient of delta_n_x, delta_n_y   (A1 projection of each)
         -> "actively suppressed by symmetry" -> c_{00} ~ 0
         -> "trivial coincidence" -> c_{00} != 0 because sign cancellation
            in quadratic quantity; relies on signed-components staying nonzero.

      2. Even-m block (m=0, +/-2, +/-4, ...) of delta_n_x and delta_n_y.
         -> Assumption 5.A says these should all be ~0 (C2(z) parity lock).

      3. Odd-m block (m=+/-1, +/-3, ...) of delta_n_x and delta_n_y.
         -> Leading contribution if Assumption 5.A holds.
            For Q=2 expect odd-m l=1 dipole to dominate (since the
            double-ring axis gives a net horizontal orientation parity).

      4. l=0 coefficient of chi and chi^2
         -> chi_l0: numerically decays (gap 2 prediction).
         -> chi2_l0: nonzero and finite (energy density surrogate stays).

      5. Per-l rms per component, the per-l absolute coefficient vector
         and the per-r signed-snapshot of (delta_n_x, delta_n_y) so the
         output is reproducible.

Inputs / outputs mirror farfield_multipole.py.

Outputs (in --out):
    sh_components.json            : full per-shell, per-component, per-l data
    sh_components_summary.csv     : compact table for quick review
    even_m_block_audit.csv       : even-m rms vs odd-m rms per component
                                   (key Assumption 5.A test)
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


def real_sph_basis(dirs, lmax=4):
    """Return unnormalized real spherical harmonics up to lmax.

    Convention (matches the "m=-l, ..., +l" ordering used in sh_decompose v1.0):

      l=0 : 1
      l=1 : (y, z, x)       i.e. m=-1, 0, +1
      l=2 : (xy, yz, 3z^2-1, xz, x^2-y^2)            m=-2,-1,0,+1,+2
      l=3 : y(3x^2-y^2), xyz, z(5z^2-3), x(5z^2-1), x(x^2-3y^2)
      l=4 : (4 Cartesian + 1 zonal), up to signs, see source.

    Returns dict {l: [basis_func_1, ..., basis_func_K]}.
    The factors only change normalization; for power-per-l they cancel.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    out = {}
    out[0] = [np.ones_like(x)]
    out[1] = [y, z, x]
    out[2] = [x * y, y * z, 3.0 * z2 - 1.0, x * z, x2 - y2]
    if lmax >= 3:
        out[3] = [
            y * (3.0 * x2 - y2),
            x * y * z,
            z * (5.0 * z2 - 3.0),
            x * (5.0 * z2 - 1.0),
            x * (x2 - 3.0 * y2),
        ]
    if lmax >= 4:
        # Real l=4 basis (9 funcs), standard Cartesian set:
        # m=-4,-3,-2,-1,0,+1,+2,+3,+4
        out[4] = [
            x * y * (x2 - y2),                  # m=-4
            y * z * (3.0 * x2 - y2),            # m=-3
            x * y * (7.0 * z2 - 1.0),           # m=-2
            y * z * (7.0 * z2 - 3.0),           # m=-1
            35.0 * z2 * z2 - 30.0 * z2 + 3.0,   # m=0
            x * z * (7.0 * z2 - 3.0),           # m=+1
            (x2 - y2) * (7.0 * z2 - 1.0),       # m=+2
            x * z * (x2 - 3.0 * y2),            # m=+3
            (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2)),  # m=+4
        ]
    return out


def is_even_m(l, m_index):
    """Return True if the m_index-th function in real_sph_basis[l] has even m."""
    # Only meaningful for l>=1. For l=0, m=0 (even).
    if l == 0:
        return True
    # The ordering above is "m=-l, m=-l+1, ..., m=+l".
    m = -l + m_index
    return (m % 2 == 0)


def lstsq_coeffs(basis_cols, vals):
    basis = np.stack(basis_cols, axis=1)
    coeff, *_ = np.linalg.lstsq(basis, vals, rcond=None)
    return coeff


def split_even_odd_blocks(coeff, l, lmax):
    """Split a coefficient vector into even-m rms and odd-m rms."""
    even_rms = 0.0
    odd_rms = 0.0
    for j, c in enumerate(coeff):
        if is_even_m(l, j):
            even_rms += float(c) * float(c)
        else:
            odd_rms += float(c) * float(c)
    return math.sqrt(even_rms), math.sqrt(odd_rms)


def main():
    parser = argparse.ArgumentParser(
        description="Per-component SH decomposition with even/odd m audit (sec 5.7.4 / qeff v1.1)"
    )
    parser.add_argument("--field", required=True, help="nfield .npy [N,N,N,3]")
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--r-shells", type=float, nargs="+",
                        default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--lmax", type=int, default=4,
                        help="up to l=4 to capture l=4 shell for Q=3/4 audits")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nfield = load_field(args.field)
    n = nfield.shape[0]
    print(f"[sh_decomp] loaded {args.field}, shape={nfield.shape}")

    dirs = fibonacci_sphere(args.n_points)
    basis = real_sph_basis(dirs, lmax=args.lmax)

    rows = []
    for r in args.r_shells:
        pts = dirs * r
        nv = trilinear_sample_vec3(nfield, pts, args.length)
        dn_x = nv[:, 0]
        dn_y = nv[:, 1]
        chi = np.sqrt(np.clip(dn_x * dn_x + dn_y * dn_y, 0.0, None))
        chi2 = dn_x * dn_x + dn_y * dn_y

        per_comp = {}
        for comp_name, comp in [("dn_x", dn_x), ("dn_y", dn_y), ("chi", chi), ("chi2", chi2)]:
            per_l = {}
            for l in range(args.lmax + 1):
                coeff = lstsq_coeffs(basis[l], comp)
                l_rms = float(np.sqrt(np.sum(coeff * coeff)))
                per_l[f"l{l}_rms"] = l_rms
                per_l[f"l{l}_coeffs"] = [float(c) for c in coeff]
                # even-m / odd-m split (only matters for l>=1)
                if l >= 1:
                    e_rms, o_rms = split_even_odd_blocks(coeff, l, args.lmax)
                    per_l[f"l{l}_even_rms"] = e_rms
                    per_l[f"l{l}_odd_rms"] = o_rms
            per_comp[comp_name] = per_l

        rows.append({
            "r": float(r),
            "per_component": per_comp,
            "chi_rms": float(np.sqrt(np.mean(chi * chi))),
            "dnx_rms": float(np.sqrt(np.mean(dn_x * dn_x))),
            "dny_rms": float(np.sqrt(np.mean(dn_y * dn_y))),
        })
        p = per_comp
        print(f"  r={r:6.2f}  "
              f"dnx_l0={p['dn_x']['l0_rms']:.3e}  dnx_l1_even={p['dn_x']['l1_even_rms']:.3e}  "
              f"dnx_l1_odd={p['dn_x']['l1_odd_rms']:.3e}  "
              f"chi_l0={p['chi']['l0_rms']:.3e}  chi_l2={p['chi']['l2_rms']:.3e}")

    # Compact CSV: main rows
    csv_path = out_dir / "sh_components_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("r,dnx_l0,dny_l0,chi_l0,chi_l1,chi_l2,chi_l3,chi_l4,chi2_l0,chi2_l2,chi_rms,dnx_rms,dny_rms\n")
        for row in rows:
            p = row["per_component"]
            chi4 = p["chi"].get("l4_rms", 0.0)
            f.write(
                f"{row['r']:.3f},"
                f"{p['dn_x']['l0_rms']:.6e},"
                f"{p['dn_y']['l0_rms']:.6e},"
                f"{p['chi']['l0_rms']:.6e},"
                f"{p['chi']['l1_rms']:.6e},"
                f"{p['chi']['l2_rms']:.6e},"
                f"{p['chi']['l3_rms']:.6e},"
                f"{chi4:.6e},"
                f"{p['chi2']['l0_rms']:.6e},"
                f"{p['chi2']['l2_rms']:.6e},"
                f"{row['chi_rms']:.6e},"
                f"{row['dnx_rms']:.6e},"
                f"{row['dny_rms']:.6e}\n"
            )

    # Even-m vs odd-m audit (key Assumption 5.A test for delta_n_x and delta_n_y)
    audit_path = out_dir / "even_m_block_audit.csv"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("r,comp,l,even_rms,odd_rms,ratio_even_over_odd\n")
        for row in rows:
            r = row["r"]
            for comp_name in ["dn_x", "dn_y"]:
                comp_data = row["per_component"][comp_name]
                for l in range(1, args.lmax + 1):
                    e = comp_data.get(f"l{l}_even_rms", 0.0)
                    o = comp_data.get(f"l{l}_odd_rms", 0.0)
                    if o > 1e-30:
                        ratio = e / o
                    else:
                        ratio = float("nan")
                    f.write(f"{r:.3f},{comp_name},l{l},{e:.6e},{o:.6e},{ratio:.6e}\n")

    # Verdict
    # If for many r values dnx_even and dny_even are << dnx_odd and dny_odd,
    # Assumption 5.A is supported.  Threshold: even_rms < 1e-2 * odd_rms.
    threshold_count = 0
    total = 0
    for row in rows:
        for comp_name in ["dn_x", "dn_y"]:
            for l in range(1, args.lmax + 1):
                e = row["per_component"][comp_name].get(f"l{l}_even_rms", 0.0)
                o = row["per_component"][comp_name].get(f"l{l}_odd_rms", 0.0)
                total += 1
                if o > 1e-30 and e < 1e-2 * o:
                    threshold_count += 1

    fraction = threshold_count / total if total else float("nan")
    if fraction > 0.95:
        verdict = "EVEN-M BLOCK IS NULL -> Assumption 5.A SUPPORTED (D2 C2(z) parity lock)"
    elif fraction > 0.7:
        verdict = "EVEN-M BLOCK IS MOSTLY SMALL -> Assumption 5.A PARTIAL (residual numerical)"
    else:
        verdict = "EVEN-M BLOCK HAS COMPARABLE POWER -> Assumption 5.A REJECTED"

    summary = {
        "field": args.field,
        "N": int(n),
        "length": args.length,
        "lmax": int(args.lmax),
        "n_points": int(args.n_points),
        "r_shells": list(args.r_shells),
        "rows": rows,
        "even_m_block_pass_fraction": fraction,
        "verdict": verdict,
    }
    with open(out_dir / "sh_components.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {audit_path}")
    print(f"Saved: {out_dir / 'sh_components.json'}")
    print(f"Even-m null fraction: {fraction:.3f}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()