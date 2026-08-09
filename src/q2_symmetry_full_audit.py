"""Full symmetry group audit for Q=2 Hopf-Skyrme field (corrected v2).

The vacuum n0 = (0,0,1) is fixed ONLY by operations whose action matrix R
preserves n0.  For a true symmetry of the field we need a JOINT operation:
    n'(x) = R @ n(T^{-1} x)
to equal n(x) at every x.

If R @ n0 != n0 then "R alone" is not a symmetry, but a JOINT operation
"R such that R @ n0 = n0" can still be a symmetry.

Strategy:
  For each space operation T (8 candidates: C2x, C2y, C2z, Mx, My, Mz,
  S4x, S4y), we sweep over the rotation subgroup that fixes n0 -- this
  is SO(2)_z, parametrized by angle theta.  For each theta, we test
  whether n(x) = R_z(theta) @ n(T^{-1} x) holds to within tolerance.
  We report the minimum L_inf residual over theta, the optimal theta,
  and the verdict.

If the minimum residual is small, the joint operation is a symmetry.
The "internal" rotation R_z(theta) absorbs the freedom of how the field
gets rotated back to its vacuum configuration.

Output:
    q2_symmetry_full_audit.json with per-op residual surface, optimal
    theta, and group verdict.
"""

import argparse
import json
from pathlib import Path

import numpy as np


N0 = np.array([0.0, 0.0, 1.0])  # the vacuum


def Rz(theta):
    """Rotation by angle theta about the z-axis (preserves n0)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# 8 space operations T as 3x3 matrices acting on coordinates (x,y,z).
# Each T is in O(3) (proper or improper).
# ---------------------------------------------------------------------------

SPACE_OPS = {
    "C2x": np.diag([1.0, -1.0, -1.0]),   # (x, y, z) -> ( x, -y, -z)
    "C2y": np.diag([-1.0, 1.0, -1.0]),   # (x, y, z) -> (-x,  y, -z)
    "C2z": np.diag([-1.0, -1.0, 1.0]),   # (x, y, z) -> (-x, -y,  z)
    "Mx":   np.diag([-1.0, 1.0, 1.0]),   # mirror in yz-plane
    "My":   np.diag([1.0, -1.0, 1.0]),   # mirror in xz-plane
    "Mz":   np.diag([1.0, 1.0, -1.0]),   # mirror in xy-plane
    "S4x":  np.array([                   # (x, y, z) -> (x, -z,  y)
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]),
    "S4y":  np.array([                   # (x, y, z) -> (-z, y,  x)
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]),
}


def build_permutation(N, T):
    """Return indices (px, py, pz) such that
        nfield[T^-1 x] = nfield[px, py, pz, :]
    where x is the integer grid index in [-L, L)^3 mapped to {0,...,N-1}^3.

    For T = diag(t_x, t_y, t_z) with t_i = +/-1:
        T^{-1} = T  (for +/-1).
        If t_i = -1, coordinate i is flipped: index i -> N-1-i.
    """
    idx = np.arange(N)
    axes = [
        (N - 1) - idx if T[0, 0] < 0 else idx,
        (N - 1) - idx if T[1, 1] < 0 else idx,
        (N - 1) - idx if T[2, 2] < 0 else idx,
    ]
    # For S4 entries we need the off-diagonal logic too; if a diagonal
    # entry of T is 0, the operation permutes axes.  S4x and S4y both
    # have one such off-diagonal: handle them explicitly.
    for i in range(3):
        if abs(T[i, i]) < 0.5:
            # The i-th row picks up from one of the j != i axes.
            # Determine which j by looking at the row.
            for j in range(3):
                if abs(T[i, j]) > 0.5:
                    axes[i] = (N - 1) - idx if T[i, j] < 0 else idx
                    break
            else:
                raise ValueError("could not parse T row")
    perm = np.indices((N, N, N))
    px = axes[0][perm[0]]
    py = axes[1][perm[1]]
    pz = axes[2][perm[2]]
    return px, py, pz


def load_field(path):
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected [N,N,N,3], got {arr.shape}")
    return arr.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Full symmetry audit (v2) for Q=2 Hopf-Skyrme field")
    parser.add_argument("--field", required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-2,
                        help="L_inf residual tolerance for 'exact' symmetry")
    parser.add_argument("--n-theta", type=int, default=181,
                        help="number of theta samples in [0, 2*pi)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nfield = load_field(args.field)
    N = nfield.shape[0]
    print(f"[q2_symm_audit v2] loaded {args.field}, shape={nfield.shape}")

    thetas = np.linspace(0.0, 2 * np.pi, args.n_theta, endpoint=False)
    rows = []

    for op_name, T in SPACE_OPS.items():
        px, py, pz = build_permutation(N, T)
        nTinv = nfield[px, py, pz, :]   # n(T^{-1} x)

        sweep = []   # list of (theta, L_inf)
        for theta in thetas:
            R = Rz(theta)
            rotated = nTinv @ R.T       # R @ n(T^{-1} x)
            residual = rotated - nfield
            L_inf = float(np.max(np.abs(residual)))
            sweep.append({"theta": float(theta), "L_inf": L_inf})

        # minimum residual and optimal theta
        min_idx = int(np.argmin([s["L_inf"] for s in sweep]))
        theta_opt = sweep[min_idx]["theta"]
        L_inf_min = sweep[min_idx]["L_inf"]

        # also L1 / L2 / chi_l0 of residual at optimal theta
        R_opt = Rz(theta_opt)
        rotated_opt = nTinv @ R_opt.T
        residual_opt = rotated_opt - nfield
        L2_opt = float(np.sqrt(np.mean(residual_opt ** 2)))
        L1_opt = float(np.mean(np.abs(residual_opt)))
        chi_l0_opt = float(np.mean(np.sqrt(
            residual_opt[..., 0] ** 2 + residual_opt[..., 1] ** 2)))

        # Vacuum-preserving status of bare T (no internal rotation)
        T_det = np.linalg.det(T)
        T_preserves_n0 = (abs(T[2, 2] - 1) < 1e-10
                          and abs(T[0, 2]) < 1e-10
                          and abs(T[1, 2]) < 1e-10
                          and abs(T[2, 0]) < 1e-10
                          and abs(T[2, 1]) < 1e-10)

        # Verdict
        if L_inf_min < args.tolerance:
            verdict = "EXACT SYMMETRY"
        elif L_inf_min < 10 * args.tolerance:
            verdict = f"NEAR SYMMETRY (residual {L_inf_min:.2e} > tolerance)"
        else:
            verdict = "NOT A SYMMETRY"

        row = {
            "op": op_name,
            "det_T": float(T_det),
            "T_preserves_vacuum": bool(T_preserves_n0),
            "L_inf_at_theta_0": sweep[0]["L_inf"],
            "L_inf_min_over_sweep": L_inf_min,
            "theta_opt_rad": theta_opt,
            "theta_opt_deg": float(np.degrees(theta_opt)),
            "L2_at_opt": L2_opt,
            "L1_at_opt": L1_opt,
            "chi_l0_at_opt": chi_l0_opt,
            "verdict": verdict,
            "sweep_sample": [
                {"theta_deg": float(np.degrees(s["theta"])),
                 "L_inf": s["L_inf"]}
                for s in sweep[::max(1, len(sweep) // 12)]
            ],
        }
        rows.append(row)
        print(
            f"  {op_name:5s} det={T_det:+.0f} vac_pres={int(T_preserves_n0)} "
            f"L_inf(theta=0)={sweep[0]['L_inf']:.3e} "
            f"L_inf_min={L_inf_min:.3e} "
            f"theta_opt={np.degrees(theta_opt):6.1f} deg  {verdict}"
        )

    exact_ops = [r["op"] for r in rows if r["verdict"] == "EXACT SYMMETRY"]
    near_ops = [r["op"] for r in rows if r["verdict"].startswith("NEAR SYMMETRY")]

    # Decide group:
    rot_set = {"C2x", "C2y", "C2z"}
    mirror_set = {"Mx", "My", "Mz"}
    s4_set = {"S4x", "S4y"}

    has_rot = rot_set <= set(exact_ops) | set(near_ops)
    has_mirror = bool(mirror_set & (set(exact_ops) | set(near_ops)))
    has_s4 = bool(s4_set & (set(exact_ops) | set(near_ops)))

    if has_rot and not has_mirror and not has_s4:
        if rot_set <= set(exact_ops):
            group_verdict = "D2 (Klein four-group, all 3 C2 exact)"
        else:
            group_verdict = "D2-like (3 C2 + near-symmetries, no mirrors)"
    elif has_rot and has_mirror and not has_s4:
        group_verdict = "D2h (added 3 mirrors)"
    elif has_rot and has_s4:
        group_verdict = "D2d (added S4 and diagonal mirrors)"
    elif {"C2z"} <= set(exact_ops) and len(exact_ops) == 1:
        group_verdict = "C2 (only C2z exact -- degenerate Q=2)"
    else:
        group_verdict = f"OTHER exact={exact_ops} near={near_ops}"

    summary = {
        "field": args.field,
        "N": int(N),
        "length": args.length,
        "tolerance": args.tolerance,
        "n_theta": args.n_theta,
        "vacuum": [0.0, 0.0, 1.0],
        "per_op": rows,
        "exact_ops": exact_ops,
        "near_ops": near_ops,
        "group_verdict": group_verdict,
    }
    with open(out_dir / "q2_symmetry_full_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nExact: {exact_ops}")
    print(f"Near  : {near_ops}")
    print(f"GROUP VERDICT: {group_verdict}")
    print(f"artifacts saved to {out_dir}/q2_symmetry_full_audit.json")


if __name__ == "__main__":
    main()