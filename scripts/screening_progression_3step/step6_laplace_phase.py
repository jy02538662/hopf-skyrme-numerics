#!/usr/bin/env python3
"""
Step 6 — Laplace-phase check on far-field dn+ = nx + i ny (default Q=1).

Continuum: in the linearized far zone, ∇² δn_⊥ ≈ 0.
Primary dimensionless diagnostic on shells:

    eta = |r² ∇² dn+| / |dn+|

(Acceptance on dimensional |∇²|/|dn+| alone is ill-posed: units 1/length².)

Also report L²|∇²|/|dn+| for box-unit comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_common():
    here = Path(__file__).resolve().parent
    common_path = here / "common.py"
    if not common_path.is_file():
        raise SystemExit(f"Cannot find common.py next to this script.\n  expected: {common_path}")
    spec = importlib.util.spec_from_file_location("screening_common", common_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_common = _load_common()
load_nfield = _common.load_nfield
resolve_field_table = _common.resolve_field_table
save_json = _common.save_json
fibonacci_sphere = _common.fibonacci_sphere
trilinear_sample = _common.trilinear_sample


def grid_spacing(n: int, length: float) -> float:
    return 2.0 * length / (n - 1)


def laplacian_3d(f: np.ndarray, h: float) -> np.ndarray:
    """Central second differences; Neumann-zero on faces (match FS prototype spirit)."""
    lap = np.zeros_like(f, dtype=np.float64)
    # interior
    lap[1:-1, :, :] += (f[2:, :, :] - 2.0 * f[1:-1, :, :] + f[:-2, :, :]) / (h * h)
    lap[:, 1:-1, :] += (f[:, 2:, :] - 2.0 * f[:, 1:-1, :] + f[:, :-2, :]) / (h * h)
    lap[:, :, 1:-1] += (f[:, :, 2:] - 2.0 * f[:, :, 1:-1] + f[:, :, :-2]) / (h * h)
    return lap


def shell_stats(nx, ny, lap_x, lap_y, length, r, n_points, floor=1e-8):
    """Sample real/imag parts, then form |dn+| and |∇² dn+| on the shell."""
    dirs = fibonacci_sphere(n_points)
    pts = r * dirs
    fx = trilinear_sample(nx, pts, length)
    fy = trilinear_sample(ny, pts, length)
    lx = trilinear_sample(lap_x, pts, length)
    ly = trilinear_sample(lap_y, pts, length)
    f = np.sqrt(fx * fx + fy * fy)
    lap = np.sqrt(lx * lx + ly * ly)

    mask = f > floor
    if not np.any(mask):
        return None
    f_m, lap_m = f[mask], lap[mask]
    eta = (r * r) * lap_m / f_m  # dimensionless |r²∇²f|/|f|
    return {
        "n_pts_used": int(np.count_nonzero(mask)),
        "mean_abs_f": float(np.mean(f_m)),
        "mean_abs_lap": float(np.mean(lap_m)),
        "rms_abs_f": float(np.sqrt(np.mean(f_m**2))),
        "rms_abs_lap": float(np.sqrt(np.mean(lap_m**2))),
        "eta_mean": float(np.mean(eta)),
        "eta_median": float(np.median(eta)),
        "eta_p90": float(np.percentile(eta, 90)),
        "frac_eta_lt_1e-2": float(np.mean(eta < 1e-2)),
        "frac_eta_lt_1e-1": float(np.mean(eta < 1e-1)),
        "L2_lap_over_f_mean": float(np.mean(lap_m / f_m) * (length**2)),
    }


def main():
    ap = argparse.ArgumentParser(description="Step6: Laplace-phase check for dn+")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="", help="override nfield path")
    ap.add_argument(
        "--r-shells",
        type=str,
        default="5,6,7,8,9",
        help="comma-separated radii",
    )
    ap.add_argument("--n-points", type=int, default=6000)
    ap.add_argument("--floor", type=float, default=1e-6, help="min |dn+| on shell")
    ap.add_argument("--eta-tol", type=float, default=1e-2, help="median |r²∇²|/|f| target")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    table = resolve_field_table(args.abs_outputs)
    meta = table[args.Q]
    path = args.field or meta["path"]
    if not Path(path).is_file():
        raise SystemExit(f"missing field: {path}")
    length = float(meta["length"])
    nfield = load_nfield(path)
    n = nfield.shape[0]
    h = grid_spacing(n, length)

    nx = nfield[..., 0]
    ny = nfield[..., 1]
    lap_x = laplacian_3d(nx, h)
    lap_y = laplacian_3d(ny, h)

    r_shells = [float(s) for s in args.r_shells.split(",") if s.strip()]
    r_shells = [r for r in r_shells if r < 0.95 * length]
    if not r_shells:
        raise SystemExit("no valid r shells inside box")

    print("=" * 72)
    print(f"STEP 6: Laplace-phase check  Q={args.Q}  L={length}  N={n}  h={h:.4f}")
    print(f"  field: {path}")
    print("  primary: eta = |r^2 ∇^2 dn+| / |dn+|   (dimensionless)")
    print(f"  target: median eta < {args.eta_tol} and not growing with r")
    print("=" * 72)

    rows = []
    for r in r_shells:
        st = shell_stats(
            nx, ny, lap_x, lap_y, length, r, args.n_points, floor=args.floor
        )
        if st is None:
            print(f"r={r:.2f}: SKIP (no points above floor)")
            continue
        st["r"] = r
        rows.append(st)
        print(
            f"r={r:5.2f}  eta_med={st['eta_median']:.4e}  eta_mean={st['eta_mean']:.4e}  "
            f"eta_p90={st['eta_p90']:.4e}  frac<1e-2={st['frac_eta_lt_1e-2']:.3f}  "
            f"|dn+|~{st['mean_abs_f']:.3e}  L^2|lap|/|f|~{st['L2_lap_over_f_mean']:.3e}"
        )

    if len(rows) < 2:
        raise SystemExit("need >=2 shells")

    etas = [row["eta_median"] for row in rows]
    drop = etas[0] / max(etas[-1], 1e-30)
    outer = etas[len(etas) // 2 :]
    pass_tol = etas[-1] < args.eta_tol
    pass_outer = float(np.median(outer)) < args.eta_tol
    not_growing = etas[-1] <= etas[0] * 1.2

    if pass_tol and pass_outer and not_growing:
        verdict = "SUPPORT"
    elif pass_outer:
        verdict = "WEAK_SUPPORT"
    else:
        verdict = "NOT_IN_LAPLACE_PHASE"

    print("-" * 72)
    print(f"eta_median trend: {[f'{e:.3e}' for e in etas]}")
    print(f"drop (first/last) = {drop:.3f}  last_eta={etas[-1]:.3e}  tol={args.eta_tol}")
    print(f"VERDICT: {verdict}")
    print(
        "  SUPPORT       : outer median eta < tol and not growing with r\n"
        "  WEAK_SUPPORT  : outer OK but trend noisy\n"
        "  NOT_IN_LAPLACE_PHASE : raise r_min / larger box\n"
        "  Note: raw |∇²|/|f| has units 1/L²; use eta=|r²∇²|/|f|."
    )
    print("=" * 72)

    out = {
        "Q": args.Q,
        "path": path,
        "length": length,
        "N": n,
        "h": h,
        "eta_tol": args.eta_tol,
        "shells": rows,
        "verdict": verdict,
        "drop_first_over_last": drop,
    }
    out_dir = Path(args.out_dir)
    save_json(out_dir / f"step6_laplace_phase_Q{args.Q}.json", out)
    print(f"saved {out_dir / f'step6_laplace_phase_Q{args.Q}.json'}")


if __name__ == "__main__":
    main()
