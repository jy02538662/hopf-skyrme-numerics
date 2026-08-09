#!/usr/bin/env python3
"""
验证2：k_ex 曲面积分双通道（线性响应）。

定义（与推导一致）:
  DeltaTheta_ex = 2 * flux(b)
  k_ex = (1/2) d(DeltaTheta)/dα = flux(δb/α)

通道 A（FD）:   flux( (b[n_α]-b[n_0])/α )
通道 B（核）:   flux( K θ̃ + M·∇θ̃ )

只检验线性响应在曲面积分上是否闭合；
不检验绝热输运 / 交换拓扑常数。

用法:
  python verify2_flux_dual_channel.py --abs-outputs --Q 1 --out-dir /tmp/k_ex_v2
  python verify2_flux_dual_channel.py --synthetic --out-dir ./out_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from berry_kernels import (
    apply_kernel,
    berry_curvature,
    finite_diff_db_dalpha,
    grid_axes,
    grid_spacing,
    response_kernels,
    synthetic_bp_skyrmion,
)

# Prefer helpers from berry_kernels if present (newer file); else local fallback
# so GPU can run after copying only this script atop an older berry_kernels.py.
try:
    from berry_kernels import make_theta_tilde, planar_disk_flux
except ImportError:

    def make_theta_tilde(n: int, length: float, mode: str, soft: float) -> np.ndarray:
        ax = grid_axes(n, length)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        R = np.sqrt(X * X + Y * Y + Z * Z)
        if mode == "soft_monopole":
            return 1.0 / np.sqrt(R * R + soft * soft)
        if mode == "gaussian":
            return np.exp(-(R * R) / (2.0 * soft * soft))
        if mode == "linear_x":
            return X.copy()
        raise ValueError(mode)

    def planar_disk_flux(vec, length, plane, radius, offset=0.0):
        n = vec.shape[0]
        h = grid_spacing(n, length)
        ax = grid_axes(n, length)
        i0 = int(np.argmin(np.abs(ax - offset)))
        if plane == "xy":
            X, Y = np.meshgrid(ax, ax, indexing="ij")
            mask = (X * X + Y * Y) <= radius * radius
            return float(np.sum(vec[:, :, i0, 2][mask]) * h * h)
        if plane == "xz":
            X, Z = np.meshgrid(ax, ax, indexing="ij")
            mask = (X * X + Z * Z) <= radius * radius
            return float(np.sum(vec[:, i0, :, 1][mask]) * h * h)
        if plane == "yz":
            Y, Z = np.meshgrid(ax, ax, indexing="ij")
            mask = (Y * Y + Z * Z) <= radius * radius
            return float(np.sum(vec[i0, :, :, 0][mask]) * h * h)
        raise ValueError(plane)


def _load_screening_common():
    common_path = _HERE.parent / "screening_progression_3step" / "common.py"
    if not common_path.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("screening_common", common_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_field(args):
    if args.synthetic:
        nfield, length = synthetic_bp_skyrmion()
        return nfield, length, "synthetic_bp_skyrmion"
    common = _load_screening_common()
    if common is None:
        raise SystemExit("need screening_progression_3step/common.py or --synthetic")
    table = common.resolve_field_table(args.abs_outputs)
    meta = table[args.Q]
    path = args.field or meta["path"]
    length = float(meta["length"])
    if not Path(path).is_file():
        raise SystemExit(f"missing field: {path}")
    return common.load_nfield(path), length, path


def parse_radii(s: str, length: float):
    vals = [float(x) for x in s.split(",") if x.strip()]
    return [r for r in vals if r < 0.95 * length]


def main():
    ap = argparse.ArgumentParser(description="V2: flux dual-channel for k_ex")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument(
        "--alphas",
        type=str,
        default="",
        help="optional comma list to scan linearity, e.g. 1e-5,1e-4,1e-3",
    )
    ap.add_argument(
        "--theta-mode",
        type=str,
        default="soft_monopole",
        choices=["soft_monopole", "gaussian", "linear_x"],
    )
    ap.add_argument("--soft", type=float, default=2.0)
    ap.add_argument(
        "--planes",
        type=str,
        default="xy,xz,yz",
        help="comma list among xy,xz,yz",
    )
    ap.add_argument(
        "--radii",
        type=str,
        default="3,5,7",
        help="disk radii for planar fluxes",
    )
    ap.add_argument("--offset", type=float, default=0.0, help="plane offset")
    ap.add_argument("--tol", type=float, default=0.15, help="rel tol on primary surfaces")
    ap.add_argument(
        "--flux-floor",
        type=float,
        default=1e-6,
        help="absolute floor on |k| (also see --rel-floor)",
    )
    ap.add_argument(
        "--rel-floor",
        type=float,
        default=0.05,
        help="skip surface if |k| < rel_floor * max_|k| over all disks",
    )
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    nfield, length, path = load_field(args)
    n = nfield.shape[0]
    h = grid_spacing(n, length)
    planes = [p.strip() for p in args.planes.split(",") if p.strip()]
    radii = parse_radii(args.radii, length)
    if not radii:
        raise SystemExit("no valid radii inside box")

    print("=" * 72)
    print("VERIFY2: flux dual-channel  (linear response only)")
    print(f"  field: {path}")
    print(f"  L={length}  N={n}  h={h:.4f}")
    print(f"  theta_mode={args.theta_mode}  soft={args.soft}  alpha={args.alpha}")
    print(f"  planes={planes}  radii={radii}  offset={args.offset}")
    print("=" * 72)

    K, M, _ = response_kernels(nfield, h)
    theta = make_theta_tilde(n, length, args.theta_mode, args.soft)
    gth = (
        np.gradient(theta, h, axis=0, edge_order=2),
        np.gradient(theta, h, axis=1, edge_order=2),
        np.gradient(theta, h, axis=2, edge_order=2),
    )
    db_kernel = apply_kernel(K, M, theta, gth)
    db_fd = finite_diff_db_dalpha(nfield, h, theta, alpha=args.alpha)
    b0 = berry_curvature(nfield, h)

    # first pass: compute all fluxes
    raw = []
    for plane in planes:
        for R in radii:
            fb0 = planar_disk_flux(b0, length, plane, R, args.offset)
            k_a = planar_disk_flux(db_fd, length, plane, R, args.offset)
            k_b = planar_disk_flux(db_kernel, length, plane, R, args.offset)
            raw.append(
                {
                    "plane": plane,
                    "radius": R,
                    "offset": args.offset,
                    "flux_b0": fb0,
                    "k_FD": k_a,
                    "k_kernel": k_b,
                    "scale": max(abs(k_a), abs(k_b)),
                }
            )

    max_scale = max((r["scale"] for r in raw), default=0.0)
    eff_floor = max(args.flux_floor, args.rel_floor * max_scale)

    rows = []
    rels = []
    print(
        f"{'plane':>5} {'R':>6} {'flux_b0':>12} {'k_FD':>12} {'k_kern':>12} {'rel':>10}  note"
    )
    print(f"  (effective |k| floor = {eff_floor:.3e}  from abs={args.flux_floor} "
          f"and {args.rel_floor:.0%}*max|k|={args.rel_floor * max_scale:.3e})")
    for r in raw:
        scale = r["scale"]
        if scale < eff_floor:
            note = "SKIP_noise"
            rel = float("nan")
        else:
            note = "ok"
            rel = abs(r["k_FD"] - r["k_kernel"]) / scale
            rels.append(rel)
        row = {**r, "rel": rel, "note": note}
        rows.append(row)
        rel_s = f"{rel:10.3e}" if note == "ok" else f"{'nan':>10}"
        print(
            f"{r['plane']:>5} {r['radius']:6.2f} {r['flux_b0']:12.4e} "
            f"{r['k_FD']:12.4e} {r['k_kernel']:12.4e} {rel_s}  {note}"
        )

    # alpha scan on strongest |k| active surface
    alpha_rows = []
    active = [r for r in rows if r["note"] == "ok"]
    if args.alphas.strip() and active:
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        primary = max(active, key=lambda r: r["scale"])
        plane0, R0 = primary["plane"], primary["radius"]
        k_b0 = planar_disk_flux(db_kernel, length, plane0, R0, args.offset)
        print("-" * 72)
        print(f"alpha scan on PRIMARY {plane0} R={R0}  (|k|={primary['scale']:.3e})")
        for a in alphas:
            db = finite_diff_db_dalpha(nfield, h, theta, alpha=a)
            k_a = planar_disk_flux(db, length, plane0, R0, args.offset)
            scale = max(abs(k_a), abs(k_b0), 1e-30)
            rel = abs(k_a - k_b0) / scale
            alpha_rows.append({"alpha": a, "k_FD": k_a, "k_kernel": k_b0, "rel": rel})
            print(f"  alpha={a:.1e}  k_FD={k_a:.6e}  k_kern={k_b0:.6e}  rel={rel:.3e}")

    # primary = max-|k|; verdict from primary + all active
    if not active:
        verdict = "NO_SIGNAL"
        med_rel = float("nan")
        max_rel = float("nan")
        primary_rel = float("nan")
    else:
        primary = max(active, key=lambda r: r["scale"])
        primary_rel = float(primary["rel"])
        max_rel = float(np.max(rels))
        med_rel = float(np.median(rels))
        # PASS if primary good; WEAK if primary ok-ish; else FAIL
        if primary_rel < args.tol and med_rel < args.tol:
            verdict = "PASS"
        elif primary_rel < args.tol:
            verdict = "WEAK_PASS"
        elif primary_rel < 2.0 * args.tol:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

    print("-" * 72)
    if active:
        print(
            f"active={len(active)}/{len(rows)}  primary={primary['plane']} R={primary['radius']} "
            f"rel={primary_rel:.3e}  median={med_rel:.3e}  max={max_rel:.3e}  tol={args.tol}"
        )
    else:
        print(f"active=0/{len(rows)}  (all below effective floor {eff_floor:.3e})")
    print(f"VERDICT: {verdict}")
    print(
        "  PASS/WEAK based mainly on PRIMARY (largest |k|) disk.\n"
        "  SKIP_noise = |k| < max(abs_floor, rel_floor*max|k|).\n"
        "  Does NOT prove adiabatic transport or path-independent k_ex."
    )
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "scale"}
        if isinstance(rr.get("rel"), float) and (rr["rel"] != rr["rel"]):
            rr["rel"] = None
        rows_out.append(rr)
    summary = {
        "path": path,
        "length": length,
        "N": n,
        "h": h,
        "theta_mode": args.theta_mode,
        "soft": args.soft,
        "alpha": args.alpha,
        "tol": args.tol,
        "flux_floor": args.flux_floor,
        "rel_floor": args.rel_floor,
        "eff_floor": eff_floor,
        "rows": rows_out,
        "alpha_scan": alpha_rows,
        "median_rel": None if not active else med_rel,
        "max_rel": None if not active else max_rel,
        "primary_rel": None if not active else primary_rel,
        "verdict": verdict,
    }
    with open(out_dir / "verify2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir / 'verify2_summary.json'}")


if __name__ == "__main__":
    main()
