#!/usr/bin/env python3
"""
验证4：k_ex 的路径/曲面相关性（诚实拆分）。

定义: k_ex = flux(δb/α) = flux(K θ̃ + M·∇θ̃)

A) Stokes 同边界（应近似成立）:
   同一圆周 C 上 flat / hemisphere / cone 的 k_ex 是否一致。

B) 平面盘半径扫描（诊断）:
   不同 R → 不同边界；若 k_ex 随 R 大变，则不是“与路径无关的拓扑常数”。

C) 平行偏移盘诊断。

主验收看 Stokes；半径非恒定单独标注，不假装成 PASS 拓扑。

用法:
  python verify4_kex_path_independence.py --abs-outputs --Q 1 --out-dir /tmp/k_ex_v4
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
    cone_flux_xy,
    finite_diff_db_dalpha,
    grid_axes,
    grid_spacing,
    hemisphere_flux_xy,
    planar_disk_flux,
    response_kernels,
    synthetic_bp_skyrmion,
)

try:
    from berry_kernels import make_theta_tilde
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


def rel_spread(vals):
    vals = np.asarray(vals, dtype=float)
    scale = max(float(np.max(np.abs(vals))), 1e-30)
    return float((np.max(vals) - np.min(vals)) / scale)


def flux_triple(field, length, R, cone_H):
    return {
        "disk": planar_disk_flux(field, length, "xy", R, 0.0),
        "hemisphere": hemisphere_flux_xy(field, length, R, sign_z=1.0),
        "cone": cone_flux_xy(field, length, R, height=cone_H),
    }


def main():
    ap = argparse.ArgumentParser(description="V4: k_ex path / surface dependence")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--alpha", type=float, default=1e-5, help="FD alpha (small)")
    ap.add_argument(
        "--theta-mode",
        type=str,
        default="soft_monopole",
        choices=["soft_monopole", "gaussian", "linear_x"],
    )
    ap.add_argument("--soft", type=float, default=2.0)
    ap.add_argument("--stokes-R", type=float, default=3.0)
    ap.add_argument("--cone-height", type=float, default=3.0)
    ap.add_argument("--radii", type=str, default="2,3,4,5")
    ap.add_argument("--offsets", type=str, default="0,-1,1")
    ap.add_argument("--stokes-tol", type=float, default=0.10)
    ap.add_argument(
        "--channel",
        type=str,
        default="both",
        choices=["kernel", "fd", "both"],
        help="which delta-b field to integrate",
    )
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    nfield, length, path = load_field(args)
    n = nfield.shape[0]
    h = grid_spacing(n, length)
    R_s = args.stokes_R
    if R_s >= 0.9 * length:
        R_s = 0.5 * length
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    radii = [r for r in radii if r < 0.9 * length]

    print("=" * 72)
    print("VERIFY4: k_ex path/surface dependence")
    print(f"  field: {path}")
    print(f"  L={length}  N={n}  h={h:.4f}")
    print(f"  theta_mode={args.theta_mode}  soft={args.soft}  alpha={args.alpha}")
    print("=" * 72)

    K, M, _ = response_kernels(nfield, h)
    theta = make_theta_tilde(n, length, args.theta_mode, args.soft)
    gth = (
        np.gradient(theta, h, axis=0, edge_order=2),
        np.gradient(theta, h, axis=1, edge_order=2),
        np.gradient(theta, h, axis=2, edge_order=2),
    )
    db_kern = apply_kernel(K, M, theta, gth)
    db_fd = None
    if args.channel in ("fd", "both"):
        db_fd = finite_diff_db_dalpha(nfield, h, theta, alpha=args.alpha)

    channels = []
    if args.channel in ("kernel", "both"):
        channels.append(("kernel", db_kern))
    if args.channel in ("fd", "both"):
        channels.append(("fd", db_fd))

    stokes_block = {}
    rad_block = {}
    off_block = {}
    stokes_verdicts = []

    for cname, db in channels:
        print(f"\n=== channel: {cname} ===")
        print(f"[A] Stokes same boundary C: R={R_s}")
        trip = flux_triple(db, length, R_s, args.cone_height)
        spread = rel_spread(list(trip.values()))
        print(f"  flat disk  = {trip['disk']:.6e}")
        print(f"  hemisphere = {trip['hemisphere']:.6e}")
        print(f"  cone       = {trip['cone']:.6e}")
        print(f"  spread     = {spread:.4e}")
        if spread < args.stokes_tol:
            sv = "STOKES_PASS"
        elif spread < 2.0 * args.stokes_tol:
            sv = "STOKES_WEAK"
        else:
            sv = "STOKES_FAIL"
        print(f"  => {sv}")
        stokes_verdicts.append(sv)
        stokes_block[cname] = {**trip, "spread": spread, "verdict": sv}

        print("[B] Planar xy radius scan of k_ex (different boundaries)")
        rows = []
        for R in radii:
            k = planar_disk_flux(db, length, "xy", R, 0.0)
            rows.append({"radius": R, "k_ex": k})
            print(f"  R={R:.2f}  k_ex={k:.6e}")
        rspread = rel_spread([r["k_ex"] for r in rows])
        if rspread < 0.10:
            rnote = "RADIUS_NEARLY_CONSTANT"
        else:
            rnote = "RADIUS_NOT_CONSTANT (k_ex is surface/boundary dependent)"
        print(f"  spread={rspread:.4e}  => {rnote}")
        rad_block[cname] = {"rows": rows, "spread": rspread, "note": rnote}

        print("[C] Parallel offsets at fixed R")
        orows = []
        for z0 in [float(x) for x in args.offsets.split(",") if x.strip()]:
            if abs(z0) >= 0.9 * length:
                continue
            k = planar_disk_flux(db, length, "xy", R_s, z0)
            orows.append({"offset": z0, "k_ex": k})
            print(f"  z0={z0:+.2f}  k_ex={k:.6e}")
        ospread = rel_spread([r["k_ex"] for r in orows]) if orows else float("nan")
        print(f"  offset spread={ospread:.4e}")
        off_block[cname] = {"rows": orows, "spread": ospread}

    # overall: require Stokes on primary channel (kernel if both)
    primary = "kernel" if "kernel" in stokes_block else list(stokes_block.keys())[0]
    sv = stokes_block[primary]["verdict"]
    rnote = rad_block[primary]["note"]
    if sv == "STOKES_PASS" and "NEARLY_CONSTANT" in rnote:
        verdict = "SUPPORT_PATH_INDEPENDENT"
    elif sv in ("STOKES_PASS", "STOKES_WEAK"):
        verdict = "SUPPORT_STOKES_ONLY"
    else:
        verdict = "FAIL_STOKES"

    print("-" * 72)
    print(f"PRIMARY channel={primary}  Stokes={sv}")
    print(f"OVERALL VERDICT: {verdict}")
    print(
        "  SUPPORT_PATH_INDEPENDENT = Stokes OK and radius nearly constant (rare).\n"
        "  SUPPORT_STOKES_ONLY = same-boundary OK; k_ex still depends on which path/R.\n"
        "  Does NOT prove config-space adiabatic exchange Berry phase."
    )
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "path": path,
        "length": length,
        "N": n,
        "h": h,
        "alpha": args.alpha,
        "theta_mode": args.theta_mode,
        "soft": args.soft,
        "stokes_R": R_s,
        "stokes": stokes_block,
        "radius_scan": rad_block,
        "offset_scan": off_block,
        "primary": primary,
        "verdict": verdict,
    }
    with open(out_dir / "verify4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir / 'verify4_summary.json'}")


if __name__ == "__main__":
    main()
