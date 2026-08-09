#!/usr/bin/env python3
"""
验证3：真空 Berry 通量稳定性（诚实拆分两个命题）。

A) Stokes / 同边界曲面无关（应成立，若 ∇·b≈0）:
   同一圆周边界 C: 平底盘 vs 半球 vs 圆锥，比较 flux(b).

B) 平面圆盘半径扫描（诊断，Hopfion 上通常不恒定）:
   不同 R 的 xy 盘 flux(b) —— 不作为“拓扑常数=4πQ”的通过条件。

C) 平行偏移盘（z=offset）诊断。

验收（默认）:
  Stokes: 相对偏差 < stokes_tol (default 0.10) → STOKES_PASS
  半径扫描: 仅报告变异系数；若误当作拓扑常数会标 RADIUS_NOT_CONSTANT

用法:
  python verify3_vacuum_flux_stability.py --abs-outputs --Q 1 --out-dir /tmp/k_ex_v3
  python verify3_vacuum_flux_stability.py --synthetic --out-dir ./out_v3
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
    berry_curvature,
    cone_flux_xy,
    divergence_b,
    grid_spacing,
    hemisphere_flux_xy,
    planar_disk_flux,
    synthetic_bp_skyrmion,
)


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


def main():
    ap = argparse.ArgumentParser(description="V3: vacuum Berry flux stability")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument(
        "--radii",
        type=str,
        default="2,3,4,5",
        help="disk radii for scans",
    )
    ap.add_argument(
        "--stokes-R",
        type=float,
        default=3.0,
        help="boundary radius for Stokes same-boundary test",
    )
    ap.add_argument("--cone-height", type=float, default=3.0)
    ap.add_argument("--offsets", type=str, default="0,-1,1", help="xy-plane z offsets")
    ap.add_argument("--stokes-tol", type=float, default=0.10)
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    nfield, length, path = load_field(args)
    n = nfield.shape[0]
    h = grid_spacing(n, length)
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    radii = [r for r in radii if r < 0.9 * length]
    R_s = args.stokes_R
    if R_s >= 0.9 * length:
        R_s = 0.5 * length

    b = berry_curvature(nfield, h)
    div = divergence_b(b, h)
    # interior RMS div (drop 2 cells)
    m = 2
    div_i = div[m : n - m, m : n - m, m : n - m]
    rms_div = float(np.sqrt(np.mean(div_i**2)))
    rms_b = float(np.sqrt(np.mean(b[m : n - m, m : n - m, m : n - m] ** 2)))
    div_rel = rms_div / (rms_b / max(h, 1e-30) + 1e-30)  # rough dimensionless

    print("=" * 72)
    print("VERIFY3: vacuum Berry flux stability")
    print(f"  field: {path}")
    print(f"  L={length}  N={n}  h={h:.4f}")
    print(f"  rms|div b|={rms_div:.4e}  rms|b|={rms_b:.4e}  (div not exactly 0 on grid)")
    print("=" * 72)

    # --- A) Stokes same-boundary ---
    print(f"\n[A] Stokes same boundary C: circle R={R_s} in z=0")
    f_disk = planar_disk_flux(b, length, "xy", R_s, 0.0)
    f_hemi = hemisphere_flux_xy(b, length, R_s, sign_z=1.0)
    f_cone = cone_flux_xy(b, length, R_s, height=args.cone_height)
    stokes_vals = [f_disk, f_hemi, f_cone]
    stokes_spread = rel_spread(stokes_vals)
    print(f"  flat disk     = {f_disk:.6e}")
    print(f"  hemisphere    = {f_hemi:.6e}")
    print(f"  cone H={args.cone_height} = {f_cone:.6e}")
    print(f"  relative spread (max-min)/max| | = {stokes_spread:.4e}")
    if stokes_spread < args.stokes_tol:
        stokes_verdict = "STOKES_PASS"
    elif stokes_spread < 2.0 * args.stokes_tol:
        stokes_verdict = "STOKES_WEAK"
    else:
        stokes_verdict = "STOKES_FAIL"
    print(f"  => {stokes_verdict}  (tol={args.stokes_tol})")

    # --- B) radius scan (diagnostic) ---
    print("\n[B] Planar xy disk radius scan (NOT expected constant for Hopfion)")
    rad_rows = []
    for R in radii:
        f = planar_disk_flux(b, length, "xy", R, 0.0)
        rad_rows.append({"radius": R, "flux": f})
        print(f"  R={R:.2f}  flux_b0={f:.6e}")
    fluxes = [r["flux"] for r in rad_rows]
    rad_spread = rel_spread(fluxes) if fluxes else float("nan")
    print(f"  relative spread across R = {rad_spread:.4e}")
    if rad_spread < 0.05:
        rad_note = "RADIUS_NEARLY_CONSTANT"
    else:
        rad_note = "RADIUS_NOT_CONSTANT (expected for 3D Hopfion planar disks)"
    print(f"  => {rad_note}")

    # orientation at fixed R_s
    print(f"\n[B2] Orientation at R={R_s}")
    for plane in ("xy", "xz", "yz"):
        f = planar_disk_flux(b, length, plane, R_s, 0.0)
        print(f"  plane {plane}: flux={f:.6e}")

    # --- C) offset scan ---
    print("\n[C] Parallel xy disks (same R, different z offset)")
    offsets = [float(x) for x in args.offsets.split(",") if x.strip()]
    off_rows = []
    for z0 in offsets:
        if abs(z0) >= 0.9 * length:
            continue
        f = planar_disk_flux(b, length, "xy", R_s, z0)
        off_rows.append({"offset": z0, "flux": f})
        print(f"  z0={z0:+.2f}  flux={f:.6e}")
    off_spread = rel_spread([r["flux"] for r in off_rows]) if off_rows else float("nan")
    print(f"  relative spread across offsets = {off_spread:.4e}")

    # overall: Stokes is the pass/fail for V3; radius is diagnostic
    if stokes_verdict == "STOKES_PASS":
        verdict = "SUPPORT_STOKES"
    elif stokes_verdict == "STOKES_WEAK":
        verdict = "WEAK_SUPPORT_STOKES"
    else:
        verdict = "FAIL_STOKES"

    print("-" * 72)
    print(f"OVERALL VERDICT: {verdict}")
    print(
        "  Stokes PASS = same-boundary surfaces agree (supports using flux of b).\n"
        "  Radius non-constant does NOT kill Stokes; it only kills naive 4*pi*Q planar claim.\n"
        "  Still does NOT prove config-space exchange Berry = this flux."
    )
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "path": path,
        "length": length,
        "N": n,
        "h": h,
        "rms_div_b": rms_div,
        "rms_b": rms_b,
        "stokes": {
            "R": R_s,
            "disk": f_disk,
            "hemisphere": f_hemi,
            "cone": f_cone,
            "spread": stokes_spread,
            "verdict": stokes_verdict,
            "tol": args.stokes_tol,
        },
        "radius_scan": rad_rows,
        "radius_spread": rad_spread,
        "radius_note": rad_note,
        "offset_scan": off_rows,
        "offset_spread": off_spread,
        "verdict": verdict,
    }
    with open(out_dir / "verify3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir / 'verify3_summary.json'}")


if __name__ == "__main__":
    main()
