#!/usr/bin/env python3
"""
路径 A：G_eff 扫描（阶段 1a 参数边界）。

在同一 Q=1（或指定 Q）构型上扫描 G_eff，记录:
  max|2Φ/c²|, dE/E, min κ_eff, Φ 远场 alpha

交通灯:
  GREEN  : max|2Φ/c²| < 0.1
  YELLOW : 0.1 <= max|2Φ/c²| < 0.3
  ORANGE : 0.3 <= max|2Φ/c²| < 1.0   (弱场黄灯区，用户阈值 0.3)
  RED    : max|2Φ/c²| >= 1.0 或 min κ_eff <= 0

用法:
  python phase1_G_scan.py --abs-outputs --Q 1 --out-dir /tmp/bp25_Gscan
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

from phase1_single_shot import discrete_energy_weighted, load_field
from poisson_phi import (
    calibrate_kappa_rho,
    chi_from_nfield,
    fit_power_law,
    grid_spacing,
    kappa_eff_from_phi,
    rho_eff,
    shell_mean_phi,
    solve_phi,
)


def traffic_light(max_abs_two: float, keff_min: float) -> str:
    if keff_min <= 0 or max_abs_two >= 1.0:
        return "RED"
    if max_abs_two >= 0.3:
        return "ORANGE"  # 黄灯区（弱场失效开始）
    if max_abs_two >= 0.1:
        return "YELLOW"
    return "GREEN"


def main():
    ap = argparse.ArgumentParser(description="BP2.5 path A: G_eff scan")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--energy", type=float, default=-1.0)
    ap.add_argument(
        "--G-list",
        type=str,
        default="1e-4,1e-3,1e-2,5e-2",
        dest="g_list",
    )
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--r-fit", type=str, default="4,5,6,7,8")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    # reuse load_field via a tiny namespace hack
    class _A:
        pass

    ns = _A()
    ns.abs_outputs = args.abs_outputs
    ns.Q = args.Q
    ns.field = args.field
    ns.synthetic = args.synthetic
    ns.energy = args.energy

    nfield, length, path, energy = load_field(ns)
    n = nfield.shape[0]
    h = grid_spacing(n, length)
    chi = chi_from_nfield(nfield)
    kappa_rho, integ = calibrate_kappa_rho(nfield, length, energy)
    rho = rho_eff(chi, kappa_rho)
    r_list = [float(x) for x in args.r_fit.split(",") if x.strip()]
    r_list = [r for r in r_list if r < 0.85 * length]

    e2_0, e4_0, et_0 = discrete_energy_weighted(
        nfield, h, np.full(chi.shape, args.kappa0), args.b
    )

    g_vals = [float(x) for x in args.g_list.split(",") if x.strip()]
    rows = []

    print("=" * 72)
    print(f"BP2.5 PATH A: G_eff scan  Q={args.Q}")
    print(f"  field: {path}")
    print(f"  kappa_rho={kappa_rho:.6e}  E0={et_0:.4f}")
    print("=" * 72)
    print(
        f"{'G_eff':>10} {'max|2Phi|':>12} {'min_keff':>10} {'dE/E':>12} "
        f"{'alpha_Phi':>10} {'R2':>8} {'light':>8}"
    )

    first_orange = None
    first_red = None
    for g in g_vals:
        phi = solve_phi(rho, h, g)
        keff = kappa_eff_from_phi(phi, args.kappa0, args.c2)
        two = 2.0 * phi / args.c2
        max_abs_two = float(np.max(np.abs(two)))
        keff_min = float(np.min(keff))
        e2_w, e4_w, et_w = discrete_energy_weighted(nfield, h, keff, args.b)
        dE_E = (et_w - et_0) / (abs(et_0) + 1e-30)
        shells = [shell_mean_phi(phi, length, r) for r in r_list]
        A, alpha, r2 = fit_power_law(np.array(r_list), np.array(shells))
        light = traffic_light(max_abs_two, keff_min)
        if light == "ORANGE" and first_orange is None:
            first_orange = g
        if light == "RED" and first_red is None:
            first_red = g
        row = {
            "G_eff": g,
            "max_abs_2phi_c2": max_abs_two,
            "keff_min": keff_min,
            "keff_max": float(np.max(keff)),
            "dE_over_E": float(dE_E),
            "phi_min": float(np.min(phi)),
            "alpha_Phi": alpha,
            "R2_Phi": r2,
            "light": light,
        }
        rows.append(row)
        print(
            f"{g:10.1e} {max_abs_two:12.4e} {keff_min:10.4e} {dE_E:12.4e} "
            f"{alpha:10.4f} {r2:8.4f} {light:>8}"
        )

    print("-" * 72)
    print(f"first ORANGE (max|2Phi|>=0.3): {first_orange}")
    print(f"first RED    (max|2Phi|>=1 or keff<=0): {first_red}")
    # linear G_c from first GREEN/any row: S = max|2Phi|/G
    if rows:
        S = rows[0]["max_abs_2phi_c2"] / max(rows[0]["G_eff"], 1e-30)
        G_c_kappa = args.kappa0 / S
        print(f"G_c^(kappa) ~ kappa0 / (max|2Phi|/G) = {G_c_kappa:.6e}  (S={S:.6e})")
    print("NOTE: fixed-n reweight only; not a dynamical instability proof.")
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "path": path,
        "Q": args.Q,
        "kappa_rho": kappa_rho,
        "integral_chi2": integ,
        "E_baseline": et_0,
        "rows": rows,
        "first_orange_G": first_orange,
        "first_red_G": first_red,
        "G_c_kappa_extrap": (args.kappa0 / (rows[0]["max_abs_2phi_c2"] / rows[0]["G_eff"]))
        if rows
        else None,
    }
    with open(out_dir / f"phase1_G_scan_Q{args.Q}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir / f'phase1_G_scan_Q{args.Q}.json'}")


if __name__ == "__main__":
    main()
