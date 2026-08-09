#!/usr/bin/env python3
"""
目标3轻量版：按 Q 对比 κ 正定墙 G_c^(κ)。

固定 n 重加权（与阶段1a / 路径A 同一字典）：
  Φ ∝ G_eff  ⇒  max|2Φ/c²| = S · G_eff
  κ_eff = κ0 (1 + 2Φ/c²)  ⇒  min κ = 0 当 S · G_c = 1 (κ0=c2=1)
  ⇒  G_c^(κ) = 1 / S

断点1推论：Q↑ → 屏蔽↑ → 引力自能弱 → S↓ → G_c↑。

用法（GPU）:
  python compare_Gc_kappa_by_Q.py --abs-outputs --Q-list 1,2 \\
    --out-dir /tmp/bp25_Gc_by_Q

可选密集确认扫:
  --G-list 1e-3,1e-2,1.5e-2,1.8e-2,2e-2,3e-2,5e-2
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

from phase1_single_shot import load_field
from poisson_phi import (
    calibrate_kappa_rho,
    chi_from_nfield,
    grid_spacing,
    kappa_eff_from_phi,
    rho_eff,
    solve_phi,
)


DEFAULT_E = {1: 434.415, 2: 700.0, 3: 1082.0, 4: 1735.0}


def measure_one(args, q: int, g_ref: float) -> dict:
    class _NS:
        pass

    ns = _NS()
    ns.abs_outputs = args.abs_outputs
    ns.Q = q
    ns.field = ""
    ns.synthetic = False
    ns.energy = args.energy if args.energy > 0 else float(DEFAULT_E.get(q, 434.415))

    nfield, length, path, energy = load_field(ns)
    n = nfield.shape[0]
    h = grid_spacing(n, length)
    chi = chi_from_nfield(nfield)
    kappa_rho, integ = calibrate_kappa_rho(nfield, length, energy)
    rho = rho_eff(chi, kappa_rho)

    phi = solve_phi(rho, h, g_ref)
    two = 2.0 * phi / args.c2
    max_abs_two = float(np.max(np.abs(two)))
    # linear scale: max|2Φ/c²| = S * G
    S = max_abs_two / max(g_ref, 1e-30)
    # κ_min = κ0 * (1 - S*G) for attracting Φ (sign: max abs of 2Φ/c2)
    # zero when S * G_c = 1  (κ0=1); general: G_c = 1/S if κ0=1
    G_c = float(args.kappa0) / S

    # confirm at a few G
    confirm = []
    for g in args.g_confirm:
        phi_g = solve_phi(rho, h, g)
        keff = kappa_eff_from_phi(phi_g, args.kappa0, args.c2)
        m2 = float(np.max(np.abs(2.0 * phi_g / args.c2)))
        kmin = float(np.min(keff))
        confirm.append(
            {
                "G_eff": g,
                "max_abs_2phi_c2": m2,
                "keff_min": kmin,
                "light": (
                    "RED"
                    if (kmin <= 0 or m2 >= 1.0)
                    else ("ORANGE" if m2 >= 0.3 else ("YELLOW" if m2 >= 0.1 else "GREEN"))
                ),
            }
        )

    return {
        "Q": q,
        "path": path,
        "N": n,
        "length": length,
        "energy_in": energy,
        "kappa_rho": kappa_rho,
        "integral_chi2": integ,
        "G_ref": g_ref,
        "max_abs_2phi_at_Gref": max_abs_two,
        "S_max_abs_2phi_over_G": S,
        "G_c_kappa": G_c,
        "confirm_rows": confirm,
    }


def verdict(rows: list[dict]) -> dict:
    by_q = {r["Q"]: r["G_c_kappa"] for r in rows}
    out = {"G_c_by_Q": by_q, "breakpoint1_cross_check": None, "note": ""}
    if 1 in by_q and 2 in by_q:
        g1, g2 = by_q[1], by_q[2]
        ratio = g2 / g1 if g1 > 0 else float("inf")
        # "显著高于": use 20% relative as soft threshold for reporting language
        if ratio >= 1.2:
            tag = "SUPPORT_BP1"
            note = (
                f"G_c(Q=2)/G_c(Q=1)={ratio:.3f} >= 1.2: "
                "higher-Q wall is higher — consistent with naive screening⇒weaker-self-energy map."
            )
        elif ratio <= 0.8:
            tag = "OPPOSE_BP1_NAIVE"
            note = (
                f"G_c(Q=2)/G_c(Q=1)={ratio:.3f} <= 0.8: "
                "higher-Q wall is LOWER. Under int(rho)=E, more compact chi raises peak |Phi| "
                "and lowers G_c — does NOT by itself falsify far-field screening (BP1)."
            )
        else:
            tag = "NO_SIGNIFICANT_DIFF"
            note = (
                f"G_c(Q=2)/G_c(Q=1)={ratio:.3f} in (0.8,1.2): "
                "no clear Q-elevation of kappa wall under this dictionary."
            )
        out["ratio_Q2_over_Q1"] = ratio
        out["breakpoint1_cross_check"] = tag
        out["note"] = note
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare G_c^(kappa) across Q")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q-list", type=str, default="1,2", dest="q_list")
    ap.add_argument("--energy", type=float, default=-1.0, help="<0 => per-Q default")
    ap.add_argument("--G-ref", type=float, default=1e-3, dest="g_ref")
    ap.add_argument(
        "--G-list",
        type=str,
        default="1e-3,1e-2,1.5e-2,1.8e-2,2e-2,3e-2,5e-2",
        dest="g_list",
    )
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()
    args.g_confirm = [float(x) for x in args.g_list.split(",") if x.strip()]
    q_list = [int(x) for x in args.q_list.split(",") if x.strip()]

    print("=" * 72)
    print("BP2.5: G_c^(kappa) by Q  (fixed-n reweight; Phi linear in G)")
    print(f"  Q_list={q_list}  G_ref={args.g_ref}")
    print("  NO adiabatic ramp; NO dynamical SC claim.")
    print("=" * 72)

    rows = []
    for q in q_list:
        r = measure_one(args, q, args.g_ref)
        rows.append(r)
        print("-" * 72)
        print(f"Q={q}  field={r['path']}")
        print(f"  N={r['N']} L={r['length']}  kappa_rho={r['kappa_rho']:.6e}")
        print(
            f"  S=max|2Phi|/G = {r['S_max_abs_2phi_over_G']:.6e}  "
            f"=>  G_c^(kappa) = {r['G_c_kappa']:.6e}"
        )
        print(f"  {'G_eff':>10} {'max|2Phi|':>12} {'min_keff':>12} {'light':>8}")
        for c in r["confirm_rows"]:
            print(
                f"  {c['G_eff']:10.3e} {c['max_abs_2phi_c2']:12.4e} "
                f"{c['keff_min']:12.4e} {c['light']:>8}"
            )

    v = verdict(rows)
    print("=" * 72)
    print("CROSS-CHECK vs breakpoint-1 (screening ⇒ weaker self-energy ⇒ higher G_c):")
    print(f"  tag: {v.get('breakpoint1_cross_check')}")
    if "ratio_Q2_over_Q1" in v:
        print(f"  G_c(Q2)/G_c(Q1) = {v['ratio_Q2_over_Q1']:.4f}")
    print(f"  {v.get('note')}")
    print("NOTE: G_c is kappa positivity wall, not dynamical SC fixed-point.")
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": rows,
        "verdict": v,
        "G_c_Q1_reference_from_dyn_scan": 1.82e-2,
        "framing": (
            "Compare G_c^(kappa) only. Do not interpret as dynamical convergence. "
            "Adiabatic G-ramp deliberately out of scope."
        ),
    }
    out_json = out_dir / "compare_Gc_kappa_by_Q.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
