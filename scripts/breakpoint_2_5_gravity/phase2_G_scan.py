#!/usr/bin/env python3
"""
目标2/4：G_eff 扫描 + 每点 freeze-kappa 对照。

对每个 G:
  1) 自洽外环 (phase2)
  2) --freeze-kappa 对照（同初始 kappa_eff）
并汇总 dE/E、dQ、min keff、是否与 freeze 分叉。

用法:
  python phase2_G_scan.py --abs-outputs --Q 1 --device cuda \\
    --G-list 3e-3,1e-2,3e-2,5e-2 --outer 40 --n-relax 20 --lr 5e-4 \\
    --out-dir /tmp/bp25_Gscan_dyn
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def run_one(args, g_eff: float, freeze: bool, out_dir: Path) -> Path:
    tag = "freeze" if freeze else "sc"
    sub = out_dir / f"G_{g_eff:g}_{tag}"
    sub.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_HERE / "phase2_self_consistent_loop.py"),
        "--Q",
        str(args.Q),
        "--G-eff",
        str(g_eff),
        "--outer",
        str(args.outer),
        "--n-relax",
        str(args.n_relax),
        "--alpha",
        str(args.alpha),
        "--lr",
        str(args.lr),
        "--etol",
        str(args.etol),
        "--qtol",
        str(args.qtol),
        "--device",
        args.device,
        "--keff-floor",
        str(args.keff_floor),
        "--out-dir",
        str(sub),
    ]
    if args.abs_outputs:
        cmd.append("--abs-outputs")
    if args.field:
        cmd.extend(["--field", args.field])
    if freeze:
        cmd.append("--freeze-kappa")
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)
    # summary path
    summary = sub / f"phase2_Q{args.Q}_{tag}_summary.json"
    return summary


def summarize_history(hist: list) -> dict:
    if not hist:
        return {}
    dE = [abs(r.get("dE_over_E", 0.0)) for r in hist if "dE_over_E" in r]
    dQ = [abs(r.get("dQ", 0.0)) for r in hist if "dQ" in r]
    keff = [r.get("keff_min_raw", r.get("keff_min")) for r in hist]
    keff = [k for k in keff if k is not None]
    phi = [r.get("max_abs_2phi_c2") for r in hist if r.get("max_abs_2phi_c2") is not None]
    rho = [r.get("rho_max") for r in hist if r.get("rho_max") is not None]
    chi = [r.get("chi_max") for r in hist if r.get("chi_max") is not None]
    # dead-knot1: core density rising with outer iters?
    rho_trend = None
    if len(rho) >= 4:
        mid = len(rho) // 2
        rho_trend = float(np.mean(rho[mid:]) - np.mean(rho[:mid]))
    return {
        "n_iter": len(hist),
        "E0": hist[0].get("E"),
        "E_final": hist[-1].get("E"),
        "Q0": hist[0].get("Q_fft"),
        "Q_final": hist[-1].get("Q_fft"),
        "mean_abs_dE_first5": float(np.mean(dE[:5])) if dE else None,
        "mean_abs_dE_last5": float(np.mean(dE[-5:])) if dE else None,
        "mean_abs_dQ_first5": float(np.mean(dQ[:5])) if dQ else None,
        "mean_abs_dQ_last5": float(np.mean(dQ[-5:])) if dQ else None,
        "keff_min_raw_min": float(np.min(keff)) if keff else None,
        "max_abs_2phi_max": float(np.max(phi)) if phi else None,
        "rho_max_first": float(rho[0]) if rho else None,
        "rho_max_last": float(rho[-1]) if rho else None,
        "rho_max_trend_late_minus_early": rho_trend,
        "chi_max_last": float(chi[-1]) if chi else None,
        "Q_drift_total": (
            float(hist[-1]["Q_fft"] - hist[0]["Q_fft"])
            if hist[0].get("Q_fft") is not None and hist[-1].get("Q_fft") is not None
            else None
        ),
    }


def classify_row(sc: dict, fr: dict, status_sc: str) -> str:
    """Rough dynamical class for one G."""
    if status_sc == "ABORT_INIT_KAPPA_NONPOSITIVE":
        return "INIT_KAPPA_WALL"
    if status_sc and status_sc.startswith("ABORT"):
        return "ABORT"
    kmin = sc.get("keff_min_raw_min")
    if kmin is not None and kmin <= 0:
        return "INIT_KAPPA_WALL"
    # dead-knot1: core rho rising under SC
    rt = sc.get("rho_max_trend_late_minus_early")
    r0, r1 = sc.get("rho_max_first"), sc.get("rho_max_last")
    if r0 is not None and r1 is not None and r0 > 0 and (r1 - r0) / r0 > 0.05:
        return "CORE_DENSITY_RISING"
    if rt is not None and r0 is not None and r0 > 0 and rt / r0 > 0.03:
        return "CORE_DENSITY_RISING"
    # bifurcation: compare last-5 mean |dE| or total Q drift
    bif = False
    if sc.get("Q_drift_total") is not None and fr.get("Q_drift_total") is not None:
        if abs(sc["Q_drift_total"] - fr["Q_drift_total"]) > 5e-4:
            bif = True
    if sc.get("mean_abs_dE_last5") is not None and fr.get("mean_abs_dE_last5") is not None:
        denom = max(abs(fr["mean_abs_dE_last5"]), 1e-30)
        if abs(sc["mean_abs_dE_last5"] - fr["mean_abs_dE_last5"]) / denom > 0.2:
            bif = True
    dE_last = sc.get("mean_abs_dE_last5")
    dE_first = sc.get("mean_abs_dE_first5")
    if dE_last is not None and dE_first is not None:
        if dE_last > 1.5 * dE_first + 1e-12:
            return "DIVERGING"
        if dE_last < 0.3 * dE_first and dE_last < 1e-4:
            return "CONVERGING_HINT"
    if bif:
        return "BIFURCATED_FROM_FREEZE"
    return "WEAK_OR_NO_BIFURCATION"


def main():
    ap = argparse.ArgumentParser(description="Phase2 G_eff dynamical scan + freeze controls")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1)
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--G-list", type=str, default="3e-3,1e-2,3e-2,5e-2", dest="g_list")
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--n-relax", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--etol", type=float, default=1e-4)
    ap.add_argument("--qtol", type=float, default=5e-4)
    ap.add_argument("--keff-floor", type=float, default=0.05)
    ap.add_argument("--skip-freeze", action="store_true", help="only run SC arm")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g_vals = [float(x) for x in args.g_list.split(",") if x.strip()]

    rows = []
    print("=" * 72)
    print("BP2.5 PHASE2 G-scan (SC + freeze control)")
    print(f"  G_list={g_vals}  outer={args.outer}  n_relax={args.n_relax}")
    print("=" * 72)

    for g in g_vals:
        sc_path = run_one(args, g, freeze=False, out_dir=out_dir)
        fr_path = None
        if not args.skip_freeze:
            fr_path = run_one(args, g, freeze=True, out_dir=out_dir)

        sc_sum = json.loads(sc_path.read_text(encoding="utf-8")) if sc_path.is_file() else {}
        fr_sum = (
            json.loads(fr_path.read_text(encoding="utf-8"))
            if fr_path and fr_path.is_file()
            else {}
        )
        sc_m = summarize_history(sc_sum.get("history", []))
        fr_m = summarize_history(fr_sum.get("history", []))
        klass = classify_row(sc_m, fr_m, sc_sum.get("status", ""))
        row = {
            "G_eff": g,
            "class": klass,
            "sc_status": sc_sum.get("status"),
            "sc_weak_verdict": sc_sum.get("weak_field_verdict"),
            "sc": sc_m,
            "freeze": fr_m,
            "sc_summary": str(sc_path),
            "freeze_summary": str(fr_path) if fr_path else None,
        }
        rows.append(row)
        print("-" * 72)
        print(
            f"G={g:g}  class={klass}  sc_status={row['sc_status']}  "
            f"dE_last={sc_m.get('mean_abs_dE_last5')}  "
            f"keff_min={sc_m.get('keff_min_raw_min')}  "
            f"dQ_tot_sc={sc_m.get('Q_drift_total')}  dQ_tot_fr={fr_m.get('Q_drift_total')}"
        )

    table = {
        "Q": args.Q,
        "G_list": g_vals,
        "outer": args.outer,
        "n_relax": args.n_relax,
        "lr": args.lr,
        "rows": rows,
        "note": (
            "class is heuristic. BIFURCATED_FROM_FREEZE means SC differs from freeze. "
            "CONVERGING_HINT is not full SC convergence."
        ),
    }
    out_json = out_dir / f"phase2_G_scan_Q{args.Q}.json"
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"saved {out_json}")
    print(
        "Classes: WEAK_OR_NO_BIFURCATION | BIFURCATED_FROM_FREEZE | CONVERGING_HINT | "
        "DIVERGING | CORE_DENSITY_RISING | INIT_KAPPA_WALL | ABORT"
    )


if __name__ == "__main__":
    main()
