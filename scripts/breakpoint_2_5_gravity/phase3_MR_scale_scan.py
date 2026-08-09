#!/usr/bin/env python3
"""
断点 2.5 阶段3：M(R) 标度扫描（零引力短弛豫 + κ 墙 G_c）。

协议见 reports/MR_scaling_PROTOCOL.md

流程（每个 ansatz scale s）:
  1) make_initial_hopf_pq / q1
  2) short Riemannian relax (a=b=1)
  3) measure E2, E4, E, Q_fft, R_chi
  4) Poisson → S = max|2Φ|/G_ref → G_c = 1/S
  5) fit E ≈ a R + b/R  and  G_c ≈ k (R/E)

用法:
  # Whitehead scale 族（旧协议）
  python phase3_MR_scale_scan.py --device cuda --Q 1 --steps 0 ...

  # 真 Derrick 空间拉伸 → 用专用入口:
  python phase3_MR_dilate.py --abs-outputs --Q 1 --device cuda \\
    --lambda-list 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0 \\
    --out-dir /tmp/bp25_MR_dilate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _add_torch_src():
    candidates = [
        _HERE / "hopf_skyrme_torch.py",
        _HERE / "src" / "hopf_skyrme_torch.py",
        _HERE.parent / "src" / "hopf_skyrme_torch.py",
        _HERE.parent.parent / "src" / "hopf_skyrme_torch.py",
        Path("/src/hopf_skyrme_torch.py"),
        Path("/hopf_skyrme_torch.py"),
        Path("/workspace/src/hopf_skyrme_torch.py"),
        Path("/hopf_skyrme_cpu/src/hopf_skyrme_torch.py"),
    ]
    cur = _HERE
    for _ in range(6):
        candidates.append(cur / "src" / "hopf_skyrme_torch.py")
        if cur.parent == cur:
            break
        cur = cur.parent
    for p in candidates:
        if p.is_file():
            sys.path.insert(0, str(p.parent))
            return p
    raise SystemExit("Cannot find hopf_skyrme_torch.py")


_TORCH_PATH = _add_torch_src()

from poisson_phi import (  # noqa: E402
    calibrate_kappa_rho,
    chi_from_nfield,
    core_radius_chi,
    grid_spacing,
    rho_eff,
    solve_phi,
)
import hopf_skyrme_torch as hst  # noqa: E402


def radius_metrics(n_np: np.ndarray, length: float) -> dict:
    """Robust radii. Prefer R_rms / R_halfmass; R_chi often saturates at 0.8L."""
    n = n_np.shape[0]
    h = grid_spacing(n, length)
    ax = np.linspace(-length, length, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = X * X + Y * Y + Z * Z
    chi = chi_from_nfield(n_np)
    w = chi * chi
    vol = h**3
    wsum = float(np.sum(w) * vol)
    if wsum <= 1e-30:
        return {
            "R_chi": float("nan"),
            "R_chi_saturated": True,
            "R_rms_chi2": float("nan"),
            "R_halfmass_chi2": float("nan"),
        }
    R_rms = float(np.sqrt(np.sum(w * r2) * vol / wsum))
    # half-mass radius of χ²
    flat_r = np.sqrt(r2).ravel()
    flat_w = (w * vol).ravel()
    order = np.argsort(flat_r)
    csum = np.cumsum(flat_w[order])
    half = 0.5 * csum[-1]
    ih = int(np.searchsorted(csum, half))
    R_half = float(flat_r[order[min(ih, len(order) - 1)]])
    R_chi = core_radius_chi(chi, length, 0.5)
    r_ceil = 0.8 * length
    saturated = bool(np.isfinite(R_chi) and abs(R_chi - r_ceil) < 1e-6)
    return {
        "R_chi": R_chi,
        "R_chi_saturated": saturated,
        "R_rms_chi2": R_rms,
        "R_halfmass_chi2": R_half,
    }


def relax_steps(u, h, a, b, n_steps, lr):
    for _ in range(n_steps):
        if u.grad is not None:
            u.grad = None
        nfield = hst.normalize(u)
        _e2, _e4, etot_t = hst.energy_parts_tensor(nfield, h, a, b)
        etot_t.backward()
        with torch.no_grad():
            grad = u.grad
            grad_t = grad - torch.sum(grad * nfield, dim=-1, keepdim=True) * nfield
            u.copy_(hst.normalize(u - lr * grad_t))
            hst.apply_boundary_(u)
    with torch.no_grad():
        nfield = hst.normalize(u)
        e2, e4, etot = hst.energy_parts_float(nfield, h, a, b)
        q = hst.hopf_charge_cpu_fft(nfield, h)
    return float(e2), float(e4), float(etot), float(q)


def measure_gc(n_np: np.ndarray, length: float, energy: float, g_ref: float, kappa0: float, c2: float):
    kappa_rho, integ = calibrate_kappa_rho(n_np, length, energy)
    chi = chi_from_nfield(n_np)
    rho = rho_eff(chi, kappa_rho)
    h = grid_spacing(n_np.shape[0], length)
    phi = solve_phi(rho, h, g_ref)
    max_abs_two = float(np.max(np.abs(2.0 * phi / c2)))
    S = max_abs_two / max(g_ref, 1e-30)
    G_c = float(kappa0) / S
    rad = radius_metrics(n_np, length)
    # primary R for scaling fits: RMS (robust); keep R_chi for diagnostics
    R_primary = rad["R_rms_chi2"]
    return {
        "kappa_rho": kappa_rho,
        "integral_chi2": integ,
        "max_abs_2phi_at_Gref": max_abs_two,
        "S_max_abs_2phi_over_G": S,
        "G_c_kappa": G_c,
        "R_primary": R_primary,
        "R_primary_def": "R_rms_chi2",
        "phi_min": float(np.min(phi)),
        **rad,
    }


def fit_ER(rows: list[dict], r_key: str = "R_primary") -> dict:
    """E ≈ a R + b/R  via  E*R = a R^2 + b."""
    pts = [
        (r.get(r_key), r.get("E"))
        for r in rows
        if r.get(r_key) is not None and r.get("E") is not None
    ]
    R = np.array([p[0] for p in pts], dtype=float)
    E = np.array([p[1] for p in pts], dtype=float)
    if len(R) < 3:
        return {"ok": False, "reason": f"need >=3 finite {r_key} points", "r_key": r_key}
    mask = (R > 1e-6) & np.isfinite(E) & np.isfinite(R)
    R, E = R[mask], E[mask]
    if len(R) < 3:
        return {"ok": False, "reason": "need >=3 valid points after mask", "r_key": r_key}
    # dynamic range guard (R_chi ceiling → no U-curve)
    if float(np.max(R) / (np.min(R) + 1e-30)) < 1.05:
        return {
            "ok": False,
            "reason": f"{r_key} nearly constant (max/min<1.05); cannot fit U-curve",
            "r_key": r_key,
            "R_min": float(np.min(R)),
            "R_max": float(np.max(R)),
        }
    y = E * R
    X = np.column_stack([R * R, np.ones_like(R)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    R0 = float(np.sqrt(b / a)) if a > 0 and b > 0 else float("nan")
    E0 = float(2.0 * np.sqrt(a * b)) if a > 0 and b > 0 else float("nan")
    return {
        "ok": True,
        "r_key": r_key,
        "a": a,
        "b": b,
        "R2_fit": r2,
        "R0_sqrt_b_over_a": R0,
        "E0_2sqrt_ab": E0,
        "n_points": int(len(R)),
        "U_shape_coefficients_positive": bool(a > 0 and b > 0),
        "model": f"E = a*{r_key} + b/{r_key}  (fit via E*R = a*R^2 + b)",
    }


def fit_Gc_vs_R_over_E(rows: list[dict], r_key: str = "R_primary") -> dict:
    """G_c ≈ k * (R/E); also report Pearson corr."""
    R, E, Gc = [], [], []
    for r in rows:
        rr, ee, gg = r.get(r_key), r.get("E"), r.get("G_c_kappa")
        if rr is None or ee is None or gg is None:
            continue
        if not (np.isfinite(rr) and np.isfinite(ee) and np.isfinite(gg) and ee > 0 and rr > 0):
            continue
        R.append(rr)
        E.append(ee)
        Gc.append(gg)
    R, E, Gc = np.asarray(R), np.asarray(E), np.asarray(Gc)
    if len(R) < 3:
        return {"ok": False, "reason": "need >=3 points", "r_key": r_key}
    x = R / E
    k = float(np.sum(x * Gc) / (np.sum(x * x) + 1e-30))
    Gc_hat = k * x
    ss_res = float(np.sum((Gc - Gc_hat) ** 2))
    ss_tot = float(np.sum((Gc - np.mean(Gc)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    X2 = np.column_stack([x, np.ones_like(x)])
    coef2, *_ = np.linalg.lstsq(X2, Gc, rcond=None)
    k1, c0 = float(coef2[0]), float(coef2[1])
    pred2 = X2 @ coef2
    r2_free = 1.0 - float(np.sum((Gc - pred2) ** 2)) / (
        float(np.sum((Gc - np.mean(Gc)) ** 2)) + 1e-30
    )
    pearson = float(np.corrcoef(x, Gc)[0, 1]) if len(x) >= 2 else float("nan")
    k_newt = 0.5
    return {
        "ok": True,
        "r_key": r_key,
        "k_through_origin": k,
        "R2_through_origin": r2,
        "k_free": k1,
        "intercept": c0,
        "R2_free": r2_free,
        "pearson_R_over_E_vs_Gc": pearson,
        "newton_heuristic_k": k_newt,
        "k_over_newton": k / k_newt if abs(k_newt) > 1e-30 else float("nan"),
        "n_points": int(len(R)),
        "G_c_vs_R_over_E_positive_trend": bool(pearson > 0.3),
        "model": f"G_c ≈ k ({r_key}/E); newton heuristic k~1/2",
        "note": "Do NOT require k==1/2; Dirichlet box + chi^2 density shift prefactor.",
    }


def make_ansatz(q: int, n: int, length: float, scale: float, device, dtype, p: int, qq: int):
    if q == 1 and p == 1 and qq == 1:
        return hst.make_initial_q1(n, length, scale, device, dtype)
    return hst.make_initial_hopf_pq(n, length, scale, p, qq, device, dtype)


def main():
    ap = argparse.ArgumentParser(description="BP2.5 phase3 M(R) scale scan")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--p", type=int, default=-1, help="Hopf p; <0 => Q for p, q=1 (Q=1) or p=q=sqrt for even")
    ap.add_argument("--q-ansatz", type=int, default=-1, dest="q_ansatz", help="Hopf q; <0 => auto")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--length", type=float, default=10.0)
    ap.add_argument(
        "--scale-list",
        type=str,
        default="0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4",
        dest="scale_list",
    )
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--G-ref", type=float, default=1e-3, dest="g_ref")
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    # ansatz (p,q)
    if args.p > 0 and args.q_ansatz > 0:
        p, qq = args.p, args.q_ansatz
    elif args.Q == 1:
        p, qq = 1, 1
    elif args.Q == 2:
        p, qq = 2, 1  # common axial Q=2; override with --p/--q-ansatz if needed
    else:
        p, qq = int(args.Q), 1

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dtype = torch.float32
    scales = [float(x) for x in args.scale_list.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BP2.5 PHASE3: M(R) scale scan")
    print(f"  torch: {_TORCH_PATH}")
    print(f"  Q={args.Q}  ansatz p={p} q={qq}  N={args.n} L={args.length}")
    print(f"  scales={scales}")
    print(f"  steps={args.steps} lr={args.lr} device={device}")
    print(f"  G_ref={args.g_ref}  (G_c from linear S; no adiabatic ramp)")
    print("  R_primary = R_rms_chi2  (R_chi often saturates at 0.8L — diagnostic only)")
    if args.steps == 0:
        print("  mode=ANSATZ_ONLY (steps=0): clean Derrick probe on ansatz family")
    print("=" * 72)

    rows = []
    t0 = time.time()
    n_sat = 0
    for s in scales:
        print("-" * 72)
        print(f"scale={s:.3g}  building ansatz + relax {args.steps} steps ...", flush=True)
        u0, h = make_ansatz(args.Q, args.n, args.length, s, device, dtype, p, qq)
        u = torch.nn.Parameter(u0.detach().clone())
        e2, e4, etot, q = relax_steps(u, h, args.a, args.b, args.steps, args.lr)
        n_np = hst.normalize(u).detach().cpu().numpy()
        grav = measure_gc(n_np, args.length, etot, args.g_ref, args.kappa0, args.c2)
        if grav.get("R_chi_saturated"):
            n_sat += 1
        Rp = grav["R_primary"]
        R_over_E = float(Rp / etot) if etot > 0 and np.isfinite(Rp) else float("nan")
        row = {
            "scale": s,
            "scale_as_R": s,
            "E2": e2,
            "E4": e4,
            "E": etot,
            "Q_fft": q,
            "R_over_E": R_over_E,
            "h": h,
            **grav,
        }
        rows.append(row)
        sat = " SAT" if grav.get("R_chi_saturated") else ""
        print(
            f"  E={etot:.4f} (E2={e2:.4f} E4={e4:.4f})  Q={q:.4f}  "
            f"R_rms={Rp:.4f}  R_half={grav['R_halfmass_chi2']:.4f}  "
            f"R_chi={grav['R_chi']:.4f}{sat}  G_c={grav['G_c_kappa']:.4e}  "
            f"Rrms/E={R_over_E:.4e}",
            flush=True,
        )
        if args.save_fields:
            np.save(
                out_dir / f"nfield_Q{args.Q}_s{s:g}.npy",
                n_np.astype(np.float32),
            )

    fit_e = fit_ER(rows, r_key="R_primary")
    fit_e_scale = fit_ER(rows, r_key="scale_as_R")
    fit_g = fit_Gc_vs_R_over_E(rows, r_key="R_primary")
    fit_g_scale = fit_Gc_vs_R_over_E(rows, r_key="scale_as_R")

    print("=" * 72)
    if n_sat:
        print(f"WARNING: R_chi saturated at 0.8L for {n_sat}/{len(rows)} points — use R_rms")
    print("FIT E = a R_rms + b/R_rms")
    if fit_e.get("ok"):
        print(
            f"  a={fit_e['a']:.6e}  b={fit_e['b']:.6e}  R2={fit_e['R2_fit']:.4f}  "
            f"R0={fit_e['R0_sqrt_b_over_a']:.4f}  E0={fit_e['E0_2sqrt_ab']:.4f}"
        )
        print(f"  U_shape_coefficients_positive: {fit_e.get('U_shape_coefficients_positive')}")
    else:
        print(f"  FAILED: {fit_e.get('reason')}")

    print("FIT E = a*scale + b/scale  (ansatz knob; Derrick family)")
    if fit_e_scale.get("ok"):
        print(
            f"  a={fit_e_scale['a']:.6e}  b={fit_e_scale['b']:.6e}  "
            f"R2={fit_e_scale['R2_fit']:.4f}  "
            f"s0={fit_e_scale['R0_sqrt_b_over_a']:.4f}  "
            f"U_pos={fit_e_scale.get('U_shape_coefficients_positive')}"
        )
    else:
        print(f"  FAILED: {fit_e_scale.get('reason')}")

    print("FIT G_c ≈ k (R_rms/E)")
    if fit_g.get("ok"):
        print(
            f"  k(origin)={fit_g['k_through_origin']:.6e}  R2={fit_g['R2_through_origin']:.4f}  "
            f"pearson={fit_g['pearson_R_over_E_vs_Gc']:.4f}"
        )
        print(
            f"  k(free)={fit_g['k_free']:.6e}  c={fit_g['intercept']:.6e}  "
            f"R2_free={fit_g['R2_free']:.4f}"
        )
        print(
            f"  newton_heuristic_k=0.5  k/newton={fit_g['k_over_newton']:.3f}  "
            f"(direction-only; prefactor free)"
        )
        print(f"  G_c_vs_R_over_E_positive_trend: {fit_g.get('G_c_vs_R_over_E_positive_trend')}")
    else:
        print(f"  FAILED: {fit_g.get('reason')}")

    print("FIT G_c ≈ k (scale/E)")
    if fit_g_scale.get("ok"):
        print(
            f"  pearson={fit_g_scale['pearson_R_over_E_vs_Gc']:.4f}  "
            f"k(free)={fit_g_scale['k_free']:.6e}  R2_free={fit_g_scale['R2_free']:.4f}  "
            f"pos={fit_g_scale.get('G_c_vs_R_over_E_positive_trend')}"
        )
    else:
        print(f"  FAILED: {fit_g_scale.get('reason')}")

    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.1f}s")
    print("NOTE: short-relax probe; not long-run vacuum. No particle/hierarchy claim.")
    print("=" * 72)

    summary = {
        "phase": "phase3_MR",
        "Q": args.Q,
        "ansatz_p": p,
        "ansatz_q": qq,
        "n": args.n,
        "length": args.length,
        "steps": args.steps,
        "lr": args.lr,
        "a": args.a,
        "b": args.b,
        "device": str(device),
        "G_ref": args.g_ref,
        "scales": scales,
        "rows": rows,
        "n_R_chi_saturated": n_sat,
        "fit_E_of_R_rms": fit_e,
        "fit_E_of_scale": fit_e_scale,
        "fit_Gc_vs_Rrms_over_E": fit_g,
        "fit_Gc_vs_scale_over_E": fit_g_scale,
        "elapsed_s": elapsed,
        "torch_module": str(_TORCH_PATH),
        "framing": (
            "G_c is kappa positivity wall from fixed-n reweight. "
            "R_primary=R_rms_chi2. No adiabatic ramp; no particle/hierarchy claim."
        ),
    }
    out_json = out_dir / f"phase3_MR_Q{args.Q}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
