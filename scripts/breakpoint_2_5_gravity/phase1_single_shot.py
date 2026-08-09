#!/usr/bin/env python3
"""
断点 2.5 阶段 1a：单次自能试水（纯后处理，不声称自洽）。

流程:
  n → χ → κ_ρ → ρ_eff → Φ (Poisson) → 远场 1/r 检验
  → κ_eff = κ0 (1 + 2Φ/c²) → 弱场违反分数
  → 用 a(x)=κ_eff 对现有构型做能量重加权（无再弛豫）对比

用法:
  python phase1_single_shot.py --abs-outputs --Q 1 --G-eff 1e-3 --out-dir /tmp/bp25_p1
  python phase1_single_shot.py --synthetic --G-eff 1e-2 --out-dir ./out_p1
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

from poisson_phi import (
    calibrate_kappa_rho,
    chi_from_nfield,
    core_radius_chi,
    fit_power_law,
    grid_spacing,
    kappa_eff_from_phi,
    rho_eff,
    shell_mean_phi,
    solve_phi,
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


def synthetic_bp(n=48, length=6.0, lam=1.5):
    ax = np.linspace(-length, length, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rho = np.sqrt(X * X + Y * Y) + 1e-30
    theta = 2.0 * np.arctan2(lam, rho)
    phi = np.arctan2(Y, X)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return np.stack([nx, ny, nz], axis=-1), length


def discrete_energy_weighted(nfield, h, a_field, b: float):
    """Numpy FS energy with spatially varying a (scalar b)."""
    gx = np.stack([np.gradient(nfield[..., c], h, axis=0, edge_order=2) for c in range(3)], axis=-1)
    gy = np.stack([np.gradient(nfield[..., c], h, axis=1, edge_order=2) for c in range(3)], axis=-1)
    gz = np.stack([np.gradient(nfield[..., c], h, axis=2, edge_order=2) for c in range(3)], axis=-1)
    grad_sq = np.sum(gx * gx + gy * gy + gz * gz, axis=-1)
    Fxy = np.sum(nfield * np.cross(gx, gy), axis=-1)
    Fxz = np.sum(nfield * np.cross(gx, gz), axis=-1)
    Fyz = np.sum(nfield * np.cross(gy, gz), axis=-1)
    f_sq = Fxy * Fxy + Fxz * Fxz + Fyz * Fyz
    vol = h**3
    e2 = 0.5 * float(np.sum(a_field * grad_sq) * vol)
    e4 = 0.25 * b * float(np.sum(f_sq) * vol)
    return e2, e4, e2 + e4


def load_field(args):
    if args.synthetic:
        nfield, length = synthetic_bp()
        energy = args.energy if args.energy > 0 else 50.0  # dummy for synthetic
        return nfield, length, "synthetic_bp", energy
    common = _load_screening_common()
    if common is None:
        raise SystemExit("need screening_progression_3step/common.py or --synthetic")
    table = common.resolve_field_table(args.abs_outputs)
    meta = table[args.Q]
    path = args.field or meta["path"]
    length = float(meta["length"])
    if not Path(path).is_file():
        raise SystemExit(f"missing field: {path}")
    # default energies from notes (approx)
    default_E = {1: 434.415, 2: 700.0, 3: 1082.0, 4: 1735.0}
    energy = args.energy if args.energy > 0 else float(default_E.get(args.Q, 434.415))
    return common.load_nfield(path), length, path, energy


def main():
    ap = argparse.ArgumentParser(description="BP2.5 phase1a single-shot Phi + kappa_eff")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--energy", type=float, default=-1.0, help="FS energy for kappa_rho; <0 => default")
    ap.add_argument("--G-eff", type=float, default=1e-3, dest="g_eff")
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--r-fit", type=str, default="4,5,6,7,8")
    ap.add_argument("--weak-tol", type=float, default=0.1, help="|2Phi/c2| threshold")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    nfield, length, path, energy = load_field(args)
    n = nfield.shape[0]
    h = grid_spacing(n, length)
    chi = chi_from_nfield(nfield)
    kappa_rho, integ = calibrate_kappa_rho(nfield, length, energy)
    rho = rho_eff(chi, kappa_rho)
    phi = solve_phi(rho, h, args.g_eff)
    keff = kappa_eff_from_phi(phi, args.kappa0, args.c2)

    # far-field Phi shells
    r_list = [float(x) for x in args.r_fit.split(",") if x.strip()]
    r_list = [r for r in r_list if r < 0.85 * length]
    phi_shells = [shell_mean_phi(phi, length, r) for r in r_list]
    A_phi, alpha_phi, r2_phi = fit_power_law(np.array(r_list), np.array(phi_shells))

    two_phi = 2.0 * phi / args.c2
    frac_weak_violate = float(np.mean(np.abs(two_phi) > args.weak_tol))
    max_abs_two = float(np.max(np.abs(two_phi)))

    # energies: baseline a=kappa0 vs weighted a=keff
    e2_0, e4_0, et_0 = discrete_energy_weighted(nfield, h, np.full(chi.shape, args.kappa0), args.b)
    e2_w, e4_w, et_w = discrete_energy_weighted(nfield, h, keff, args.b)

    r_core = core_radius_chi(chi, length, 0.5)

    print("=" * 72)
    print("BP2.5 PHASE1a: single-shot self-energy (postprocess only)")
    print(f"  field: {path}")
    print(f"  L={length} N={n} h={h:.4f}  E_in={energy:.4f}")
    print(f"  kappa_rho={kappa_rho:.6e}  int_chi2={integ:.6e}")
    print(f"  G_eff={args.g_eff}  kappa0={args.kappa0}  c2={args.c2}")
    print("=" * 72)
    print(f"Phi: min={phi.min():.4e}  max={phi.max():.4e}  mean={phi.mean():.4e}")
    print(f"Phi shells:")
    for r, v in zip(r_list, phi_shells):
        print(f"  r={r:.2f}  <Phi>={v:.6e}")
    print(f"Phi power fit: A={A_phi:.4e}  alpha={alpha_phi:.4f}  R2={r2_phi:.6f}")
    # Newtonian attracting mass => Phi should be negative if rho>0 and our sign convention
    print(f"kappa_eff: min={keff.min():.4e} max={keff.max():.4e} mean={keff.mean():.4e}")
    print(f"weak-field: max|2Phi/c2|={max_abs_two:.4e}  frac>|tol|={frac_weak_violate:.4e}")
    print(f"core radius (chi half) ~ {r_core:.4f}")
    print(f"E(a=kappa0): E2={e2_0:.4f} E4={e4_0:.4f} Etot={et_0:.4f}")
    print(f"E(a=keff):   E2={e2_w:.4f} E4={e4_w:.4f} Etot={et_w:.4f}  dE/E={(et_w-et_0)/(abs(et_0)+1e-30):.4e}")

    # qualitative flags
    flags = []
    if alpha_phi == alpha_phi and 0.7 < alpha_phi < 1.4 and r2_phi > 0.95:
        flags.append("PHI_FAR_LIKE_1_OVER_R")
    elif alpha_phi == alpha_phi and r2_phi > 0.9:
        flags.append("PHI_POWER_OK_NOT_EXACT_1")
    else:
        flags.append("PHI_FAR_INCONCLUSIVE")
    if max_abs_two < args.weak_tol:
        flags.append("WEAK_FIELD_OK")
    elif frac_weak_violate < 0.05:
        flags.append("WEAK_FIELD_MOSTLY_OK")
    else:
        flags.append("WEAK_FIELD_VIOLATED")
    if et_w > 0 and np.isfinite(et_w) and abs(et_w - et_0) / (abs(et_0) + 1e-30) < 10:
        flags.append("NO_IMMEDIATE_ENERGY_BLOWUP")
    else:
        flags.append("ENERGY_SUSPECT")

    # collapse proxy: if keff goes negative → instability
    if float(np.min(keff)) <= 0:
        flags.append("KAPPA_EFF_NONPOSITIVE_RISK")
    else:
        flags.append("KAPPA_EFF_POSITIVE")

    print("-" * 72)
    print("FLAGS:", ", ".join(flags))
    print(
        "NOTE: reweighted energy on FIXED n is not a relaxed self-consistent soliton.\n"
        "      No claim of iteration convergence. k_ex from BP2 is NOT used."
    )
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"phase1_Q{args.Q}_maps.npz",
        phi=phi.astype(np.float32),
        keff=keff.astype(np.float32),
        rho=rho.astype(np.float32),
        chi=chi.astype(np.float32),
    )
    summary = {
        "path": path,
        "Q": args.Q,
        "length": length,
        "N": n,
        "h": h,
        "energy_input": energy,
        "kappa_rho": kappa_rho,
        "integral_chi2": integ,
        "G_eff": args.g_eff,
        "kappa0": args.kappa0,
        "c2": args.c2,
        "phi_min": float(phi.min()),
        "phi_max": float(phi.max()),
        "phi_shells": [{"r": r, "phi": v} for r, v in zip(r_list, phi_shells)],
        "phi_fit": {"A": A_phi, "alpha": alpha_phi, "R2": r2_phi},
        "keff_min": float(keff.min()),
        "keff_max": float(keff.max()),
        "max_abs_2phi_c2": max_abs_two,
        "frac_weak_violate": frac_weak_violate,
        "r_core_chi_half": r_core,
        "E_baseline": {"E2": e2_0, "E4": e4_0, "Etot": et_0},
        "E_reweighted": {"E2": e2_w, "E4": e4_w, "Etot": et_w},
        "flags": flags,
    }
    with open(out_dir / f"phase1_Q{args.Q}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir / f'phase1_Q{args.Q}_summary.json'}")


if __name__ == "__main__":
    main()
