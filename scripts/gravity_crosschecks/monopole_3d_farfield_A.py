#!/usr/bin/env python3
"""
3D soft monopole far-field interaction: extract A in U = A/R (+ B/R^3).

Migrated from hopf_linking/vortex_monopole_3d_interaction/.

IMPORTANT (hopf_skyrme_cpu / breakpoint 2.5):
  The G_eff / kappa_t printed below are *legacy toy* quantities from
  A_fit / m_Q^2. They are NOT the BP2.5 dynamical kappa-wall G_c^(κ)
  from Poisson Φ and κ_eff = κ0(1+2Φ/c²). Do not merge into BP2.5 claims.
  See scripts/gravity_crosschecks/README.md and
  scripts/breakpoint_2_5_gravity/PREREQUISITES.md.

Point-source limit:
    U(R) = Q^2 / (4 pi R)     (rho_s = 1, E = 1/2 int |grad theta|^2
                               => U_int = int grad1.grad2)

Original finite-box cross-term (grid_size=50, step=0.8) truncates 1/r^2
    tails: A_fit ~40% low, U*R drifts ~15%. Fix: Green identity

    int grad th1 . grad th2 = int th1 (-lap th2)

with -lap[Q/(4 pi s)] = Q/(4 pi) * 3 xi^2 / s^5, integrate on a core-centered
grid (integrand localized). Then A -> A_theory and alpha -> 1.

Also reports E_core = 3/(128 xi) for this soft monopole (Q=1, rho_s=1),
and derived G_eff, kappa_t from A_fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def trap3(f, x, y, z):
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(trap(trap(f, x), y), z))


def model_1overR_corr(R, A, B):
    return A / R + B / R**3


def model_1overR(R, A):
    return A / R


def U_finite_box_cross(R, Q, xi, grid_size, grid_step):
    """Original method (diagnostic): analytic soft grads on a finite box."""
    x = np.arange(-grid_size, grid_size, grid_step)
    y = np.arange(-grid_size, grid_size, grid_step)
    z = np.arange(-grid_size, grid_size, grid_step)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    def grad(x0, y0, z0):
        dx, dy, dz = X - x0, Y - y0, Z - z0
        s = np.sqrt(dx * dx + dy * dy + dz * dz + xi * xi)
        coeff = -Q / (4.0 * np.pi) / (s**3)
        return coeff * dx, coeff * dy, coeff * dz

    g1x, g1y, g1z = grad(-R / 2.0, 0.0, 0.0)
    g2x, g2y, g2z = grad(R / 2.0, 0.0, 0.0)
    return trap3(g1x * g2x + g1y * g2y + g1z * g2z, x, y, z)


def U_laplacian_identity(R, Q, xi, core_box, grid_step):
    """
    U = int th1 (-lap th2), vortex 2 at origin, vortex 1 at (-R,0,0).
    """
    x = np.arange(-core_box, core_box + 0.5 * grid_step, grid_step)
    y = np.arange(-core_box, core_box + 0.5 * grid_step, grid_step)
    z = np.arange(-core_box, core_box + 0.5 * grid_step, grid_step)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    s2 = np.sqrt(X * X + Y * Y + Z * Z + xi * xi)
    s1 = np.sqrt((X + R) ** 2 + Y * Y + Z * Z + xi * xi)
    integrand = (Q / (4.0 * np.pi)) ** 2 * (1.0 / s1) * (3.0 * xi * xi) / (s2**5)
    return trap3(integrand, x, y, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=float, default=1.0)
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--R-min", type=float, default=10.0)
    ap.add_argument("--R-max", type=float, default=30.0)
    ap.add_argument("--nR", type=int, default=12)
    ap.add_argument("--core-box", type=float, default=12.0)
    ap.add_argument("--core-step", type=float, default=0.08)
    ap.add_argument("--diag-box", type=float, default=50.0)
    ap.add_argument("--diag-step", type=float, default=0.8)
    ap.add_argument("--skip-diag", action="store_true")
    ap.add_argument(
        "--out",
        type=str,
        default="monopole_3d_farfield_A.png",
    )
    args = ap.parse_args()

    Q, xi = args.Q, args.xi
    A_theory = Q**2 / (4.0 * np.pi)
    R_list = np.linspace(args.R_min, args.R_max, args.nR)

    print(f"Q={Q}, xi={xi}")
    print(f"A_theory = Q^2/(4 pi) = {A_theory:.6f}")
    print()

    if not args.skip_diag:
        print("=== DIAGNOSTIC: finite-box cross term (original) ===")
        U_diag = np.array(
            [U_finite_box_cross(float(R), Q, xi, args.diag_box, args.diag_step) for R in R_list]
        )
        p_d, _ = curve_fit(model_1overR_corr, R_list, U_diag, p0=[A_theory, 0.0])
        fluct_d = float(np.std(U_diag * R_list) / np.mean(U_diag * R_list) * 100)
        print(f"A_fit={p_d[0]:.6f}  err={abs(p_d[0]-A_theory)/A_theory*100:.2f}%  U*R fluct={fluct_d:.2f}%")
        print(">> box truncation dominates; do not use this A for G_eff.\n")

    print("=== FIXED: Laplacian identity (core-centered) ===")
    print(f"core_box={args.core_box}, step={args.core_step}, R={args.R_min}..{args.R_max}")
    U_list = np.array(
        [U_laplacian_identity(float(R), Q, xi, args.core_box, args.core_step) for R in R_list]
    )

    popt, _ = curve_fit(model_1overR_corr, R_list, U_list, p0=[A_theory, 0.0])
    A_fit, B_fit = map(float, popt)
    pA, _ = curve_fit(model_1overR, R_list, U_list, p0=[A_theory])
    A_pure = float(pA[0])

    UR = U_list * R_list
    platform_fluct = float(np.std(UR) / np.mean(UR) * 100)
    rel = abs(A_fit - A_theory) / A_theory * 100

    print(f"A_fit (A/R + B/R^3) = {A_fit:.6f}")
    print(f"A_fit (pure A/R)    = {A_pure:.6f}")
    print(f"A_theory            = {A_theory:.6f}")
    print(f"rel_err(A_corr)     = {rel:.3f}%")
    print(f"B_fit               = {B_fit:.6f}")
    print(f"U*R platform fluct  = {platform_fluct:.3f}%")
    print(f"U*R                 = {UR}")

    # Core energy for soft monopole Q=1, E=1/2 int |grad|^2
    E_core = 3.0 / (128.0 * xi)
    m_Q = E_core
    G_eff = abs(A_fit) / (m_Q**2)
    kappa_t = 4.0 * np.pi * G_eff * m_Q

    print("\n=== Core mass and effective coupling (from A_fit) ===")
    print(f"E_core = 3/(128 xi) = {E_core:.6f}")
    print(f"m_Q = {m_Q:.6f}")
    print(f"G_eff = |A|/m_Q^2 = {G_eff:.6f}")
    print(f"kappa_t = 4 pi G_eff m_Q = {kappa_t:.6f}")

    ok = rel < 5.0 and platform_fluct < 5.0
    print()
    print("PASS" if ok else "FAIL", f"(rel_err={rel:.2f}%, UR_fluct={platform_fluct:.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.loglog(R_list, U_list, "o", label="U numerical")
    ax.loglog(R_list, model_1overR_corr(R_list, A_fit, B_fit), "--", label=f"A/R+B/R3 A={A_fit:.4f}")
    ax.loglog(R_list, model_1overR(R_list, A_theory), ":", label=f"theory A/R A={A_theory:.4f}")
    ax.set_xlabel("R")
    ax.set_ylabel("U")
    ax.set_title("3D monopole far-field interaction")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax2 = axes[1]
    ax2.plot(R_list, UR, "o-", label="U*R")
    ax2.axhline(A_theory, color="k", ls="--", label="A_theory")
    ax2.axhline(A_fit, color="C1", ls=":", label="A_fit")
    ax2.set_xlabel("R")
    ax2.set_ylabel("U*R")
    ax2.set_title(f"plateau  rel_err={rel:.2f}%")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved figure: {out.resolve()}")


if __name__ == "__main__":
    main()
