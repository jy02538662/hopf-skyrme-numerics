#!/usr/bin/env python3
"""
Poisson solver: ∇² Φ = 4π G_eff ρ, Dirichlet Φ=0 on box faces.

Grid: x = linspace(-L, L, N), h = 2L/(N-1).
Uses DST-I on the interior (N-2)^3.
"""

from __future__ import annotations

import numpy as np


def grid_spacing(n: int, length: float) -> float:
    return 2.0 * length / (n - 1)


def chi_from_nfield(nfield: np.ndarray) -> np.ndarray:
    return np.sqrt(nfield[..., 0] ** 2 + nfield[..., 1] ** 2)


def rho_eff(chi: np.ndarray, kappa_rho: float) -> np.ndarray:
    return kappa_rho * (chi * chi)


def calibrate_kappa_rho(nfield: np.ndarray, length: float, energy: float) -> float:
    """κ_ρ = E / ∫χ² d³x with h = 2L/(N-1)."""
    n = nfield.shape[0]
    h = grid_spacing(n, length)
    chi = chi_from_nfield(nfield)
    integ = float(np.sum(chi * chi) * (h**3))
    if integ <= 0:
        raise RuntimeError("∫χ² <= 0")
    return float(energy / integ), integ


def poisson_dirichlet(rhs: np.ndarray, h: float) -> np.ndarray:
    """
    Solve ∇² u = rhs with u=0 on boundary.
    rhs, u shape (N,N,N).
    """
    from scipy.fft import dstn, idstn

    n = rhs.shape[0]
    if n < 4:
        raise ValueError("N too small")
    # interior
    f = rhs[1:-1, 1:-1, 1:-1].astype(np.float64, copy=True)
    m = f.shape[0]
    # DST-I eigenmodes for 2nd difference / h²
    # λ_j = (2/h²) * (cos(π j /(m+1)) - 1), j=1..m
    j = np.arange(1, m + 1, dtype=np.float64)
    lam1d = (2.0 / (h * h)) * (np.cos(np.pi * j / (m + 1)) - 1.0)
    lx, ly, lz = np.meshgrid(lam1d, lam1d, lam1d, indexing="ij")
    lam = lx + ly + lz
    # avoid exact zero (should not occur for Dirichlet)
    lam = np.where(np.abs(lam) < 1e-30, 1e-30, lam)

    # scipy dstn type=1: unnormalized; idstn undoes with proper norm='ortho' or factor
    # Use norm='forward'/'backward' carefully — use ortho for round-trip safety.
    fh = dstn(f, type=1, axes=(0, 1, 2), norm="ortho")
    uh = fh / lam
    ui = idstn(uh, type=1, axes=(0, 1, 2), norm="ortho")

    u = np.zeros_like(rhs, dtype=np.float64)
    u[1:-1, 1:-1, 1:-1] = ui
    return u


def solve_phi(rho: np.ndarray, h: float, g_eff: float) -> np.ndarray:
    """∇² Φ = 4π G_eff ρ  →  rhs = 4π G_eff ρ."""
    rhs = (4.0 * np.pi * g_eff) * rho
    return poisson_dirichlet(rhs, h)


def kappa_eff_from_phi(phi: np.ndarray, kappa0: float, c2: float = 1.0) -> np.ndarray:
    """κ_eff = κ0 (1 + 2 Φ / c²)."""
    return kappa0 * (1.0 + 2.0 * phi / c2)


def shell_mean_phi(phi: np.ndarray, length: float, r: float, n_points: int = 4000) -> float:
    """Fibonacci shell mean of Φ."""
    n = phi.shape[0]
    h = grid_spacing(n, length)
    idx = np.arange(n_points, dtype=np.float64) + 0.5
    polar = np.arccos(1.0 - 2.0 * idx / n_points)
    azim = np.pi * (1.0 + 5.0**0.5) * idx
    x = r * np.sin(polar) * np.cos(azim)
    y = r * np.sin(polar) * np.sin(azim)
    z = r * np.cos(polar)
    pts = np.stack([x, y, z], axis=1)
    grid = (pts + length) / h
    i0 = np.floor(grid).astype(np.int64)
    frac = grid - i0
    inside = np.all((i0 >= 0) & (i0 + 1 <= n - 1), axis=1)
    if not np.any(inside):
        return float("nan")
    ii = i0[inside]
    ff = frac[inside]
    vals = np.zeros(ii.shape[0], dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (ff[:, 0] if dx else 1.0 - ff[:, 0])
                    * (ff[:, 1] if dy else 1.0 - ff[:, 1])
                    * (ff[:, 2] if dz else 1.0 - ff[:, 2])
                )
                vals += w * phi[ii[:, 0] + dx, ii[:, 1] + dy, ii[:, 2] + dz]
    return float(np.mean(vals))


def fit_power_law(r: np.ndarray, y: np.ndarray):
    """y ~ A r^{-alpha}; use |y| and sign separately. Returns A, alpha, r2 on log|y|."""
    y = np.asarray(y, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = np.abs(y) > 1e-30
    if np.count_nonzero(mask) < 3:
        return float("nan"), float("nan"), float("nan")
    lr = np.log(r[mask])
    ly = np.log(np.abs(y[mask]))
    # ly = log|A| - alpha log r
    coef = np.polyfit(lr, ly, 1)
    alpha = -float(coef[0])
    logA = float(coef[1])
    A = float(np.exp(logA) * np.sign(np.median(y[mask])))
    pred = logA - alpha * lr
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return A, alpha, r2


def core_radius_chi(chi: np.ndarray, length: float, frac: float = 0.5) -> float:
    """Radius where shell-mean χ drops to frac * max shell mean (coarse)."""
    n = chi.shape[0]
    h = grid_spacing(n, length)
    ax = np.linspace(-length, length, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    rs = np.linspace(0.5 * h, 0.8 * length, 40)
    means = []
    for r in rs:
        band = (R > r - 0.5 * h) & (R < r + 0.5 * h)
        means.append(float(np.mean(chi[band])) if np.any(band) else 0.0)
    means = np.asarray(means)
    peak = float(np.max(means))
    if peak <= 0:
        return float("nan")
    thr = frac * peak
    # first r where mean < thr after peak
    ip = int(np.argmax(means))
    for i in range(ip, len(rs)):
        if means[i] <= thr:
            return float(rs[i])
    return float(rs[-1])
