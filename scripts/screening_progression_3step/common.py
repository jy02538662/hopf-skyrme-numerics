#!/usr/bin/env python3
"""Shared helpers for the 3-step screening-progression verification.

Grid convention (hopf_skyrme_torch / farfield_multipole):
    x = linspace(-length, length, N)   # length = box half-length
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def eval_sph_harm(m, l, polar, azim):
        return _sph_harm_y(l, m, polar, azim)

except ImportError:
    from scipy.special import sph_harm as _sph_harm_legacy

    def eval_sph_harm(m, l, polar, azim):
        return _sph_harm_legacy(m, l, azim, polar)


# Canonical GPU paths (relative to hopf_skyrme_cpu, or under /outputs on server)
# Fit windows: OUTER far field only. Mid-field (r~5-7) for Q>=2 has bumps in
# <chi>(r) that destroy power-law R^2 (see sh_decompose chi_l0 profiles).
DEFAULT_FIELDS = {
    1: {
        "path": "outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy",
        "length": 10.0,
        "r_min": 7.0,
        "r_max": 9.0,
    },
    2: {
        "path": "outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy",
        "length": 12.0,
        "r_min": 8.0,
        "r_max": 10.5,
    },
    3: {
        "path": "outputs/Q3_L12_phase2/nfield_q3_torch.npy",
        "length": 12.0,
        "r_min": 8.0,
        "r_max": 10.5,
    },
    4: {
        "path": "outputs/Q4_L12_phase2/nfield_q4_torch.npy",
        "length": 12.0,
        "r_min": 8.0,
        "r_max": 10.5,
    },
}

# Absolute /outputs mirrors (GPU container layout)
DEFAULT_FIELDS_ABS = {
    1: {
        "path": "/outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy",
        "length": 10.0,
        "r_min": 7.0,
        "r_max": 9.0,
    },
    2: {
        "path": "/outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy",
        "length": 12.0,
        "r_min": 8.0,
        "r_max": 10.5,
    },
    3: {
        "path": "/outputs/Q3_L12_phase2/nfield_q3_torch.npy",
        "length": 12.0,
        "r_min": 8.0,
        "r_max": 10.5,
    },
    4: {
        "path": "/outputs/Q4_L12_phase2/nfield_q4_torch.npy",
        "length": 12.0,
        "r_min": 8.0,
        "r_max": 10.5,
    },
}


def load_nfield(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (N,N,N,3), got {arr.shape} from {path}")
    return arr.astype(np.float64)


def chi_from_nfield(nfield: np.ndarray) -> np.ndarray:
    return np.sqrt(nfield[..., 0] ** 2 + nfield[..., 1] ** 2)


def fibonacci_sphere(num: int) -> np.ndarray:
    idx = np.arange(num, dtype=np.float64) + 0.5
    polar = np.arccos(1.0 - 2.0 * idx / num)
    azim = np.pi * (1.0 + 5.0**0.5) * idx
    x = np.sin(polar) * np.cos(azim)
    y = np.sin(polar) * np.sin(azim)
    z = np.cos(polar)
    return np.stack([x, y, z], axis=1)


def trilinear_sample(vol: np.ndarray, pts_xyz: np.ndarray, length: float) -> np.ndarray:
    n = vol.shape[0]
    h = 2.0 * length / (n - 1)
    grid = (pts_xyz + length) / h
    out = np.zeros(pts_xyz.shape[0], dtype=np.float64)
    i0 = np.floor(grid).astype(np.int64)
    frac = grid - i0
    inside = np.all((i0 >= 0) & (i0 + 1 <= n - 1), axis=1)
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
                vals += w * vol[ii[:, 0] + dx, ii[:, 1] + dy, ii[:, 2] + dz]
    out[inside] = vals
    return out


def shell_mean(vol: np.ndarray, length: float, r: float, n_points: int = 4000) -> float:
    dirs = fibonacci_sphere(n_points)
    return float(np.mean(trilinear_sample(vol, r * dirs, length)))


def shell_samples(vol: np.ndarray, length: float, r: float, n_points: int = 4000):
    """Return (values, polar, azim) on Fibonacci shell."""
    dirs = fibonacci_sphere(n_points)
    vals = trilinear_sample(vol, r * dirs, length)
    polar = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    azim = np.arctan2(dirs[:, 1], dirs[:, 0])
    return vals, polar, azim


def sh_coeffs_power_by_lm(f, polar, azim, l_max: int):
    """Return dict power[(l,m)] and power_per_l[l]."""
    npts = len(f)
    dOmega = 4.0 * np.pi / npts
    power_lm = {}
    power_l = np.zeros(l_max + 1, dtype=np.float64)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Ylm = eval_sph_harm(m, l, polar, azim)
            coeff = np.sum(f * np.conj(Ylm)) * dOmega
            p = float(np.abs(coeff) ** 2)
            power_lm[(l, m)] = p
            power_l[l] += p
    return power_lm, power_l


def odd_even_m_power(power_lm, l: int):
    odd = 0.0
    even = 0.0
    for m in range(-l, l + 1):
        p = power_lm.get((l, m), 0.0)
        if m % 2 == 0:
            even += p
        else:
            odd += p
    return odd, even


def resolve_field_table(use_abs_outputs: bool):
    return DEFAULT_FIELDS_ABS if use_abs_outputs else DEFAULT_FIELDS


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
