#!/usr/bin/env python3
"""
Berry curvature b[n] and linear-response kernels under polar tilt δn = α θ̃ θ̂.

Convention:
  b_i = (1/2) ε_ijk n·(∂_j n × ∂_k n)
      = n·(∂_u n × ∂_v n)  for the positive cyclic pair (u,v) of i
        (e.g. b_x = n·(∂_y n × ∂_z n)).

Corrected response (draft M_ij was wrong):
  δb_i/α = K_i θ̃ + M_{im} ∂_m θ̃

  K_i = (1/2) ε_ijk [ θ̂·(∂_j n × ∂_k n)
                     + n·(∂_j θ̂ × ∂_k n)
                     + n·(∂_j n × ∂_k θ̂) ]

  M_{im} = ε_{imk} n · (θ̂ × ∂_k n)
"""

from __future__ import annotations

import numpy as np


def grid_axes(n: int, length: float) -> np.ndarray:
    return np.linspace(-length, length, n)


def grid_spacing(n: int, length: float) -> float:
    return 2.0 * length / (n - 1)


def gradients_n(nfield: np.ndarray, h: float):
    gx = np.gradient(nfield, h, axis=0, edge_order=2)
    gy = np.gradient(nfield, h, axis=1, edge_order=2)
    gz = np.gradient(nfield, h, axis=2, edge_order=2)
    return gx, gy, gz


def berry_curvature(nfield: np.ndarray, h: float) -> np.ndarray:
    """Shape (N,N,N,3)."""
    n = nfield
    dx, dy, dz = gradients_n(n, h)
    # (1/2)ε sum ≡ cyclic single cross
    bx = np.sum(n * np.cross(dy, dz), axis=-1)
    by = np.sum(n * np.cross(dz, dx), axis=-1)
    bz = np.sum(n * np.cross(dx, dy), axis=-1)
    return np.stack([bx, by, bz], axis=-1)


def theta_hat_from_n(nfield: np.ndarray) -> np.ndarray:
    nx, ny, nz = nfield[..., 0], nfield[..., 1], nfield[..., 2]
    theta = np.arccos(np.clip(nz, -1.0, 1.0))
    phi = np.arctan2(ny, nx)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    return np.stack([ct * cp, ct * sp, -st], axis=-1)


def _half_eps_n_cross(n, left_grads, right_grads):
    """(1/2) ε_ijk n·(L_j × R_k) as vector — equals n·(L_u × R_v) cyclic."""
    fx = np.sum(n * np.cross(left_grads[1], right_grads[2]), axis=-1)
    fy = np.sum(n * np.cross(left_grads[2], right_grads[0]), axis=-1)
    fz = np.sum(n * np.cross(left_grads[0], right_grads[1]), axis=-1)
    return np.stack([fx, fy, fz], axis=-1)


def response_kernels(nfield: np.ndarray, h: float):
    """K (N,N,N,3), M (N,N,N,3,3) with M[...,i,m]=M_im, and θ̂."""
    n = nfield
    th = theta_hat_from_n(n)
    dn = gradients_n(n, h)
    dth = gradients_n(th, h)

    # piece1: (1/2)ε θ̂·(∂j n × ∂k n)
    k1 = _half_eps_n_cross(th, dn, dn)
    # pieces 2+3
    k2 = _half_eps_n_cross(n, dth, dn)
    k3 = _half_eps_n_cross(n, dn, dth)
    K = k1 + k2 + k3

    th_cross_dn = [np.cross(th, dn[k]) for k in range(3)]
    ndot = [np.sum(n * th_cross_dn[k], axis=-1) for k in range(3)]

    eps = np.zeros((3, 3, 3), dtype=np.float64)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

    M = np.zeros(n.shape[:3] + (3, 3), dtype=np.float64)
    for i in range(3):
        for m in range(3):
            for k in range(3):
                e = eps[i, m, k]
                if e == 0.0:
                    continue
                M[..., i, m] += e * ndot[k]

    return K, M, th


def apply_kernel(K, M, theta_tilde, grad_theta_tilde):
    out = K * theta_tilde[..., None]
    for m in range(3):
        out = out + M[..., :, m] * grad_theta_tilde[m][..., None]
    return out


def finite_diff_db_dalpha(nfield, h, theta_tilde, alpha=1e-4):
    th = theta_hat_from_n(nfield)
    n1 = nfield + alpha * theta_tilde[..., None] * th
    n1 = n1 / np.linalg.norm(n1, axis=-1, keepdims=True).clip(1e-30)
    return (berry_curvature(n1, h) - berry_curvature(nfield, h)) / alpha


def synthetic_bp_skyrmion(n: int = 48, length: float = 6.0, lam: float = 1.5):
    ax = grid_axes(n, length)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rho = np.sqrt(X * X + Y * Y) + 1e-30
    theta = 2.0 * np.arctan2(lam, rho)
    phi = np.arctan2(Y, X)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return np.stack([nx, ny, nz], axis=-1), float(length)


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


def planar_disk_flux(
    vec: np.ndarray,
    length: float,
    plane: str,
    radius: float,
    offset: float = 0.0,
) -> float:
    """
    Flux of vector field `vec` (N,N,N,3) through a flat disk.

    plane:
      xy → normal +z, disk in z=offset, integrate v_z
      xz → normal +y, disk in y=offset, integrate v_y
      yz → normal +x, disk in x=offset, integrate v_x
    """
    n = vec.shape[0]
    h = grid_spacing(n, length)
    ax = grid_axes(n, length)
    # nearest plane index
    i0 = int(np.argmin(np.abs(ax - offset)))

    if plane == "xy":
        # axes: (i=x, j=y, k=z)
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        mask = (X * X + Y * Y) <= radius * radius
        slab = vec[:, :, i0, 2]
        return float(np.sum(slab[mask]) * h * h)
    if plane == "xz":
        X, Z = np.meshgrid(ax, ax, indexing="ij")
        mask = (X * X + Z * Z) <= radius * radius
        slab = vec[:, i0, :, 1]
        return float(np.sum(slab[mask]) * h * h)
    if plane == "yz":
        Y, Z = np.meshgrid(ax, ax, indexing="ij")
        mask = (Y * Y + Z * Z) <= radius * radius
        slab = vec[i0, :, :, 0]
        return float(np.sum(slab[mask]) * h * h)
    raise ValueError(plane)


def trilinear_sample_scalar(vol: np.ndarray, pts_xyz: np.ndarray, length: float) -> np.ndarray:
    """vol (N,N,N); pts (P,3) -> (P,)."""
    n = vol.shape[0]
    h = 2.0 * length / (n - 1)
    grid = (pts_xyz + length) / h
    out = np.zeros(pts_xyz.shape[0], dtype=np.float64)
    i0 = np.floor(grid).astype(np.int64)
    frac = grid - i0
    inside = np.all((i0 >= 0) & (i0 + 1 <= n - 1), axis=1)
    if not np.any(inside):
        return out
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


def trilinear_sample_vec(vec: np.ndarray, pts_xyz: np.ndarray, length: float) -> np.ndarray:
    """vec (N,N,N,3); pts (P,3) -> (P,3)."""
    comps = [
        trilinear_sample_scalar(vec[..., c], pts_xyz, length) for c in range(3)
    ]
    return np.stack(comps, axis=-1)


def hemisphere_flux_xy(
    vec: np.ndarray,
    length: float,
    radius: float,
    n_theta: int = 48,
    n_phi: int = 96,
    sign_z: float = 1.0,
) -> float:
    """
    Flux through northern (sign_z>0) or southern hemisphere of radius R
    centered at origin; boundary = equator in z=0 (same as flat xy disk).
    dS = R^2 sinθ dθ dφ \\hat{r}
    """
    # θ from 0..π/2 (north) or π/2..π (south)
    if sign_z >= 0:
        theta = np.linspace(0.0, 0.5 * np.pi, n_theta)
    else:
        theta = np.linspace(0.5 * np.pi, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    st, ct = np.sin(tt), np.cos(tt)
    cp, sp = np.cos(pp), np.sin(pp)
    # points on sphere
    x = radius * st * cp
    y = radius * st * sp
    z = radius * ct
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    # outward radial for north is +rhat; for Stokes same orientation as +z on disk,
    # use rhat with positive z-hemisphere (north)
    rhat = np.stack([st * cp, st * sp, ct], axis=-1).reshape(-1, 3)
    b = trilinear_sample_vec(vec, pts, length)
    integrand = np.sum(b * rhat, axis=-1) * (radius * radius) * st.ravel()
    return float(np.sum(integrand) * dtheta * dphi)


def cone_flux_xy(
    vec: np.ndarray,
    length: float,
    radius: float,
    height: float,
    n_r: int = 40,
    n_phi: int = 96,
) -> float:
    """
    Flux through cone from apex (0,0,height) to base circle r=radius, z=0.
    Same boundary as flat xy disk. Parametrize by (s,φ), s in [0,1] along generators.
    """
    s = np.linspace(0.0, 1.0, n_r)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    ds = s[1] - s[0]
    dphi = phi[1] - phi[0]
    ss, pp = np.meshgrid(s, phi, indexing="ij")
    # position: (s*R cosφ, s*R sinφ, (1-s)*H)
    x = ss * radius * np.cos(pp)
    y = ss * radius * np.sin(pp)
    z = (1.0 - ss) * height
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    # ∂r/∂s = (R cosφ, R sinφ, -H), ∂r/∂φ = (-s R sinφ, s R cosφ, 0)
    # N = ∂s × ∂φ
    cp, sp = np.cos(pp), np.sin(pp)
    # N_x = (R sinφ)*0 - (-H)(s R cosφ) = H s R cosφ
    # N_y = (-H)(-s R sinφ) - (R cosφ)*0 = H s R sinφ
    # N_z = (R cosφ)(s R cosφ) - (R sinφ)(-s R sinφ) = s R^2
    Nx = (height * ss * radius * cp).ravel()
    Ny = (height * ss * radius * sp).ravel()
    Nz = (ss * radius * radius).ravel()
    # at s=0 apex N=0; orientation: Nz>=0 matches +z on flat disk near base
    b = trilinear_sample_vec(vec, pts, length)
    integrand = b[:, 0] * Nx + b[:, 1] * Ny + b[:, 2] * Nz
    return float(np.sum(integrand) * ds * dphi)


def divergence_b(vec: np.ndarray, h: float) -> np.ndarray:
    """∇·vec on grid."""
    dx = np.gradient(vec[..., 0], h, axis=0, edge_order=2)
    dy = np.gradient(vec[..., 1], h, axis=1, edge_order=2)
    dz = np.gradient(vec[..., 2], h, axis=2, edge_order=2)
    return dx + dy + dz
