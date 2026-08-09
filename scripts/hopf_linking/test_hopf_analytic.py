"""
test_hopf_analytic.py
Generate an analytic Q=1 Hopf field and verify signed linking number.

After Jacobian orientation + Fourier/Whitehead-matched Gauss sign,
all regular-value pairs must give the *same* signed integer
Q_link = -1 (matches hopf_skyrme_torch Q_fft on the same field).

Runs both grid conventions used in this monorepo:
  - half: hopf_skyrme_cpu / breakpoint_2_5_gravity
  - side: legacy hopf_linking cell-centered box
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hopf_linking import compute_hopf_invariant, grid_coords, _format_orientation


def generate_hopf_field(N=64, L=8.0, grid_mode="half"):
    """
    Generate the standard Q=1 Hopfion field via stereographic Hopf map.

    Returns (N,N,N,3) unit vector field on the requested grid.
    """
    _h, coords = grid_coords(N, L, grid_mode=grid_mode)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Inverse stereographic projection R^3 -> S^3
    r2 = X**2 + Y**2 + Z**2
    denom = 1.0 + r2
    w = (1.0 - r2) / denom
    xi = 2.0 * X / denom
    yi = 2.0 * Y / denom
    zi = 2.0 * Z / denom

    # Hopf map S^3 -> S^2
    nx = 2.0 * (w * yi + xi * zi)
    ny = 2.0 * (xi * yi - w * zi)
    nz = w**2 + xi**2 - yi**2 - zi**2

    n_field = np.stack([nx, ny, nz], axis=-1)
    norms = np.linalg.norm(n_field, axis=-1, keepdims=True)
    n_field = n_field / np.maximum(norms, 1e-15)
    return n_field


def _run_pairs(n_field, L, grid_mode):
    test_pairs = [
        (np.array([1.0, 0.0, 0.1]), np.array([0.0, 1.0, 0.1])),
        (np.array([1.0, 0.2, 0.0]), np.array([0.0, 1.0, 0.2])),
        (np.array([0.5, 0.5, 0.1]), np.array([-0.5, 0.5, 0.1])),
    ]
    results = []
    for i, (p1, p2) in enumerate(test_pairs):
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)
        print(f"\n--- Pair {i+1}: p1={p1.round(3)}, p2={p2.round(3)} ---")
        result, _g1, _g2 = compute_hopf_invariant(
            n_field, L, p1, p2, grid_mode=grid_mode
        )
        print(f"  Q_link={result['Q_link']}  link_raw={result.get('link_raw')}")
        print(f"  orient1: {_format_orientation(result.get('orientation_gamma1'))}")
        print(f"  orient2: {_format_orientation(result.get('orientation_gamma2'))}")
        print(f"  status:  {result['status']}")
        results.append(result)
    return results


def main():
    print("=" * 60)
    print("TEST: Analytic Hopf field (signed Q_link = Q_fft)")
    print("=" * 60)

    N = 64
    configs = [
        # half-box L=4 → physical domain ≈ [-4,4], comparable to legacy side L=8
        ("half", 4.0),
        ("side", 8.0),
    ]

    all_ok = True
    for grid_mode, L in configs:
        print(f"\n### grid_mode={grid_mode}, L={L}, N={N}")
        n_field = generate_hopf_field(N, L, grid_mode=grid_mode)
        norms = np.linalg.norm(n_field, axis=-1)
        print(f"  |n| range: [{norms.min():.8f}, {norms.max():.8f}]")
        results = _run_pairs(n_field, L, grid_mode)
        qs = [r["Q_link"] for r in results]
        if not all(q == -1 for q in qs):
            all_ok = False
            print(f"FAIL: expected all Q_link=-1, got {qs}")
        else:
            print(f"PASS: all pairs Q_link=-1 under grid_mode={grid_mode}")

    if not all_ok:
        raise SystemExit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
