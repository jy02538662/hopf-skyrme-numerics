"""
hopf_linking.py
Compute exact integer Hopf invariant via preimage linking number.

Algorithm:
  1. Choose two regular values p1, p2 on S^2
  2. For each pi, construct two scalar fields f1, f2 such that
     n(x)=pi iff f1(x)=0 AND f2(x)=0
  3. Extract the preimage curve as the intersection of the
     f1=0 isosurface with the f2=0 level (edge interpolation on triangles)
  4. Orient each curve along the Jacobian tangent t0 = grad(f1) x grad(f2)
  5. Compute Gauss linking integral with Fourier/Whitehead-matched sign
     -> signed integer Hopf charge Q_H = Lk = Q_fft (same sign)

Orientation convention (canonical, unique up to field geometry):
  f1 = n·e1, f2 = n·e2 with right-handed frame {e1, e2, p}.
  t0 = ∇f1 × ∇f2 is fixed by the local Jacobian of n.
  If the chained polyline's tangent disagrees with t0 (score < 0),
  reverse the whole curve. This removes the arbitrary polyline sign.

Sign convention (match hopf_skyrme_torch Whitehead FFT):
  With Jacobian fiber orientation, the textbook Gauss integrand
  (r1-r2)·(dr1×dr2) equals −Q_fft on the standard Hopf map and on
  the project's relaxed fields. We therefore use (r2-r1)·(dr1×dr2)
  so that the reported Q_link shares the Fourier sign:
      Q_H = Q_link = Q_fft   (signed equality, up to discretization)

Migrated into hopf_skyrme_cpu from the sibling hopf_linking package.
See scripts/hopf_linking/README.md for grid conventions and BP2.5 role.

Dependencies: numpy, scipy, scikit-image
"""
import argparse
import json
import numpy as np
from pathlib import Path


def grid_coords(N, L, grid_mode="half"):
    """
    Return (h, coords) for the chosen cubic-grid convention.

    Parameters
    ----------
    N : int
        Points per axis.
    L : float
        Half-box length if grid_mode='half' (field on [-L, L]^3),
        or full side length if grid_mode='side' (legacy hopf_linking).
    grid_mode : {'half', 'side'}
        'half' matches hopf_skyrme_torch / breakpoint_2_5_gravity:
            coords = linspace(-L, L, N), h = 2L/(N-1).
        'side' is the legacy cell-centered box of side L:
            h = L/N, coords in [-L/2+h/2, L/2-h/2].
    """
    N = int(N)
    L = float(L)
    if grid_mode == "half":
        if N < 2:
            raise ValueError("half-mode grid needs N>=2")
        h = 2.0 * L / (N - 1)
        coords = np.linspace(-L, L, N)
    elif grid_mode == "side":
        h = L / float(N)
        coords = np.linspace(-L / 2.0 + h / 2.0, L / 2.0 - h / 2.0, N)
    else:
        raise ValueError(f"Unknown grid_mode={grid_mode!r}; use 'half' or 'side'")
    return h, coords


def _empty_orientation_info():
    return {
        'orientation_score': None,
        'mean_alignment': None,
        'orientation_reversed': False,
        'orientation_weak': False,
    }


def extract_preimage_curve(n_field, L, target_point, return_info=False, grid_mode="half"):
    """
    Extract preimage curve n^{-1}(p) by intersecting two zero-level sets.

    Method: marching cubes on f1 gives a triangle mesh. On each triangle
    edge, linearly interpolate f2 to find zero-crossings. Connect them
    to form ordered curve segments, then orient along ∇f1 × ∇f2.
    """
    from skimage.measure import marching_cubes

    N = n_field.shape[0]
    h, _coords = grid_coords(N, L, grid_mode=grid_mode)

    p = np.asarray(target_point, dtype=np.float64)
    p = p / np.linalg.norm(p)

    # Build right-handed orthonormal frame {e1, e2, p}
    if abs(p[2]) < 0.9:
        v = np.array([0.0, 0.0, 1.0])
    else:
        v = np.array([1.0, 0.0, 0.0])
    e1 = v - np.dot(v, p) * p
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(p, e1)
    e2 = e2 / np.linalg.norm(e2)

    # Scalar fields. The simultaneous zeros contain both n=p and n=-p;
    # fp separates the requested preimage from its antipodal preimage.
    f1 = np.einsum('ijkl,l->ijk', n_field, e1)
    f2 = np.einsum('ijkl,l->ijk', n_field, e2)
    fp = np.einsum('ijkl,l->ijk', n_field, p)

    # Canonical tangent: t0 = ∇f1 × ∇f2 (Jacobian co-orientation).
    grad_f1 = np.stack(np.gradient(f1, h, h, h, edge_order=2), axis=-1)
    grad_f2 = np.stack(np.gradient(f2, h, h, h, edge_order=2), axis=-1)
    tangent_field = np.cross(grad_f1, grad_f2)

    # Marching cubes on f1=0
    try:
        verts, faces, _, _ = marching_cubes(f1, level=0.0, spacing=(h, h, h))
    except (ValueError, RuntimeError):
        empty = np.zeros((0, 3))
        info = _empty_orientation_info()
        return (empty, info) if return_info else empty

    # marching_cubes places index 0 at coordinate 0 with the given spacing.
    # Map to the physical grid used by this repository (or legacy side mode).
    if grid_mode == "half":
        verts = verts - L
    else:
        verts = verts - L / 2.0 + h / 2.0

    # Interpolate f2 and fp at each vertex of the isosurface.
    f2_at_verts = _interpolate_at_points(f2, verts, L, N, grid_mode=grid_mode)
    fp_at_verts = _interpolate_at_points(fp, verts, L, N, grid_mode=grid_mode)

    # For each triangle, find edges where f2 changes sign. Retain only
    # intersections on the n=p branch (fp>0), excluding n=-p.
    segments = []
    for tri in faces:
        f2_tri = f2_at_verts[tri]  # (3,)
        v_tri = verts[tri]         # (3, 3)

        crossings = []
        for i in range(3):
            j = (i + 1) % 3
            if f2_tri[i] * f2_tri[j] < 0:
                # Linear interpolation for the simultaneous zero crossing.
                t = f2_tri[i] / (f2_tri[i] - f2_tri[j])
                fp_cross = fp_at_verts[tri[i]] + t * (
                    fp_at_verts[tri[j]] - fp_at_verts[tri[i]]
                )
                if fp_cross > 0.0:
                    pt = v_tri[i] + t * (v_tri[j] - v_tri[i])
                    crossings.append(pt)

        if len(crossings) == 2:
            segments.append(crossings)

    if not segments:
        empty = np.zeros((0, 3))
        info = _empty_orientation_info()
        return (empty, info) if return_info else empty

    curve = _chain_segments(segments)
    curve, info = _orient_curve_canonically(
        curve, tangent_field, L, N, grid_mode=grid_mode
    )
    return (curve, info) if return_info else curve


def _interpolate_vector_at_points(vector_field, points, L, N, grid_mode="half"):
    """Trilinear interpolation of a vector field."""
    return np.column_stack([
        _interpolate_at_points(
            vector_field[..., component], points, L, N, grid_mode=grid_mode
        )
        for component in range(3)
    ])


def _orient_curve_canonically(curve, tangent_field, L, N, grid_mode="half"):
    """
    Orient a closed preimage curve along t0 = ∇f1 × ∇f2.

    Score = sum_i Δr_i · t0(mid_i). If negative, reverse the polyline.
    Also report mean cosine alignment as a robustness diagnostic.
    """
    info = _empty_orientation_info()
    if len(curve) < 3:
        return curve, info

    closed = np.vstack([curve, curve[0]])
    segment_tangents = np.diff(closed, axis=0)
    midpoints = 0.5 * (closed[:-1] + closed[1:])
    canonical = _interpolate_vector_at_points(
        tangent_field, midpoints, L, N, grid_mode=grid_mode
    )

    # Length-weighted agreement with Jacobian tangent.
    dots = np.einsum('ij,ij->i', segment_tangents, canonical)
    score = float(np.sum(dots))

    # Mean cosine: ignores amplitude of t0, checks directional consistency.
    seg_norm = np.linalg.norm(segment_tangents, axis=1)
    can_norm = np.linalg.norm(canonical, axis=1)
    valid = (seg_norm > 1e-15) & (can_norm > 1e-15)
    if np.any(valid):
        cosines = dots[valid] / (seg_norm[valid] * can_norm[valid])
        mean_alignment = float(np.mean(cosines))
    else:
        mean_alignment = 0.0

    reversed_curve = score < 0.0
    if reversed_curve:
        curve = curve[::-1].copy()
        score = -score
        mean_alignment = -mean_alignment

    # Weak orientation: |mean cosine| too small → target may be near-critical.
    orientation_weak = abs(mean_alignment) < 0.2

    info = {
        'orientation_score': score,
        'mean_alignment': mean_alignment,
        'orientation_reversed': reversed_curve,
        'orientation_weak': orientation_weak,
    }
    return curve, info


def _interpolate_at_points(scalar_field, points, L, N, grid_mode="half"):
    """Trilinear interpolation of scalar field at arbitrary points."""
    from scipy.interpolate import RegularGridInterpolator

    _h, coords = grid_coords(N, L, grid_mode=grid_mode)
    interp = RegularGridInterpolator(
        (coords, coords, coords), scalar_field,
        method='linear', bounds_error=False, fill_value=0.0
    )
    return interp(points)


def _chain_segments(segments):
    """
    Chain a list of line segments [(p1,p2), ...] into an ordered curve.
    Uses endpoint matching with tolerance.
    """
    if not segments:
        return np.zeros((0, 3))

    segments = [(np.array(s[0]), np.array(s[1])) for s in segments]
    n_seg = len(segments)

    if n_seg == 0:
        return np.zeros((0, 3))

    # Build all endpoints
    all_pts = []
    for s in segments:
        all_pts.append(s[0])
        all_pts.append(s[1])
    all_pts = np.array(all_pts)

    # Median segment length for tolerance
    lengths = np.array([np.linalg.norm(s[1] - s[0]) for s in segments])
    tol = np.median(lengths) * 2.0

    # Greedy chaining
    used = np.zeros(n_seg, dtype=bool)
    chain = list(segments[0])
    used[0] = True

    for _ in range(n_seg - 1):
        tail = chain[-1]
        best_idx = -1
        best_dist = np.inf
        flip = False

        for i in range(n_seg):
            if used[i]:
                continue
            d0 = np.linalg.norm(segments[i][0] - tail)
            d1 = np.linalg.norm(segments[i][1] - tail)
            if d0 < best_dist:
                best_dist = d0
                best_idx = i
                flip = False
            if d1 < best_dist:
                best_dist = d1
                best_idx = i
                flip = True

        if best_idx < 0 or best_dist > tol:
            break

        used[best_idx] = True
        if flip:
            chain.append(segments[best_idx][0])
        else:
            chain.append(segments[best_idx][1])

    return np.array(chain)


def gauss_linking_number(gamma1, gamma2):
    """
    Compute Gauss linking integral between two closed curves.

    Fourier/Whitehead-matched form (see module docstring):

        link = (1/4π) ∬ (r2-r1) · (dr1 × dr2) / |r1-r2|^3

    With Jacobian-oriented fibers this equals the hopf_skyrme_torch
    Whitehead Q_fft (same global sign). Returns (integer, raw_float).
    """
    if len(gamma1) < 3 or len(gamma2) < 3:
        return 0, 0.0

    # Close the curves
    g1 = np.vstack([gamma1, gamma1[0:1]])
    g2 = np.vstack([gamma2, gamma2[0:1]])

    dr1 = np.diff(g1, axis=0)  # (M1, 3)
    dr2 = np.diff(g2, axis=0)  # (M2, 3)

    mid1 = 0.5 * (g1[:-1] + g1[1:])  # (M1, 3)
    mid2 = 0.5 * (g2[:-1] + g2[1:])  # (M2, 3)

    # r21[i,j] = mid2[j] - mid1[i]  (Fourier/Whitehead-matched displacement)
    r21 = mid2[None, :, :] - mid1[:, None, :]  # (M1, M2, 3)
    dist = np.linalg.norm(r21, axis=2, keepdims=True)
    dist = np.maximum(dist, 1e-12)

    # cross[i,j] = dr1[i] x dr2[j]
    cross = np.cross(dr1[:, None, :], dr2[None, :, :])

    # integrand[i,j] = r21 . cross / |r21|^3
    integrand = np.sum(r21 * cross, axis=2) / (dist[:, :, 0] ** 3)

    link_raw = integrand.sum() / (4.0 * np.pi)
    link_int = int(np.round(link_raw))

    return link_int, link_raw


def _format_orientation(info):
    """One-line orientation diagnostic."""
    if info is None or info.get('orientation_score') is None:
        return 'n/a'
    score = info['orientation_score']
    align = info.get('mean_alignment')
    rev = info.get('orientation_reversed', False)
    weak = info.get('orientation_weak', False)
    align_s = f'{align:.4f}' if align is not None else 'n/a'
    flags = []
    if rev:
        flags.append('reversed')
    if weak:
        flags.append('WEAK')
    flag_s = ','.join(flags) if flags else 'ok'
    return f'score={score:.4g}, mean_cos={align_s}, {flag_s}'


def compute_hopf_invariant(n_field, L, p1=None, p2=None, grid_mode="half"):
    """
    Compute Hopf invariant Q_H = Lk(preimage(p1), preimage(p2)).

    Fibers are Jacobian-oriented; the Gauss integrand uses the
    Fourier/Whitehead-matched sign so that Q_link and Q_fft agree
    (including sign) up to discretization error.

    Parameters
    ----------
    n_field : (N,N,N,3) unit vector field
    L : half-box length (grid_mode='half') or side length (grid_mode='side')
    p1, p2 : target points on S^2 (auto-chosen if None)
    grid_mode : {'half','side'}
        Prefer 'half' for fields from hopf_skyrme_cpu.

    Returns
    -------
    result : dict
    gamma1, gamma2 : canonically oriented preimage curves
    """
    if p1 is None:
        # Choose two well-separated points near equator
        theta1, phi1 = np.pi/2, 0.0
        p1 = np.array([np.sin(theta1)*np.cos(phi1),
                       np.sin(theta1)*np.sin(phi1),
                       np.cos(theta1)])
    if p2 is None:
        theta2, phi2 = np.pi/2, np.pi/2
        p2 = np.array([np.sin(theta2)*np.cos(phi2),
                       np.sin(theta2)*np.sin(phi2),
                       np.cos(theta2)])

    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    gamma1, orientation1 = extract_preimage_curve(
        n_field, L, p1, return_info=True, grid_mode=grid_mode
    )
    gamma2, orientation2 = extract_preimage_curve(
        n_field, L, p2, return_info=True, grid_mode=grid_mode
    )

    base = {
        'p1': p1.tolist(),
        'p2': p2.tolist(),
        'grid_mode': grid_mode,
        'n_pts_gamma1': len(gamma1),
        'n_pts_gamma2': len(gamma2),
        'orientation_gamma1': orientation1,
        'orientation_gamma2': orientation2,
        'orientation_weak': bool(
            orientation1.get('orientation_weak')
            or orientation2.get('orientation_weak')
        ),
    }

    if len(gamma1) < 3 or len(gamma2) < 3:
        return {
            **base,
            'Q_link': None,
            'link_raw': None,
            'status': 'FAILED: insufficient preimage points',
        }, gamma1, gamma2

    Q_link, link_raw = gauss_linking_number(gamma1, gamma2)

    status = 'OK'
    if base['orientation_weak']:
        status = 'OK_WEAK_ORIENTATION'

    return {
        **base,
        'Q_link': Q_link,
        'link_raw': float(link_raw),
        'rounding_error': float(abs(link_raw - Q_link)),
        'status': status,
    }, gamma1, gamma2


def main():
    parser = argparse.ArgumentParser(
        description='Compute Hopf invariant via preimage linking number'
    )
    parser.add_argument('--field', required=True, help='Path to nfield.npy')
    parser.add_argument(
        '--length',
        type=float,
        required=True,
        help="Half-box L for --grid-mode half (default); side length for 'side'",
    )
    parser.add_argument(
        '--grid-mode',
        choices=('half', 'side'),
        default='half',
        help="Coordinate convention (default: half = hopf_skyrme_cpu)",
    )
    parser.add_argument('--out', default='results/', help='Output directory')
    parser.add_argument('--p1', type=float, nargs=3, default=None)
    parser.add_argument('--p2', type=float, nargs=3, default=None)
    args = parser.parse_args()

    print(f"Loading field: {args.field}")
    n_field = np.load(args.field)

    if n_field.ndim == 4 and n_field.shape[0] == 3:
        n_field = np.moveaxis(n_field, 0, -1)
        print("  Transposed (3,N,N,N) -> (N,N,N,3)")

    N = n_field.shape[0]
    h, _ = grid_coords(N, args.length, grid_mode=args.grid_mode)
    print(f"  Grid: {N}^3, L={args.length}, mode={args.grid_mode}, h={h:.6g}")

    norms = np.linalg.norm(n_field, axis=-1)
    print(f"  |n| range: [{norms.min():.6f}, {norms.max():.6f}]")

    p1 = np.array(args.p1) if args.p1 else None
    p2 = np.array(args.p2) if args.p2 else None

    result, gamma1, gamma2 = compute_hopf_invariant(
        n_field, args.length, p1, p2, grid_mode=args.grid_mode
    )

    print(f"\n=== RESULT ===")
    print(f"  Q_link (integer): {result['Q_link']}")
    print(f"  link_raw (float): {result.get('link_raw', 'N/A')}")
    print(f"  rounding error:   {result.get('rounding_error', 'N/A')}")
    print(f"  gamma1 points:    {result['n_pts_gamma1']}")
    print(f"  gamma2 points:    {result['n_pts_gamma2']}")
    print(f"  orient gamma1:    {_format_orientation(result.get('orientation_gamma1'))}")
    print(f"  orient gamma2:    {_format_orientation(result.get('orientation_gamma2'))}")
    print(f"  status:           {result['status']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'gamma1.npy', gamma1)
    np.save(out_dir / 'gamma2.npy', gamma2)
    with open(out_dir / 'linking_report.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
