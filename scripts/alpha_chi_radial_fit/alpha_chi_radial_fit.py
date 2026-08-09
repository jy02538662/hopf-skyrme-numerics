#!/usr/bin/env python3
"""
Far-field radial fit: spherical averages of tilt angle α and transverse χ,
then log–log power-law fit ~ A / r^β.

Physics
-------
Under |n|=1: χ = sin(α) exactly (pointwise). If α ~ A/r is NOT small in the
fit window, a naive power-law fit to χ = sin(α) gives an *apparent* β < 1:

    d ln sin(α) / d ln r  =  − α cot(α)     (for α = A/r)

so β_eff(χ) = α cot(α) → 1 only when α → 0. For Q=1 with A~5 on L=10,
α(r=6..9) ~ 0.5–0.9 rad ⇒ β_eff ~ 0.7–0.9 — matching the “anomaly”.

Primary proxy for the 1/r monopole is therefore α (or arcsin(χ)), not raw χ.

Grid: x = linspace(-length, length, N)  (box half-length = length)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


def load_nxyz(path: Path):
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 4 and arr.shape[-1] == 3:
            return (
                arr[..., 0].astype(np.float64),
                arr[..., 1].astype(np.float64),
                arr[..., 2].astype(np.float64),
            )
        raise ValueError(
            f"Expected nfield shape (N,N,N,3), got {arr.shape}. "
            "Or pass --nx/--ny/--nz separately."
        )
    raise ValueError(f"Unsupported file: {path}")


def normalize_field(nx, ny, nz, eps=1e-30):
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    return nx / np.maximum(norm, eps), ny / np.maximum(norm, eps), nz / np.maximum(norm, eps)


def spherical_bin_means(r_grid, fields, length, num_bins):
    r_edges = np.linspace(0.0, length, num_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    means = {k: np.full(num_bins, np.nan) for k in fields}
    for i in range(num_bins):
        mask = (r_grid >= r_edges[i]) & (r_grid < r_edges[i + 1])
        if not np.any(mask):
            continue
        for k, vol in fields.items():
            means[k][i] = float(np.mean(vol[mask]))
    return r_centers, means


def power_law_fit(r, y):
    ok = np.isfinite(r) & np.isfinite(y) & (r > 0) & (y > 0)
    if np.count_nonzero(ok) < 2:
        raise RuntimeError("Not enough finite positive points for power-law fit.")
    log_r = np.log(r[ok])
    log_y = np.log(y[ok])
    coeff = np.polyfit(log_r, log_y, 1)
    beta = -float(coeff[0])
    A = float(np.exp(coeff[1]))
    return beta, A, coeff, log_r, log_y


def fit_window(r_centers, series, r_min, r_max):
    m = (r_centers >= r_min) & (r_centers <= r_max)
    return power_law_fit(r_centers[m], series[m])


def beta_eff_sin_of_monopole(alpha):
    """Local effective index of sin(α) when α ∝ 1/r: β_eff = α cot(α)."""
    a = np.asarray(alpha, dtype=np.float64)
    # stable: α cot(α) = α cos(α)/sin(α); → 1 as α→0
    out = np.empty_like(a)
    small = np.abs(a) < 1e-8
    out[small] = 1.0
    as_ = a[~small]
    out[~small] = as_ * np.cos(as_) / np.sin(as_)
    return out


def predict_beta_chi_from_alpha_fit(r_fit, A_alpha, beta_alpha):
    """
    If <α> ≈ A/r^β, predict apparent β of sin(<α>) by fitting
    sin(A/r^β) over the same r points (radial proxy; ignores angular Jensen gap).
    """
    alpha_model = A_alpha / np.power(r_fit, beta_alpha)
    chi_model = np.sin(alpha_model)
    beta_pred, A_pred, _, _, _ = power_law_fit(r_fit, chi_model)
    beta_local = beta_eff_sin_of_monopole(alpha_model)
    return beta_pred, A_pred, float(np.mean(alpha_model)), float(np.mean(beta_local))


def print_diagnostics(r_grid, nx, ny, nz, alpha_angle, chi, L):
    print("-" * 60)
    print("Diagnostics (numerical integrity)")
    print("-" * 60)

    r0, r1 = 0.60 * L, 0.90 * L
    mask = (r_grid >= r0) & (r_grid <= r1)
    if not np.any(mask):
        print("  [skip] empty diagnostic shell")
        return

    norm_n = np.sqrt(nx**2 + ny**2 + nz**2)
    print(f"  shell r ∈ [{r0:.2f}, {r1:.2f}]")
    print(f"  |n| mean     = {np.mean(norm_n[mask]):.8f}")
    print(f"  |n| std      = {np.std(norm_n[mask]):.8f}")
    print(f"  |n|-1 max    = {np.max(np.abs(norm_n[mask] - 1.0)):.8f}")
    print(f"  <α>          = {np.mean(alpha_angle[mask]):.4f} rad  "
          f"(NOT ≪ 1 ⇒ sinα power-law ≠ α power-law)")

    rc = 0.70 * L
    half = max(0.02 * L, 0.1)
    mask_t = (r_grid > rc - half) & (r_grid < rc + half)
    if np.any(mask_t):
        chi_t = chi[mask_t]
        sin_t = np.sin(alpha_angle[mask_t])
        rel = np.abs(chi_t - sin_t) / np.maximum(chi_t, 1e-30)
        a_mean = float(np.mean(alpha_angle[mask_t]))
        print(f"  shell r ≈ {rc:.2f} (±{half:.2f})")
        print(f"  <χ>          = {np.mean(chi_t):.6f}")
        print(f"  <sin α>      = {np.mean(sin_t):.6f}")
        print(f"  mean |χ-sinα|/χ = {np.mean(rel):.4%}")
        print(f"  α cot(α) @ <α>  = {float(beta_eff_sin_of_monopole(a_mean)):.3f}  "
              f"(expected local β_eff for χ if α∝1/r)")
        if np.mean(rel) > 1e-3:
            print("  >> χ ≉ sinα: unit-norm broken; try --normalize")
        else:
            print("  >> χ = sinα OK; if β_χ < β_α with α=O(1), that is geometry not noise")

    # Compare outer shell to sin(A/r) expectation will be done after α fit
    r_edges = np.linspace(0.0, L, 51)
    outer = (r_grid >= r_edges[-2]) & (r_grid < r_edges[-1])
    mid = (r_grid >= 0.55 * L) & (r_grid < 0.65 * L)
    if np.any(outer) and np.any(mid):
        chi_out = float(np.mean(chi[outer]))
        chi_mid = float(np.mean(chi[mid]))
        a_out = float(np.mean(alpha_angle[outer]))
        a_mid = float(np.mean(alpha_angle[mid]))
        print(f"  <α>,<χ>(r~{0.6*L:.1f}) = {a_mid:.4f}, {chi_mid:.6g}")
        print(f"  <α>,<χ>(r~{L:.1f} edge) = {a_out:.4f}, {chi_out:.6g}")
        print(f"  sin(<α>_edge) = {np.sin(a_out):.6g}  (if ≈ <χ>_edge, no extra constant floor)")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser(description="α / χ spherical radial far-field fit")
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--nx", type=str, default="")
    ap.add_argument("--ny", type=str, default="")
    ap.add_argument("--nz", type=str, default="")
    ap.add_argument("--length", type=float, required=True, help="box half-length L")
    ap.add_argument("--r-min-fit", type=float, default=None)
    ap.add_argument("--r-max-fit", type=float, default=None)
    ap.add_argument("--num-bins", type=int, default=50)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--scan-windows", action="store_true")
    ap.add_argument("--out", type=str, default="alpha_radial_fit.png")
    ap.add_argument("--save-csv", type=str, default="")
    args = ap.parse_args()

    L = float(args.length)
    r_min_fit = args.r_min_fit if args.r_min_fit is not None else 0.55 * L
    r_max_fit = args.r_max_fit if args.r_max_fit is not None else 0.85 * L

    if args.field:
        nx, ny, nz = load_nxyz(Path(args.field))
        field_label = Path(args.field).name
    elif args.nx and args.ny and args.nz:
        nx = np.load(args.nx).astype(np.float64)
        ny = np.load(args.ny).astype(np.float64)
        nz = np.load(args.nz).astype(np.float64)
        field_label = f"{Path(args.nx).name}+ny+nz"
    else:
        ap.error("Provide --field nfield.npy  OR  --nx/--ny/--nz")

    if nx.shape != ny.shape or nx.shape != nz.shape:
        raise ValueError(f"nx/ny/nz shape mismatch: {nx.shape}, {ny.shape}, {nz.shape}")
    if nx.ndim != 3 or nx.shape[0] != nx.shape[1] or nx.shape[1] != nx.shape[2]:
        raise ValueError(f"Expected cubic 3D arrays, got {nx.shape}")

    N = int(nx.shape[0])
    x = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r_grid = np.sqrt(X**2 + Y**2 + Z**2)

    print_diagnostics(
        r_grid,
        nx,
        ny,
        nz,
        alpha_angle=np.arccos(np.clip(nz, -1.0, 1.0)),
        chi=np.sqrt(nx**2 + ny**2),
        L=L,
    )

    if args.normalize:
        nx, ny, nz = normalize_field(nx, ny, nz)
        print("[+] applied --normalize (n ← n/|n|)")

    nz_clipped = np.clip(nz, -1.0, 1.0)
    alpha_angle = np.arccos(nz_clipped)
    chi = np.sqrt(nx**2 + ny**2)
    sin_alpha = np.sin(alpha_angle)
    # Recovered tilt from χ (equals α when |n|=1 and nz≥0)
    alpha_from_chi = np.arcsin(np.clip(chi, 0.0, 1.0))

    r_centers, means = spherical_bin_means(
        r_grid,
        {
            "alpha": alpha_angle,
            "chi": chi,
            "sin_alpha": sin_alpha,
            "alpha_from_chi": alpha_from_chi,
        },
        L,
        args.num_bins,
    )
    mean_alpha = means["alpha"]
    mean_chi = means["chi"]
    mean_sin = means["sin_alpha"]
    mean_a_from_chi = means["alpha_from_chi"]

    beta_alpha, A_alpha, coeff_alpha, log_r_a, log_alpha = fit_window(
        r_centers, mean_alpha, r_min_fit, r_max_fit
    )
    beta_chi, A_chi, coeff_chi, log_r_c, log_chi = fit_window(
        r_centers, mean_chi, r_min_fit, r_max_fit
    )
    beta_sin, A_sin, _, _, _ = fit_window(r_centers, mean_sin, r_min_fit, r_max_fit)
    beta_afc, A_afc, _, _, _ = fit_window(r_centers, mean_a_from_chi, r_min_fit, r_max_fit)

    fit_m = (r_centers >= r_min_fit) & (r_centers <= r_max_fit)
    r_fit = r_centers[fit_m]
    beta_pred, A_pred, a_mean_model, beta_local_mean = predict_beta_chi_from_alpha_fit(
        r_fit, A_alpha, beta_alpha
    )
    a_data_mean = float(np.nanmean(mean_alpha[fit_m]))

    print("=" * 60)
    print(f"field: {field_label}")
    print(f"grid:  N={N}, L={L:.4g}  (linspace(-L,L,N))")
    print(f"primary fit window r = {r_min_fit:.2f} .. {r_max_fit:.2f}")
    print(f"<α> in window (data) = {a_data_mean:.4f} rad")
    print("=" * 60)
    print(f"Tilt angle α:          β = {beta_alpha:.3f},  A = {A_alpha:.6g}   ← primary 1/r proxy")
    print(f"arcsin(<χ>) proxy:     β = {beta_afc:.3f},  A = {A_afc:.6g}   ← should ≈ α if |n|=1")
    print(f"Transverse χ=sinα:     β = {beta_chi:.3f},  A = {A_chi:.6g}   (apparent; α not ≪1)")
    print(f"sin(α) shell avg:      β = {beta_sin:.3f},  A = {A_sin:.6g}")
    print("-" * 60)
    print("Geometry check (if α ≈ A/r^β, what β should raw χ show?):")
    print(f"  predict β from sin(A/r^β) fit = {beta_pred:.3f}  (A_sin_model={A_pred:.4g})")
    print(f"  mean α·cot(α) on model curve  = {beta_local_mean:.3f}")
    print(f"  measured β_χ                  = {beta_chi:.3f}")
    if abs(beta_chi - beta_pred) < 0.08:
        print("  >> β_χ matches sin(monopole α) prediction — NOT a numerical bug")
    else:
        print("  >> residual mismatch may include angular averaging (Jensen) / truncation")
    print(f"A_α / A_χ = {A_alpha / A_chi:.3f}  (not meaningful as same-power amplitude ratio)")
    print("=" * 60)

    if args.scan_windows:
        windows = [
            (0.65 * L, 0.95 * L, "outer"),
            (0.55 * L, 0.85 * L, "mid"),
            (0.50 * L, 0.80 * L, "inner"),
            (0.55 * L, 0.75 * L, "narrow"),
        ]
        print("Window scan: β_α | β_arcsinχ | β_χ | predicted_β_χ")
        for a, b, tag in windows:
            try:
                ba, Aa, _, _, _ = fit_window(r_centers, mean_alpha, a, b)
                bc, _, _, _, _ = fit_window(r_centers, mean_chi, a, b)
                bf, _, _, _, _ = fit_window(r_centers, mean_a_from_chi, a, b)
                rf = r_centers[(r_centers >= a) & (r_centers <= b)]
                bp, _, _, _ = predict_beta_chi_from_alpha_fit(rf, Aa, ba)
                print(
                    f"  [{a:.2f},{b:.2f}]  β_α={ba:.3f}  β_asinχ={bf:.3f}  "
                    f"β_χ={bc:.3f}  pred={bp:.3f}  ({tag})"
                )
            except RuntimeError as e:
                print(f"  [{a:.2f},{b:.2f}]  FAILED: {e}")
        print("=" * 60)

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            csv_path,
            np.column_stack([r_centers, mean_alpha, mean_chi, mean_sin, mean_a_from_chi]),
            header="r,<alpha>,<chi>,<sin_alpha>,<arcsin_chi>",
            comments="",
            delimiter=",",
        )
        print(f"saved CSV: {csv_path}")

    plt.rcParams["figure.dpi"] = 120
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(r_centers, mean_alpha, "o-", label="α", markersize=4)
    ax1.loglog(r_centers, mean_a_from_chi, "d-", label="arcsin χ", markersize=3, alpha=0.8)
    ax1.loglog(r_centers, mean_chi, "s-", label="χ=sinα", markersize=4)
    r_line = np.linspace(r_min_fit, r_max_fit, 100)
    ax1.loglog(r_line, A_alpha / r_line**beta_alpha, "k--", label=f"α: 1/r^{beta_alpha:.2f}")
    ax1.loglog(r_line, A_chi / r_line**beta_chi, "r--", label=f"χ: 1/r^{beta_chi:.2f}")
    ax1.loglog(
        r_line,
        np.sin(A_alpha / r_line**beta_alpha),
        "g:",
        lw=2,
        label="sin(α_fit) prediction",
    )
    ax1.axvline(r_min_fit, color="gray", ls=":", lw=1)
    ax1.axvline(r_max_fit, color="gray", ls=":", lw=1)
    ax1.set_xlabel("r")
    ax1.set_ylabel("Spherical average")
    ax1.set_title("Radial decay (log-log)")
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(log_r_a, log_alpha, "o", label="α")
    ax2.plot(log_r_a, np.polyval(coeff_alpha, log_r_a), "k--", label=f"α slope={-beta_alpha:.3f}")
    ax2.plot(log_r_c, log_chi, "s", label="χ", markersize=4)
    ax2.plot(log_r_c, np.polyval(coeff_chi, log_r_c), "r--", label=f"χ slope={-beta_chi:.3f}")
    ax2.set_xlabel("ln(r)")
    ax2.set_ylabel("ln(<·>)")
    ax2.set_title("Far-field fit detail")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"N={N}, L={L}, fit [{r_min_fit:.2f},{r_max_fit:.2f}] | {field_label}"
        + (" | normalized" if args.normalize else ""),
        fontsize=10,
    )
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved figure: {out_path.resolve()}")


if __name__ == "__main__":
    main()
