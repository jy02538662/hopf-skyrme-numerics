#!/usr/bin/env python3
"""
Step 1 — Far-field shell mean ~ A r^{-alpha} for Q=1..4.

Primary quantity: <chi> = <sqrt(nx^2+ny^2)>  (positive transverse amplitude).
Optional cross-check: <tilt> = <arccos(nz)> (closer to paper ~1/r for Q=1).

Writable: Q=1 long-range; Q>=2 screened; alpha increases with Q.
NOT a signed-component l=0 monopole-forbid test.

Fit windows are OUTER far field (see common.DEFAULT_FIELDS). Mid-field
r~5-7 for Q>=2 has non-monotonic <chi> bumps — do not use for power laws.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def _load_common():
    here = Path(__file__).resolve().parent
    common_path = here / "common.py"
    if not common_path.is_file():
        raise SystemExit(
            f"Cannot find common.py next to this script.\n"
            f"  expected: {common_path}\n"
            f"Copy the whole screening_progression_3step/ folder."
        )
    spec = importlib.util.spec_from_file_location("screening_common", common_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_common = _load_common()
chi_from_nfield = _common.chi_from_nfield
load_nfield = _common.load_nfield
resolve_field_table = _common.resolve_field_table
save_json = _common.save_json
shell_mean = _common.shell_mean


def power_law(r, A, alpha):
    return A * np.power(r, -alpha)


def fit_alpha(r, y, p0_alpha):
    y = np.asarray(y, dtype=float)
    r = np.asarray(r, dtype=float)
    if np.any(y <= 0):
        raise RuntimeError("non-positive shell mean; cannot log-fit")
    popt, pcov = curve_fit(
        power_law,
        r,
        y,
        p0=[float(y[0] * r[0] ** p0_alpha), p0_alpha],
        bounds=([0.0, 0.2], [np.inf, 20.0]),
        maxfev=20000,
    )
    err = float(np.sqrt(np.diag(pcov))[1])
    pred = power_law(r, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return float(popt[0]), float(popt[1]), err, r2, pred


def main():
    ap = argparse.ArgumentParser(description="Step1: far-field alpha fit")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--n-points", type=int, default=4000)
    ap.add_argument("--n-shells", type=int, default=9)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument(
        "--quantity",
        choices=["chi", "tilt", "both"],
        default="both",
        help="chi=<sqrt(nx^2+ny^2)>; tilt=<arccos(nz)>; both=report both",
    )
    ap.add_argument("--field-q1", type=str, default="")
    ap.add_argument("--field-q2", type=str, default="")
    ap.add_argument("--field-q3", type=str, default="")
    ap.add_argument("--field-q4", type=str, default="")
    args = ap.parse_args()

    table = resolve_field_table(args.abs_outputs)
    overrides = {1: args.field_q1, 2: args.field_q2, 3: args.field_q3, 4: args.field_q4}
    ref_alpha = {1: 1.04, 2: 3.17, 3: 5.59, 4: 8.02}
    p0_alpha = {1: 1.0, 2: 3.0, 3: 5.5, 4: 8.0}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quantities = ["chi", "tilt"] if args.quantity == "both" else [args.quantity]
    fit_results = {qname: {} for qname in quantities}

    print("=" * 64)
    print("STEP 1: outer-far-field power law  (mid-field r~5-7 excluded)")
    print("  Primary writable proxy: <chi>; tilt = <arccos(nz)> is cross-check")
    print("=" * 64)

    for Q in [1, 2, 3, 4]:
        meta = table[Q]
        path = overrides[Q] or meta["path"]
        if not Path(path).is_file():
            print(f"[SKIP] Q={Q}: missing {path}")
            continue
        length = float(meta["length"])
        r_min, r_max = float(meta["r_min"]), float(meta["r_max"])
        nfield = load_nfield(path)
        vols = {}
        if "chi" in quantities:
            vols["chi"] = chi_from_nfield(nfield)
        if "tilt" in quantities:
            vols["tilt"] = np.arccos(np.clip(nfield[..., 2], -1.0, 1.0))

        r_fit = np.linspace(r_min, r_max, args.n_shells)
        print(f"\n--- Q={Q}  L={length}  window=[{r_min},{r_max}]  {Path(path).name} ---")

        for qname, vol in vols.items():
            y = np.array([shell_mean(vol, length, float(r), args.n_points) for r in r_fit])
            print(f"  {qname} shell means: " + ", ".join(f"{v:.4g}" for v in y))
            A, alpha, alpha_err, r2, pred = fit_alpha(r_fit, y, p0_alpha[Q])
            rec = {
                "path": path,
                "length": length,
                "r_min": r_min,
                "r_max": r_max,
                "r": r_fit.tolist(),
                "shell_mean": y.tolist(),
                "A": A,
                "alpha": alpha,
                "alpha_err": alpha_err,
                "R2": r2,
                "ref_alpha": ref_alpha[Q],
                "rel_dev_vs_ref_pct": abs(alpha - ref_alpha[Q]) / ref_alpha[Q] * 100.0,
            }
            fit_results[qname][Q] = rec
            print(
                f"  {qname}: alpha={alpha:.4f}+/-{alpha_err:.4f}  R2={r2:.6f}  "
                f"A={A:.4g}  vs_ref={ref_alpha[Q]:.2f} ({rec['rel_dev_vs_ref_pct']:.1f}%)"
            )

    # plot chi if present else tilt
    plot_key = "chi" if "chi" in fit_results and fit_results["chi"] else quantities[0]
    plt.figure(figsize=(9, 6))
    for Q, rec in sorted(fit_results[plot_key].items()):
        r = np.array(rec["r"])
        y = np.array(rec["shell_mean"])
        pred = power_law(r, rec["A"], rec["alpha"])
        plt.loglog(r, y, "o", markersize=6, label=f"Q={Q} {plot_key}")
        plt.loglog(
            r,
            pred,
            "--",
            lw=2,
            label=f"Q={Q} α={rec['alpha']:.2f}±{rec['alpha_err']:.2f}",
        )
    plt.xlabel("r")
    plt.ylabel(f"shell mean <{plot_key}>")
    plt.title(f"Outer far-field screening progression ({plot_key})")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "step1_screening_alpha_fit.png"
    plt.savefig(fig_path, dpi=150)

    # acceptance on primary chi if available
    primary = "chi" if fit_results.get("chi") else plot_key
    block = fit_results[primary]
    if not block:
        raise SystemExit("No fields loaded.")
    qs = sorted(block)
    alphas = [block[Q]["alpha"] for Q in qs]
    mono = all(alphas[i] < alphas[i + 1] for i in range(len(alphas) - 1))
    r2_ok = all(block[Q]["R2"] > 0.95 for Q in qs)
    print("\n" + "=" * 64)
    print(f"PRIMARY ({primary}) acceptance")
    print(f"monotonic alpha: {'PASS' if mono else 'FAIL'}  {dict(zip(qs, [round(a,3) for a in alphas]))}")
    print(
        f"all R2>0.95:     {'PASS' if r2_ok else 'CHECK'}  "
        + str({Q: round(block[Q]['R2'], 6) for Q in qs})
    )
    print(
        "NOTE: paper ref alphas (~1.04/3.17/5.59/8.02) come from farfield_multipole "
        "windows; <chi> outer-window should recover the same ordering. "
        "Q=1 <chi> may sit below 1 if tilt is O(1) (sin geometry)."
    )
    print("=" * 64)

    save_json(out_dir / "step1_alpha_fit.json", fit_results)
    print(f"saved {fig_path}")
    print(f"saved {out_dir / 'step1_alpha_fit.json'}")


if __name__ == "__main__":
    main()
