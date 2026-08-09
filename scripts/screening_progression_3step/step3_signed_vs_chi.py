#!/usr/bin/env python3
"""
Step 3 — Clarify signed n_x vs positive chi multipole content (Q=2 default).

Writable: chi is positive-definite => natural l=0; screening is read from the
radial exponent of <chi>, NOT from whether chi's l=0 amplitude exists.
Signed n_x can have near-noise l=0 under symmetry.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


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
sh_coeffs_power_by_lm = _common.sh_coeffs_power_by_lm
shell_samples = _common.shell_samples


def main():
    ap = argparse.ArgumentParser(description="Step3: signed nx vs positive chi SH")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--r-shell", type=float, default=8.0)
    ap.add_argument("--n-points", type=int, default=6000)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--field", type=str, default="", help="override nfield path")
    args = ap.parse_args()

    table = resolve_field_table(args.abs_outputs)
    meta = table[args.Q]
    path = args.field or meta["path"]
    if not Path(path).is_file():
        raise SystemExit(f"missing field: {path}")

    length = float(meta["length"])
    r = float(args.r_shell)
    if r >= 0.95 * length:
        r = 0.80 * length
        print(f"[NOTE] r clamped to {r:.2f} (L={length})")

    nfield = load_nfield(path)
    chi = chi_from_nfield(nfield)
    nx = nfield[..., 0]

    nx_f, polar, azim = shell_samples(nx, length, r, args.n_points)
    chi_f, _, _ = shell_samples(chi, length, r, args.n_points)

    _, p_nx = sh_coeffs_power_by_lm(nx_f, polar, azim, args.l_max)
    _, p_chi = sh_coeffs_power_by_lm(chi_f, polar, azim, args.l_max)

    # normalize by max for display (as in user's draft)
    p_nx_n = p_nx / (np.max(p_nx) + 1e-30)
    p_chi_n = p_chi / (np.max(p_chi) + 1e-30)

    print("=" * 64)
    print(f"STEP 3: Q={args.Q}  r={r}  signed n_x vs positive chi")
    print("  chi l=0 is expected (positive definite); NOT a monopole-forbid test")
    print("=" * 64)
    print("l\t n_x power\t n_x frac_max\t chi power\t chi frac_max")
    print("-" * 64)
    for l in range(args.l_max + 1):
        print(
            f"{l}\t {p_nx[l]:.6e}\t {p_nx_n[l]:.6f}\t "
            f"{p_chi[l]:.6e}\t {p_chi_n[l]:.6f}"
        )
    print("=" * 64)
    print(
        f"CHECK: n_x l=0 / max(n_x) = {p_nx_n[0]:.3e}  (expect ~noise for symmetric far field)"
    )
    print(
        f"CHECK: chi l=0 / max(chi) = {p_chi_n[0]:.3f}  (expect O(1); natural for chi>=0)"
    )

    out = {
        "Q": args.Q,
        "path": path,
        "r": r,
        "length": length,
        "power_nx": p_nx.tolist(),
        "power_chi": p_chi.tolist(),
        "norm_nx": p_nx_n.tolist(),
        "norm_chi": p_chi_n.tolist(),
    }
    out_dir = Path(args.out_dir)
    save_json(out_dir / f"step3_signed_vs_chi_Q{args.Q}.json", out)
    print(f"saved {out_dir / f'step3_signed_vs_chi_Q{args.Q}.json'}")


if __name__ == "__main__":
    main()
