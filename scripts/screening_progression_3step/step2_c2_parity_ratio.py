#!/usr/bin/env python3
"""
Step 2 — Odd-m vs even-m power ratio of signed n_x on a far shell.

Writable: for Q>=2, odd-m power >> even-m (significant suppression),
consistent with C2(z) antisymmetry — mechanism side evidence.
Does NOT claim even-m is exactly zero; does NOT use this to separate Q=1 from Q>=2.
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
load_nfield = _common.load_nfield
odd_even_m_power = _common.odd_even_m_power
resolve_field_table = _common.resolve_field_table
save_json = _common.save_json
sh_coeffs_power_by_lm = _common.sh_coeffs_power_by_lm
shell_samples = _common.shell_samples


def main():
    ap = argparse.ArgumentParser(description="Step2: C2 odd/even-m power ratio")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--r-shell", type=float, default=8.0)
    ap.add_argument("--n-points", type=int, default=6000)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--field-q1", type=str, default="")
    ap.add_argument("--field-q2", type=str, default="")
    ap.add_argument("--field-q3", type=str, default="")
    ap.add_argument("--field-q4", type=str, default="")
    args = ap.parse_args()

    table = resolve_field_table(args.abs_outputs)
    overrides = {1: args.field_q1, 2: args.field_q2, 3: args.field_q3, 4: args.field_q4}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    print("=" * 72)
    print(f"STEP 2: signed n_x odd/even-m power ratios at r={args.r_shell}")
    print("  (side evidence for C2(z); no 'exactly zero'; no Q1-vs-Q>=2 discriminator)")
    print("=" * 72)
    header = "Q\t" + "\t".join([f"l={l} odd/even" for l in range(1, args.l_max + 1)])
    print(header)
    print("-" * 72)

    for Q in [1, 2, 3, 4]:
        meta = table[Q]
        path = overrides[Q] or meta["path"]
        if not Path(path).is_file():
            print(f"[SKIP] Q={Q}: missing {path}")
            continue
        # Q=1 box is L=10; r=8 still ok; if r >= length, skip
        length = float(meta["length"])
        r = float(args.r_shell)
        if r >= 0.95 * length:
            # auto clamp for Q=1
            r = 0.80 * length
            print(f"[NOTE] Q={Q}: r clamped to {r:.2f} (L={length})")

        nfield = load_nfield(path)
        nx = nfield[..., 0]
        f, polar, azim = shell_samples(nx, length, r, args.n_points)
        power_lm, _ = sh_coeffs_power_by_lm(f, polar, azim, args.l_max)

        ratios = {}
        row = [str(Q)]
        for l in range(1, args.l_max + 1):
            odd, even = odd_even_m_power(power_lm, l)
            if even < 1e-18:
                ratio = float("inf")
                row.append("inf")
            else:
                ratio = odd / even
                row.append(f"{ratio:.2e}")
            ratios[l] = {
                "odd": odd,
                "even": even,
                "ratio": None if ratio == float("inf") else float(ratio),
            }
        print("\t".join(row))
        results[Q] = {"path": path, "length": length, "r": r, "ratios": ratios}

        # soft check for Q>=2: at least one of l=1..3 has ratio > 10
        if Q >= 2:
            ratios_list = []
            for l in ratios:
                r = ratios[l]["ratio"]
                if r is None:  # inf
                    ratios_list.append(float("inf"))
                else:
                    ratios_list.append(r)
            best = max(ratios_list) if ratios_list else 0.0
            flag = (
                "PASS (>=10x on some l)"
                if best >= 10 or best == float("inf")
                else "CHECK (ratio<10)"
            )
            best_s = "inf" if best == float("inf") else f"{best:.3g}"
            print(f"      Q>=2 odd/even peak={best_s}  {flag}")

    print("=" * 72)
    save_json(out_dir / "step2_c2_parity_ratio.json", results)
    print(f"saved {out_dir / 'step2_c2_parity_ratio.json'}")


if __name__ == "__main__":
    main()
