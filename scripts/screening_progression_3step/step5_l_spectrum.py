#!/usr/bin/env python3
"""
Step 5 — l-spectrum inside the fixed |m|=1 sector of dn+ = nx + i ny.

Hypothesis (post step4):
  For Q>=2, power sits in |m|=1; screening progression may be
  elevation of effective l within that sector (not climbing |m|).

Reuses the same shell sampling + SH convention as step4_m_spectrum.py.
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
        raise SystemExit(f"Cannot find common.py next to this script.\n  expected: {common_path}")
    spec = importlib.util.spec_from_file_location("screening_common", common_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_common = _load_common()
eval_sph_harm = _common.eval_sph_harm
load_nfield = _common.load_nfield
resolve_field_table = _common.resolve_field_table
save_json = _common.save_json
shell_samples = _common.shell_samples


def complex_sh_power_lm(f_complex, polar, azim, l_max: int):
    npts = len(f_complex)
    dOmega = 4.0 * np.pi / npts
    power_lm = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Ylm = eval_sph_harm(m, l, polar, azim)
            coeff = np.sum(f_complex * np.conj(Ylm)) * dOmega
            power_lm[(l, m)] = float(np.abs(coeff) ** 2)
    return power_lm


def analyze_l_in_m_sector(power_lm, l_max: int, target_m_abs: int = 1):
    """Aggregate P(l) for |m|=target_m_abs; return dist + stats."""
    l_power = np.zeros(l_max + 1, dtype=np.float64)
    for (l, m), p in power_lm.items():
        if abs(m) == target_m_abs:
            l_power[l] += p

    tot_all = float(sum(power_lm.values())) + 1e-30
    tot_m = float(np.sum(l_power)) + 1e-30
    l_dist = {int(l): float(l_power[l] / tot_m) for l in range(l_max + 1)}

    ls = np.arange(l_max + 1, dtype=np.float64)
    pn = l_power / tot_m
    mean_l = float(np.sum(ls * pn))
    std_l = float(np.sqrt(np.sum((ls - mean_l) ** 2 * pn)))
    peak_l = int(np.argmax(pn))

    # cumulative from low l: smallest L with sum_{l<=L} P >= 0.9
    cum = np.cumsum(pn)
    l90_low = int(np.searchsorted(cum, 0.9))
    l90_low = min(l90_low, l_max)

    return l_dist, {
        "mean_l": mean_l,
        "std_l": std_l,
        "peak_l": peak_l,
        "l90_from_low": l90_low,
        "sector_power": float(tot_m - 1e-30),
        "sector_frac_of_total": float(tot_m / tot_all),
        "l_power_raw": {str(l): float(l_power[l]) for l in range(l_max + 1)},
    }


def main():
    ap = argparse.ArgumentParser(description="Step5: l-spectrum in fixed |m| sector")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument(
        "--r-shells",
        type=str,
        default="8",
        help="comma-separated radii, e.g. 8,9.5",
    )
    ap.add_argument("--l-max", type=int, default=8)
    ap.add_argument("--n-points", type=int, default=8000)
    ap.add_argument("--target-m-abs", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--field-q1", type=str, default="")
    ap.add_argument("--field-q2", type=str, default="")
    ap.add_argument("--field-q3", type=str, default="")
    ap.add_argument("--field-q4", type=str, default="")
    args = ap.parse_args()

    table = resolve_field_table(args.abs_outputs)
    overrides = {1: args.field_q1, 2: args.field_q2, 3: args.field_q3, 4: args.field_q4}
    r_shells = [float(x) for x in args.r_shells.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"STEP 5: l-spectrum in fixed |m|={args.target_m_abs} sector")
    print("  Hypothesis: mean_l / peak_l increase with Q (m fixed)")
    print("=" * 72)

    all_results = {}
    for r0 in r_shells:
        r_key = f"r{r0:.2f}"
        all_results[r_key] = {}
        print(f"\n{'=' * 72}\n  requested r = {r0:.2f}\n{'=' * 72}")

        for Q in [1, 2, 3, 4]:
            meta = table[Q]
            path = overrides[Q] or meta["path"]
            if not Path(path).is_file():
                print(f"[SKIP] Q={Q}: missing {path}")
                continue
            length = float(meta["length"])
            r = float(r0)
            if r >= 0.95 * length:
                r = 0.80 * length
                print(f"[NOTE] Q={Q}: r clamped to {r:.2f} (L={length})")

            nfield = load_nfield(path)
            nx_s, polar, azim = shell_samples(nfield[..., 0], length, r, args.n_points)
            ny_s, _, _ = shell_samples(nfield[..., 1], length, r, args.n_points)
            f = nx_s + 1j * ny_s

            power_lm = complex_sh_power_lm(f, polar, azim, args.l_max)
            l_dist, stats = analyze_l_in_m_sector(power_lm, args.l_max, args.target_m_abs)

            print(f"\n--- Q={Q}  r={r:.2f}  |m|={args.target_m_abs}  {Path(path).name} ---")
            print(f"  sector frac of total dn+ power = {stats['sector_frac_of_total']:.4f}")
            print(f"  peak_l = {stats['peak_l']}  (frac={l_dist[stats['peak_l']]:.4f})")
            print(f"  mean_l = {stats['mean_l']:.3f} ± {stats['std_l']:.3f}")
            print(f"  l90_from_low (cum P from small l) = {stats['l90_from_low']}")
            print("  P(l | |m|=target) top:")
            for l, frac in sorted(l_dist.items(), key=lambda kv: -kv[1])[:6]:
                if frac > 1e-4:
                    print(f"    l={l:2d}: {frac:.4f}")

            all_results[r_key][Q] = {
                "path": path,
                "r_used": r,
                "length": length,
                "l_dist": l_dist,
                **stats,
            }

    print("\n" + "=" * 72)
    print(f"SUMMARY  mean_l vs Q  (|m|={args.target_m_abs})")
    print("-" * 72)
    header = f"{'r':>8s}" + "".join(f"{'Q='+str(q):>10s}" for q in [1, 2, 3, 4])
    print(header)
    for r0 in r_shells:
        r_key = f"r{r0:.2f}"
        row = f"{r0:8.2f}"
        for Q in [1, 2, 3, 4]:
            if Q in all_results.get(r_key, {}):
                row += f"{all_results[r_key][Q]['mean_l']:10.3f}"
            else:
                row += f"{'N/A':>10s}"
        print(row)

    print("\n" + "=" * 72)
    print("HYPOTHESIS VERDICT  (l-elevation with Q, fixed |m|)")
    print("-" * 72)
    for r0 in r_shells:
        r_key = f"r{r0:.2f}"
        block = all_results.get(r_key, {})
        qs = sorted(block)
        if len(qs) < 2:
            continue
        means = [block[Q]["mean_l"] for Q in qs]
        peaks = [block[Q]["peak_l"] for Q in qs]
        mono_mean = all(means[i] < means[i + 1] for i in range(len(means) - 1))
        mono_peak = all(peaks[i] <= peaks[i + 1] for i in range(len(peaks) - 1)) and (
            peaks[-1] > peaks[0]
        )
        # Q>=2 only (Q1 may sit in different symmetry class)
        qs2 = [Q for Q in qs if Q >= 2]
        mono_mean_ge2 = False
        if len(qs2) >= 2:
            m2 = [block[Q]["mean_l"] for Q in qs2]
            mono_mean_ge2 = all(m2[i] < m2[i + 1] for i in range(len(m2) - 1))

        if mono_mean:
            verdict = "SUPPORT"
        elif mono_mean_ge2:
            verdict = "SUPPORT_Qge2"
        else:
            verdict = "NOT_MONOTONIC"

        print(
            f"  r={r0:.2f}: mean_l {dict(zip(qs, [round(x,3) for x in means]))}  "
            f"peak_l {dict(zip(qs, peaks))}  [{verdict}]"
        )
        print(
            f"         mono_mean_allQ={mono_mean}  mono_mean_Qge2={mono_mean_ge2}  "
            f"peak_nondecreasing_up={mono_peak}"
        )
    print("=" * 72)
    print(
        "Notes:\n"
        "  SUPPORT       : mean_l rises Q1→Q4\n"
        "  SUPPORT_Qge2  : rises on Q=2,3,4 (enough for screening progression among screened)\n"
        "  NOT_MONOTONIC : do not claim l-elevation mechanism yet\n"
        "  sector_frac≈1 : consistent with step4 (|m|=1 dominance)"
    )

    # JSON keys as strings for Q
    dump = {}
    for r_key, block in all_results.items():
        dump[r_key] = {str(Q): rec for Q, rec in block.items()}
    save_json(out_dir / "step5_l_spectrum.json", dump)
    print(f"saved {out_dir / 'step5_l_spectrum.json'}")


if __name__ == "__main__":
    main()
