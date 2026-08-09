#!/usr/bin/env python3
"""
Step 4 — m-spectrum of complex transverse field dn+ = nx + i ny.

Purpose
-------
Falsify / support the angular-matching hypothesis used in the screening
progression narrative:

  H0: far-field power of dn+ is dominated by |m| = Q
      (for even Q under C2(z), the matched |m|=Q is even and forbidden,
       so the *next* odd candidate is |m|=Q+1)

This is the missing non-circular check: it must NOT be inferred from alpha(Q).

Outputs per Q: power vs m, power vs |m|, top (l,m) modes, verdict lines.
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
            f"Cannot find common.py next to this script.\n  expected: {common_path}"
        )
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


def complex_sh_powers(f_complex, polar, azim, l_max: int):
    """
    a_lm = int f Y_lm* dOmega on Fibonacci shell.
    Returns:
      power_lm[(l,m)], power_m[m] (sum over l>=|m|), power_abs_m[|m|]
    """
    npts = len(f_complex)
    dOmega = 4.0 * np.pi / npts
    power_lm = {}
    power_m = {m: 0.0 for m in range(-l_max, l_max + 1)}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Ylm = eval_sph_harm(m, l, polar, azim)
            coeff = np.sum(f_complex * np.conj(Ylm)) * dOmega
            p = float(np.abs(coeff) ** 2)
            power_lm[(l, m)] = p
            power_m[m] += p

    power_abs = {k: 0.0 for k in range(0, l_max + 1)}
    for m, p in power_m.items():
        power_abs[abs(m)] += p
    return power_lm, power_m, power_abs


def top_modes(power_lm, n=8):
    items = sorted(power_lm.items(), key=lambda kv: kv[1], reverse=True)
    return [(int(l), int(m), float(p)) for (l, m), p in items[:n]]


def verdict_for_Q(Q, power_abs, tot):
    """
    Return structured verdict without overclaiming.
    """
    # dominant |m|
    abs_items = sorted(power_abs.items(), key=lambda kv: kv[1], reverse=True)
    m_star = int(abs_items[0][0])
    frac_star = abs_items[0][1] / tot if tot > 0 else 0.0
    frac_Q = power_abs.get(Q, 0.0) / tot if tot > 0 else 0.0
    frac_Qp1 = power_abs.get(Q + 1, 0.0) / tot if tot > 0 else 0.0

    # expected under matching + C2 story
    if Q % 2 == 1:
        expected = Q
        hyp = f"|m|={Q} (odd Q: matching mode odd, C2-allowed)"
    else:
        expected = Q + 1
        hyp = f"|m|={Q}+1={Q+1} (even Q: |m|=Q even forbidden by C2; next odd)"

    # also report naive |m|=Q without C2 lift
    match_Q = m_star == Q
    match_exp = m_star == expected

    if match_exp and frac_star >= 0.35:
        status = "SUPPORT"
    elif match_exp and frac_star >= 0.20:
        status = "WEAK_SUPPORT"
    elif match_Q and Q % 2 == 0:
        status = "CONTRADICT_C2_LIFT"  # even |m|=Q dominates — bad for C2+matching story
    else:
        status = "NOT_SUPPORTED"

    return {
        "m_star": m_star,
        "frac_star": frac_star,
        "frac_Q": frac_Q,
        "frac_Q_plus_1": frac_Qp1,
        "expected_under_C2_matching": expected,
        "hypothesis": hyp,
        "status": status,
    }


def main():
    ap = argparse.ArgumentParser(description="Step4: m-spectrum of dn+=nx+i*ny")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--r-shell", type=float, default=8.0)
    ap.add_argument("--n-points", type=int, default=8000)
    ap.add_argument("--l-max", type=int, default=8)
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
    print("STEP 4: m-spectrum of dn+ = nx + i ny  (angular matching test)")
    print("  Hypothesis: dominant |m| ~ Q (odd Q) or Q+1 (even Q under C2)")
    print("=" * 72)

    for Q in [1, 2, 3, 4]:
        meta = table[Q]
        path = overrides[Q] or meta["path"]
        if not Path(path).is_file():
            print(f"[SKIP] Q={Q}: missing {path}")
            continue
        length = float(meta["length"])
        r = float(args.r_shell)
        if r >= 0.95 * length:
            r = 0.80 * length
            print(f"[NOTE] Q={Q}: r clamped to {r:.2f} (L={length})")

        nfield = load_nfield(path)
        nx = nfield[..., 0]
        ny = nfield[..., 1]
        # sample both on same angles
        nx_s, polar, azim = shell_samples(nx, length, r, args.n_points)
        ny_s, _, _ = shell_samples(ny, length, r, args.n_points)
        f = nx_s + 1j * ny_s

        power_lm, power_m, power_abs = complex_sh_powers(f, polar, azim, args.l_max)
        tot = float(sum(power_abs.values())) + 1e-30
        tops = top_modes(power_lm, n=8)
        verd = verdict_for_Q(Q, power_abs, tot)

        # odd/even m totals
        odd = sum(p for m, p in power_m.items() if m % 2 != 0)
        even = sum(p for m, p in power_m.items() if m % 2 == 0)

        print(f"\n--- Q={Q}  r={r:.2f}  L={length}  {Path(path).name} ---")
        print(f"  odd-m power frac  = {odd/tot:.4f}")
        print(f"  even-m power frac = {even/tot:.4f}")
        print("  power by |m| (frac of total):")
        for k in range(0, args.l_max + 1):
            frac = power_abs[k] / tot
            mark = ""
            if k == Q:
                mark += "  <- Q"
            if k == Q + 1:
                mark += "  <- Q+1"
            if k == verd["m_star"]:
                mark += "  <- DOMINANT"
            if frac >= 0.01 or k <= Q + 1:
                print(f"    |m|={k:2d}: {frac:.4f}{mark}")
        print("  top (l,m) modes:")
        for l, m, p in tops[:6]:
            print(f"    (l={l}, m={m:+d}): P={p:.4e}  frac={p/tot:.4f}")
        print(f"  hypothesis: {verd['hypothesis']}")
        print(
            f"  dominant |m|={verd['m_star']} (frac={verd['frac_star']:.3f})  "
            f"frac(|m|=Q)={verd['frac_Q']:.3f}  frac(|m|=Q+1)={verd['frac_Q_plus_1']:.3f}"
        )
        print(f"  VERDICT: {verd['status']}")

        results[Q] = {
            "path": path,
            "r": r,
            "length": length,
            "l_max": args.l_max,
            "power_abs_m": {str(k): float(v) for k, v in power_abs.items()},
            "power_m": {str(k): float(v) for k, v in power_m.items()},
            "top_lm": [{"l": l, "m": m, "P": p} for l, m, p in tops],
            "odd_frac": odd / tot,
            "even_frac": even / tot,
            "verdict": verd,
        }

    print("\n" + "=" * 72)
    print("SUMMARY (angular matching)")
    print("-" * 72)
    for Q in sorted(results):
        v = results[Q]["verdict"]
        print(
            f"Q={Q}: dominant|m|={v['m_star']}  expect={v['expected_under_C2_matching']}  "
            f"status={v['status']}"
        )
    print("=" * 72)
    print(
        "Interpretation guide:\n"
        "  SUPPORT / WEAK_SUPPORT : matching(+C2 lift) hypothesis OK at this shell\n"
        "  NOT_SUPPORTED          : do NOT claim |m|=Q drives alpha(Q)\n"
        "  CONTRADICT_C2_LIFT     : even |m|=Q dominates for even Q — check C2 premise\n"
        "C2 even-m forbid can still hold even if |m|=Q matching fails."
    )

    save_json(out_dir / "step4_m_spectrum.json", results)
    print(f"saved {out_dir / 'step4_m_spectrum.json'}")


if __name__ == "__main__":
    main()
