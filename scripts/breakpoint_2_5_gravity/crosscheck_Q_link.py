#!/usr/bin/env python3
"""
Topology cross-check for BP2.5 input fields: Q_link vs Q_fft.

This is NOT a gravity / G_c^(κ) diagnostic. It only verifies that the
unit field still carries the expected Hopf charge before Poisson / κ scans.

Usage:
  python crosscheck_Q_link.py --field /path/to/nfield.npy --length 10
  python crosscheck_Q_link.py --abs-outputs --Q 1
  python crosscheck_Q_link.py --synthetic   # analytic Hopf, no external file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_LINK = _REPO / "scripts" / "hopf_linking"
_SRC = _REPO / "src"
_SCREEN = _REPO / "scripts" / "screening_progression_3step"

for p in (_HERE, _LINK, _SRC, _SCREEN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hopf_linking import compute_hopf_invariant, grid_coords  # noqa: E402
from hopf_skyrme import hopf_charge_fft  # noqa: E402


def _load_nfield(path: Path) -> np.ndarray:
    n = np.load(path)
    if n.ndim == 4 and n.shape[0] == 3:
        n = np.moveaxis(n, 0, -1)
    if n.shape[-1] != 3:
        raise SystemExit(f"expected (...,3) nfield, got {n.shape}")
    return np.asarray(n, dtype=np.float64)


def _analytic_field(N: int, L: float, grid_mode: str) -> np.ndarray:
    from test_hopf_analytic import generate_hopf_field

    return generate_hopf_field(N=N, L=L, grid_mode=grid_mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", type=str, default=None)
    ap.add_argument("--length", type=float, default=None)
    ap.add_argument("--grid-mode", choices=("half", "side"), default="half")
    ap.add_argument("--Q", type=int, default=1, help="canonical field Q when using table")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=64, help="synthetic grid size")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.synthetic:
        L = float(args.length if args.length is not None else (4.0 if args.grid_mode == "half" else 8.0))
        n_field = _analytic_field(args.n, L, args.grid_mode)
        field_path = "<synthetic_hopf>"
    elif args.field:
        L = float(args.length if args.length is not None else 10.0)
        field_path = args.field
        n_field = _load_nfield(Path(args.field))
    else:
        try:
            import common as screen_common
        except ImportError as exc:
            raise SystemExit(
                "need --field PATH, --synthetic, or screening common.py + --abs-outputs"
            ) from exc
        table = screen_common.resolve_field_table(args.abs_outputs)
        meta = table[int(args.Q)]
        field_path = meta["path"]
        L = float(args.length if args.length is not None else meta["length"])
        n_field = _load_nfield(Path(field_path))

    N = n_field.shape[0]
    h, _ = grid_coords(N, L, grid_mode=args.grid_mode)
    q_fft = float(hopf_charge_fft(n_field, h))
    result, _g1, _g2 = compute_hopf_invariant(
        n_field, L, grid_mode=args.grid_mode
    )
    q_link = result.get("Q_link")
    link_raw = result.get("link_raw")

    same_sign = (
        q_link is not None
        and np.sign(q_link) == np.sign(q_fft)
        and abs(q_fft) > 0.1
    )
    near_int = q_link is not None and abs(q_fft - q_link) < 0.25
    verdict = "PASS" if (same_sign and near_int and result.get("status", "").startswith("OK")) else "CHECK"

    report = {
        "field": str(field_path),
        "N": N,
        "length": L,
        "grid_mode": args.grid_mode,
        "h": h,
        "Q_fft": q_fft,
        "Q_link": q_link,
        "link_raw": link_raw,
        "rounding_error": result.get("rounding_error"),
        "status_link": result.get("status"),
        "same_sign": bool(same_sign),
        "near_integer_match": bool(near_int),
        "verdict": verdict,
        "note": (
            "Topology only — not BP2.5 G_c(kappa) / Poisson gravity. "
            "See scripts/hopf_linking/README.md"
        ),
    }

    print("=== BP2.5 topology cross-check (Q_link vs Q_fft) ===")
    print(f"  field:     {field_path}")
    print(f"  grid:      N={N}, L={L}, mode={args.grid_mode}, h={h:.6g}")
    print(f"  Q_fft:     {q_fft:.6f}")
    print(f"  Q_link:    {q_link}")
    print(f"  link_raw:  {link_raw}")
    print(f"  status:    {result.get('status')}")
    print(f"  verdict:   {verdict}")
    print(f"  note:      {report['note']}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  wrote:     {out}")

    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
