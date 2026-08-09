#!/usr/bin/env python3
"""
验证1：响应核 K、M 的空间分布 + 与有限差 δb/α 的交叉检验。

不验证绝热输运假设；只检验「给定 δn=α θ̃ θ̂ 时，解析核 vs FD」。

用法（GPU）:
  python verify1_kernel_maps.py --abs-outputs --Q 1 --out-dir /tmp/k_ex_v1

本地无场时可:
  python verify1_kernel_maps.py --synthetic --out-dir ./out_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from berry_kernels import (
    apply_kernel,
    berry_curvature,
    finite_diff_db_dalpha,
    grid_axes,
    grid_spacing,
    response_kernels,
    synthetic_bp_skyrmion,
)


def _load_screening_common():
    common_path = (
        _HERE.parent / "screening_progression_3step" / "common.py"
    )
    if not common_path.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("screening_common", common_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def make_theta_tilde(shape, length, mode: str, soft: float):
    n = shape[0]
    ax = grid_axes(n, length)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    if mode == "soft_monopole":
        # regularized 1/r — smooth, nonzero ∇
        return 1.0 / np.sqrt(R * R + soft * soft)
    if mode == "gaussian":
        return np.exp(-(R * R) / (2.0 * soft * soft))
    if mode == "linear_x":
        return X.copy()
    raise ValueError(mode)


def interior_mask(n: int, margin: int):
    m = np.zeros((n, n, n), dtype=bool)
    m[margin : n - margin, margin : n - margin, margin : n - margin] = True
    return m


def main():
    ap = argparse.ArgumentParser(description="V1: K/M maps + FD cross-check")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument(
        "--theta-mode",
        type=str,
        default="soft_monopole",
        choices=["soft_monopole", "gaussian", "linear_x"],
    )
    ap.add_argument("--soft", type=float, default=2.0, help="θ̃ soft length")
    ap.add_argument("--margin", type=int, default=2, help="drop FD edge layers")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        nfield, length = synthetic_bp_skyrmion()
        path = "synthetic_bp_skyrmion"
        print(f"SYNTHETIC BP skyrmion  N={nfield.shape[0]}  L={length}")
    else:
        common = _load_screening_common()
        if common is None:
            raise SystemExit("need screening_progression_3step/common.py or --synthetic")
        table = common.resolve_field_table(args.abs_outputs)
        meta = table[args.Q]
        path = args.field or meta["path"]
        length = float(meta["length"])
        if not Path(path).is_file():
            raise SystemExit(f"missing field: {path}\n(use --synthetic for local smoke test)")
        nfield = common.load_nfield(path)
        print(f"Q={args.Q}  L={length}  N={nfield.shape[0]}\n  field: {path}")

    n = nfield.shape[0]
    h = grid_spacing(n, length)
    b = berry_curvature(nfield, h)
    K, M, _th = response_kernels(nfield, h)

    # spatial diagnostics on |K|, |M|, |b|
    absK = np.linalg.norm(K, axis=-1)
    absM = np.linalg.norm(M.reshape(n, n, n, 9), axis=-1)
    absB = np.linalg.norm(b, axis=-1)

    def peak_and_shell(vol, r):
        ax = grid_axes(n, length)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        R = np.sqrt(X * X + Y * Y + Z * Z)
        peak = float(np.max(vol))
        band = (R > r - 0.5 * h) & (R < r + 0.5 * h)
        shell = float(np.mean(vol[band])) if np.any(band) else float("nan")
        return peak, shell

    print("=" * 72)
    print("VERIFY1: response-kernel maps (corrected K, M)")
    print("=" * 72)
    for r in [0.0, 2.0, 4.0, 6.0, 8.0]:
        if r > 0.9 * length:
            continue
        pk, sh = peak_and_shell(absK, r if r > 0 else 0.5)
        pm, sm = peak_and_shell(absM, r if r > 0 else 0.5)
        pb, sb = peak_and_shell(absB, r if r > 0 else 0.5)
        if r == 0.0:
            print(f"  max|K|={pk:.4e}  max|M|={pm:.4e}  max|b|={pb:.4e}")
        else:
            print(
                f"  r~{r:.1f}  <|K|>={sh:.4e}  <|M|>={sm:.4e}  <|b|>={sb:.4e}"
            )

    # FD cross-check
    theta = make_theta_tilde(nfield.shape, length, args.theta_mode, args.soft)
    gth = (
        np.gradient(theta, h, axis=0, edge_order=2),
        np.gradient(theta, h, axis=1, edge_order=2),
        np.gradient(theta, h, axis=2, edge_order=2),
    )
    db_an = apply_kernel(K, M, theta, gth)
    db_fd = finite_diff_db_dalpha(nfield, h, theta, alpha=args.alpha)

    mask = interior_mask(n, args.margin)
    # weight by |θ̃|+|∇θ̃| to focus where signal lives
    w = np.abs(theta) + np.abs(gth[0]) + np.abs(gth[1]) + np.abs(gth[2])
    w = np.where(mask, w, 0.0)
    # relative L2 on vector field
    diff = db_an - db_fd
    num = np.sqrt(np.sum((diff[mask] ** 2) * w[mask, None]))
    den = np.sqrt(np.sum((db_fd[mask] ** 2) * w[mask, None])) + 1e-30
    rel = float(num / den)
    # also unweighted
    num_u = np.linalg.norm(diff[mask])
    den_u = np.linalg.norm(db_fd[mask]) + 1e-30
    rel_u = float(num_u / den_u)

    print("-" * 72)
    print(
        f"FD cross-check  theta_mode={args.theta_mode}  alpha={args.alpha}  soft={args.soft}"
    )
    print(f"  relL2(weighted)   = {rel:.4e}")
    print(f"  relL2(unweighted) = {rel_u:.4e}")
    if rel < 0.05:
        verdict = "PASS"
    elif rel < 0.15:
        verdict = "WEAK_PASS"
    else:
        verdict = "FAIL"
    print(f"VERDICT: {verdict}  (kernel vs FD; NOT adiabatic hypothesis)")
    print("=" * 72)

    # save mid-plane maps for quick look
    mid = n // 2
    np.savez_compressed(
        out_dir / "verify1_kernel_maps.npz",
        absK_xy=absK[:, :, mid],
        absM_xy=absM[:, :, mid],
        absB_xy=absB[:, :, mid],
        absK_xz=absK[:, mid, :],
        length=length,
        h=h,
    )
    summary = {
        "path": path,
        "length": length,
        "N": n,
        "h": h,
        "theta_mode": args.theta_mode,
        "alpha": args.alpha,
        "relL2_weighted": rel,
        "relL2_unweighted": rel_u,
        "verdict": verdict,
        "max_absK": float(np.max(absK)),
        "max_absM": float(np.max(absM)),
        "max_absB": float(np.max(absB)),
        "note": (
            "PASS only means linearized δb under polar tilt matches K,M. "
            "Does NOT validate config-space Berry / Hopf exchange."
        ),
    }
    with open(out_dir / "verify1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir / 'verify1_summary.json'}")
    print(f"saved {out_dir / 'verify1_kernel_maps.npz'}")


if __name__ == "__main__":
    main()
