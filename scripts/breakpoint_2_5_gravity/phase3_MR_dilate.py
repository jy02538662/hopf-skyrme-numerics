#!/usr/bin/env python3
"""
断点 2.5 阶段3：真 Derrick 空间拉伸 M(R)。

对已弛豫参考场 n_ref 做
  n_λ(x) = n_ref(x/λ)   （盒外填真空）
在同一物理网格上直接算 E（不弛豫）。连续极限下应有
  E2(λ)=λ E2(1),  E4(λ)=E4(1)/λ,  E=aλ+b/λ 呈 U 型。

能量用标准 ∫… h³ —— 雅可比已含在物理坐标积分中，不再额外乘因子。

用法:
  # 填盒场必须先真空垫高再扫 λ
  python phase3_MR_dilate.py --abs-outputs --Q 1 --device cuda \\
    --embed-length 20 \\
    --lambda-list 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0 \\
    --out-dir /tmp/bp25_MR_dilate_pad
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# reuse torch discovery + fits + G_c from scale-scan module
import phase3_MR_scale_scan as mr  # noqa: E402
import hopf_skyrme_torch as hst  # noqa: E402


def _load_screening_common():
    candidates = [
        _HERE / "common.py",
        _HERE.parent / "screening_progression_3step" / "common.py",
    ]
    cur = _HERE
    for _ in range(6):
        candidates.append(cur / "scripts" / "screening_progression_3step" / "common.py")
        if cur.parent == cur:
            break
        cur = cur.parent
    for common_path in candidates:
        if common_path.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("screening_common", common_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    return None


def load_reference(args):
    common = _load_screening_common()
    table = common.resolve_field_table(args.abs_outputs) if common is not None else None

    if args.dilate_field:
        path = args.dilate_field
        length = float(args.length)
        if table is not None and args.Q in table and not args.length_forced:
            # adopt table L when using canonical Q unless user forces --length
            if abs(args.length - 10.0) < 1e-12:
                length = float(table[args.Q]["length"])
    else:
        if table is None:
            raise SystemExit("need --dilate-field PATH or screening common.py + --abs-outputs")
        if args.Q not in table:
            raise SystemExit(f"Q={args.Q} not in field table")
        path = table[args.Q]["path"]
        length = float(table[args.Q]["length"])
        if args.length_forced:
            length = float(args.length)

    if not Path(path).is_file():
        raise SystemExit(f"missing reference field: {path}")
    n_ref = np.load(path).astype(np.float64)
    if n_ref.ndim != 4 or n_ref.shape[-1] != 3:
        raise SystemExit(f"bad nfield shape {n_ref.shape}")
    norm = np.linalg.norm(n_ref, axis=-1, keepdims=True)
    n_ref = n_ref / np.clip(norm, 1e-12, None)
    return n_ref, length, path


def embed_in_larger_box(
    n_ref: np.ndarray, length_src: float, length_dst: float, n_dst: int
) -> tuple[np.ndarray, float]:
    """
    Re-sample reference into a larger vacuum-padded box (same physical units).
    Needed when R_rms ~ L_src (box-filling); Derrick on a tight box creates fake walls.
    """
    from scipy.interpolate import RegularGridInterpolator

    if length_dst < length_src - 1e-12:
        raise SystemExit(f"--embed-length ({length_dst}) must be >= source L ({length_src})")
    if n_dst < 8:
        raise SystemExit("--embed-n too small")
    n_src = n_ref.shape[0]
    ax_s = np.linspace(-length_src, length_src, n_src)
    ax_d = np.linspace(-length_dst, length_dst, n_dst)
    vacuum = (0.0, 0.0, 1.0)
    interps = [
        RegularGridInterpolator(
            (ax_s, ax_s, ax_s),
            n_ref[..., c],
            bounds_error=False,
            fill_value=vacuum[c],
            method="linear",
        )
        for c in range(3)
    ]
    X, Y, Z = np.meshgrid(ax_d, ax_d, ax_d, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    out = np.stack([interps[c](pts).reshape(n_dst, n_dst, n_dst) for c in range(3)], axis=-1)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.clip(norm, 1e-12, None)
    n0 = np.array(vacuum, dtype=out.dtype)
    out[0] = out[-1] = out[:, 0] = out[:, -1] = out[:, :, 0] = out[:, :, -1] = n0
    return out.astype(np.float64), float(length_dst)


def dilate_nfield(n_ref: np.ndarray, length: float, lam: float) -> np.ndarray:
    """n_λ(x) = n_ref(x/λ); outside ref box → vacuum (0,0,1)."""
    from scipy.interpolate import RegularGridInterpolator

    if lam <= 0:
        raise ValueError("lambda must be > 0")
    n = n_ref.shape[0]
    ax = np.linspace(-length, length, n)
    vacuum = (0.0, 0.0, 1.0)
    interps = [
        RegularGridInterpolator(
            (ax, ax, ax),
            n_ref[..., c],
            bounds_error=False,
            fill_value=vacuum[c],
            method="linear",
        )
        for c in range(3)
    ]
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([(X / lam).ravel(), (Y / lam).ravel(), (Z / lam).ravel()], axis=-1)
    out = np.stack([interps[c](pts).reshape(n, n, n) for c in range(3)], axis=-1)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.clip(norm, 1e-12, None)
    n0 = np.array(vacuum, dtype=out.dtype)
    out[0, :, :, :] = n0
    out[-1, :, :, :] = n0
    out[:, 0, :, :] = n0
    out[:, -1, :, :] = n0
    out[:, :, 0, :] = n0
    out[:, :, -1, :] = n0
    return out.astype(np.float64)


def derrick_diagnostics(rows: list[dict]) -> dict:
    """Check E2/λ ~ const, E4*λ ~ const; locate E minimum."""
    lams = np.array([r["lambda"] for r in rows], dtype=float)
    E = np.array([r["E"] for r in rows], dtype=float)
    E2 = np.array([r["E2"] for r in rows], dtype=float)
    E4 = np.array([r["E4"] for r in rows], dtype=float)
    Q = np.array([r["Q_fft"] for r in rows], dtype=float)
    e2_s = E2 / lams
    e4_s = E4 * lams
    imin = int(np.argmin(E))
    # relative spread of scaled invariants (exclude endpoints optionally)
    def rel_spread(x):
        m = float(np.mean(x))
        return float(np.std(x) / (abs(m) + 1e-30))

    return {
        "E_min": float(E[imin]),
        "lambda_at_E_min": float(lams[imin]),
        "E2_over_E4_at_min": float(E2[imin] / (E4[imin] + 1e-30)),
        "Q_at_E_min": float(Q[imin]),
        "E2_over_lambda_mean": float(np.mean(e2_s)),
        "E2_over_lambda_rel_std": rel_spread(e2_s),
        "E4_times_lambda_mean": float(np.mean(e4_s)),
        "E4_times_lambda_rel_std": rel_spread(e4_s),
        "Q_abs_min": float(np.min(np.abs(Q))),
        "Q_abs_max": float(np.max(np.abs(Q))),
        "has_interior_minimum": bool(0 < imin < len(lams) - 1),
        "note": (
            "Continuum Derrick: E2/λ and E4*λ constant; U-min near E2≈E4. "
            "Finite box + vacuum fill break exact invariance for λ far from 1."
        ),
    }


def default_lambda_list():
    return [round(0.5 + 0.1 * i, 10) for i in range(16)]  # 0.5 .. 2.0


def main():
    ap = argparse.ArgumentParser(description="BP2.5 phase3 Derrick dilation M(R)")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument(
        "--dilate-field",
        type=str,
        default="",
        help="reference nfield .npy; default: table path for --Q",
    )
    ap.add_argument("--length", type=float, default=10.0, help="box half-length (override)")
    ap.add_argument(
        "--force-length",
        action="store_true",
        dest="length_forced",
        help="force --length even when table has metadata",
    )
    ap.add_argument(
        "--lambda-list",
        type=str,
        default="",
        dest="lambda_list",
        help="comma list; default 0.5..2.0 step 0.1",
    )
    ap.add_argument(
        "--embed-length",
        type=float,
        default=-1.0,
        dest="embed_length",
        help="pad into larger box half-length before dilation (e.g. 20). <0 = off",
    )
    ap.add_argument(
        "--embed-n",
        type=int,
        default=-1,
        dest="embed_n",
        help="grid N after embed; <0 => keep source h (~ 2*L_embed/h_src+1)",
    )
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--G-ref", type=float, default=1e-3, dest="g_ref")
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    n_ref, length, path = load_reference(args)
    n_src = n_ref.shape[0]
    h_src = 2.0 * length / (n_src - 1)

    # diagnose box-filling before optional embed
    rad0 = mr.radius_metrics(n_ref, length)
    print("=" * 72)
    print("BP2.5 PHASE3: Derrick dilation  n(x) -> n(x/λ)")
    print(f"  torch: {mr._TORCH_PATH}")
    print(f"  ref: {path}")
    print(
        f"  src N={n_src} L={length} h={h_src:.4f}  "
        f"R_rms={rad0['R_rms_chi2']:.4f}  R_rms/L={rad0['R_rms_chi2']/length:.3f}"
    )
    if rad0["R_rms_chi2"] / length > 0.55 and args.embed_length <= 0:
        print(
            "  WARNING: R_rms/L > 0.55 (box-filling). Derrick on this box is unreliable.\n"
            "           Re-run with e.g. --embed-length 20 (vacuum pad)."
        )

    if args.embed_length > 0:
        L_dst = float(args.embed_length)
        if args.embed_n > 0:
            n_dst = int(args.embed_n)
        else:
            n_dst = int(round(2.0 * L_dst / h_src)) + 1
        n_ref, length = embed_in_larger_box(n_ref, length, L_dst, n_dst)
        print(f"  EMBEDDED -> N={n_ref.shape[0]} L={length} h={2*length/(n_ref.shape[0]-1):.4f}")

    n = n_ref.shape[0]
    h = 2.0 * length / (n - 1)
    lams = (
        [float(x) for x in args.lambda_list.split(",") if x.strip()]
        if args.lambda_list.strip()
        else default_lambda_list()
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dtype = torch.float32

    print(f"  work N={n} L={length} h={h:.4f}  Q_tag={args.Q}")
    print(f"  lambdas={lams}")
    print(f"  device={device}  NO relax; energy on dilated field only")
    print("  Jacobian: included via physical-grid ∫ h³ (no extra factor)")
    print("=" * 72)
    # baseline λ=1 energy for sanity
    u1 = torch.tensor(n_ref, device=device, dtype=dtype)
    e2_1, e4_1, e_1 = hst.energy_parts_float(hst.normalize(u1), h, args.a, args.b)
    q_1 = hst.hopf_charge_cpu_fft(hst.normalize(u1), h)
    print(f"  ref λ=1: E={e_1:.4f} E2={e2_1:.4f} E4={e4_1:.4f} Q={q_1:.4f}")

    rows = []
    t0 = time.time()
    n_sat = 0
    for lam in lams:
        print("-" * 72)
        print(f"lambda={lam:.3g}  dilating ...", flush=True)
        n_np = dilate_nfield(n_ref, length, lam)
        u = torch.tensor(n_np, device=device, dtype=dtype)
        with torch.no_grad():
            nf = hst.normalize(u)
            e2, e4, etot = hst.energy_parts_float(nf, h, args.a, args.b)
            q = hst.hopf_charge_cpu_fft(nf, h)
            n_np = nf.detach().cpu().numpy()
        grav = mr.measure_gc(n_np, length, etot, args.g_ref, args.kappa0, args.c2)
        if grav.get("R_chi_saturated"):
            n_sat += 1
        Rp = grav["R_primary"]
        row = {
            "lambda": lam,
            "scale_as_R": lam,
            "E2": float(e2),
            "E4": float(e4),
            "E": float(etot),
            "Q_fft": float(q),
            "E2_over_lambda": float(e2 / lam),
            "E4_times_lambda": float(e4 * lam),
            "R_over_E": float(Rp / etot) if etot > 0 else float("nan"),
            "h": h,
            **grav,
        }
        rows.append(row)
        sat = " SAT" if grav.get("R_chi_saturated") else ""
        print(
            f"  E={etot:.4f} (E2={e2:.4f} E4={e4:.4f})  Q={q:.4f}  "
            f"E2/λ={e2/lam:.4f} E4*λ={e4*lam:.4f}  "
            f"R_rms={Rp:.4f}  G_c={grav['G_c_kappa']:.4e}{sat}",
            flush=True,
        )
        if args.save_fields:
            np.save(out_dir / f"nfield_dilate_l{lam:g}.npy", n_np.astype(np.float32))

    # fits: use lambda as R-proxy AND R_rms
    fit_e_lam = mr.fit_ER(rows, r_key="scale_as_R")
    fit_e_rms = mr.fit_ER(rows, r_key="R_primary")
    fit_g_lam = mr.fit_Gc_vs_R_over_E(rows, r_key="scale_as_R")
    fit_g_rms = mr.fit_Gc_vs_R_over_E(rows, r_key="R_primary")
    derr = derrick_diagnostics(rows)

    print("=" * 72)
    if n_sat:
        print(f"WARNING: R_chi saturated for {n_sat}/{len(rows)} points")
    print("DERRICK CHECK")
    print(
        f"  E_min={derr['E_min']:.4f} at λ={derr['lambda_at_E_min']:.3g}  "
        f"interior_min={derr['has_interior_minimum']}  "
        f"E2/E4(at min)={derr['E2_over_E4_at_min']:.4f}"
    )
    print(
        f"  E2/λ: mean={derr['E2_over_lambda_mean']:.4f}  rel_std={derr['E2_over_lambda_rel_std']:.4f}"
    )
    print(
        f"  E4*λ: mean={derr['E4_times_lambda_mean']:.4f}  rel_std={derr['E4_times_lambda_rel_std']:.4f}"
    )
    print(f"  |Q| range: [{derr['Q_abs_min']:.4f}, {derr['Q_abs_max']:.4f}]")

    print("FIT E = a λ + b/λ")
    if fit_e_lam.get("ok"):
        print(
            f"  a={fit_e_lam['a']:.6e}  b={fit_e_lam['b']:.6e}  R2={fit_e_lam['R2_fit']:.4f}  "
            f"λ0={fit_e_lam['R0_sqrt_b_over_a']:.4f}  "
            f"U_pos={fit_e_lam.get('U_shape_coefficients_positive')}"
        )
    else:
        print(f"  FAILED: {fit_e_lam.get('reason')}")

    print("FIT E = a R_rms + b/R_rms")
    if fit_e_rms.get("ok"):
        print(
            f"  a={fit_e_rms['a']:.6e}  b={fit_e_rms['b']:.6e}  R2={fit_e_rms['R2_fit']:.4f}  "
            f"U_pos={fit_e_rms.get('U_shape_coefficients_positive')}"
        )
    else:
        print(f"  FAILED: {fit_e_rms.get('reason')}")

    print("FIT G_c ≈ k (λ/E) and (R_rms/E)")
    for name, fg in (("lambda", fit_g_lam), ("R_rms", fit_g_rms)):
        if fg.get("ok"):
            print(
                f"  [{name}] pearson={fg['pearson_R_over_E_vs_Gc']:.4f}  "
                f"k_free={fg['k_free']:.4e}  R2_free={fg['R2_free']:.4f}  "
                f"pos={fg.get('G_c_vs_R_over_E_positive_trend')}"
            )
        else:
            print(f"  [{name}] FAILED: {fg.get('reason')}")

    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.1f}s")
    print("NOTE: fixed-box dilation; far λ truncated by vacuum fill. Not GR.")
    print("=" * 72)

    summary = {
        "phase": "phase3_MR_dilate",
        "mode": "DERRICK_DILATION",
        "Q": args.Q,
        "reference_path": path,
        "n": n,
        "length": length,
        "lambdas": lams,
        "ref_E": float(e_1),
        "ref_E2": float(e2_1),
        "ref_E4": float(e4_1),
        "ref_Q": float(q_1),
        "rows": rows,
        "n_R_chi_saturated": n_sat,
        "derrick": derr,
        "fit_E_of_lambda": fit_e_lam,
        "fit_E_of_R_rms": fit_e_rms,
        "fit_Gc_vs_lambda_over_E": fit_g_lam,
        "fit_Gc_vs_Rrms_over_E": fit_g_rms,
        "elapsed_s": elapsed,
        "framing": (
            "True Derrick probe: n_λ(x)=n_ref(x/λ), no relax. "
            "Energy on physical grid (Jacobian in h³). "
            "G_c is kappa wall from fixed-n reweight."
        ),
    }
    out_json = out_dir / f"phase3_MR_dilate_Q{args.Q}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
