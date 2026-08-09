#!/usr/bin/env python3
"""
路径 B：阶段 1b — 固定单次 κ_eff 下的变刚度短弛豫。

流程:
  1) 由 n0 算 ρ → Φ → κ_eff（固定，不每步更新）
  2) 用 a(x)=κ_eff 的 FS 能量做 Riemannian / Adam 短弛豫
  3) 记录 E、Q_fft、core 诊断 vs 无修正对照（可选）

依赖: hopf_skyrme_cpu/src/hopf_skyrme_torch.py（已支持 a 为空间场）

用法:
  python phase1b_short_relax.py --abs-outputs --Q 1 --G-eff 1e-3 \\
    --steps 150 --lr 1e-3 --device cuda --out-dir /tmp/bp25_1b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _add_torch_src():
    """Find hopf_skyrme_torch.py whether script lives in repo or flat / on GPU."""
    candidates = [
        _HERE / "hopf_skyrme_torch.py",
        _HERE / "src" / "hopf_skyrme_torch.py",
        _HERE.parent / "src" / "hopf_skyrme_torch.py",
        _HERE.parent.parent / "src" / "hopf_skyrme_torch.py",
        Path("/src/hopf_skyrme_torch.py"),
        Path("/workspace/src/hopf_skyrme_torch.py"),
        Path("/hopf_skyrme_cpu/src/hopf_skyrme_torch.py"),
    ]
    # walk up a few levels looking for .../src/hopf_skyrme_torch.py
    cur = _HERE
    for _ in range(6):
        candidates.append(cur / "src" / "hopf_skyrme_torch.py")
        if cur.parent == cur:
            break
        cur = cur.parent
    for p in candidates:
        if p.is_file():
            sys.path.insert(0, str(p.parent))
            return p
    raise SystemExit(
        "Cannot find hopf_skyrme_torch.py.\n"
        "Copy src/hopf_skyrme_torch.py next to this script, or keep repo layout "
        "(.../hopf_skyrme_cpu/src/hopf_skyrme_torch.py)."
    )


_TORCH_PATH = _add_torch_src()

from poisson_phi import (  # noqa: E402
    calibrate_kappa_rho,
    chi_from_nfield,
    core_radius_chi,
    grid_spacing,
    kappa_eff_from_phi,
    rho_eff,
    solve_phi,
)

import hopf_skyrme_torch as hst  # noqa: E402


def _load_screening_common():
    # flat GPU: common.py beside this script; or repo layout
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


def load_nfield_np(args):
    common = _load_screening_common()
    if common is None:
        raise SystemExit("need screening_progression_3step/common.py")
    table = common.resolve_field_table(args.abs_outputs)
    meta = table[args.Q]
    path = args.field or meta["path"]
    length = float(meta["length"])
    if not Path(path).is_file():
        raise SystemExit(f"missing field: {path}")
    default_E = {1: 434.415, 2: 700.0, 3: 1082.0, 4: 1735.0}
    energy = args.energy if args.energy > 0 else float(default_E.get(args.Q, 434.415))
    return common.load_nfield(path), length, path, energy


def run_relax(u0, h, a_field, b, steps, lr, device, print_every, hopf_every, optimizer_name):
    u = torch.nn.Parameter(u0.clone())
    if optimizer_name == "adam":
        opt = torch.optim.Adam([u], lr=lr)
    else:
        opt = None  # riemannian manual

    history = []
    nfield = hst.normalize(u)
    e2, e4, et = hst.energy_parts_float(nfield, h, a_field, b)
    q0 = hst.hopf_charge_cpu_fft(nfield, h)
    e_init = float(et)
    print(f"  initial: E={et:.6f} E2={e2:.6f} E4={e4:.6f} Q={q0:.6f}")

    t0 = time.time()
    for step in range(1, steps + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        elif u.grad is not None:
            u.grad = None
        nfield = hst.normalize(u)
        e2_t, e4_t, etot_t = hst.energy_parts_tensor(nfield, h, a_field, b)
        etot_t.backward()
        with torch.no_grad():
            if optimizer_name == "riemannian":
                grad = u.grad
                grad_t = grad - torch.sum(grad * nfield, dim=-1, keepdim=True) * nfield
                u.copy_(hst.normalize(u - lr * grad_t))
            else:
                opt.step()
                u.copy_(hst.normalize(u))
            hst.apply_boundary_(u)

        if step == 1 or step % print_every == 0 or step == steps:
            nfield = hst.normalize(u)
            e2, e4, et = hst.energy_parts_float(nfield, h, a_field, b)
            q = (
                hst.hopf_charge_cpu_fft(nfield, h)
                if (step == 1 or step % hopf_every == 0 or step == steps)
                else float("nan")
            )
            diag = hst.core_diagnostics(nfield, h)
            history.append(
                {
                    "step": step,
                    "E": float(et),
                    "E2": float(e2),
                    "E4": float(e4),
                    "Q_fft": float(q) if math.isfinite(q) else None,
                    "core_volume": float(diag.get("core_volume", float("nan"))),
                    "max_deviation": float(diag.get("max_deviation", float("nan"))),
                }
            )
            print(
                f"  step={step:4d} E={et:12.6f} E2={e2:10.6f} E4={e4:10.6f} "
                f"Q={q: .5f} coreV={diag.get('core_volume', float('nan')):.4f}"
            )

    elapsed = time.time() - t0
    n_final = hst.normalize(u).detach()
    return n_final, history, elapsed, float(q0), e_init


def main():
    ap = argparse.ArgumentParser(description="BP2.5 phase1b short relax with a(x)")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--energy", type=float, default=-1.0)
    ap.add_argument("--G-eff", type=float, default=1e-3, dest="g_eff")
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", type=str, default="riemannian", choices=["riemannian", "adam"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--hopf-every", type=int, default=25)
    ap.add_argument("--keff-floor", type=float, default=0.05, help="clamp kappa_eff >= floor")
    ap.add_argument("--also-baseline", action="store_true", help="also run a=kappa0 control")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    n_np, length, path, energy = load_nfield_np(args)
    n = n_np.shape[0]
    h = grid_spacing(n, length)
    chi = chi_from_nfield(n_np)
    kappa_rho, integ = calibrate_kappa_rho(n_np, length, energy)
    rho = rho_eff(chi, kappa_rho)
    phi = solve_phi(rho, h, args.g_eff)
    keff_np = kappa_eff_from_phi(phi, args.kappa0, args.c2)
    keff_min_raw = float(np.min(keff_np))
    keff_np = np.maximum(keff_np, args.keff_floor)
    max_abs_two = float(np.max(np.abs(2.0 * phi / args.c2)))
    r_core0 = core_radius_chi(chi, length)

    print("=" * 72)
    print("BP2.5 PHASE1b: fixed-kappa_eff short relax")
    print(f"  torch module: {_TORCH_PATH}")
    print(f"  field: {path}")
    print(f"  device={device}  G_eff={args.g_eff}  steps={args.steps}  lr={args.lr}")
    print(f"  kappa_rho={kappa_rho:.6e}  max|2Phi/c2|={max_abs_two:.4e}")
    print(f"  keff raw min={keff_min_raw:.4e}  clamped floor={args.keff_floor}")
    print(f"  r_core(chi half) init ~ {r_core0:.4f}")
    print("=" * 72)

    dtype = torch.float32
    u0 = torch.tensor(n_np, device=device, dtype=dtype)
    a_field = torch.tensor(keff_np, device=device, dtype=dtype)

    print("\n--- run: a(x)=kappa_eff (fixed) ---")
    n_fin, hist, elapsed, q0, e0 = run_relax(
        u0,
        h,
        a_field,
        args.b,
        args.steps,
        args.lr,
        device,
        args.print_every,
        args.hopf_every,
        args.optimizer,
    )
    n_fin_np = n_fin.detach().cpu().numpy()
    chi_f = chi_from_nfield(n_fin_np)
    r_core1 = core_radius_chi(chi_f, length)
    q1 = hist[-1]["Q_fft"] if hist[-1]["Q_fft"] is not None else float("nan")
    e1 = hist[-1]["E"]
    dQ = (q1 - q0) if (q1 == q1 and q0 == q0) else float("nan")
    dE_E = (e1 - e0) / (abs(e0) + 1e-30)

    baseline = None
    if args.also_baseline:
        print("\n--- control: a=kappa0 ---")
        a0 = torch.full_like(a_field, float(args.kappa0))
        n_b, hist_b, _, q0b, e0b = run_relax(
            u0,
            h,
            a0,
            args.b,
            args.steps,
            args.lr,
            device,
            args.print_every,
            args.hopf_every,
            args.optimizer,
        )
        baseline = {
            "history": hist_b,
            "Q0": q0b,
            "E0": e0b,
            "Q_final": hist_b[-1]["Q_fft"],
            "E_final": hist_b[-1]["E"],
        }

    # trend flags
    flags = []
    if max_abs_two < 0.1:
        flags.append("WEAK_FIELD_OK")
    elif max_abs_two < 0.3:
        flags.append("WEAK_FIELD_YELLOW")
    else:
        flags.append("WEAK_FIELD_ORANGE_OR_RED")
    if abs(dQ) < 1e-3 if dQ == dQ else False:
        flags.append("Q_STABLE")
    elif abs(dQ) < 1e-2 if dQ == dQ else False:
        flags.append("Q_MILD_DRIFT")
    else:
        flags.append("Q_DRIFT_WARN")
    if dE_E < -1e-6:
        flags.append("ENERGY_DECREASING")
    elif abs(dE_E) < 1e-4:
        flags.append("ENERGY_FLAT")
    else:
        flags.append("ENERGY_INCREASING")
    if r_core1 == r_core1 and r_core0 == r_core0:
        if r_core1 < r_core0 * 0.98:
            flags.append("CORE_SHRINK_HINT")
        elif r_core1 > r_core0 * 1.02:
            flags.append("CORE_EXPAND_HINT")
        else:
            flags.append("CORE_RADIUS_STABLE_HINT")

    print("-" * 72)
    print(f"elapsed={elapsed:.1f}s")
    print(f"Q: {q0:.6f} -> {q1}  dQ={dQ}")
    print(f"E: {e0:.6f} -> {e1:.6f}  dE/E={dE_E:.4e}")
    print(f"r_core: {r_core0:.4f} -> {r_core1:.4f}")
    print("FLAGS:", ", ".join(flags))
    print(
        "NOTE: kappa_eff FIXED from initial n (single-shot). Not a self-consistent loop.\n"
        "      chi-half core radius is a coarse proxy only."
    )
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"phase1b_Q{args.Q}_nfield_final.npy", n_fin_np.astype(np.float32))
    summary = {
        "path": path,
        "Q": args.Q,
        "G_eff": args.g_eff,
        "steps": args.steps,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "device": str(device),
        "kappa_rho": kappa_rho,
        "max_abs_2phi_c2": max_abs_two,
        "keff_min_raw": keff_min_raw,
        "keff_floor": args.keff_floor,
        "Q0": q0,
        "Q_final": q1,
        "dQ": dQ,
        "E0": e0,
        "E_final": e1,
        "dE_over_E": dE_E,
        "r_core0": r_core0,
        "r_core1": r_core1,
        "history": hist,
        "baseline": baseline,
        "flags": flags,
        "elapsed_s": elapsed,
    }
    with open(out_dir / f"phase1b_Q{args.Q}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir / f'phase1b_Q{args.Q}_summary.json'}")


if __name__ == "__main__":
    main()
