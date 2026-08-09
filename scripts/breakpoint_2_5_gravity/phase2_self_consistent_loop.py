#!/usr/bin/env python3
"""
断点 2.5 阶段2：半经典自洽迭代（牛顿 Φ + 变刚度 a(x)）。

循环:
  n → ρ=κ_ρ χ² → Φ → κ_eff → damp → 弛豫 n_relax 步 → 重算 …

κ 阻尼: κ ← (1-α) κ_old + α κ_new
Q_H: 冻结为 hopf_skyrme_torch 的 Q_fft（死结2开放项不阻塞）

收敛判据（可调）:
  |ΔE/E| < etol
  |Δx_core| < xtol * h
  |ΔQ| < qtol

用法（建议先 G=1e-3, alpha=0.4）:
  python phase2_self_consistent_loop.py --abs-outputs --Q 1 --G-eff 1e-3 \\
    --outer 30 --n-relax 40 --alpha 0.4 --lr 1e-3 --device cuda \\
    --out-dir /tmp/bp25_p2
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
    candidates = [
        _HERE / "hopf_skyrme_torch.py",
        _HERE / "src" / "hopf_skyrme_torch.py",
        _HERE.parent / "src" / "hopf_skyrme_torch.py",
        _HERE.parent.parent / "src" / "hopf_skyrme_torch.py",
        Path("/src/hopf_skyrme_torch.py"),
        Path("/hopf_skyrme_torch.py"),
        Path("/workspace/src/hopf_skyrme_torch.py"),
        Path("/hopf_skyrme_cpu/src/hopf_skyrme_torch.py"),
    ]
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
    raise SystemExit("Cannot find hopf_skyrme_torch.py")


_TORCH_PATH = _add_torch_src()

from poisson_phi import (  # noqa: E402
    calibrate_kappa_rho,
    chi_from_nfield,
    grid_spacing,
    kappa_eff_from_phi,
    rho_eff,
    solve_phi,
)
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


def load_nfield_np(args):
    common = _load_screening_common()
    if common is None:
        raise SystemExit("need common.py (screening_progression_3step)")
    table = common.resolve_field_table(args.abs_outputs)
    meta = table[args.Q]
    path = args.field or meta["path"]
    length = float(meta["length"])
    if not Path(path).is_file():
        raise SystemExit(f"missing field: {path}")
    default_E = {1: 434.415, 2: 700.0, 3: 1082.0, 4: 1735.0}
    energy = args.energy if args.energy > 0 else float(default_E.get(args.Q, 434.415))
    return common.load_nfield(path), length, path, energy


def core_com(nfield_np: np.ndarray, length: float):
    """COM of (1-n_z) mass as core position proxy."""
    n = nfield_np.shape[0]
    h = grid_spacing(n, length)
    ax = np.linspace(-length, length, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    w = np.clip(1.0 - nfield_np[..., 2], 0.0, None)
    s = float(np.sum(w))
    if s < 1e-30:
        return np.zeros(3), 0.0
    com = np.array(
        [
            float(np.sum(w * X) / s),
            float(np.sum(w * Y) / s),
            float(np.sum(w * Z) / s),
        ]
    )
    return com, s * (h**3)


def compute_keff_np(n_np, length, kappa_rho, g_eff, kappa0, c2, keff_floor):
    h = grid_spacing(n_np.shape[0], length)
    chi = chi_from_nfield(n_np)
    rho = rho_eff(chi, kappa_rho)
    phi = solve_phi(rho, h, g_eff)
    keff = kappa_eff_from_phi(phi, kappa0, c2)
    keff_min_raw = float(np.min(keff))
    keff = np.maximum(keff, keff_floor)
    max_abs_two = float(np.max(np.abs(2.0 * phi / c2)))
    # dead-knot1 proxy: peak effective density + chi peak
    rho_max = float(np.max(rho))
    chi_max = float(np.max(chi))
    return keff, keff_min_raw, max_abs_two, float(np.min(phi)), rho_max, chi_max


def relax_block(u, h, a_field, b, n_steps, lr, optimizer_name):
    """In-place-ish: returns updated Parameter u and last (E,Q,coreV)."""
    if optimizer_name == "adam":
        opt = torch.optim.Adam([u], lr=lr)
    else:
        opt = None
    e_last = q_last = core_v = float("nan")
    for _ in range(n_steps):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        elif u.grad is not None:
            u.grad = None
        nfield = hst.normalize(u)
        _e2, _e4, etot_t = hst.energy_parts_tensor(nfield, h, a_field, b)
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
    with torch.no_grad():
        nfield = hst.normalize(u)
        e2, e4, e_last = hst.energy_parts_float(nfield, h, a_field, b)
        q_last = hst.hopf_charge_cpu_fft(nfield, h)
        diag = hst.core_diagnostics(nfield, h)
        core_v = float(diag["core_volume"])
    return float(e_last), float(q_last), core_v, float(e2), float(e4)


def main():
    ap = argparse.ArgumentParser(description="BP2.5 phase2 self-consistent loop")
    ap.add_argument("--abs-outputs", action="store_true")
    ap.add_argument("--Q", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--field", type=str, default="")
    ap.add_argument("--energy", type=float, default=-1.0)
    ap.add_argument("--G-eff", type=float, default=1e-3, dest="g_eff")
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.4, help="kappa damping in (0,1]")
    ap.add_argument("--outer", type=int, default=30, help="outer self-consistent iterations")
    ap.add_argument("--n-relax", type=int, default=40, help="micro relax steps per outer")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", type=str, default="riemannian", choices=["riemannian", "adam"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--keff-floor", type=float, default=0.05)
    ap.add_argument("--etol", type=float, default=1e-5)
    ap.add_argument("--qtol", type=float, default=1e-4)
    ap.add_argument("--xtol", type=float, default=1e-3, help="core COM shift / h")
    ap.add_argument("--q-guard", type=float, default=0.3, help="abort if |Q| below this")
    ap.add_argument(
        "--freeze-kappa",
        action="store_true",
        help="control: fix a(x)=initial kappa_eff; do NOT recompute Phi each outer",
    )
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("--alpha must be in (0,1]")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dtype = torch.float32

    n_np, length, path, energy = load_nfield_np(args)
    n = n_np.shape[0]
    h = grid_spacing(n, length)
    kappa_rho, integ = calibrate_kappa_rho(n_np, length, energy)

    mode = "FREEZE_KAPPA_CONTROL" if args.freeze_kappa else "SELF_CONSISTENT"
    print("=" * 72)
    print("BP2.5 PHASE2: self-consistent loop")
    print(f"  torch: {_TORCH_PATH}")
    print(f"  field: {path}")
    print(f"  mode={mode}")
    print(
        f"  G_eff={args.g_eff}  alpha={args.alpha}  outer={args.outer}  "
        f"n_relax={args.n_relax}  lr={args.lr}"
    )
    print(f"  device={device}  kappa_rho={kappa_rho:.6e}")
    print(f"  tol: dE/E<{args.etol}  dQ<{args.qtol}  dx_core/h<{args.xtol}")
    if args.freeze_kappa:
        print("  CONTROL: a(x) frozen at initial kappa_eff; no Phi backreaction updates")
    print("=" * 72)

    u = torch.nn.Parameter(torch.tensor(n_np, device=device, dtype=dtype))
    # initial kappa (even in freeze mode: build once from G_eff, then hold)
    g_for_init = args.g_eff
    keff_np, kmin0, m2_0, _, rho_max0, chi_max0 = compute_keff_np(
        n_np, length, kappa_rho, g_for_init, args.kappa0, args.c2, args.keff_floor
    )
    a_np = keff_np.copy()
    a_field = torch.tensor(a_np, device=device, dtype=dtype)

    e2, e4, e_prev = hst.energy_parts_float(hst.normalize(u), h, a_field, args.b)
    q_prev = hst.hopf_charge_cpu_fft(hst.normalize(u), h)
    com_prev, _ = core_com(n_np, length)
    print(
        f"  init: E={e_prev:.6f} Q={q_prev:.6f} "
        f"max|2Phi|={m2_0:.4e} keff_raw_min={kmin0:.4e} "
        f"rho_max={rho_max0:.4e} chi_max={chi_max0:.4e}"
    )

    history = []
    status = "MAX_OUTER"
    t0 = time.time()

    # Hard wall: initial kappa already non-positive (cold-start G too large)
    if kmin0 <= 0 and not args.freeze_kappa:
        status = "ABORT_INIT_KAPPA_NONPOSITIVE"
        print(f"ABORT: init keff_raw_min={kmin0:.4e} <= 0 (G_eff too large for this n)")
        history.append(
            {
                "iter": 0,
                "status": status,
                "E": float(e_prev),
                "Q_fft": float(q_prev),
                "keff_min_raw": kmin0,
                "max_abs_2phi_c2": m2_0,
                "rho_max": rho_max0,
                "chi_max": chi_max0,
            }
        )
    elif kmin0 <= 0 and args.freeze_kappa:
        status = "ABORT_INIT_KAPPA_NONPOSITIVE"
        print(
            f"ABORT: init keff_raw_min={kmin0:.4e} <= 0; "
            "freeze+keff_floor is NOT a valid physical control — refusing to run"
        )
        history.append(
            {
                "iter": 0,
                "status": status,
                "E": float(e_prev),
                "Q_fft": float(q_prev),
                "keff_min_raw": kmin0,
                "max_abs_2phi_c2": m2_0,
                "rho_max": rho_max0,
                "chi_max": chi_max0,
                "note": "refused_freeze_with_negative_raw_kappa",
            }
        )

    start_outer = 1 if status == "MAX_OUTER" else args.outer + 1
    for it in range(start_outer, args.outer + 1):
        e_cur, q_cur, core_v, e2, e4 = relax_block(
            u, h, a_field, args.b, args.n_relax, args.lr, args.optimizer
        )
        n_np = hst.normalize(u).detach().cpu().numpy()
        com_cur, core_mass = core_com(n_np, length)

        if args.freeze_kappa:
            # diagnostic Phi from current n, but do NOT update a
            _, kmin_raw, max_abs_two, phi_min, rho_max, chi_max = compute_keff_np(
                n_np, length, kappa_rho, args.g_eff, args.kappa0, args.c2, args.keff_floor
            )
            e_with_new_a = e_cur
        else:
            keff_new, kmin_raw, max_abs_two, phi_min, rho_max, chi_max = compute_keff_np(
                n_np, length, kappa_rho, args.g_eff, args.kappa0, args.c2, args.keff_floor
            )
            if kmin_raw <= 0:
                status = "ABORT_KAPPA_NONPOSITIVE"
                print(f"it={it}: ABORT keff_raw_min={kmin_raw:.4e}")
                history.append(
                    {
                        "iter": it,
                        "status": status,
                        "keff_min_raw": kmin_raw,
                        "max_abs_2phi_c2": max_abs_two,
                        "rho_max": rho_max,
                        "chi_max": chi_max,
                    }
                )
                break

            a_np = (1.0 - args.alpha) * a_np + args.alpha * keff_new
            a_np = np.maximum(a_np, args.keff_floor)
            a_field = torch.tensor(a_np, device=device, dtype=dtype)
            e_with_new_a = hst.energy_parts_float(
                hst.normalize(u), h, a_field, args.b
            )[2]

        dE_E = (e_cur - e_prev) / (abs(e_prev) + 1e-30)
        dQ = q_cur - q_prev
        dx = float(np.linalg.norm(com_cur - com_prev)) / max(h, 1e-30)

        row = {
            "iter": it,
            "E": e_cur,
            "E2": e2,
            "E4": e4,
            "E_after_kappa_update": float(e_with_new_a),
            "Q_fft": q_cur,
            "dE_over_E": dE_E,
            "dQ": dQ,
            "dx_core_over_h": dx,
            "core_volume": core_v,
            "core_com": com_cur.tolist(),
            "keff_min": float(np.min(a_np)),
            "keff_min_raw": kmin_raw,
            "max_abs_2phi_c2": max_abs_two,
            "phi_min": phi_min,
            "rho_max": rho_max,
            "chi_max": chi_max,
            "freeze_kappa": bool(args.freeze_kappa),
        }
        history.append(row)
        print(
            f"it={it:3d} E={e_cur:12.6f} Q={q_cur: .6f} "
            f"dE/E={dE_E: .3e} dQ={dQ: .3e} dx/h={dx: .3e} "
            f"|2Phi|={max_abs_two:.3e} keff_min={np.min(a_np):.3e}"
        )

        if not math.isfinite(e_cur) or not math.isfinite(q_cur):
            status = "ABORT_NONFINITE"
            break
        if abs(q_cur) < args.q_guard:
            status = "ABORT_Q_GUARD"
            break

        converged = (
            abs(dE_E) < args.etol
            and abs(dQ) < args.qtol
            and dx < args.xtol
        )
        if converged and it > 1:
            status = "CONVERGED"
            print(f"CONVERGED at outer iter {it}")
            break

        e_prev, q_prev, com_prev = e_cur, q_cur, com_cur

    elapsed = time.time() - t0
    n_final = hst.normalize(u).detach().cpu().numpy()

    # Weak-field backreaction stability (not full SC convergence)
    if history and "Q_fft" in history[0]:
        dQ_abs = [abs(r["dQ"]) for r in history if "dQ" in r]
        q_amplifying = (
            len(dQ_abs) >= 10
            and float(np.mean(dQ_abs[-5:])) > 2.0 * float(np.mean(dQ_abs[:5])) + 1e-12
        )
        kappa_ok = all(
            (r.get("keff_min_raw") is None) or (r["keff_min_raw"] > 0)
            for r in history
            if "keff_min_raw" in r
        )
        no_abort = status in ("MAX_OUTER", "CONVERGED")
        if args.g_eff <= 3e-3 and no_abort and kappa_ok and not q_amplifying:
            weak_verdict = "WEAK_FIELD_BACKREACTION_STABLE"
        elif no_abort and kappa_ok:
            weak_verdict = "NUMERICALLY_STABLE_NOT_CONVERGED"
        else:
            weak_verdict = "UNSTABLE_OR_ABORT"
    else:
        weak_verdict = "NO_DATA"
        q_amplifying = False

    dead = {
        "note": "征兆记录，非判定",
        "max_abs_2phi_c2_last": history[-1].get("max_abs_2phi_c2") if history else None,
        "keff_min_raw_last": history[-1].get("keff_min_raw") if history else None,
        "Q_drift_total": (
            float(history[-1]["Q_fft"] - history[0]["Q_fft"])
            if len(history) >= 1 and "Q_fft" in history[0]
            else None
        ),
        "G_eff": args.g_eff,
        "q_drift_amplifying": q_amplifying,
    }

    print("-" * 72)
    print(f"STATUS: {status}  elapsed={elapsed:.1f}s  outers_done={len(history)}")
    print(f"WEAK_FIELD_VERDICT: {weak_verdict}")
    print(
        "  WEAK_FIELD_BACKREACTION_STABLE = loop OK, kappa>0, Q drift not amplifying\n"
        "  (does NOT mean SC fixed-point converged; see STATUS)\n"
        "NOTE: semiclassical Newtonian backreaction only; not GR.\n"
        "      Q_fft proxy for Hopf; k_ex from BP2 not used."
    )
    print("=" * 72)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "freeze" if args.freeze_kappa else "sc"
    np.save(out_dir / f"phase2_Q{args.Q}_{tag}_nfield_final.npy", n_final.astype(np.float32))
    np.save(out_dir / f"phase2_Q{args.Q}_{tag}_a_final.npy", a_np.astype(np.float32))
    summary = {
        "path": path,
        "Q": args.Q,
        "mode": mode,
        "freeze_kappa": bool(args.freeze_kappa),
        "G_eff": args.g_eff,
        "alpha": args.alpha,
        "outer": args.outer,
        "n_relax": args.n_relax,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "device": str(device),
        "kappa_rho": kappa_rho,
        "integral_chi2": integ,
        "etol": args.etol,
        "qtol": args.qtol,
        "xtol": args.xtol,
        "status": status,
        "weak_field_verdict": weak_verdict,
        "elapsed_s": elapsed,
        "history": history,
        "dead_knot_log": dead,
        "torch_module": str(_TORCH_PATH),
    }
    out_json = out_dir / f"phase2_Q{args.Q}_{tag}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
