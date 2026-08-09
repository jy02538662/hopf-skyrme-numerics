import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hopf_skyrme_torch import apply_boundary_, energy_parts_float, energy_parts_tensor, hopf_charge_torch_fft, normalize


def device_from_args(args):
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def load_field(path, device, dtype):
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"invalid nfield shape: {arr.shape}")
    n = torch.as_tensor(arr, device=device, dtype=dtype)
    n = normalize(n)
    apply_boundary_(n)
    return n


def tangent_project(v, n):
    v = v - torch.sum(v * n, dim=-1, keepdim=True) * n
    v = v.clone()
    v[0, :, :, :] = 0.0
    v[-1, :, :, :] = 0.0
    v[:, 0, :, :] = 0.0
    v[:, -1, :, :] = 0.0
    v[:, :, 0, :] = 0.0
    v[:, :, -1, :] = 0.0
    return v


def inner(a, b, h):
    return torch.sum(a * b) * (h ** 3)


def norm(v, h):
    return torch.sqrt(torch.clamp(inner(v, v, h), min=0.0))


def orthogonalize(v, basis, h, n, reorth="full", passes=1):
    v = tangent_project(v, n)
    if reorth == "full":
        if basis:
            for _ in range(max(1, passes)):
                for q in basis:
                    v = v - inner(q, v, h) * q
                v = tangent_project(v, n)
        else:
            v = tangent_project(v, n)
    elif reorth == "none":
        v = tangent_project(v, n)
    else:
        raise ValueError(f"unknown reorth mode: {reorth}")
    nv = norm(v, h)
    if float(nv.detach().cpu()) < 1e-12:
        return None, nv
    return v / nv, nv


def objective_tensor(nfield, h, a, b, q_penalty, q_target):
    e = energy_parts_tensor(nfield, h, a, b)[2]
    if q_penalty == 0.0:
        return e
    q_fft = hopf_charge_torch_fft(nfield, h)
    return e + q_penalty * (q_fft - q_target) ** 2


def objective_float(nfield, h, a, b, q_penalty, q_target):
    e2, e4, etot = energy_parts_float(nfield, h, a, b)
    with torch.no_grad():
        q_fft = float(hopf_charge_torch_fft(nfield, h).detach().cpu())
    obj = etot + q_penalty * (q_fft - q_target) ** 2
    return e2, e4, etot, q_fft, obj


def make_normalized_pullback_hvp(n0, h, a, b, damping, q_penalty, q_target):
    u0 = n0.detach().clone().requires_grad_(True)

    def hvp(v):
        if u0.grad is not None:
            u0.grad = None
        n = normalize(u0)
        obj = objective_tensor(n, h, a, b, q_penalty, q_target)
        grad = torch.autograd.grad(obj, u0, create_graph=True)[0]
        gv = torch.sum(grad * v)
        hv = torch.autograd.grad(gv, u0, retain_graph=False)[0] / (h ** 3)
        hv = tangent_project(hv, n0)
        if damping != 0.0:
            hv = hv + damping * v
        return hv.detach()

    return hvp


def make_constrained_hvp(n0, h, a, b, damping, q_penalty, q_target):
    n_ref = n0.detach().clone().requires_grad_(True)

    def hvp(v):
        if n_ref.grad is not None:
            n_ref.grad = None
        obj = objective_tensor(n_ref, h, a, b, q_penalty, q_target)
        grad = torch.autograd.grad(obj, n_ref, create_graph=True)[0]
        lambda_raw = torch.sum(n_ref * grad, dim=-1, keepdim=True)
        gv = torch.sum(grad * v)
        dgrad_v = torch.autograd.grad(gv, n_ref, retain_graph=False)[0]
        hv = (dgrad_v - lambda_raw * v) / (h ** 3)
        hv = tangent_project(hv, n0)
        if damping != 0.0:
            hv = hv + damping * v
        return hv.detach()

    return hvp


def make_hvp(n0, h, a, b, damping, mode, q_penalty, q_target):
    if mode == "constrained":
        return make_constrained_hvp(n0, h, a, b, damping, q_penalty, q_target)
    if mode == "normalized-pullback":
        return make_normalized_pullback_hvp(n0, h, a, b, damping, q_penalty, q_target)
    raise ValueError(f"unknown hvp mode: {mode}")


def lanczos(n0, h, a, b, num_iters, seed, damping, hvp_mode, keep_basis, q_penalty, q_target, reorth, reorth_passes):
    torch.manual_seed(seed)
    hvp = make_hvp(n0, h, a, b, damping, hvp_mode, q_penalty, q_target)
    q = torch.randn_like(n0)
    q, _ = orthogonalize(q, [], h, n0, reorth=reorth, passes=reorth_passes)
    if q is None:
        raise RuntimeError("failed to initialize Lanczos vector")

    basis = []
    alphas = []
    betas = []
    beta_prev = torch.tensor(0.0, device=n0.device, dtype=n0.dtype)
    q_prev = torch.zeros_like(q)
    t0 = time.time()
    orthogonality_errors = []

    for k in range(num_iters):
        z = hvp(q)
        if k > 0:
            z = z - beta_prev * q_prev
        alpha = inner(q, z, h)
        z = z - alpha * q
        ortho_basis = basis + [q]
        z, nz = orthogonalize(z, ortho_basis, h, n0, reorth=reorth, passes=reorth_passes)
        max_orth_err = 0.0
        if z is not None and ortho_basis:
            max_orth_err = max(abs(float(inner(qj, z, h).detach().cpu())) for qj in ortho_basis)
        orthogonality_errors.append(max_orth_err)
        basis.append(q.detach().clone() if keep_basis else q.detach())
        alphas.append(float(alpha.detach().cpu()))
        print(
            f"lanczos={k + 1:4d}/{num_iters} alpha={alphas[-1]: .8e} "
            f"beta={float(nz.detach().cpu()): .8e} orth={max_orth_err:.2e} "
            f"elapsed={time.time() - t0:.1f}s"
        )
        if z is None:
            break
        if k < num_iters - 1:
            betas.append(float(nz.detach().cpu()))
        q_prev = q
        q = z
        beta_prev = nz

    m = len(alphas)
    tri = np.zeros((m, m), dtype=np.float64)
    for i, alpha in enumerate(alphas):
        tri[i, i] = alpha
    for i, beta in enumerate(betas[: max(0, m - 1)]):
        tri[i, i + 1] = beta
        tri[i + 1, i] = beta
    evals, evecs = np.linalg.eigh(tri)
    return evals, evecs, alphas, betas, basis, orthogonality_errors


def reconstruct_ritz_mode(basis, evecs, mode_idx, n0, h):
    coeff = torch.as_tensor(evecs[:, mode_idx], device=n0.device, dtype=n0.dtype)
    mode = torch.zeros_like(n0)
    for i, q in enumerate(basis):
        mode = mode + coeff[i] * q
    mode = tangent_project(mode, n0)
    mode_norm = norm(mode, h)
    if float(mode_norm.detach().cpu()) > 0.0:
        mode = mode / mode_norm
    return mode.detach()


def parse_scan_eps(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def energy_line_scan(n0, mode, h, a, b, eps_values, eigenvalue, q_penalty, q_target):
    _, _, e0, q0, obj0 = objective_float(n0, h, a, b, q_penalty, q_target)
    rows = []
    with torch.no_grad():
        for eps in eps_values:
            trial = normalize(n0 + eps * mode)
            apply_boundary_(trial)
            e2, e4, etot, q_fft, obj = objective_float(trial, h, a, b, q_penalty, q_target)
            rows.append(
                {
                    "eps": eps,
                    "E2": e2,
                    "E4": e4,
                    "E": etot,
                    "delta_E": etot - e0,
                    "Q_fft": q_fft,
                    "delta_Q_fft": q_fft - q0,
                    "objective": obj,
                    "delta_objective": obj - obj0,
                    "quadratic_prediction_delta_objective": 0.5 * eigenvalue * eps * eps,
                }
            )
    return rows


def save_modes_and_scans(out_dir, n0, h, a, b, evals, evecs, basis, save_modes, scan_modes, scan_eps, q_penalty, q_target):
    saved = []
    scans = []
    count = min(max(save_modes, scan_modes), len(evals), evecs.shape[1])
    for idx in range(count):
        mode = reconstruct_ritz_mode(basis, evecs, idx, n0, h)
        mode_norm = float(norm(mode, h).detach().cpu())
        mode_info = {"mode_index": idx + 1, "eigenvalue": float(evals[idx]), "norm": mode_norm}
        if idx < save_modes:
            mode_path = out_dir / f"ritz_mode_{idx + 1:03d}.npy"
            np.save(mode_path, mode.detach().cpu().numpy().astype(np.float32))
            mode_info["path"] = str(mode_path)
        if idx < scan_modes:
            scan = energy_line_scan(n0, mode, h, a, b, scan_eps, float(evals[idx]), q_penalty, q_target)
            scans.append({"mode_index": idx + 1, "eigenvalue": float(evals[idx]), "scan": scan})
        saved.append(mode_info)
    return saved, scans


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix-free constrained Hessian low-spectrum probe for Hopf-Skyrme fields")
    parser.add_argument("--field", required=True, help="Input .npy nfield from hopf_skyrme_torch.py")
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--damping", type=float, default=0.0, help="Optional diagonal shift for numerical probes")
    parser.add_argument("--hvp-mode", choices=["constrained", "normalized-pullback"], default="constrained")
    parser.add_argument("--q-penalty", type=float, default=0.0, help="Optional Hopf-charge penalty included in Hessian objective")
    parser.add_argument("--q-target", type=float, default=-1.0, help="Target Q for --q-penalty")
    parser.add_argument(
        "--reorth",
        choices=["full", "none"],
        default="full",
        help="Lanczos reorthogonalization mode. Use full for long runs to suppress ghost Ritz values.",
    )
    parser.add_argument(
        "--reorth-passes",
        type=int,
        default=1,
        help="Number of modified Gram-Schmidt passes when --reorth=full. Use 2 for long float32 runs.",
    )
    parser.add_argument("--save-modes", type=int, default=0, help="Save the first K reconstructed Ritz modes as .npy")
    parser.add_argument("--scan-modes", type=int, default=0, help="Run +/- epsilon energy scans for the first K Ritz modes")
    parser.add_argument("--scan-eps", default="-0.05,-0.02,-0.01,0,0.01,0.02,0.05")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = device_from_args(args)
    dtype = torch.float64 if args.float64 else torch.float32
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n0 = load_field(args.field, device, dtype)
    n = n0.shape[0]
    h = 2.0 * args.length / (n - 1)
    e2, e4, etot, q_fft, obj = objective_float(n0, h, args.a, args.b, args.q_penalty, args.q_target)
    keep_basis = args.save_modes > 0 or args.scan_modes > 0
    print(
        f"device={device}, dtype={dtype}, n={n}, h={h:.8f}, E={etot:.8f}, "
        f"Q={q_fft:.6f}, objective={obj:.8f}, hvp_mode={args.hvp_mode}, "
        f"q_penalty={args.q_penalty:g}, q_target={args.q_target:g}, "
        f"reorth={args.reorth}, reorth_passes={args.reorth_passes}"
    )
    evals, evecs, alphas, betas, basis, orthogonality_errors = lanczos(
        n0=n0,
        h=h,
        a=args.a,
        b=args.b,
        num_iters=args.iters,
        seed=args.seed,
        damping=args.damping,
        hvp_mode=args.hvp_mode,
        keep_basis=keep_basis,
        q_penalty=args.q_penalty,
        q_target=args.q_target,
        reorth=args.reorth,
        reorth_passes=args.reorth_passes,
    )

    scan_eps = parse_scan_eps(args.scan_eps)
    saved_modes = []
    mode_scans = []
    if keep_basis:
        saved_modes, mode_scans = save_modes_and_scans(
            out_dir=out_dir,
            n0=n0,
            h=h,
            a=args.a,
            b=args.b,
            evals=evals,
            evecs=evecs,
            basis=basis,
            save_modes=args.save_modes,
            scan_modes=args.scan_modes,
            scan_eps=scan_eps,
            q_penalty=args.q_penalty,
            q_target=args.q_target,
        )

    summary = {
        "field": args.field,
        "device": str(device),
        "n": n,
        "length": args.length,
        "h": h,
        "a": args.a,
        "b": args.b,
        "E2": e2,
        "E4": e4,
        "E": etot,
        "Q_fft": q_fft,
        "objective": obj,
        "q_penalty": args.q_penalty,
        "q_target": args.q_target,
        "iters_requested": args.iters,
        "iters_completed": len(alphas),
        "seed": args.seed,
        "damping": args.damping,
        "hvp_mode": args.hvp_mode,
        "reorth": args.reorth,
        "reorth_passes": args.reorth_passes,
        "orthogonality_errors": orthogonality_errors,
        "max_orthogonality_error": max(orthogonality_errors) if orthogonality_errors else 0.0,
        "lowest_eigenvalues": evals[: min(40, len(evals))].tolist(),
        "highest_eigenvalues": evals[max(0, len(evals) - 10):].tolist(),
        "alphas": alphas,
        "betas": betas,
        "saved_modes": saved_modes,
        "mode_scans": mode_scans,
    }
    with open(out_dir / "hessian_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.save(out_dir / "ritz_eigenvalues.npy", evals)
    np.save(out_dir / "tridiagonal_eigenvectors.npy", evecs)
    print("lowest eigenvalues:")
    for i, value in enumerate(summary["lowest_eigenvalues"][:20], start=1):
        print(f"{i:3d}: {value: .10e}")
    if mode_scans:
        print("energy line scans:")
        for item in mode_scans:
            print(f"mode {item['mode_index']} eigenvalue={item['eigenvalue']: .10e}")
            for row in item["scan"]:
                print(
                    f"  eps={row['eps']: .5f} dObj={row['delta_objective']: .10e} "
                    f"quad={row['quadratic_prediction_delta_objective']: .10e} "
                    f"dE={row['delta_E']: .10e} Q={row['Q_fft']: .6f} dQ={row['delta_Q_fft']: .3e}"
                )


if __name__ == "__main__":
    main()
