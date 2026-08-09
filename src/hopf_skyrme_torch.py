import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


N0 = torch.tensor([0.0, 0.0, 1.0])


def device_from_args(args):
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def normalize(u, eps=1e-12):
    return u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=eps)


def apply_boundary_(u):
    n0 = N0.to(device=u.device, dtype=u.dtype)
    u.data[0, :, :, :] = n0
    u.data[-1, :, :, :] = n0
    u.data[:, 0, :, :] = n0
    u.data[:, -1, :, :] = n0
    u.data[:, :, 0, :] = n0
    u.data[:, :, -1, :] = n0


def make_initial_hopf_pq(n, length, scale, p, q, device, dtype):
    if p < 1 or q < 1:
        raise ValueError("p and q must be positive integers")
    x = torch.linspace(-length, length, n, device=device, dtype=dtype)
    h = float((x[1] - x[0]).detach().cpu())
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    xs = X / scale
    ys = Y / scale
    zs = Z / scale
    r2 = xs * xs + ys * ys + zs * zs

    denom = 2.0 * zs + 1j * (r2 - 1.0)
    numer = 2.0 * (xs + 1j * ys)
    W = (numer ** p) / (denom ** q + 1e-20)
    absW2 = torch.abs(W) ** 2
    Z1 = 1.0 / torch.sqrt(1.0 + absW2)
    Z2 = W / torch.sqrt(1.0 + absW2)

    u = torch.empty((n, n, n, 3), device=device, dtype=dtype)
    u[..., 0] = 2.0 * torch.real(Z1 * torch.conj(Z2))
    u[..., 1] = 2.0 * torch.imag(Z1 * torch.conj(Z2))
    u[..., 2] = torch.abs(Z1) ** 2 - torch.abs(Z2) ** 2
    u = normalize(u)
    apply_boundary_(u)
    return u, h


def make_initial_hopf(n, length, scale, charge, device, dtype):
    return make_initial_hopf_pq(n, length, scale, charge, 1, device, dtype)


def make_initial_q1(n, length, scale, device, dtype):
    return make_initial_hopf(n, length, scale, 1, device, dtype)


def central_gradient(f, h, axis):
    return (torch.roll(f, shifts=-1, dims=axis) - torch.roll(f, shifts=1, dims=axis)) / (2.0 * h)


def gradients(nfield, h):
    gx = central_gradient(nfield, h, 0)
    gy = central_gradient(nfield, h, 1)
    gz = central_gradient(nfield, h, 2)
    gx = gx.clone()
    gy = gy.clone()
    gz = gz.clone()
    gx[0, :, :, :] = 0.0
    gx[-1, :, :, :] = 0.0
    gy[:, 0, :, :] = 0.0
    gy[:, -1, :, :] = 0.0
    gz[:, :, 0, :] = 0.0
    gz[:, :, -1, :] = 0.0
    return gx, gy, gz


def curvature(nfield, h):
    gx, gy, gz = gradients(nfield, h)
    Fxy = torch.sum(nfield * torch.cross(gx, gy, dim=-1), dim=-1)
    Fxz = torch.sum(nfield * torch.cross(gx, gz, dim=-1), dim=-1)
    Fyz = torch.sum(nfield * torch.cross(gy, gz, dim=-1), dim=-1)
    return Fxy, Fxz, Fyz, gx, gy, gz


def energy_parts_tensor(nfield, h, a, b, mu=0.0):
    """FS energy. `a` may be a float/0-d tensor or a spatial field (N,N,N)."""
    Fxy, Fxz, Fyz, gx, gy, gz = curvature(nfield, h)
    grad_sq = torch.sum(gx * gx + gy * gy + gz * gz, dim=-1)
    f_sq = Fxy * Fxy + Fxz * Fxz + Fyz * Fyz
    vol = h ** 3
    if torch.is_tensor(a) and a.ndim >= 3:
        e2 = 0.5 * torch.sum(a * grad_sq) * vol
    else:
        e2 = 0.5 * a * torch.sum(grad_sq) * vol
    e4 = 0.25 * b * torch.sum(f_sq) * vol
    if mu > 0:
        pot = (1.0 - nfield[..., 2]) ** 2
        e_pot = 0.5 * mu * mu * torch.sum(pot) * vol
        return e2, e4, e_pot, e2 + e4 + e_pot
    return e2, e4, e2 + e4


def energy_parts_float(nfield, h, a, b, mu=0.0):
    with torch.no_grad():
        result = energy_parts_tensor(nfield, h, a, b, mu)
        if mu > 0:
            e2, e4, e_pot, etot = result
            return float(e2.detach().cpu()), float(e4.detach().cpu()), float(e_pot.detach().cpu()), float(etot.detach().cpu())
        e2, e4, etot = result
    return float(e2.detach().cpu()), float(e4.detach().cpu()), float(etot.detach().cpu())


def energy_parts_float4(nfield, h, a, b, mu=0.0):
    result = energy_parts_float(nfield, h, a, b, mu)
    if mu > 0:
        return result
    e2, e4, etot = result
    return e2, e4, 0.0, etot


def hopf_charge_torch_fft(nfield, h):
    Fxy, Fxz, Fyz, _, _, _ = curvature(nfield, h)
    Bx = Fyz
    By = -Fxz
    Bz = Fxy
    n = nfield.shape[0]

    Bxh = torch.fft.fftn(Bx)
    Byh = torch.fft.fftn(By)
    Bzh = torch.fft.fftn(Bz)

    k = 2.0 * math.pi * torch.fft.fftfreq(n, d=h, device=nfield.device, dtype=nfield.dtype)
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
    k2 = KX * KX + KY * KY + KZ * KZ
    k2 = torch.where(k2 == 0, torch.ones_like(k2), k2)

    Axh = -1j * (KY * Bzh - KZ * Byh) / k2
    Ayh = -1j * (KZ * Bxh - KX * Bzh) / k2
    Azh = -1j * (KX * Byh - KY * Bxh) / k2
    Axh = Axh.clone()
    Ayh = Ayh.clone()
    Azh = Azh.clone()
    Axh[0, 0, 0] = 0.0
    Ayh[0, 0, 0] = 0.0
    Azh[0, 0, 0] = 0.0

    Ax = torch.real(torch.fft.ifftn(Axh))
    Ay = torch.real(torch.fft.ifftn(Ayh))
    Az = torch.real(torch.fft.ifftn(Azh))
    integrand = Ax * Fyz - Ay * Fxz + Az * Fxy
    return torch.sum(integrand) * h ** 3 / (16.0 * math.pi ** 2)


def hopf_charge_cpu_fft(nfield, h):
    with torch.no_grad():
        Fxy, Fxz, Fyz, _, _, _ = curvature(nfield, h)
        Fxy = Fxy.detach().cpu().numpy()
        Fxz = Fxz.detach().cpu().numpy()
        Fyz = Fyz.detach().cpu().numpy()

    Bx = Fyz
    By = -Fxz
    Bz = Fxy
    n = Fxy.shape[0]
    Bxh = np.fft.fftn(Bx)
    Byh = np.fft.fftn(By)
    Bzh = np.fft.fftn(Bz)

    k = 2.0 * math.pi * np.fft.fftfreq(n, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    k2 = KX * KX + KY * KY + KZ * KZ
    k2[0, 0, 0] = 1.0

    Axh = -1j * (KY * Bzh - KZ * Byh) / k2
    Ayh = -1j * (KZ * Bxh - KX * Bzh) / k2
    Azh = -1j * (KX * Byh - KY * Bxh) / k2
    Axh[0, 0, 0] = 0.0
    Ayh[0, 0, 0] = 0.0
    Azh[0, 0, 0] = 0.0

    Ax = np.real(np.fft.ifftn(Axh))
    Ay = np.real(np.fft.ifftn(Ayh))
    Az = np.real(np.fft.ifftn(Azh))
    integrand = Ax * Fyz - Ay * Fxz + Az * Fxy
    return float(np.sum(integrand) * h ** 3 / (16.0 * math.pi ** 2))


def core_diagnostics(nfield, h):
    with torch.no_grad():
        n0 = N0.to(device=nfield.device, dtype=nfield.dtype)
        deviation = torch.linalg.norm(nfield - n0, dim=-1)
        core_volume = float((torch.sum(deviation > 0.5) * h ** 3).detach().cpu())
        max_deviation = float(torch.max(deviation).detach().cpu())
    return {"core_volume": core_volume, "max_deviation": max_deviation}


def save_numpy(path, nfield):
    np.save(path, nfield.detach().cpu().numpy().astype(np.float32))


def make_optimizer(args, params):
    if args.optimizer == "riemannian":
        return None  # manual projected gradient flow in the loop
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    raise ValueError(f"unknown optimizer: {args.optimizer}")


def load_initial_from_file(path, device, dtype):
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"invalid initial field shape: {arr.shape}")
    u = torch.as_tensor(arr, device=device, dtype=dtype)
    u = normalize(u)
    apply_boundary_(u)
    return u


def run(args):
    device = device_from_args(args)
    dtype = torch.float64 if args.float64 else torch.float32
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.p is None:
        args.p = args.charge
    if args.q is None:
        args.q = 1
    args.charge = int(args.p * args.q)
    if args.q_target is None:
        args.q_target = -float(args.charge)

    if args.init:
        u0 = load_initial_from_file(args.init, device, dtype)
        h = 2.0 * args.length / (u0.shape[0] - 1)
        if u0.shape[0] != args.n:
            print(f"init file has n={u0.shape[0]}; overriding --n={args.n}")
            args.n = int(u0.shape[0])
    else:
        u0, h = make_initial_hopf_pq(args.n, args.length, args.scale, args.p, args.q, device, dtype)
    u = torch.nn.Parameter(u0.clone())
    optimizer = make_optimizer(args, [u])

    nfield = normalize(u)
    e2, e4, e_pot, etot = energy_parts_float4(nfield, h, args.a, args.b, args.mu)
    q = hopf_charge_cpu_fft(nfield, h) if not args.no_hopf else float("nan")
    print(f"device={device}, dtype={dtype}, h={h:.6f}")
    print(f"initial: E2={e2:.8f} E4={e4:.8f} Epot={e_pot:.8f} E={etot:.8f} Q_fft={q:.6f}")

    history = []
    best_e = etot
    best_state = nfield.detach().clone()
    last_good_q = q
    last_good_step = 0
    last_good_state = nfield.detach().clone() if math.isfinite(q) and abs(q) >= args.q_guard else None
    t0 = time.time()
    stopped_by_q_guard = False

    for step in range(1, args.steps + 1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        elif u.grad is not None:
            u.grad = None
        nfield = normalize(u)
        energy_result = energy_parts_tensor(nfield, h, args.a, args.b, args.mu)
        if args.mu > 0:
            e2_t, e4_t, e_pot_t, etot_t = energy_result
        else:
            e2_t, e4_t, etot_t = energy_result
            e_pot_t = torch.zeros((), device=nfield.device, dtype=nfield.dtype)
        loss_t = etot_t
        if args.q_penalty > 0:
            q_t = hopf_charge_torch_fft(nfield, h)
            loss_t = loss_t + args.q_penalty * (q_t - args.q_target) ** 2
        loss_t.backward()
        with torch.no_grad():
            if args.optimizer == "riemannian":
                grad = u.grad
                grad_t = grad - torch.sum(grad * nfield, dim=-1, keepdim=True) * nfield
                if args.grad_clip > 0:
                    gnorm = torch.linalg.norm(grad_t)
                    if float(gnorm) > args.grad_clip:
                        grad_t = grad_t * (args.grad_clip / gnorm)
                u.copy_(normalize(u - args.lr * grad_t))
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([u], args.grad_clip)
                optimizer.step()
                u.copy_(normalize(u))
            apply_boundary_(u)

        if step == 1 or step % args.print_every == 0 or step == args.steps:
            nfield = normalize(u)
            e2, e4, e_pot, etot = energy_parts_float4(nfield, h, args.a, args.b, args.mu)
            q = hopf_charge_cpu_fft(nfield, h) if not args.no_hopf and (step == 1 or step % args.hopf_every == 0 or step == args.steps) else float("nan")
            diag = core_diagnostics(nfield, h)
            norm_err = float(torch.max(torch.abs(torch.linalg.norm(nfield, dim=-1) - 1.0)).detach().cpu())
            row = {
                "step": step,
                "E2": e2,
                "E4": e4,
                "Epot": e_pot,
                "E": etot,
                "Q_fft": q,
                "max_norm_err": norm_err,
                **diag,
            }
            history.append(row)
            print(
                f"step={step:6d} E={etot:12.6f} E2={e2:10.6f} E4={e4:10.6f} Epot={e_pot:10.6f} "
                f"Q={q: .5f} coreV={diag['core_volume']:.4f}"
            )
            if etot < best_e:
                best_e = etot
                best_state = nfield.detach().clone()
            if math.isfinite(q) and abs(q) >= args.q_guard:
                last_good_q = q
                last_good_step = step
                last_good_state = nfield.detach().clone()
            if args.q_guard > 0 and math.isfinite(q) and abs(q) < args.q_guard:
                print(f"Q guard triggered: |Q|={abs(q):.6f} < {args.q_guard:.6f}. Stopping.")
                stopped_by_q_guard = True
                break

    final = last_good_state if last_good_state is not None else best_state
    e2, e4, e_pot, etot = energy_parts_float4(final, h, args.a, args.b, args.mu)
    q = hopf_charge_cpu_fft(final, h) if not args.no_hopf else float("nan")
    summary = {
        "backend": "torch",
        "device": str(device),
        "n": args.n,
        "length": args.length,
        "h": h,
        "a": args.a,
        "b": args.b,
        "scale": args.scale,
        "charge": args.charge,
        "p": args.p,
        "q": args.q,
        "init": args.init,
        "steps_requested": args.steps,
        "float64": args.float64,
        "lr": args.lr,
        "q_penalty": args.q_penalty,
        "q_target": args.q_target,
        "mu": args.mu,
        "E2": e2,
        "E4": e4,
        "Epot": e_pot,
        "E": etot,
        "Q_fft": q,
        "stopped_by_q_guard": stopped_by_q_guard,
        "last_good_step": last_good_step,
        "last_good_q": last_good_q,
        "elapsed_sec": time.time() - t0,
        **core_diagnostics(final, h),
    }
    save_numpy(out_dir / f"nfield_q{abs(args.p * args.q)}_torch.npy", final)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("final summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="PyTorch autograd prototype for Q=1 Hopf-Skyrme minimization")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--length", type=float, default=6.0)
    parser.add_argument("--scale", type=float, default=1.8)
    parser.add_argument("--charge", type=int, default=1, help="Compatibility shortcut: sets p=charge, q=1 when --p/--q are omitted")
    parser.add_argument("--p", type=int, default=None, help="Axial Hopf ansatz integer p")
    parser.add_argument("--q", type=int, default=None, help="Axial Hopf ansatz integer q")
    parser.add_argument("--init", default="", help="Optional .npy field to continue from")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--optimizer", choices=["adam", "sgd", "riemannian"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--hopf-every", type=int, default=100)
    parser.add_argument("--q-guard", type=float, default=0.5)
    parser.add_argument("--q-penalty", type=float, default=0.0, help="Soft penalty weight for differentiable Hopf charge")
    parser.add_argument("--q-target", type=float, default=None, help="Target Hopf charge for soft penalty; defaults to -charge")
    parser.add_argument("--mu", type=float, default=0.0, help="Potential barrier strength (stabilizes topology)")
    parser.add_argument("--no-hopf", action="store_true")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="outputs/q1_torch")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
