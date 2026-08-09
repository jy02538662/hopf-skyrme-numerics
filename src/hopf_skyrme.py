import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


N0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def make_grid(n: int, length: float):
    x = np.linspace(-length, length, n, dtype=np.float64)
    h = float(x[1] - x[0])
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, h


def normalize_field(nfield: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(nfield, axis=-1, keepdims=True)
    return nfield / np.maximum(norm, 1e-14)


def apply_boundary(nfield: np.ndarray) -> None:
    nfield[0, :, :, :] = N0
    nfield[-1, :, :, :] = N0
    nfield[:, 0, :, :] = N0
    nfield[:, -1, :, :] = N0
    nfield[:, :, 0, :] = N0
    nfield[:, :, -1, :] = N0


def hopf_q1_initial(n: int, length: float, scale: float = 1.0) -> tuple[np.ndarray, float]:
    X, Y, Z, h = make_grid(n, length)
    xs = X / scale
    ys = Y / scale
    zs = Z / scale
    r2 = xs * xs + ys * ys + zs * zs

    denom = 2.0 * zs + 1j * (r2 - 1.0)
    W = 2.0 * (xs + 1j * ys) / (denom + 1e-30)
    absW2 = np.abs(W) ** 2

    Z1 = 1.0 / np.sqrt(1.0 + absW2)
    Z2 = W / np.sqrt(1.0 + absW2)

    nfield = np.empty((n, n, n, 3), dtype=np.float64)
    nfield[..., 0] = 2.0 * np.real(Z1 * np.conj(Z2))
    nfield[..., 1] = 2.0 * np.imag(Z1 * np.conj(Z2))
    nfield[..., 2] = np.abs(Z1) ** 2 - np.abs(Z2) ** 2
    nfield = normalize_field(nfield)
    apply_boundary(nfield)
    return nfield, h


def gradients(nfield: np.ndarray, h: float):
    gx, gy, gz = np.gradient(nfield, h, h, h, axis=(0, 1, 2), edge_order=2)
    return gx, gy, gz


def curvature_components(nfield: np.ndarray, h: float):
    gx, gy, gz = gradients(nfield, h)
    Fxy = np.einsum("...i,...i->...", nfield, np.cross(gx, gy))
    Fxz = np.einsum("...i,...i->...", nfield, np.cross(gx, gz))
    Fyz = np.einsum("...i,...i->...", nfield, np.cross(gy, gz))
    return Fxy, Fxz, Fyz, gx, gy, gz


def energy_parts(nfield: np.ndarray, h: float, a: float = 1.0, b: float = 1.0):
    Fxy, Fxz, Fyz, gx, gy, gz = curvature_components(nfield, h)
    grad_sq = np.sum(gx * gx + gy * gy + gz * gz, axis=-1)
    f_sq = Fxy * Fxy + Fxz * Fxz + Fyz * Fyz
    vol = h ** 3
    e2 = 0.5 * a * float(np.sum(grad_sq) * vol)
    e4 = 0.25 * b * float(np.sum(f_sq) * vol)
    return e2, e4, e2 + e4


def energy(nfield: np.ndarray, h: float, a: float = 1.0, b: float = 1.0) -> float:
    return energy_parts(nfield, h, a, b)[2]


def laplacian(nfield: np.ndarray, h: float) -> np.ndarray:
    lap = (
        np.roll(nfield, 1, axis=0)
        + np.roll(nfield, -1, axis=0)
        + np.roll(nfield, 1, axis=1)
        + np.roll(nfield, -1, axis=1)
        + np.roll(nfield, 1, axis=2)
        + np.roll(nfield, -1, axis=2)
        - 6.0 * nfield
    ) / (h * h)
    lap[0, :, :, :] = 0.0
    lap[-1, :, :, :] = 0.0
    lap[:, 0, :, :] = 0.0
    lap[:, -1, :, :] = 0.0
    lap[:, :, 0, :] = 0.0
    lap[:, :, -1, :] = 0.0
    return lap


def numerical_gradient(nfield: np.ndarray, h: float, a: float, b: float, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(nfield)
    base = nfield.copy()
    n = nfield.shape[0]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            for k in range(1, n - 1):
                ni = base[i, j, k].copy()
                helper = np.array([1.0, 0.0, 0.0]) if abs(ni[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
                e1 = helper - np.dot(helper, ni) * ni
                e1 /= max(np.linalg.norm(e1), 1e-14)
                e2 = np.cross(ni, e1)
                for basis in (e1, e2):
                    plus = base.copy()
                    minus = base.copy()
                    plus[i, j, k] = ni + eps * basis
                    minus[i, j, k] = ni - eps * basis
                    plus[i, j, k] /= np.linalg.norm(plus[i, j, k])
                    minus[i, j, k] /= np.linalg.norm(minus[i, j, k])
                    ep = energy(plus, h, a, b)
                    em = energy(minus, h, a, b)
                    deriv = (ep - em) / (2.0 * eps)
                    grad[i, j, k] += deriv * basis / (h ** 3)
    return grad


def smooth_relax_step(nfield: np.ndarray, h: float, tau: float, smooth: float) -> np.ndarray:
    lap = laplacian(nfield, h)
    tangent_lap = lap - np.sum(lap * nfield, axis=-1, keepdims=True) * nfield
    trial = nfield + tau * smooth * tangent_lap
    trial = normalize_field(trial)
    apply_boundary(trial)
    return trial


def hopf_charge_fft(nfield: np.ndarray, h: float) -> float:
    Fxy, Fxz, Fyz, _, _, _ = curvature_components(nfield, h)
    Bx = Fyz
    By = -Fxz
    Bz = Fxy
    n = nfield.shape[0]

    Bxh = np.fft.fftn(Bx)
    Byh = np.fft.fftn(By)
    Bzh = np.fft.fftn(Bz)

    k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
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


def core_diagnostics(nfield: np.ndarray, h: float):
    deviation = np.linalg.norm(nfield - N0, axis=-1)
    mask = deviation > 0.5
    core_volume = float(np.sum(mask) * h ** 3)
    max_dev = float(np.max(deviation))
    return {"core_volume": core_volume, "max_deviation": max_dev}


def relax(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    nfield, h = hopf_q1_initial(args.n, args.length, args.scale)
    e2, e4, etot = energy_parts(nfield, h, args.a, args.b)
    q = hopf_charge_fft(nfield, h) if args.hopf_every > 0 else float("nan")
    print(f"initial: E2={e2:.8f} E4={e4:.8f} E={etot:.8f} Q_fft={q:.6f}")

    history = []
    tau = args.tau
    best_e = etot
    best = nfield.copy()
    t0 = time.time()

    for step in range(1, args.steps + 1):
        accepted = False
        current_e = energy(nfield, h, args.a, args.b)
        local_tau = tau
        for _ in range(12):
            trial = smooth_relax_step(nfield, h, local_tau, args.smooth)
            trial_e = energy(trial, h, args.a, args.b)
            if np.isfinite(trial_e) and trial_e <= current_e:
                nfield = trial
                tau = min(local_tau * 1.02, args.tau_max)
                accepted = True
                if trial_e < best_e:
                    best_e = trial_e
                    best = trial.copy()
                break
            local_tau *= 0.5
        if not accepted:
            tau *= 0.5

        if step == 1 or step % args.print_every == 0 or step == args.steps:
            e2, e4, etot = energy_parts(nfield, h, args.a, args.b)
            q = hopf_charge_fft(nfield, h) if args.hopf_every > 0 and (step % args.hopf_every == 0 or step == 1) else float("nan")
            diag = core_diagnostics(nfield, h)
            max_norm_err = float(np.max(np.abs(np.linalg.norm(nfield, axis=-1) - 1.0)))
            row = {
                "step": step,
                "E2": e2,
                "E4": e4,
                "E": etot,
                "Q_fft": q,
                "tau": tau,
                "max_norm_err": max_norm_err,
                **diag,
            }
            history.append(row)
            print(
                f"step={step:6d} E={etot:12.6f} E2={e2:10.6f} E4={e4:10.6f} "
                f"Q={q: .5f} tau={tau:.2e} coreV={diag['core_volume']:.4f}"
            )

    nfield = best
    e2, e4, etot = energy_parts(nfield, h, args.a, args.b)
    q = hopf_charge_fft(nfield, h) if args.hopf_every > 0 else float("nan")
    summary = {
        "n": args.n,
        "length": args.length,
        "h": h,
        "a": args.a,
        "b": args.b,
        "scale": args.scale,
        "steps": args.steps,
        "E2": e2,
        "E4": e4,
        "E": etot,
        "Q_fft": q,
        "elapsed_sec": time.time() - t0,
        **core_diagnostics(nfield, h),
    }
    np.save(out_dir / "nfield_q1.npy", nfield.astype(np.float32))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("final summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def smoke_gradient(args):
    nfield, h = hopf_q1_initial(args.n, args.length, args.scale)
    print("computing numerical gradient on tiny grid; this is for debugging only")
    grad = numerical_gradient(nfield, h, args.a, args.b)
    print(f"grad_norm_inf={np.max(np.linalg.norm(grad, axis=-1)):.6e}")


def main():
    parser = argparse.ArgumentParser(description="CPU prototype for Q=1 Hopf-Skyrme relaxation")
    parser.add_argument("--mode", choices=["relax", "smoke-gradient"], default="relax")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--length", type=float, default=6.0)
    parser.add_argument("--scale", type=float, default=1.8)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--tau-max", type=float, default=0.05)
    parser.add_argument("--smooth", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--hopf-every", type=int, default=100)
    parser.add_argument("--out", default="outputs/q1_cpu_smoke")
    args = parser.parse_args()

    if args.mode == "relax":
        relax(args)
    else:
        smoke_gradient(args)


if __name__ == "__main__":
    main()
