import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit("CuPy is required. Install cupy-cuda12x or cupy-cuda11x on the GPU machine.") from exc


N0_CPU = [0.0, 0.0, 1.0]


def xp_float(x):
    return float(cp.asnumpy(x))


def make_grid(n: int, length: float, dtype=cp.float32):
    x = cp.linspace(-length, length, n, dtype=dtype)
    h = float(cp.asnumpy(x[1] - x[0]))
    X, Y, Z = cp.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, h


def normalize_field(nfield):
    norm = cp.linalg.norm(nfield, axis=-1, keepdims=True)
    return nfield / cp.maximum(norm, 1e-12)


def apply_boundary(nfield):
    n0 = cp.asarray(N0_CPU, dtype=nfield.dtype)
    nfield[0, :, :, :] = n0
    nfield[-1, :, :, :] = n0
    nfield[:, 0, :, :] = n0
    nfield[:, -1, :, :] = n0
    nfield[:, :, 0, :] = n0
    nfield[:, :, -1, :] = n0


def hopf_q1_initial(n: int, length: float, scale: float = 1.0, dtype=cp.float32):
    X, Y, Z, h = make_grid(n, length, dtype=dtype)
    xs = X / scale
    ys = Y / scale
    zs = Z / scale
    r2 = xs * xs + ys * ys + zs * zs

    denom = 2.0 * zs + 1j * (r2 - 1.0)
    W = 2.0 * (xs + 1j * ys) / (denom + 1e-20)
    absW2 = cp.abs(W) ** 2

    Z1 = 1.0 / cp.sqrt(1.0 + absW2)
    Z2 = W / cp.sqrt(1.0 + absW2)

    nfield = cp.empty((n, n, n, 3), dtype=dtype)
    nfield[..., 0] = 2.0 * cp.real(Z1 * cp.conj(Z2))
    nfield[..., 1] = 2.0 * cp.imag(Z1 * cp.conj(Z2))
    nfield[..., 2] = cp.abs(Z1) ** 2 - cp.abs(Z2) ** 2
    nfield = normalize_field(nfield)
    apply_boundary(nfield)
    return nfield, h


def central_gradient(f, h: float, axis: int):
    return (cp.roll(f, -1, axis=axis) - cp.roll(f, 1, axis=axis)) / (2.0 * h)


def gradients(nfield, h: float):
    gx = central_gradient(nfield, h, 0)
    gy = central_gradient(nfield, h, 1)
    gz = central_gradient(nfield, h, 2)
    gx[0, :, :, :] = 0.0
    gx[-1, :, :, :] = 0.0
    gy[:, 0, :, :] = 0.0
    gy[:, -1, :, :] = 0.0
    gz[:, :, 0, :] = 0.0
    gz[:, :, -1, :] = 0.0
    return gx, gy, gz


def curvature_components(nfield, h: float):
    gx, gy, gz = gradients(nfield, h)
    Fxy = cp.sum(nfield * cp.cross(gx, gy, axis=-1), axis=-1)
    Fxz = cp.sum(nfield * cp.cross(gx, gz, axis=-1), axis=-1)
    Fyz = cp.sum(nfield * cp.cross(gy, gz, axis=-1), axis=-1)
    return Fxy, Fxz, Fyz, gx, gy, gz


def energy_parts(nfield, h: float, a: float = 1.0, b: float = 1.0):
    Fxy, Fxz, Fyz, gx, gy, gz = curvature_components(nfield, h)
    grad_sq = cp.sum(gx * gx + gy * gy + gz * gz, axis=-1)
    f_sq = Fxy * Fxy + Fxz * Fxz + Fyz * Fyz
    vol = h ** 3
    e2 = 0.5 * a * xp_float(cp.sum(grad_sq) * vol)
    e4 = 0.25 * b * xp_float(cp.sum(f_sq) * vol)
    return e2, e4, e2 + e4


def energy(nfield, h: float, a: float = 1.0, b: float = 1.0):
    return energy_parts(nfield, h, a, b)[2]


def laplacian(nfield, h: float):
    lap = (
        cp.roll(nfield, 1, axis=0)
        + cp.roll(nfield, -1, axis=0)
        + cp.roll(nfield, 1, axis=1)
        + cp.roll(nfield, -1, axis=1)
        + cp.roll(nfield, 1, axis=2)
        + cp.roll(nfield, -1, axis=2)
        - 6.0 * nfield
    ) / (h * h)
    lap[0, :, :, :] = 0.0
    lap[-1, :, :, :] = 0.0
    lap[:, 0, :, :] = 0.0
    lap[:, -1, :, :] = 0.0
    lap[:, :, 0, :] = 0.0
    lap[:, :, -1, :] = 0.0
    return lap


def smooth_relax_step(nfield, h: float, tau: float, smooth: float):
    lap = laplacian(nfield, h)
    tangent_lap = lap - cp.sum(lap * nfield, axis=-1, keepdims=True) * nfield
    trial = nfield + tau * smooth * tangent_lap
    trial = normalize_field(trial)
    apply_boundary(trial)
    return trial


def hopf_charge_cpu_fft_from_components(Fxy_gpu, Fxz_gpu, Fyz_gpu, h: float):
    Fxy = cp.asnumpy(Fxy_gpu)
    Fxz = cp.asnumpy(Fxz_gpu)
    Fyz = cp.asnumpy(Fyz_gpu)
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


def hopf_charge_fft(nfield, h: float, fallback_cpu: bool = True):
    Fxy, Fxz, Fyz, _, _, _ = curvature_components(nfield, h)
    Bx = Fyz
    By = -Fxz
    Bz = Fxy
    n = nfield.shape[0]

    try:
        Bxh = cp.fft.fftn(Bx)
        Byh = cp.fft.fftn(By)
        Bzh = cp.fft.fftn(Bz)
    except ImportError as exc:
        if not fallback_cpu:
            print(f"Hopf FFT skipped because cuFFT is unavailable: {exc}")
            return float("nan")
        print(f"cuFFT unavailable; using CPU FFT fallback for Hopf charge: {exc}")
        return hopf_charge_cpu_fft_from_components(Fxy, Fxz, Fyz, h)

    k = 2.0 * math.pi * cp.fft.fftfreq(n, d=h)
    KX, KY, KZ = cp.meshgrid(k, k, k, indexing="ij")
    k2 = KX * KX + KY * KY + KZ * KZ
    k2[0, 0, 0] = 1.0

    Axh = -1j * (KY * Bzh - KZ * Byh) / k2
    Ayh = -1j * (KZ * Bxh - KX * Bzh) / k2
    Azh = -1j * (KX * Byh - KY * Bxh) / k2
    Axh[0, 0, 0] = 0.0
    Ayh[0, 0, 0] = 0.0
    Azh[0, 0, 0] = 0.0

    Ax = cp.real(cp.fft.ifftn(Axh))
    Ay = cp.real(cp.fft.ifftn(Ayh))
    Az = cp.real(cp.fft.ifftn(Azh))

    integrand = Ax * Fyz - Ay * Fxz + Az * Fxy
    return xp_float(cp.sum(integrand) * h ** 3 / (16.0 * math.pi ** 2))


def core_diagnostics(nfield, h: float):
    n0 = cp.asarray(N0_CPU, dtype=nfield.dtype)
    deviation = cp.linalg.norm(nfield - n0, axis=-1)
    mask = deviation > 0.5
    core_volume = xp_float(cp.sum(mask) * h ** 3)
    max_dev = xp_float(cp.max(deviation))
    return {"core_volume": core_volume, "max_deviation": max_dev}


def save_field(path: Path, nfield):
    cp.save(str(path), nfield.astype(cp.float32))


def relax(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = cp.float64 if args.float64 else cp.float32
    nfield, h = hopf_q1_initial(args.n, args.length, args.scale, dtype=dtype)
    cp.cuda.Stream.null.synchronize()

    e2, e4, etot = energy_parts(nfield, h, args.a, args.b)
    q = hopf_charge_fft(nfield, h, fallback_cpu=not args.no_cpu_hopf_fallback) if args.hopf_every > 0 and not args.no_hopf else float("nan")
    print(f"initial: E2={e2:.8f} E4={e4:.8f} E={etot:.8f} Q_fft={q:.6f}")

    history = []
    tau = args.tau
    best_e = etot
    best = nfield.copy()
    t0 = time.time()

    for step in range(1, args.steps + 1):
        current_e = energy(nfield, h, args.a, args.b)
        local_tau = tau
        accepted = False
        for _ in range(14):
            trial = smooth_relax_step(nfield, h, local_tau, args.smooth)
            trial_e = energy(trial, h, args.a, args.b)
            if math.isfinite(trial_e) and trial_e <= current_e:
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
            q = hopf_charge_fft(nfield, h, fallback_cpu=not args.no_cpu_hopf_fallback) if args.hopf_every > 0 and not args.no_hopf and (step % args.hopf_every == 0 or step == 1) else float("nan")
            diag = core_diagnostics(nfield, h)
            norm_err = xp_float(cp.max(cp.abs(cp.linalg.norm(nfield, axis=-1) - 1.0)))
            row = {
                "step": step,
                "E2": e2,
                "E4": e4,
                "E": etot,
                "Q_fft": q,
                "tau": tau,
                "max_norm_err": norm_err,
                **diag,
            }
            history.append(row)
            print(
                f"step={step:6d} E={etot:12.6f} E2={e2:10.6f} E4={e4:10.6f} "
                f"Q={q: .5f} tau={tau:.2e} coreV={diag['core_volume']:.4f}"
            )

    nfield = best
    e2, e4, etot = energy_parts(nfield, h, args.a, args.b)
    q = hopf_charge_fft(nfield, h, fallback_cpu=not args.no_cpu_hopf_fallback) if args.hopf_every > 0 and not args.no_hopf else float("nan")
    summary = {
        "backend": "cupy",
        "n": args.n,
        "length": args.length,
        "h": h,
        "a": args.a,
        "b": args.b,
        "scale": args.scale,
        "steps": args.steps,
        "float64": args.float64,
        "E2": e2,
        "E4": e4,
        "E": etot,
        "Q_fft": q,
        "elapsed_sec": time.time() - t0,
        **core_diagnostics(nfield, h),
    }
    save_field(out_dir / "nfield_q1_gpu.npy", nfield)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("final summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="CuPy GPU prototype for Q=1 Hopf-Skyrme relaxation")
    parser.add_argument("--n", type=int, default=64)
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
    parser.add_argument("--no-hopf", action="store_true", help="Skip FFT Hopf charge calculation")
    parser.add_argument("--no-cpu-hopf-fallback", action="store_true", help="Do not fall back to NumPy FFT when cuFFT is unavailable")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out", default="outputs/q1_gpu_smoke")
    args = parser.parse_args()
    relax(args)


if __name__ == "__main__":
    main()
