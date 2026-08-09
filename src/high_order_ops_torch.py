"""High-order (4th-order) finite-difference operators for the Hopf-Skyrme field.

Layout convention matches hopf_skyrme_torch.py:
    nfield: [Nx, Ny, Nz, 3], vacuum n0 = (0,0,1), h = 2*length/(N-1).

Internally we convert to channel-first [1, 3, Nx, Ny, Nz] and use grouped
conv3d so the same 1D stencil acts on all three field components at once.

Boundary policy: 'valid' cropping. conv3d with zero-padding is WRONG here
because the vacuum is (0,0,1), not the zero vector. We therefore compute on
the interior only and crop `width` cells from every face.

Run this file directly to print an O(h^4) convergence self-test table.
"""

import argparse

import torch
import torch.nn.functional as F


# 4th-order central stencils (5-point)
STENCIL_D1 = [1.0, -8.0, 0.0, 8.0, -1.0]          # * 1/(12 h)   -> d/dx
STENCIL_D2 = [-1.0, 16.0, -30.0, 16.0, -1.0]       # * 1/(12 h^2) -> d2/dx2
STENCIL_HALF_WIDTH = 2


def to_channels_first(nfield):
    """[Nx, Ny, Nz, C] -> [1, C, Nx, Ny, Nz]."""
    return nfield.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def to_channels_last(u5):
    """[1, C, Nx, Ny, Nz] -> [Nx, Ny, Nz, C]."""
    return u5.squeeze(0).permute(1, 2, 3, 0).contiguous()


def _grouped_kernel(stencil, axis, channels, device, dtype):
    """Build a [C, 1, k, k, k] grouped conv kernel placing `stencil` along `axis`."""
    k = len(stencil)
    c = STENCIL_HALF_WIDTH
    coeff = torch.as_tensor(stencil, device=device, dtype=dtype)
    weight = torch.zeros((channels, 1, k, k, k), device=device, dtype=dtype)
    if axis == 0:
        weight[:, 0, :, c, c] = coeff
    elif axis == 1:
        weight[:, 0, c, :, c] = coeff
    elif axis == 2:
        weight[:, 0, c, c, :] = coeff
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return weight


def _apply(u5, stencil, axis, scale):
    channels = u5.shape[1]
    weight = _grouped_kernel(stencil, axis, channels, u5.device, u5.dtype)
    out = F.conv3d(u5, weight, padding=0, groups=channels)
    return out * scale


def grad4(u5, h):
    """4th-order gradient. Input [1,C,Nx,Ny,Nz]; returns gx,gy,gz each cropped by 2 cells."""
    s = 1.0 / (12.0 * h)
    gx = _apply(u5, STENCIL_D1, 0, s)
    gy = _apply(u5, STENCIL_D1, 1, s)
    gz = _apply(u5, STENCIL_D1, 2, s)
    return gx, gy, gz


def lap4(u5, h):
    """4th-order Laplacian. Input [1,C,Nx,Ny,Nz]; returns [1,C,Nx-4,Ny-4,Nz-4]."""
    s = 1.0 / (12.0 * h * h)
    return _apply(u5, STENCIL_D2, 0, s) + _apply(u5, STENCIL_D2, 1, s) + _apply(u5, STENCIL_D2, 2, s)


def crop_valid(u5, width=STENCIL_HALF_WIDTH):
    """Crop `width` cells from every spatial face of a [1,C,Nx,Ny,Nz] tensor."""
    w = width
    return u5[:, :, w:-w, w:-w, w:-w].contiguous()


def stress_tensor4(gx, gy, gz):
    """Build S_ij = d_i n . d_j n - 0.5 delta_ij (d_k n . d_k n).

    gx,gy,gz: [1, C, X, Y, Z]. Returns (G, S), each [1, 3, 3, X, Y, Z].
    """
    grad = torch.stack([gx, gy, gz], dim=2)  # [B, C, I, X, Y, Z]
    # G_ij = sum_a d_i n_a d_j n_a  (contract over field component C)
    G = torch.einsum("bcixyz,bcjxyz->bijxyz", grad, grad)
    trace = G[:, 0, 0] + G[:, 1, 1] + G[:, 2, 2]
    S = G.clone()
    for i in range(3):
        S[:, i, i] = S[:, i, i] - 0.5 * trace
    return G, S


# ---------------------------------------------------------------------------
# Self-test: analytic convergence table (run this file directly)
# ---------------------------------------------------------------------------

def _make_test_field(n, length, device, dtype):
    """Scalar-ish test: put f=(x, x^2, sin(x)+cos(y)+... ) into 3 channels."""
    x = torch.linspace(-length, length, n, device=device, dtype=dtype)
    h = float((x[1] - x[0]).item())
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    f = torch.empty((n, n, n, 3), device=device, dtype=dtype)
    f[..., 0] = X                      # d/dx = 1 ; lap = 0
    f[..., 1] = X * X + Y * Y + Z * Z  # lap = 6
    f[..., 2] = torch.sin(X) * torch.cos(Y)  # lap = -2 sin x cos y
    return f, h, (X, Y, Z)


def _self_test(device, dtype):
    print(f"convergence self-test  device={device} dtype={dtype}")
    print(f"{'N':>6} {'h':>10} {'err_d1':>14} {'err_lap':>14} {'rate_d1':>9} {'rate_lap':>9}")
    prev = None
    for n in (48, 72, 108, 162):
        length = 3.14159265358979
        f, h, (X, Y, Z) = _make_test_field(n, length, device, dtype)
        u5 = to_channels_first(f)
        gx, gy, gz = grad4(u5, h)
        lap = lap4(u5, h)
        w = STENCIL_HALF_WIDTH

        # d/dx of channel 0 should be 1
        d1_num = gx[0, 0]
        err_d1 = (d1_num - 1.0).abs().max().item()

        # laplacian of channel 1 should be 6
        lap_ch1 = lap[0, 1]
        err_lap1 = (lap_ch1 - 6.0).abs().max().item()

        # laplacian of channel 2 should be -2 sin x cos y (cropped)
        Xc = X[w:-w, w:-w, w:-w]
        Yc = Y[w:-w, w:-w, w:-w]
        target = -2.0 * torch.sin(Xc) * torch.cos(Yc)
        err_lap2 = (lap[0, 2] - target).abs().max().item()
        err_lap = max(err_lap1, err_lap2)

        if prev is None:
            rate_d1 = rate_lap = float("nan")
        else:
            import math
            ratio = prev[0] / h
            rate_d1 = math.log(prev[1] / max(err_d1, 1e-300)) / math.log(ratio)
            rate_lap = math.log(prev[2] / max(err_lap, 1e-300)) / math.log(ratio)
        print(f"{n:>6} {h:>10.5f} {err_d1:>14.3e} {err_lap:>14.3e} {rate_d1:>9.3f} {rate_lap:>9.3f}")
        prev = (h, err_d1, err_lap)
    print("expected convergence rate ~ 4.0 for both columns (linear-in-x d1 may be ~machine).")


def main():
    parser = argparse.ArgumentParser(description="4th-order FD operators + convergence self-test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))
    dtype = torch.float64 if args.float64 else torch.float32
    _self_test(device, dtype)


if __name__ == "__main__":
    main()
