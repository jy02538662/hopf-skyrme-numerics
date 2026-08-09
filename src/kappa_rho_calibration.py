#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kappa_rho_calibration.py
========================

任务：使用已有 Q=1 长跑结果标定耦合常数 kappa_rho

公式：
    kappa_rho = E / \int chi^2 d^3x
    chi = |delta n_perp| = sqrt(n_x^2 + n_y^2)
    delta n = n - n0, n0 = (0, 0, 1)

输入：outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy
     （或任何 Q=1 主场文件，shape (N, N, N, 3)）

输出：终端打印 kappa_rho 数值 + 量纲分析
      写到 outputs/kappa_rho_calibration/kappa_rho.json
"""

import json
import sys
from pathlib import Path
import numpy as np


# --------------- 配置 ---------------
DEFAULT_FIELD_PATHS = [
    "outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy",
    "outputs/q1_torch_N96_L12_s30_qpen0_short/nfield_q1_torch.npy",
    "outputs/Q1_p1q1_N96_L12_s30_qpen0_short/nfield_q1_torch.npy",
]
DEFAULT_LENGTH = 10.0   # 盒子尺寸 L （与场文件对应）
DEFAULT_ENERGY = 434.415  # Q=1 已知能量（v1.6.5 §7.3）

# --------------- 主流程 ---------------

def calibrate(field_path, length, energy, out_dir):
    p = Path(field_path)
    if not p.exists():
        print(f"[!] 找不到文件：{field_path}")
        print(f"    请检查路径或自行指定 --field")
        return None

    print(f"[+] 读取场：{field_path}")
    nfield = np.load(p).astype(np.float64)
    if nfield.ndim != 4 or nfield.shape[-1] != 3:
        print(f"[!] 场形状异常：{nfield.shape}，预期 (N, N, N, 3)")
        return None

    N = nfield.shape[0]
    print(f"    shape = {nfield.shape}, dtype = {nfield.dtype}")
    print(f"    盒子 L = {length}, 网格 N = {N}, dx = {length/N:.6f}")

    # 计算 chi = |delta n_perp|
    # 真空方向 n0 = (0, 0, 1)
    nx = nfield[..., 0]
    ny = nfield[..., 1]
    # nz 在原场里已经相对于"原始基准"存储；为稳妥起见不去减 baseline，
    # 因为我们约定：基准方向是 (0,0,1)，所以 delta n_perp 就是 (nx, ny) 本身
    chi = np.sqrt(nx**2 + ny**2)

    print(f"    chi: min={chi.min():.4e}, max={chi.max():.4e}, "
          f"mean={chi.mean():.4e}, std={chi.std():.4e}")

    # 数值积分 ∫ chi^2 d^3x
    dx = length / N
    integral_chi2 = np.sum(chi**2) * (dx**3)
    print(f"    ∫ χ² d³x = {integral_chi2:.6e}")

    # 标定 kappa_rho
    kappa_rho = energy / integral_chi2
    print(f"\n[+] 标定结果：")
    print(f"    kappa_rho = {kappa_rho:.6e}")
    print(f"    量纲：[energy] / [length]³")

    # 量纲检查
    # [kappa_rho]·[length] = [energy density]·[length] = [energy]/[length]²
    # 这正是 Skyrme 拉氏量中 (∇n)² 项的量纲形式
    surface_density = kappa_rho * chi.mean()**2
    print(f"    检查：kappa_rho * <chi^2> = {surface_density:.4e}（应为能量密度量级）")

    # 等效长程引力质量（Q=1 未屏蔽态）
    # M_eff = (4π/3) * kappa_rho * A_chi_0^2 / r_core
    # 其中 A_chi_0 ≈ lim r * <chi>。我们用近场 r ~ N/2 球面作为粗估
    center = N // 2
    r_vec = np.arange(N) - center
    r3d = np.sqrt(r_vec[None, None, :]**2 + r_vec[None, :, None]**2 + r_vec[:, None, None]**2)
    r_sphere = r3d * dx
    r_outer = length / 3  # r ~ L/3 当作"核心边界"
    mask = (r_sphere > r_outer * 0.5) & (r_sphere < r_outer * 1.5)
    if mask.sum() > 0:
        chi_at_r = chi[mask]
        r_at_r = r_sphere[mask]
        # <chi> 在 r ~ r_outer 球面上的平均
        chi_ball = chi_at_r.mean()
        # 粗估 A_chi_0 ≈ r_outer * chi_ball（注意：1/r^α 衰减去限后这只是 proxy）
        A_chi_0_est = r_outer * chi_ball
        M_eff_est = (4 * np.pi / 3) * kappa_rho * A_chi_0_est**2 / r_outer
        print(f"\n[+] Q=1 等效引力质量粗估：")
        print(f"    r_core 范围 = {0.5*r_outer:.3f} ~ {1.5*r_outer:.3f}")
        print(f"    <chi>|_{{.}}    = {chi_ball:.4e}")
        print(f"    A_chi_0 粗估 = {A_chi_0_est:.4e}")
        print(f"    M_eff 粗估  = {M_eff_est:.4e}")
        print(f"    （量纲同 [energy]·[time]²/[length]；需与已知粒子质量比较才能定标）")
    else:
        A_chi_0_est = None
        M_eff_est = None

    # 保存结果
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "field": field_path,
        "N": int(N),
        "L": float(length),
        "energy": float(energy),
        "integral_chi2": float(integral_chi2),
        "kappa_rho": float(kappa_rho),
        "kappa_rho_units": "[energy] / [length]^3",
        "A_chi_0_estimate": float(A_chi_0_est) if A_chi_0_est is not None else None,
        "M_eff_estimate": float(M_eff_est) if M_eff_est is not None else None,
        "notes": "粗估来自 N/2 附近球壳，仅用作量纲一致性检查",
    }
    out_json = out_dir / "kappa_rho.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 写入：{out_json}")

    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="标定 κ_ρ —— Faddeev-Skyrme 模型能量密度到远场的耦合常数"
    )
    ap.add_argument("--field", type=str, default=None,
                    help="Q=1 主场文件路径 (.npy)；不指定则按顺序尝试默认路径")
    ap.add_argument("--length", type=float, default=DEFAULT_LENGTH,
                    help=f"盒子尺寸 L（默认 {DEFAULT_LENGTH}）")
    ap.add_argument("--energy", type=float, default=DEFAULT_ENERGY,
                    help=f"Q=1 已知能量 E（默认 {DEFAULT_ENERGY}）")
    ap.add_argument("--out", type=str, default="outputs/kappa_rho_calibration",
                    help="输出目录")
    args = ap.parse_args()

    print("=" * 60)
    print("  κ_ρ 标定任务（v1.6.6 §G.5）")
    print("=" * 60)

    if args.field is not None:
        calibrate(args.field, args.length, args.energy, Path(args.out))
    else:
        # 自动尝试默认路径
        for f in DEFAULT_FIELD_PATHS:
            if Path(f).exists():
                calibrate(f, args.length, args.energy, Path(args.out))
                return
        print("[!] 没找到任何默认 Q=1 场文件，请用 --field 指定")
        print("    默认查找列表：")
        for f in DEFAULT_FIELD_PATHS:
            print(f"      - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
