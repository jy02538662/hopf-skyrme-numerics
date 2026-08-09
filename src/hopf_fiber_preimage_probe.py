#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hopf_fiber_preimage_probe.py
=============================

任务：可视化 Q=2 主场的 Hopf 纤维 preimage，验证对称群论断

输出：
  outputs/hopf_fiber_preimage_Q2/
    preimage_*.vtk          # 等值面（ParaView 查看）
    summary.json            # 对称操作检测结果

依赖：numpy, scipy, scikit-image（pyvista 可选，仅用于 .vtk 输出）
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


# --------------- 工具函数 ---------------

def n_to_ab_vec(nfield):
    """
    从 S^2 场 n 重构 S^3 Cayley-Klein 参数 (a, b)。
    返回 a (实数, shape N³) 和 b (复数, shape N³)。

    标准 Hopf 纤维化：
      n_x = 2 Re(conj(a) b)
      n_y = 2 Im(conj(a) b)
      n_z = |a|^2 - |b|^2
    取全局规范 a >= 0（实数）。
    """
    nx = nfield[..., 0].astype(np.float64)
    ny = nfield[..., 1].astype(np.float64)
    nz = nfield[..., 2].astype(np.float64)

    a_mag = np.sqrt(np.clip((1.0 + nz) / 2.0, 0.0, 1.0))
    b_mag = np.sqrt(np.clip((1.0 - nz) / 2.0, 0.0, 1.0))
    b_phase = np.arctan2(ny, nx)

    a = a_mag  # 实数，全局规范
    b = b_mag * np.exp(1j * b_phase)
    return a, b


def hopf_w(a, b):
    """Hopf 坐标 W = b/a，复数"""
    return b / (a + 1e-12)


def extract_phase_unwrapped(W):
    """
    取 W 的相位 Phi，并对每个 z 切片做 1D unwrap（沿 phi 方向）。
    返回 Phi shape (N, N, N)，连续无跳变。
    """
    Phi_raw = np.angle(W)
    Phi = np.zeros_like(Phi_raw)
    # 沿 phi 方向（第二维）逐 z 切片 unwrap
    for iz in range(Phi_raw.shape[2]):
        Phi[:, :, iz] = np.unwrap(Phi_raw[:, :, iz], axis=1)
    # 沿 z 方向再做一次匹配相邻切片（避免层间跳变）
    for ix in range(Phi.shape[0]):
        for iy in range(Phi.shape[1]):
            Phi[ix, iy, :] = np.unwrap(Phi[ix, iy, :])
    return Phi


# --------------- 等值面提取 ---------------

def extract_isosurface(Phi, level, length):
    """
    从 3D 标量场 Phi 提取 Phi = level 的等值面。
    返回 (verts, faces), verts 居中到 [-L/2, L/2]^3。

    用 marching_cubes；Phi 有连续相位时该函数会正常工作。
    """
    from skimage.measure import marching_cubes
    N = Phi.shape[0]
    dx = length / N
    # marching_cubes 的 level 参数：以 Phi 减去 level，得到 field=0 的等值面
    diff = Phi - level
    try:
        verts, faces, _, _ = marching_cubes(diff, level=0.0, step_size=2)
        verts = verts * dx - length / 2.0  # 居中到原点
        return verts, faces
    except (ValueError, RuntimeError) as e:
        # 等值面不存在（断缺区域）
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)


# --------------- 对称操作测试 ---------------

# 注意：我们在 *物理坐标* 上施加操作。
# C2(z) 旋转 (x,y,z) -> (-x,-y,z)
# C2(x) 旋转 (x,y,z) -> (x,-y,-z)
# C2(y) 旋转 (x,y,z) -> (-x,y,-z)
SYMMETRY_OPS = {
    "C2z": np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]),
    "C2x": np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
    "C2y": np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]),
}


def apply_op(verts, R):
    """对 verts 应用旋转 R"""
    if len(verts) == 0:
        return verts
    return verts @ R.T


def test_symmetry_equivalence(verts_orig, verts_transformed, length,
                              max_mean_dist_frac=0.05):
    """
    判断 verts_transformed 是否与 verts_orig 几何重合（位置重叠）。
    返回 (pass_bool, mean_distance)。
    """
    from scipy.spatial import cKDTree
    if len(verts_orig) == 0 or len(verts_transformed) == 0:
        return False, float("inf")
    tree = cKDTree(verts_orig)
    dists, _ = tree.query(verts_transformed, k=1)
    mean_dist = float(dists.mean())
    return mean_dist < max_mean_dist_frac * length, mean_dist


# --------------- 主流程 ---------------

def main():
    ap = argparse.ArgumentParser(
        description="Hopf 纤维 preimage 探针（v1.6.6 §B.2）"
    )
    ap.add_argument("--field", required=True,
                    help="Q=2 主场 .npy，shape (N,N,N,3)")
    ap.add_argument("--length", type=float, default=12.0,
                    help="盒子尺寸 L")
    ap.add_argument("--n-phases", type=int, default=8,
                    help="提取的等值面数量")
    ap.add_argument("--out", default="outputs/hopf_fiber_preimage_Q2",
                    help="输出目录")
    ap.add_argument("--tolerance-frac", type=float, default=0.05,
                    help="preimage 重叠容差（相对 L）")
    args = ap.parse_args()

    print("=" * 60)
    print("  Hopf 纤维 preimage 探针（v1.6.6 §B.2）")
    print("=" * 60)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载场 ----
    nfield = np.load(args.field).astype(np.float64)
    print(f"[+] 场 shape = {nfield.shape}, L = {args.length}")
    if nfield.ndim != 4 or nfield.shape[-1] != 3:
        print(f"[!] 形状异常")
        return 1
    N = nfield.shape[0]

    # ---- Cayley-Klein 重建 ----
    print("[+] Cayley-Klein 参数化 n -> (a, b)")
    a, b = n_to_ab_vec(nfield)
    W = hopf_w(a, b)
    print(f"    |W|: min={np.abs(W).min():.3e}, max={np.abs(W).max():.3e}")

    # ---- 相位解缠 ----
    print("[+] 相位解缠（沿 phi 与 z 方向逐切片 unwrap）")
    Phi = extract_phase_unwrapped(W)
    print(f"    Phi: min={Phi.min():.3f}, max={Phi.max():.3f}")

    # ---- 等值面提取 ----
    phase_values = np.linspace(0.0, 2.0 * np.pi, args.n_phases, endpoint=False)
    isosurfaces = []  # list of (Phi_0, verts, faces)

    print(f"[+] 提取 {args.n_phases} 个等值面...")
    for Phi_0 in phase_values:
        verts, faces = extract_isosurface(Phi, Phi_0, args.length)
        isosurfaces.append((Phi_0, verts, faces))
        print(f"    Phi_0 = {Phi_0:.3f}: verts = {len(verts)}")

    # ---- 对称操作测试 ----
    print("\n[+] 对称操作测试")
    sym_results = {}
    summary_lines = []

    for op_name, R in SYMMETRY_OPS.items():
        sym_results[op_name] = {}
        op_pass_count = 0
        for Phi_0, verts_i, _ in isosurfaces:
            if len(verts_i) == 0:
                sym_results[op_name][f"{Phi_0:.3f}"] = {
                    "pass": False,
                    "mean_dist": None,
                    "reason": "empty isosurface"
                }
                continue

            # 施加旋转
            verts_T = apply_op(verts_i, R)

            # 对 C2(z)：Phi -> Phi + pi，等值面 i 应等值面 Phi_0 + pi
            if op_name == "C2z":
                target_level = (Phi_0 + np.pi) % (2.0 * np.pi)
            # 对 C2(x), C2(y)：Phi -> -Phi（或更复杂），这里用 *距离最近* 等值面判定
            else:
                # 找到"几何最像"的目标等值面（按点数匹配）
                target_level = (-Phi_0) % (2.0 * np.pi)

            # 在已提取的等值面里找最贴近 target 的
            best_target = None
            best_target_label = None
            best_label_diff = float("inf")
            for j, (Pj, verts_j, _) in enumerate(isosurfaces):
                diff = abs(((Pj - target_level + np.pi) % (2 * np.pi)) - np.pi)
                if diff < best_label_diff:
                    best_label_diff = diff
                    best_target = verts_j
                    best_target_label = Pj

            if best_target is None or len(best_target) == 0:
                sym_results[op_name][f"{Phi_0:.3f}"] = {
                    "pass": False,
                    "mean_dist": None,
                    "reason": "no target isosurface"
                }
                continue

            # 几何匹配
            pass_bool, mean_dist = test_symmetry_equivalence(
                best_target, verts_T, args.length,
                max_mean_dist_frac=args.tolerance_frac
            )
            sym_results[op_name][f"{Phi_0:.3f}"] = {
                "pass": bool(pass_bool),
                "mean_dist": mean_dist,
                "target_label": f"{best_target_label:.3f}"
            }
            if pass_bool:
                op_pass_count += 1

        # 总结
        total = args.n_phases
        sym_pass_str = f"{op_pass_count}/{total}"
        if op_pass_count == total:
            verdict = "PASS"
        elif op_pass_count == 0:
            verdict = "FAIL"
        else:
            verdict = f"PARTIAL ({op_pass_count}/{total})"
        line = f"    {op_name}: {verdict}"
        print(line)
        summary_lines.append(line)

    # ---- 结论 ----
    n_pass = sum(
        1 for op in SYMMETRY_OPS
        if all(
            v["pass"] for v in sym_results[op].values()
            if v.get("reason") is None
        )
    )
    print(f"\n[+] 总计通过对称操作数：{n_pass}/{len(SYMMETRY_OPS)}")
    print(f"    预期结果：仅 C2z 通过")

    conclusion = {
        "C2z": "preimage 不变（相位平移），预期 PASS",
        "C2x": "preimage 镜像反转，预期 FAIL",
        "C2y": "preimage 镜像反转，预期 FAIL",
    }
    print(f"[+] 结论对照：")
    for k, v in conclusion.items():
        actual = "PASS" if all(
            v_["pass"] for v_ in sym_results[k].values()
            if v_.get("reason") is None
        ) else "FAIL"
        expected = "PASS" if k == "C2z" else "FAIL"
        match = "✓" if actual == expected else "✗"
        print(f"    {k}: 实际 {actual} / 预期 {expected}  {match}")

    # ---- 保存 .vtk（若 pyvista 可用） ----
    try:
        import pyvista as pv
        HAVE_PV = True
    except ImportError:
        HAVE_PV = False

    if HAVE_PV:
        print(f"\n[+] 保存 .vtk 文件到 {out_dir}")
        for i, (Phi_0, verts, faces) in enumerate(isosurfaces):
            if len(verts) > 0:
                # pyvista 期望 faces 形状为 (n, 4)，每行 [3, v1, v2, v3]
                if len(faces) > 0:
                    faces_pv = np.hstack(
                        [np.full((len(faces), 1), 3, dtype=int), faces]
                    ).ravel()
                else:
                    faces_pv = np.array([], dtype=int)
                mesh = pv.PolyData(verts, faces_pv)
                mesh.save(str(out_dir / f"preimage_{i}_Phi_{Phi_0:.3f}.vtk"))
        print(f"    在 ParaView 中打开查看双环几何")
    else:
        print(f"\n[~] 未安装 pyvista，跳过 .vtk 输出")
        print(f"    如需 .vtk：pip install pyvista")

    # ---- summary.json ----
    summary = {
        "field": args.field,
        "L": args.length,
        "N": int(N),
        "n_phases": args.n_phases,
        "tolerance_frac": args.tolerance_frac,
        "symmetry_results": sym_results,
        "conclusion": conclusion,
        "expected_outcome": "Only C2z PASS",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 完成！")
    print(f"    详细结果：{out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
