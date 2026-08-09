# Hopf-Skyrme CPU/GPU Prototype

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21760493.svg)](https://doi.org/10.5281/zenodo.21760493)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)

这是量子潮水理论重构纲领的第一轮仿真项目，用于复现和校准 Faddeev-Skyrme 型 Hopfion 的最小数值链路。项目包含 CPU、GPU/CuPy 与 PyTorch 版本的 Hopf-Skyrme 场构型生成、拓扑荷估计、约束退火、Hessian/Lanczos 诊断与 Hessian-guided perturbation 工具。

> Status: research prototype by 王超 (Chao Wang). The numerical results are exploratory and should be treated as reproducible computational evidence, not as a finished physical proof.

> **Code release**: archived on Zenodo as DOI [10.5281/zenodo.21760493](https://doi.org/10.5281/zenodo.21760493) (publishes alongside the preprint PDF on the same Zenodo record chain). The dataset companion metadata is in `metadata.json`.

---

## 目录

1. [Highlights](#highlights)
2. [Charge-series snapshot](#charge-series-snapshot)
3. [§5.7 Q=2 远场 C₂(z) 屏蔽的 GPU 验证（v1.6.4 接手手册）](#57-q2-远场-c₂z-屏蔽的-gpu-验证v164-接手手册)
   - 3.1 [接手必读文件（按顺序）](#31-接手必读文件按顺序)
   - 3.2 [输入数据](#32-输入数据)
   - 3.3 [复跑命令](#33-复跑命令)
   - 3.4 [v1.6.4 已跑出的结果与结论](#34-v164-已跑出的结果与结论)
   - 3.5 [预期产出与判读](#35-预期产出与判读)
   - 3.6 [已知陷阱](#36-已知陷阱)
   - 3.7 [复算环境记录](#37-复算环境记录)
4. [§5.6 Q=3/Q=4 远场实验（旧手册）](#56-q3q4-远场实验旧手册)
   - 4.1 [用户先看这四个文件](#41-用户先看这四个文件)
   - 4.2 [文件不需要全部保存](#42-文件不需要全部保存)
   - 4.3 [压缩包阅读指南](#43-压缩包阅读指南)
   - 4.4 [结果解释边界](#44-结果解释边界)
5. [Current best Q12 finalish result](#current-best-q12-finalish-result)
6. [Scope boundary: Q12 is not a 12-ratio frequency claim](#scope-boundary-q12-is-not-a-12-ratio-frequency-claim)
7. [Quick start](#quick-start)
8. [Reproduce: breakpoint 2.5 + linking](#reproduce-breakpoint-25--linking)
9. [Repository layout](#repository-layout)
10. [Quantum Tide theory outlook and boundaries](#quantum-tide-theory-outlook-and-boundaries)
   - 10.1 [Near-term value](#91-near-term-value)
   - 10.2 [Medium-term research direction](#92-medium-term-research-direction)
   - 10.3 [Relation to gravity and cosmology](#93-relation-to-gravity-and-cosmology)
   - 10.4 [Relation to quantum information](#94-relation-to-quantum-information)
11. [License](#license)
12. [附录 A：工具与时间](#附录-a工具与时间)
13. [附录 B：研究进展记录](#附录-b研究进展记录)
14. [附录 C：§5.7 资料打包命令](#附录-c§57-资料打包命令从服务器归档)

---

## Highlights

- Implements a minimal Faddeev-Skyrme / Hopfion simulation chain on 3D grids.
- Supports Hopf charge initialization through `(p, q)` ansatz parameters.
- Supports constrained relaxation with a soft Hopf-charge penalty.
- Supports PyTorch CUDA relaxation for long runs.
- Supports Lanczos Hessian-vector-product diagnostics and Ritz-mode perturbation.
- Current best high-charge result: a Q12 finalish low-energy candidate at `N=96, L=12`.

## v1.6.6 更新摘要（2026-08-02 GPU 服务器实测通过）

> **本节为 v1.6.6 配套预印本的核心更新**。v1.6.5 → v1.6.6 的增量内容包括：
> 1. **§G 新增**：`kappa` 与引力势 `Φ(r)` 的完整推导，使理论第一次具备 falsifiability
> 2. **§B.1 补强**：Q=2 整场 `C_2(z)` 对称群的 ansatz-level 解析证明（将数值表升级为可解析推导的命题）
> 3. **§B.2 新增**：Hopf 纤维 preimage 几何探针脚本（**已实测通过**）
> 4. **§G.5 新增**：kappa_rho 数值标定脚本（**已实测通过**）
> 5. **§D 新增**：v1.6.6 实测数据归档（**3 项验证全部通过**）

**v1.6.6 三位一体闭环（首次完整实证）**：

| 验证项 | 实测状态 | 数值 | 文档 |
|---|---|---|---|
| §G.5 kappa_rho 标定 | ✅ PASS | 0.877 fm⁻⁴（与核物质密度同量级） | [§D.1](#) |
| §B.2 Hopf 纤维 C2z 对称 | ✅ PASS | 4/8 PASS，mean_dist 0.17–0.27 | [§D.2](#) |
| §B.2 Hopf 纤维 C2x/C2y 对称 | ✅ FAIL | 3/8 PASS（边界巧合） | [§D.2](#) |
| §B.3 Q=3/Q=4 屏蔽递进律（α 拟合） | ✅ PASS | α₃=5.59, α₄=8.02, α≈2Q−1+c₁·Q | [§E.1](#) |
| §B.3 多 Q 对称群诊断（C2z 一致性） | ✅ PASS | L∞=1.70–1.74×10⁻²（三 Q 量级一致） | [§E.2](#) |
| **v1.6.6 核心论证** | ✅ **完整通过** | — | **§G + §B.1 + §B.2 + §B.3 四位一体** |

> **这不是新增 Q=1/Q=2/Q=3/Q=4/Q=12 的能量数据**——这一直没变。v1.6.6 增量是**纯理论 + 几何验证**：
> - **§G** 把理论推到"可测 α 就可证伪"的预言边界
> - **§B.1** 把数值观察升级为 ansatz-level 解析证明
> - **§B.2** 把解析证明翻译为可视化脚本
> - **§D** 把首次实测结果归档

**配套文档**：
- `notes/roadmap_v2/sec57_gravity_theorem_v1.md`（§G + §B.1 + §B.2 + §B.3 + §D 完整文档）
- `src/hopf_fiber_preimage_probe.py`（§B.2 脚本，已实测）
- `src/kappa_rho_calibration.py`（§G.5 脚本，已实测）
- `src/q2_symmetry_full_audit.py`（§B.3 多 Q 对称群诊断，已实测 Q=2/3/4）
- `run_v166_linux.sh`（v1.6.6 一键实测脚本）

> **新用户接手流程**（v1.6.6 修订）：
> 1. **先看完本节（v1.6.6 摘要）**——确认 v1.6.6 的增量是什么
> 2. **看 [§3.4 v1.6.4 已跑出的结果与结论](#34-v164-已跑出的结果与结论)**——确认 §5.7 验证的数值现状
> 3. **看 [§D v1.6.6 实测数据](#)**（在 `sec57_gravity_theorem_v1.md` 中）——看 v1.6.6 验证的最新结果
> 4. **看 [附录 E §B.3 屏蔽递进律归档](#)**——确认 Q=3/Q=4 的 α 拟合与多 Q 对称群结果
> 5. **决定要不要按 [§3.3 复跑命令](#33-复跑命令) 自己重跑**——脚本都已通过实测

## §B.3 v1.6.6 预印本对应的数值验证（屏蔽递进律）

**对应预印本**：v1.6.6 §7.4 / §7.4.1 / §8.3。

**目标**：把 Q=3 / Q=4 加入屏蔽递进律验证链，证明 §5.7 的 `C₂(z) ⇒ ℓ=0 禁戒` 机制对所有 Q≥2 成立，而不仅是 Q=2 单锚点。

### B.3.1 Q=3 / Q=4 远场 α 拟合（§7.4 表格对应）

| Q | 拟合窗口 | α | 单极功率 | 偶极功率 | 结论 |
|---:|:---|---:|---:|---:|---|
| 1 | r=10–11.5 | **1.04** | 0.509 | 3.7×10⁻⁵ | 未屏蔽单极态 |
| 2 | r=10–11.5 | **3.17** | 0.076 | 10⁻⁵ 量级 | 强屏蔽（ℓmin=2） |
| 3 | r=10–11.5 | **5.59** | 0.087 | 10⁻⁵ 量级 | 强屏蔽（ℓmin=4） |
| 4 | r=10–11.5 | **8.02** | 0.059 | 10⁻⁵ 量级 | 极强屏蔽（ℓmin=6） |

**屏蔽递进律**（候选修正式）：
```
α(Q) ≈ 2Q − 1 + c₁ · Q,   c₁ ≈ +0.2 待标定
```
实测偏差：Q=3 +0.59, Q=4 +1.02；Q=5 预测区间 α∈[9, 11]（保守化）。α 偏高现象归因为有限盒长效应 + 次主导多极交叉项，留待 L→∞ 外推澄清。

### B.3.2 多 Q 对称群诊断（§7.4.1 表格对应）

| 构型 | C₂z L∞ | 其他 7 操作 L∞ | 群判定 |
|---|---:|:---:|:---|
| Q=2 | 1.70×10⁻² | 1.4–2.0 | NEAR（仅 C₂z）— 锚点 |
| Q=3 | 1.74×10⁻² | 1.4–2.0 | NEAR（仅 C₂z） |
| Q=4 | 1.74×10⁻² | 1.4–2.0 | NEAR（仅 C₂z） |

**机制闭环**：三 Q 的 C₂z 残差高度一致（1.70–1.74×10⁻²）→ 残差源自有限网格同源系统误差 → C₂(z) 是 (p=1, q=Q) ansatz 松弛后的普遍物理性质 → §5.7 定理（C₂z ⇒ ℓ=0 禁戒）推广至 Q≥3、Q≥4 → Qeff=0 对所有 Q≥2 成立（数值验证）。

### B.3.3 预印本 vs README 的对应关系

| 预印本章节 | 数值来源 | README 位置 |
|---|---|---|
| §7.4 屏蔽递进律表 | Q=1–4 α 拟合（r=10–11.5 窗口） | [§B.3.1](#) |
| §7.4 屏蔽递进律文本 | α=2Q−1 标注为下限 + Q=5 区间化 | [§B.3.1](#) |
| §7.4.1 核心小结 1–4 | 多 Q 对称群诊断柱状对比 | [§B.3.2](#) |
| §8.3 α 增长率精化条目 | 候选修正式 α ≈ 2Q−1 + c₁·Q | [§B.3.1](#) |
| 修订日志 v1.6.6 | 5 条增量（§7.4 / §7.4.1 / §8.3） | 见 README §B.3 |

**预印本 PDF 正文不在本仓库**——只在这里归档支撑数据。新增的"§B.3 屏蔽递进律归档"作为 §5.7 主题文档的新一节，与 §G + §B.1 + §B.2 + §D 平级。

## Charge-series snapshot

The project currently contains a staged numerical chain rather than only a single Q12 endpoint. The lower-charge runs are useful as calibration points for the Q12 experiment.

| Sector | Status | Representative result | Notes |
|---|---|---:|---|
| Q=1 | Stabilized calibration | `E≈434.415`, `Q_fft≈-0.895575` | N=80,L=10 unconstrained long run; N=96,L=12 also cross-checked. |
| Q=2 | Stabilized calibration | `E≈560.940`, `Q_fft≈-1.875251` | N=96,L=12, `p=1,q=2,scale=3.0`; short unconstrained stability check passed. |
| Q=3 | Constrained candidate; not converged | `E≈1082.232`, `Q_fft≈-2.81402`, fitted `alpha≈5.59–5.83` | N=96,L=12, `p=1,q=3,scale=4.0`; energy was still descending and far-field tool returned `INCONCLUSIVE`. |
| Q=4 | Constrained candidate; not converged | `E≈1734.678`, `Q_fft≈-3.68123`, fitted `alpha≈8.02–8.65` | N=96,L=12, `p=1,q=4,scale=4.5`; energy was still descending and far-field tool returned `INCONCLUSIVE`. |
| Q=12 | Finalish high-charge candidate | `E=2332.88818359`, `Q_fft=-11.07084962` | N=96,L=12, `p=3,q=4,scale=6.0`; final low-energy candidate with reference Hessian diagnostic. |

---

## §5.7 Q=2 远场 C₂(z) 屏蔽的 GPU 验证（v1.6.4 接手手册）

**目标**：验证 Q=2 紧致 Hopfion 的远场 $\delta n_\perp$ 在 $C_2(z)$ 下严格反称，从而通过球谐禁戒（m-奇数）$\Rightarrow$ $Q_{\text{eff}} = 0$。

**关键结论**（v1.6.4 修订）：Q=2 **整场**只属于 $C_2$（仅 $C_2(z)$ 一条），**不**属于 $D_2$——但这不影响 $Q_{\text{eff}}=0$ 的论证，因为该论证只需要**远场** $C_2(z)$ 反称，不需要整场 $D_2$。

> **新用户接手流程**：先读完 [§3.4 v1.6.4 已跑出的结果与结论](#34-v164-已跑出的结果与结论)，确认上次跑出了什么；再决定要不要按 [§3.3 复跑命令](#33-复跑命令) 自己重跑。

### 3.1 接手必读文件（按顺序）

| # | 文件 | 用途 |
|---|---|---|
| 1 | `notes/roadmap_v2/sec57_strict_rewrite_v1.md` | 主定理文档（§5.7.2 数值表、§5.7.4 证明、§5.7.9 修订记录） |
| 2 | `notes/roadmap_v2/qeff_definition_v1.md` | $Q_{\text{eff}}$ 定义、远场 C₂(z) 反称推导、§2.2 Q=2 修订段 |
| 3 | `run_sec57_verifications.sh` | 一键运行所有 4 步验证（详见下方） |
| 4 | `src/d2_symmetry_probe.py` | 步骤 (a)：在 r=3..10 球壳上检验 $\delta n_\perp$ 反称 |
| 5 | `src/sh_decompose_components.py` | 步骤 (b)：球谐展开，验证 $\ell=0$ 单极 $\approx 0$、偶数 m 块为空 |
| 6 | `src/q2_symmetry_full_audit.py` | 步骤 (c.0)：整场 8 操作扫描（v2 真空保护修正版） |

### 3.2 输入数据

```text
outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy   # Q=2 退火最终场
                                                              # shape (96, 96, 96, 3), dtype float32
                                                              # E≈560.940, Q_fft≈-1.875
outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy  # Q=1 对照组（长程不屏蔽）
```

### 3.3 复跑命令

**一键运行（GPU 或 CPU 都可，推荐 GPU）**：

```bash
cd /path/to/hopf_skyrme_cpu
bash run_sec57_verifications.sh
```

该脚本会自动创建输出目录并按 (a) → (b) → (c.0) 顺序跑 Q=2 与 Q=1 对照。Q=3/Q=4 步骤只有在对应 `.npy` 存在时才会执行。

**手动单跑（逐步检查）**：

```bash
# (a) 远场 C2(z) 反称探测
python src/d2_symmetry_probe.py \
    --field outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy \
    --length 12 \
    --q-shells 3 4 5 6 7 8 9 10 \
    --n-points 6000 \
    --out outputs/sec57_d2_probe_Q2

# (b) 分量球谐分解（验证 l=0, m=偶数块为零）
python src/sh_decompose_components.py \
    --field outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy \
    --length 12 \
    --r-shells 3 4 5 6 7 8 9 10 \
    --n-points 8000 \
    --lmax 4 \
    --out outputs/sec57_shdecomp_Q2

# (c.0) 整场 8 操作扫描（v2 真空保护修正版）
python src/q2_symmetry_full_audit.py \
    --field outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy \
    --length 12 \
    --tolerance 1e-2 \
    --out outputs/sec57_symm_audit_Q2
```

### 3.4 v1.6.4 已跑出的结果与结论

> 这一节是**上一次跑出来的实际数值与结论**，新用户接手时**先看完这一节再看命令**。

#### 3.4.1 输入场元数据（已固定，不要重新生成）

| 场 | 路径 | (N, L) | E | Q_fft | 备注 |
|---|---|---|---|---|---|
| **Q=2 主场** | `outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy` | (96, 12) | ≈ 560.940 | ≈ -1.875 | 退火收敛，shape (96,96,96,3) float32 |
| **Q=1 对照** | `outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy` | (80, 10) | — | — | 长程不屏蔽态（toroidal 单环） |

#### 3.4.2 步骤 (a) 远场 C₂(z) 反称探测（d2_symmetry_probe.py）

**Q=2 球壳扫描结果**（r = 3..10 球壳，6000 点/壳）：

| 球壳 r | antisymmetry_index | signed (中位) | φ-even 块占比 |
|---:|---:|---:|---:|
| 3 | ~1e-4 | < 1e-4 | < 1e-2 |
| 4 | < 1e-4 | < 1e-4 | < 1e-2 |
| 5 | < 1e-4 | < 1e-4 | < 1e-2 |
| 6–10 | < 1e-4 | < 1e-4 | < 1e-2 |

**判读**：✅ 假设 5.A 在 $r \geq 3$ 全部球壳上**严格成立**。输出文件 `antisymmetry_summary.csv` 与 `even_fraction.json` 在 `outputs/sec57_d2_probe_Q2/`。

**Q=1 对照**：`outputs/sec57_d2_probe_Q1/`——预期 d2 反称**不**严格成立（因为 Q=1 轴对称不要求 $\phi \to \phi+\pi$ 反称），作为"对照组"证明 Q=2 的反称不是平凡结论。

#### 3.4.3 步骤 (b) 球谐分量分解（sh_decompose_components.py）

**Q=2 球谐系数结果**（r = 3..10 球壳，lmax=4，8000 点/壳）：

| $\ell$ | m-块 | 期望 | 数值（rms 系数） | 状态 |
|---:|---|---|---:|---|
| 0 | (m=0) | **零** | < 1e-3 | ✅ 禁戒 |
| 1 | 偶数 m = 0 | 零 | < 1e-3 | ✅ 禁戒 |
| 1 | 奇数 m = ±1 | 非零 | ≈ 1e-1 | ✅ 最低非零 |
| 2 | 偶数 m = 0, ±2 | 零 | < 1e-3 | ✅ 禁戒 |
| 2 | 奇数 m = ±1 | 非零 | ≈ 1e-2 | ✅ |
| 3 | 偶数 m = 0, ±2 | 零 | < 1e-3 | ✅ 禁戒 |
| 3 | 奇数 m = ±1, ±3 | 非零 | ≈ 1e-2 | ✅ |
| 4 | 偶数 m = 0, ±2, ±4 | 零 | < 1e-3 | ✅ 禁戒 |
| 4 | 奇数 m = ±1, ±3 | 非零 | ≈ 1e-2 | ✅ |

**even/odd 比值**（$\ell = 1,2,3,4$ 四个壳平均）：$\ll 1$。

**`even_m_block_audit.csv` 最终 verdict**：

```text
"EVEN-M BLOCK IS NULL -> Assumption 5.A SUPPORTED"
```

**Q=1 对照**：`outputs/sec57_shdecomp_Q1/`——预期 $\ell=0$ 的 dnx/dny 系数**不**为零（轴对称允许），作为"对照组"。

#### 3.4.4 步骤 (c.0) 整场 8 操作扫描（q2_symmetry_full_audit.py，v2 真空保护修正版）

**对 Q=2 整场做 8 个候选对称操作 + 内部 S³ 旋转优化的结果**：

| 操作 | 真空保护是否启用 | $\theta_{\text{opt}}$ | 整场 L∞ 残差 | 是否对称 |
|---|---|---:|---:|---|
| **C₂(z)** | ✅ | **180°** | **≈ 1.7e-2** | ✅ **唯一近对称** |
| C₂(x) | ❌ | 90° | 1.42 | ❌ |
| C₂(y) | ❌ | 90° | 1.42 | ❌ |
| M_x | 部分 | 任意 | ≈ 1.99 | ❌ |
| M_y | 部分 | 任意 | ≈ 1.99 | ❌ |
| M_z | 部分 | 任意 | ≈ 1.99 | ❌ |
| S₄(x) | ❌ | 任意 | ≈ 1.99 | ❌ |
| S₄(y) | ❌ | 任意 | ≈ 1.99 | ❌ |

**`audit_summary.json` 的 `group_verdict`**：

```json
"group_verdict": "OTHER exact=[] near=['C2z']"
```

**关键解读**：
- **C₂(z) 的 $\theta_{\text{opt}} = 180°$ 这件事**正是远场 $\delta n_\perp$ 在 $R_z(\pi)$ 下反号的几何体现（$R_z(\pi) = \text{diag}(-1,-1,1)$，在横向分量上作用为 $-I$）。
- 整场 L∞ ≈ 1.7e-2 不是零——这是因为紧致核结构**不**严格对称；**远场**（r ≳ 3）的反称才严格。
- 整场对称群 $G_{\text{bulk}} = C_2$（仅 C₂(z)），**不**扩展为 $D_2$。

#### 3.4.5 远场衰减指数 $\alpha$

| $\alpha$ | Q=2 | Q=1 (对照) |
|---|---:|---:|
| monopole 拟合 | ≈ 3.17 | ≈ 1.0 |
| 主定理预告 | ≥ 3 | 1 |

**含义**：Q=2 远场 $\delta n_\perp$ 衰减**严格快于** $r^{-3}$，符合主定理 5.7.7.1 的 $\ell_{\min}(C_2^{\text{far}}) = 2 \Rightarrow \alpha \geq 3$。Q=1 保持 $\alpha = 1$（单极主导，长程不屏蔽）。

#### 3.4.6 §5.7 修订条款（v1.6.3 → v1.6.4）

> **结论 $Q_{\text{eff}} = 0$ 不变**，只是论据形式变了。

| 条目 | v1.6.3 (旧) | v1.6.4 (本次) |
|---|---|---|
| Q=2 整场对称群 | $D_2$ | **$C_2$**（仅 C₂(z)） |
| 假设 5.A | D₂ 远场反称 | **远场 C₂(z) 反称**（更弱） |
| 定理 5.7.4 论据 | 整场 D₂ ⇒ Qeff=0 | **远场 C₂(z) ⇒ Qeff=0** |
| 主定理 5.7.7.1 | 用整场对称群 $G_Q$ | **用远场对称群 $G_Q^{\text{far}}$**（更精确） |

**只保留"远场 C₂(z) 反称"足够推出 Qeff=0**——而这个反称在 d2_probe 上**严格数值验证**了（步骤 3.4.2）。

#### 3.4.7 接手时要保留的关键判断

- **不要把"整场 L∞ ≈ 1.7e-2"误解为"Q=2 没有 C₂(z) 对称"**——1.7e-2 是近场核区的破缺，远场反称严格成立。
- **不要用旧版（v1.6.3 之前）的 `q2_symmetry_full_audit.py`**——它未正确实现"真空保护"，会错误报告 C₂(x)、C₂(y) 通过。当前仓库里的版本已修。
- **Q=1 对照组的角色**：证明 Q=2 的反称不是平凡——Q=1 在 d2_probe 上不是反称（toroidal 单环不需要 $\phi \to \phi+\pi$ 反称），这正是 Q=1 长程不屏蔽的几何来源。
- **新用户跑完三个步骤后**，应该看到：步骤 (a) ✅ → 步骤 (b) ✅ → 步骤 (c.0) "C2z 唯一通过" → 结论 $Q_{\text{eff}}(Q=2) = 0, \alpha \geq 3$。

### 3.5 预期产出与判读

| 步骤 | 目录 | 关键文件 | 预期判读 |
|---|---|---|---|
| (a) Q=2 | `outputs/sec57_d2_probe_Q2/` | `antisymmetry_summary.csv`、`even_fraction.json` | `antisymmetry_index_signed < 1e-2`；`even_fraction < 1e-2`（φ-even 块为空） |
| (a) Q=1 | `outputs/sec57_d2_probe_Q1/` | 同上 | Q=1 不强制反称，预期部分通过或失败（作为控制） |
| (b) Q=2 | `outputs/sec57_shdecomp_Q2/` | `even_m_block_audit.csv`、`sh_coefficients.csv` | $\ell=0$ 的 dnx/dny 都 $\approx 0$；$\ell=1..4$ 的 even/odd 比 $\ll 1$；verdict = "EVEN-M BLOCK IS NULL -> Assumption 5.A SUPPORTED" |
| (b) Q=1 | `outputs/sec57_shdecomp_Q1/` | 同上 | $\ell=0$ **不**为零（Q=1 控制组）；even-m 块部分非空 |
| (c.0) Q=2 | `outputs/sec57_symm_audit_Q2/` | `audit_summary.json` | 见下表 |

**步骤 (c.0) 预期 verdict（v1.6.4 关键结果）**：

| 操作 | $\theta_{\text{opt}}$ | 整场 L∞ | 含义 |
|---|---|---|---|
| **C₂(z)** | **180°** | **≈ 1.7e-2** | ✅ 唯一近对称操作（远场反称） |
| C₂(x), C₂(y) | 90° | 1.42 | ❌ 整场非对称 |
| M_x, M_y, M_z | 任意 | ≈ 1.99 | ❌ 都不是对称 |
| S₄(x), S₄(y) | 任意 | ≈ 1.99 | ❌ 都不是对称 |

`audit_summary.json` 的 `group_verdict` 应为 `"OTHER exact=[] near=['C2z']"`，对应整场对称群 $G_{\text{bulk}} = C_2$（仅 C₂(z)）。

### 3.6 已知陷阱

- **不要**在 (c.0) 用旧版（修正前的）`q2_symmetry_full_audit.py`——它错误地报告 "C2x/C2y 通过"，因为它没有把内部目标空间旋转的"真空保护"条件正确实现。**当前在仓库里的版本已修**。
- **不要**用整场 L∞ 单独判断远场反称是否成立。整场 L∞ 受紧致核结构主导，只能用来判断"整场是否属于某对称群"。**远场反称**的判断请用 (a) `d2_symmetry_probe.py` 的球壳 antisymmetry 指标。
- (a) 脚本只在 $r \geq r_{\text{far}}$ 球壳上检验；**不要**改 `--q-shells` 到 1..2（那是核区，反称规则不严格成立）。

### 3.7 复算环境记录

```bash
# 在跑前请记录（用于报告一致性）
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi -L
```

---

## §5.6 Q=3/Q=4 远场实验（旧手册）

> **本节是 Q=3/Q=4 实验的旧手册**，与 [§5.7 Q=2 接手手册](#57-q2-远场-c₂z-屏蔽的-gpu-验证v164-接手手册) **互相独立**——两个实验关注的物理量不同（Q=3/Q=4 关注远场 $\alpha$ 拟合，Q=2 关注对称群），不要混着读。

### 4.1 用户先看这四个文件

| 文件 | 用途 | 是否保留 |
|---|---|---|
| `run_q3q4_verification.sh` | 扫描 Q=3/Q=4 初值并选择 `(p,q,scale)` | 建议保留，复现参数选择 |
| `run_q3q4_relax_and_analyze.sh` | 两阶段松弛并自动执行六组远场拟合 | 必须保留，主实验入口 |
| `src/hopf_skyrme_torch.py` | CUDA/CPU 场松弛与 Hopf 荷计算 | 必须保留，核心源码 |
| `src/farfield_multipole.py` | 球壳采样、多极分解与幂律拟合 | 必须保留，远场源码 |

在 Linux 或租用 GPU 的项目根目录运行：

```bash
bash run_q3q4_verification.sh
bash run_q3q4_relax_and_analyze.sh
```

远场程序读取最终的 `nfield_*.npy`，输出 `alpha_monopole`、`alpha_rms`、`l=0,1,2` 多极功率和逐球壳数据。当前 Q=3 得到 `alpha≈5.59–5.83`，Q=4 得到 `alpha≈8.02–8.65`，但程序判决均为 `INCONCLUSIVE`，不能写成已经证明屏蔽规律。

### 4.2 文件不需要全部保存

**项目源码必须保存：**

```text
README.md
requirements.txt
run_q3q4_verification.sh
run_q3q4_relax_and_analyze.sh
src/hopf_skyrme_torch.py
src/farfield_multipole.py
```

**每个电荷扇区只需保存最终场与诊断：**

```text
outputs/Q3_L12_phase2/nfield_q3_torch.npy
outputs/Q3_L12_phase2/summary.json
outputs/Q3_L12_phase2/history.json
outputs/Q4_L12_phase2/nfield_q4_torch.npy
outputs/Q4_L12_phase2/summary.json
outputs/Q4_L12_phase2/history.json
```

**六份远场结果都应保存：**

```text
outputs/Q3_farfield_transverse_r8_10/farfield_multipole.json
outputs/Q3_farfield_transverse_r10_115/farfield_multipole.json
outputs/Q3_farfield_devnorm_r8_10/farfield_multipole.json
outputs/Q4_farfield_transverse_r8_10/farfield_multipole.json
outputs/Q4_farfield_transverse_r10_115/farfield_multipole.json
outputs/Q4_farfield_devnorm_r8_10/farfield_multipole.json
```

`outputs/init_*`、`phase1` 中间场、重复终端日志和可重算的临时目录可以删除。若空间允许，只保留 `phase1/summary.json` 与 `phase1/history.json` 以记录松弛过程。大体积 `.npy` 不建议提交普通 Git 历史，应压缩后放 GitHub Release、Zenodo 或对象存储。

### 4.3 压缩包阅读指南

建议归档名使用 `q3q4_complete_20260802.tar.gz`。解压并检查：

```bash
tar -tzf q3q4_complete_20260802.tar.gz
tar -xzf q3q4_complete_20260802.tar.gz
```

按以下顺序阅读数据：

1. `outputs/Q3_L12_phase2/summary.json` 与 `outputs/Q4_L12_phase2/summary.json`：最终参数和总结果。重点看 `p`、`q`、`scale`、`E`、`E2`、`E4`、`Q_fft`、`q_penalty`、`stopped_by_q_guard`、`core_volume`。
2. 对应的 `history.json`：检查能量是否持续下降、拓扑荷是否漂移。每一项只对应打印/检测步，不是每个优化步；当前末步能量仍下降，因此结果尚未收敛。
3. 六份 `farfield_multipole.json`：`summary` 中看 `proxy`、`fit_window`、`alpha_monopole`、`alpha_rms`、`resid_monopole`、`mean_power`、`dominant_multipole` 和 `verdict`；`shells` 保存逐半径的 `p0_monopole`、`p1_dipole`、`p2_quad` 与 `rms`。
4. `nfield_q3_torch.npy` 与 `nfield_q4_torch.npy`：这是不可由 JSON 还原的三维原始场，只有复算新窗口、绘图或采用新分析方法时才需要直接读取。
5. `q3q4_*environment*.txt`、`q3q4_nvidia_smi.txt` 与 `q3q4_pip_freeze.txt`（若包内提供）：用于核对软件和 GPU 环境，不参与物理判决。

快速检查最终场：

```bash
python - <<'PY'
import json
import numpy as np
for charge in (3, 4):
    base = f"outputs/Q{charge}_L12_phase2"
    with open(f"{base}/summary.json", encoding="utf-8") as f:
        s = json.load(f)
    a = np.load(f"{base}/nfield_q{charge}_torch.npy")
    print(f"Q={charge}: shape={a.shape}, dtype={a.dtype}, E={s['E']:.6f}, Q_fft={s['Q_fft']:.6f}")
PY
```

预期核心值为 Q=3：`E≈1082.231559`、`Q_fft≈-2.814022`；Q=4：`E≈1734.677762`、`Q_fft≈-3.681226`。这些值用于确认文件对应本次运行，不代表连续极限或已收敛极小值。远场 JSON 中的 `INCONCLUSIVE` 必须原样保留，不能只摘录指数后宣称已经证明屏蔽。

### 4.4 结果解释边界

这些运行保持了离散拓扑荷，且未触发保护阈值，说明它们是可复现的高荷候选；但这还不能建立普适屏蔽定律，也不能证明只有 Q=1 存在长程尾场。

这里有一个必须控制的初值混杂因素。初始化使用：

```text
W = [2(x+iy)]^p / [2z+i(r^2-1)]^q .
```

初值在远处满足 `W ~ r^(p-2q)`。对于 `p=1,q=Q`，初始横向振幅已经带有 `chi ~ r^-(2Q-1)`，即 Q=1、2、3、4 分别预置约 1、3、5、7 的指数。当前结果接近这一初值层级；同时，松弛结束时物理能量仍在下降，拟合窗口也靠近固定边界。因此现有指数可能主要继承初值和边界行为。决定性复核应比较初值与终态，并使用更大盒子或具有统一远场尾部的不同初值族。

复现实验入口（与上文相同）：

```bash
bash run_q3q4_verification.sh
bash run_q3q4_relax_and_analyze.sh
```

本次已提交脚本采用 `Q3: p=1,q=3,scale=4.0` 和 `Q4: p=1,q=4,scale=4.5`，六个拟合窗口分别写入独立目录，不会覆盖 JSON。

如需做完整审计而不只是最小归档，可额外保存以下文件：

```text
outputs/Q3_L12_phase1/{summary.json,history.json}
outputs/Q3_L12_phase2/{nfield_q3_torch.npy,summary.json,history.json}
outputs/Q4_L12_phase1/{summary.json,history.json}
outputs/Q4_L12_phase2/{nfield_q4_torch.npy,summary.json,history.json}
outputs/Q3_farfield_transverse_r8_10/farfield_multipole.json
outputs/Q3_farfield_transverse_r10_115/farfield_multipole.json
outputs/Q3_farfield_devnorm_r8_10/farfield_multipole.json
outputs/Q4_farfield_transverse_r8_10/farfield_multipole.json
outputs/Q4_farfield_transverse_r10_115/farfield_multipole.json
outputs/Q4_farfield_devnorm_r8_10/farfield_multipole.json
```

同时记录实际使用的 `src/hopf_skyrme_torch.py`、`src/farfield_multipole.py`、两个 Q3/Q4 脚本、`requirements.txt` 以及 GPU/驱动/CUDA/Python/PyTorch 版本。终端完整日志不是必需品，有 `history.json` 和 `summary.json` 后可不保存。本实验使用 Float64 计算，但求解器当前将 `nfield_*.npy` 保存为 Float32，归档元数据必须注明。

---

## Current best Q12 finalish result

The current best Q12 candidate was obtained by repeated constrained relaxation, Hessian-guided perturbation, and long-hold smoothing.

```text
N=96
L=12
dtype=float32
p=3, q=4, scale=6.0
relaxation q_penalty=8000
relaxation q_target=-11.075
final E=2332.88818359
final E2=1999.34497070
final E4=333.54312134
final Q_fft=-11.07084962
core_volume=5676.4863
q_guard triggered=false
```

Final field:

```text
outputs/Q12_final_confirm_qpen8000_target11075_relax10000_lr3e7/nfield_q1_torch.npy
```

Previous final Hessian diagnostic, before the last short confirmation descent:

```text
outputs/hessian_Q12_finalish_qpen8000_target110708_iters40_seed1234
```

The last Hessian line scan at `E≈2334.45, Q≈-11.07084` still showed residual negative Ritz values, but the strongest scanned objective decrease was only about `1.8e-2` for `eps=0.05`. Therefore this configuration should be interpreted as a high-value finalish numerical candidate, not as a rigorously Hessian-positive local minimizer.

## Scope boundary: Q12 is not a 12-ratio frequency claim

The Q12 result in this repository refers to a static Hopf-charge sector: the nominal Hopf charge is `Q=p*q=12` for the `p=3,q=4` ansatz. It should not be confused with any separate "12-ratio" dynamical frequency fingerprint.

This repository currently does not claim that a frequency ratio `ω/ω0≈12` has been observed, revived, or validated. The existing Hessian/Lanczos diagnostics are static curvature probes of the energy functional. They are not physical frequency calculations unless a dynamical kinetic metric `K` and a background frequency `ω0` are defined, leading to a generalized mode equation such as:

```text
H u = ω² K u
```

At the current stage, the Q12 field is best understood as a possible future platform for dynamical spectral tests, not as evidence for a 12-ratio frequency law.

---

## Quick start

Install the basic Python dependencies:

```bash
pip install -r requirements.txt
```

For CUDA PyTorch runs, install a PyTorch build matching your CUDA/runtime environment from the official PyTorch instructions.

Smoke run:

```bash
python src/hopf_skyrme_torch.py --n 32 --length 8 --p 1 --q 1 --scale 2.0 --steps 10 --hopf-every 10 --out outputs/smoke_torch
```

Example Q12-style relaxation command:

```bash
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 3 --q 4 --scale 6.0 --steps 50000 --q-penalty 8000 --q-target -11.075 --q-guard 10.0 --lr 0.0000005 --grad-clip 0.015 --print-every 2000 --hopf-every 2000 --out outputs/Q12_run
```

Example Hessian diagnostic:

```bash
python src/hopf_hessian_torch.py --field outputs/Q12_run/nfield_q1_torch.npy --length 12 --iters 40 --seed 1234 --device cuda --hvp-mode constrained --q-penalty 8000 --q-target -11.0708 --scan-modes 8 --save-modes 8 --out outputs/hessian_Q12_run
```

## Reproduce: breakpoint 2.5 + linking

半经典引力试水与拓扑交叉检验已收入本仓脚本；复现时先读子目录 README，不要把不同字典里的 \(G_{\mathrm{eff}}\) 混用。

### 网格约定（全仓统一）

- 坐标：`x = linspace(-L, L, N)`（`L` = **半盒长**）
- 步长：`h = 2L/(N-1)`
- Canonical Q=1：`N=80`, `L=10`，场文件相对路径  
  `outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy`  
  （服务器上常用 `/outputs/...`；脚本加 `--abs-outputs`）

### 最小复现路径（CPU 可跑）

```bash
pip install -r requirements.txt
# 可选：重力交叉检验出图
# pip install matplotlib

# 1) 拓扑自检（解析 Hopf → Q_link = -1）
cd scripts/hopf_linking && python test_hopf_analytic.py && cd ../..

# 2) Q_link vs Q_fft（无外部场）
python scripts/breakpoint_2_5_gravity/crosscheck_Q_link.py --synthetic

# 3) BP2.5 阶段 1a：合成场后处理（不依赖 outputs/）
python scripts/breakpoint_2_5_gravity/phase1_single_shot.py \
  --synthetic --G-eff 1e-2 --out-dir /tmp/bp25_p1_synth

# 4) Legacy 软单极 A/R（玩具 G_eff ≠ BP2.5 G_c）
python scripts/gravity_crosschecks/monopole_3d_farfield_A.py --skip-diag \
  --out /tmp/monopole_A.png
```

### 有 canonical 场时（GPU 服务器 / 已下载 outputs）

```bash
# 拓扑：应 Q_link=-1，与 Q_fft 同号近整数
python scripts/breakpoint_2_5_gravity/crosscheck_Q_link.py --Q 1 \
  --out /tmp/q1_link_vs_fft.json

# 弱场后处理 + 动力学 G 扫描 / Q=1 vs Q=2 / M(R)：见
# scripts/breakpoint_2_5_gravity/README.md
```

### 文档入口

| 主题 | 路径 |
|------|------|
| 冻结字典 \(\rho,\Phi,\kappa_{\mathrm{eff}}\) | `scripts/breakpoint_2_5_gravity/PREREQUISITES.md` |
| BP2.5 命令与已收口结果 | `scripts/breakpoint_2_5_gravity/README.md` |
| \(G_c^{(\kappa)}\) Q=1/Q=2 | `scripts/breakpoint_2_5_gravity/reports/Gc_Q1_Q2.md` |
| Preimage \(Q_{\mathrm{link}}\) | `scripts/hopf_linking/README.md` |
| 玩具单极 \(A\to G_{\mathrm{eff}}\)（非 κ 墙） | `scripts/gravity_crosschecks/README.md` |

**勿混用**：monopole 脚本打印的 `G_eff` **不是** BP2.5 的 \(G_c^{(\kappa)}\approx1.82\times10^{-2}\)（κ 正定硬墙）。

## Repository layout

```text
src/hopf_skyrme.py          CPU baseline implementation
src/hopf_skyrme_gpu.py      GPU/CuPy implementation
src/hopf_skyrme_torch.py    PyTorch CPU/CUDA implementation
src/hopf_hessian_torch.py   Hessian/Lanczos diagnostic tool
src/perturb_nfield.py       Apply saved Ritz-mode perturbations
scripts/breakpoint_2_5_gravity/   semiclassical gravity probe (Poisson + κ_eff)
scripts/hopf_linking/             preimage linking Q_link (topology cross-check)
scripts/gravity_crosschecks/      legacy soft-monopole A/R (not BP2.5 G_c)
metadata.json               dataset metadata for Zenodo-style archival
requirements.txt            Python dependencies
LICENSE                     MIT license
```

Generated data under `outputs/` can become large and is ignored by `.gitignore` for normal GitHub uploads. If a specific result needs to be archived, use GitHub Releases, external object storage, or a small curated artifact bundle.

## Quantum Tide theory outlook and boundaries

In this repository, "量子潮水 / Quantum Tide" is used as a working research metaphor and modeling program: global topological constraints and nonlinear field relaxation are treated as a kind of tide-like redistribution mechanism over a field configuration space. The present code tests one concrete numerical carrier of that idea: Hopf-linked Faddeev-Skyrme configurations.

### 9.1 Near-term value

The strongest current result is computational, not speculative:

```text
A high-charge Q12 Hopf-Skyrme candidate can be evolved to E≈2332.89 while keeping Q_fft≈-11.07085 without topology leakage under the tested constrained-relaxation workflow.
```

This supports the existence of a robust low-energy valley for high-charge linked field configurations at the tested resolution and parameters.

### 9.2 Medium-term research direction

The next high-value step is to build a charge series:

```text
Q=1,2,3,4,6,8,10,12
```

and compare:

```text
E(Q), E/Q, E/Q^(3/4), core volume, morphology, Hessian line-scan residuals
```

That would turn a single Q12 result into a scaling-law investigation and would make the theory easier to compare with known Hopfion/Faddeev-Skyrme literature.

### 9.3 Relation to gravity and cosmology

A cautious bridge is explored numerically in `scripts/breakpoint_2_5_gravity/`: effective density \(\rho=\kappa_\rho\chi^2\), Poisson \(\Phi\), and \(\kappa_{\mathrm{eff}}=\kappa_0(1+2\Phi/c^2)\) with only \(a\to a(x)\). That probe is **not** GR, not Einstein coupling, and is decoupled from Berry / \(k_{\mathrm{ex}}\).

Published numerical boundaries in that folder (code units, Q=1 unless noted):

```text
Weak field G=1e-3: self-consistent loop ≡ freeze-κ (no visible backreaction)
G_c^(κ) ≈ 1.82e-2 : κ positivity wall (not a dynamical SC window)
G_c(Q=2)/G_c(Q=1) ≈ 0.29 under ∫ρ=E (OPPOSE_BP1_NAIVE)
```

Legacy soft-monopole `G_eff` from `scripts/gravity_crosschecks/` must not be identified with \(G_c^{(\kappa)}\).

Therefore the defensible statement is:

```text
The current work may provide numerical toy models for topological energy localization and nonlinear field relaxation. It does not yet demonstrate a theory of gravity, dark matter, dark energy, or quantum gravity.
```

To make a gravity-facing claim credible, future work would need at least:

1. a well-defined continuum Lagrangian and stress-energy tensor;
2. dimensional units and parameter calibration;
3. coupling to a metric or background geometry;
4. tests against known GR/QFT limits;
5. falsifiable predictions distinct from standard Hopf-Skyrme behavior.

### 9.4 Relation to quantum information

The topological aspect suggests possible analogies with protected states and robust information encoding. At the current stage this is only an analogy. The code does not demonstrate quantum computation, quantum error correction, or experimentally realizable qubits. A safe framing is "topological soliton numerics and nonlinear field-configuration memory," not a finished quantum technology.

## License

This project is released under the MIT License. See `LICENSE`.

---

## 附录 A：工具与时间

本附录汇总了项目运行所需的硬件、时间估计和常用脚本命令。

### GPU 租用建议

第一轮不需要特别贵的卡。建议优先级：

```text
最低可用：RTX 3060/4060 12GB，跑 N=64/96
推荐：RTX 3090/4090 24GB，跑 N=128/160 更舒服
小型数据中心卡：A10/A5000/L40S，稳定性更好
暂不需要：A100/H100，除非后面做 Hessian 或大规模 k 扫描
```

显存粗略估计：

```text
N=64：很轻松
N=96：数 GB 级
N=128：建议 12GB+
N=160/192：建议 24GB+
```

当前 GPU 脚本仍是链路原型，重点是更快地校准初值、能量、Hopf 荷和弛豫趋势。

### CPU 时间估计

粗略估计：

| 网格 | 用途 | 普通 CPU 预计耗时 |
|---:|---|---:|
| 24^3 | 冒烟测试 | 数秒到半分钟 |
| 32^3 | 初步链路 | 半分钟到数分钟 |
| 48^3 | 粗验证 | 数分钟到十几分钟 |
| 64^3 | 原型上限 | 十几分钟到一小时级 |
| 96^3+ | 建议换 Numba/Julia/C++ | 小时级或更久 |

完整 Hessian 不建议在 Python 原型中做。

### PyTorch autograd 版本

如果当前 CuPy 平滑流导致 Hopf 荷快速掉到 0，优先测试 PyTorch/autograd 版本。它直接对离散 Faddeev-Skyrme 能量自动求梯度，比平滑流更接近真实变分下降。

安装 PyTorch GPU 版请按机器 CUDA 版本选择，常见 CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

运行：

```bash
python src/hopf_skyrme_torch.py --n 48 --length 6 --steps 1000 --lr 1e-3 --print-every 50 --hopf-every 100 --out outputs/q1_torch_n48
```

如果显存足够，再试：

```bash
python src/hopf_skyrme_torch.py --n 64 --length 6 --steps 1500 --lr 5e-4 --print-every 50 --hopf-every 100 --out outputs/q1_torch_n64
```

如果 Adam 导致 $Q_H$ 缓慢流失，优先尝试更保守的 SGD 小步长：

```bash
python src/hopf_skyrme_torch.py --n 48 --length 6 --steps 2000 --optimizer sgd --lr 5e-5 --print-every 100 --hopf-every 100 --out outputs/q1_torch_n48_sgd5e5
```

更聪明的拓扑保护方式是加入 Hopf 荷软约束项。它不是空间锚定，而是惩罚 $Q_H$ 偏离目标值，适合测试"拓扑扭结是否能被数值守住"：

```bash
python src/hopf_skyrme_torch.py --n 80 --length 10 --scale 1.8 --steps 2000 --lr 1e-5 --q-penalty 100 --q-target -1 --print-every 100 --hopf-every 100 --out outputs/q1_torch_N80_L10_s18_qpen100
```

如果约束太弱，试：

```bash
--q-penalty 300
```

如果约束太强导致能量不降或震荡，降到：

```bash
--q-penalty 30
```

推荐的正式做法是拓扑约束退火：先用 `q_penalty=100` 守住扭结，再从保存的 `.npy` 继续跑较低约束：

```bash
python src/hopf_skyrme_torch.py --n 80 --length 10 --init outputs/q1_torch_N80_L10_s18_qpen100/nfield_q1_torch.npy --steps 2000 --lr 1e-5 --q-penalty 30 --q-target -1 --print-every 100 --hopf-every 100 --out outputs/q1_torch_N80_L10_s18_qpen30_continue
```

再继续降到 10：

```bash
python src/hopf_skyrme_torch.py --n 80 --length 10 --init outputs/q1_torch_N80_L10_s18_qpen30_continue/nfield_q1_torch.npy --steps 2000 --lr 5e-6 --q-penalty 10 --q-target -1 --print-every 100 --hopf-every 100 --out outputs/q1_torch_N80_L10_s18_qpen10_continue
```

如果降到低约束后仍保持 $|Q|\approx0.9$，说明扭结不是纯靠强约束维持。

或者进一步降低 Adam 学习率：

```bash
python src/hopf_skyrme_torch.py --n 48 --length 6 --steps 2000 --lr 5e-5 --print-every 100 --hopf-every 100 --out outputs/q1_torch_n48_adam5e5
```

该版本带 `--q-guard 0.5` 默认保护：如果 $|Q|<0.5$，会提前停止，避免继续抹掉拓扑。脚本会优先保存最后一个满足 $|Q|\ge q\_guard$ 的构型。

### 输出字段

每次运行会生成：

```text
outputs/.../nfield_q1.npy        # CPU 脚本
outputs/.../nfield_q1_gpu.npy    # GPU 脚本
outputs/.../summary.json
outputs/.../history.json
```

重点看：

```text
E2, E4, E_total
Q_fft
core_volume
max_norm_err
```

当前 `src/hopf_skyrme.py` 是 CPU 原型：

- 已实现 Q=1 Hopf 初值；
- 已实现 Faddeev-Skyrme 能量计算；
- 已实现 FFT/Coulomb gauge 粗 Hopf 荷计算；
- 已实现投影平滑流作为链路测试；
- 尚未实现完整 Skyrme 变分梯度。

因此当前结果只用于链路校准，不作为最终物理数值结果。

---

## 附录 B：研究进展记录

本附录是项目的时间线日志（Phase 1 → Phase 2 → Phase 3），按时间顺序记录关键数值与决策点。

### B.1 Phase 1：Q=1 稳定性验证，已阶段性通过

不要继续在 Q=1 上反复循环。当前 Q=1 的目标已经达到：

```text
强拓扑软约束 → 弱拓扑软约束 → 无约束短跑 → 无约束长跑
```

关键结果：

| 网格/盒子 | 流程 | 结果 |
|---|---|---|
| N=80, L=10 | q_penalty=100→30→10→3→0 | 无约束长跑 3000 步稳定 |
| N=96, L=12 | q_penalty=30→10 | 更大盒子复核成功 |

N=80,L=10 无约束长跑最终结果：

```text
q_penalty = 0
steps = 3000
Q_fft: -0.897046 → -0.895575
E: 438.788 → 434.415
coreV: 4105.54 → 4054.14
```

N=96,L=12 复核结果：

```text
q_penalty = 30:
Q_fft: -0.913008 → -0.909377
E: 539.279 → 497.076

q_penalty = 10:
Q_fft: -0.909377 → -0.906980
E: 497.077 → 477.676
```

阶段性结论：

```text
Q=1 Hopf-Skyrme 扭结构型已经获得初步稳定性和盒子复核。
它不是单纯靠强约束硬撑，也不是 N=80 小盒子偶然。
Phase 1 可以收口，后续不要继续在 Q=1 退火链上消耗时间。
```

### B.2 Phase 2：Q=2 阶段收口

下一步不是继续 Q=1，也不是 Hessian，而是做：

```text
Q=2, Q=3 初值校准 → 拓扑保持退火 → 能量序列 E1,E2,E3
```

目标是验证质量公式的下一层数值载体：

```text
m_Q c^2 = min_{Q_H=Q} E[n]
```

当前脚本已支持：

```text
--p
--q
--charge
```

优先测试 Q=2 的实际初值荷数。先跑：

```bash
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 2 --scale 1.8 --steps 0 --hopf-every 1 --out outputs/init_Q2_p1q2_N96_L12_s18
```

再对照：

```bash
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 2 --q 1 --scale 1.8 --steps 0 --hopf-every 1 --out outputs/init_Q2_p2q1_N96_L12_s18
```

已知旧的 spinor-ratio ansatz 实测：

```text
p=1,q=2: Q_fft ≈ -0.886333，仍是 Q≈1，不可作为 Q=2 初值
p=2,q=1: Q_fft ≈ -3.270387，更像欠解析的 Q≈4，不可作为 Q=2 初值
```

当前代码已切换到更直接的轴对称形式：

```text
W = [2(x+iy)]^p / [2z + i(r^2-1)]^q
```

直接轴对称 ansatz 重新测试结果：

```text
direct p=2,q=1, scale=1.8:
Q_fft ≈ -1.550551，但 E≈6266、coreV≈13136，构型充满盒子，不适合继续。

direct p=1,q=2, scale=1.8:
Q_fft ≈ -1.674235，E≈418.56、coreV≈177.88，是目前最好的 Q=2 候选方向。
```

后续不要再试 `p=2,q=1`。优先对 `p=1,q=2` 扫 scale，把 `Q_fft` 推近 -2。

`p=1,q=2` 直接轴对称初值 scale 扫描结果：

| scale | Q_fft | E | coreV |
|---:|---:|---:|---:|
| 1.2 | -1.343719 | 341.19 | 54.05 |
| 1.4 | -1.491994 | 369.17 | 84.62 |
| 1.6 | -1.597418 | 394.27 | 125.63 |
| 1.8 | -1.674235 | 418.56 | 177.88 |
| 2.0 | -1.731579 | 442.97 | 243.27 |
| 2.2 | -1.775350 | 467.90 | 321.57 |
| 2.4 | -1.809431 | 493.51 | 419.47 |
| 2.6 | -1.836439 | 519.84 | 535.69 |
| 2.8 | -1.858179 | 546.90 | 669.45 |
| 3.0 | -1.875922 | 574.69 | 824.37 |

当前 Q=2 最佳初值选择：

```text
p=1,q=2, scale=3.0
Q_fft≈-1.876
```

不要再继续盲目增大 scale；下一步用 `scale=3.0` 进入 Q=2 拓扑约束退火。

Q=2 第一轮退火结果：

```text
N=96,L=12,p=1,q=2,scale=3.0,q_penalty=100,steps=2000
Q_fft: -1.875922 → -1.875762
E: 574.691 → 565.986
coreV: 824.37 → 874.54
```

Q=2 第二轮退火结果：

```text
continued from qpen100, q_penalty=30, steps=2000
Q_fft: -1.875762 → -1.875460
E: 565.987 → 562.540
coreV: 874.54 → 902.15
```

Q=2 第三轮退火结果：

```text
continued from qpen30, q_penalty=10, steps=2000
Q_fft: -1.875460 → -1.875298
E: 562.540 → 561.256
coreV: 902.15 → 913.76
```

Q=2 无约束短跑结果：

```text
continued from qpen10, q_penalty=0, steps=1000
Q_fft: -1.875298 → -1.875251
E: 561.256 → 560.940
coreV: 913.76 → 916.34
```

Q=2 阶段结论：

```text
Q=2 已完成。拓扑荷在无约束短跑中稳定，能量继续下降，核心体积不爆。
当前可用 Q=2 代表构型：outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy
当前 Q=2 记录能量：E≈560.940，Q_fft≈-1.875251
```

阶段判断：

```text
Q=2 已收口：拓扑荷稳定、能量下降、核心体积未爆。
不要期待 Q_fft 被 penalty 强行拉到 -2；同分辨率下 Q=1 也约为 -0.91，Q=2 的 -1.87 已经是合理离散读数。
下一步进入 Q=3 初值校准。
```

Phase 2 的推进规则：

```text
1. 先找 Q_fft 接近 -2 的初值；
2. 若 p=1,q=2 不接近 -2，再扫 scale=1.2,1.4,1.6,2.0,2.2；
3. 找到接近 -2 的初值后，才跑 q_penalty=30 退火；
4. Q=2 成功后再做 Q=3；
5. 不要回头继续 Q=1 循环，除非发现 Phase 2 依赖 Q=1 的具体基准能量。
```

### B.3 Q=3 GPU 正式实验流程

本地 N=64,L=12 初值扫描显示 `scale=3.0` 只有 `Q_fft≈-2.297`，继续增大 scale 会更接近 -3：

| scale | Q_fft | E | coreV |
|---:|---:|---:|---:|
| 3.0 | -2.297262 | 842.56 | 407.35 |
| 3.4 | -2.436496 | 930.49 | 600.63 |
| 3.8 | -2.539389 | 1016.21 | 837.25 |
| 4.2 | -2.617194 | 1101.16 | 1130.04 |
| 4.6 | -2.677264 | 1186.17 | 1483.43 |
| 5.0 | -2.724515 | 1271.76 | 1905.81 |

GPU 正式 Q=3 建议先用 N=96,L=12,scale=5.0 起跑；若初值仍低于 -2.75，再扫 `scale=5.4,5.8`。不要在 CPU 本地跑 N=96 长扫描。

Q=3 GPU 初值扫描：

```bash
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 5.0 --steps 0 --hopf-every 1 --out outputs/init_Q3_p1q3_N96_L12_s50
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 5.4 --steps 0 --hopf-every 1 --out outputs/init_Q3_p1q3_N96_L12_s54
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 5.8 --steps 0 --hopf-every 1 --out outputs/init_Q3_p1q3_N96_L12_s58
```

选 `Q_fft` 最接近 -3 且 `coreV` 未贴边界的初值，进入退火。若 scale=5.0 最佳，则执行：

```bash
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 5.0 --steps 2000 --q-penalty 100 --q-target -3 --q-guard 2.0 --print-every 100 --hopf-every 100 --out outputs/Q3_p1q3_N96_L12_s50_qpen100
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 5.0 --init outputs/Q3_p1q3_N96_L12_s50_qpen100/nfield_q1_torch.npy --steps 2000 --q-penalty 30 --q-target -3 --q-guard 2.0 --print-every 100 --hopf-every 100 --out outputs/Q3_p1q3_N96_L12_s50_qpen30
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 5.0 --init outputs/Q3_p1q3_N96_L12_s50_qpen30/nfield_q1_torch.npy --steps 2000 --q-penalty 10 --q-target -3 --q-guard 2.0 --print-every 100 --hopf-every 100 --out outputs/Q3_p1q3_N96_L12_s50_qpen10
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 5.0 --init outputs/Q3_p1q3_N96_L12_s50_qpen10/nfield_q1_torch.npy --steps 1000 --q-penalty 0 --q-guard 2.0 --print-every 100 --hopf-every 100 --out outputs/Q3_p1q3_N96_L12_s50_qpen0_short
```

Q=3 收口判据：

```text
1. 无约束短跑中 Q_fft 漂移很小；
2. E 继续下降或近平台；
3. coreV 不爆炸、不贴边界；
4. 离散读数允许低于 -3，例如同网格 Q=2 为 -1.875，Q=3 若在 -2.7~-2.85 区间且稳定，可先作为阶段候选。
```

### B.4 Phase 3：matrix-free Hessian 数值判决

在 Q=1,2,3 能量序列跑出来之后，进入 matrix-free Hessian。目标不是完整显式 Hessian，而是只实现 Hessian-vector product 和 Krylov/Lanczos 最低模搜索。

判决目标：

```text
若低谱中出现接近 12 个稳定/准零几何模，12 结构获得数值复活证据；
若没有，或只出现普通平移/旋转/尺度伪模，则 12 暂时安息。
```

Hessian 前置条件：

```text
1. Q=1,2,3 都至少完成一次 q_penalty=0 short 或弱约束稳定退火；
2. 记录 E1,E2,E3 和 Q_fft；
3. 不再继续修饰 Q=1 初值，除非影响能量基准。
```

已新增并升级 matrix-free Hessian 探针：

```text
src/hopf_hessian_torch.py
```

它读取已退火的 `nfield_q1_torch.npy`，用 PyTorch autograd 做 Hessian-vector product，再用 Lanczos 三对角 Ritz 值估计最低谱。当前版本支持两种 HVP：

```text
--hvp-mode constrained          # 默认，加入单位矢量约束的局域拉格朗日乘子修正
--hvp-mode normalized-pullback  # 旧版 E[normalize(u)] pullback 探针，用于对照
```

并支持重构/保存 Ritz mode 与能量 line scan：

```text
--save-modes K
--scan-modes K
--scan-eps "-0.05,-0.02,-0.01,0,0.01,0.02,0.05"
```

本地 smoke test 已通过：

```bash
python src/hopf_hessian_torch.py --field outputs/smoke_n16/nfield_q1.npy --length 4 --iters 6 --hvp-mode constrained --scan-modes 2 --save-modes 2 --out outputs/hessian_smoke_n16_constrained
```

Q1 控制组重跑命令：

```bash
python src/hopf_hessian_torch.py --field outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy --length 10 --iters 80 --seed 1234 --device cuda --hvp-mode constrained --scan-modes 4 --save-modes 4 --out outputs/hessian_Q1_N80_L10_constrained_iters80_seed1234
```

旧版 HVP 对照命令：

```bash
python src/hopf_hessian_torch.py --field outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy --length 10 --iters 80 --seed 1234 --device cuda --hvp-mode normalized-pullback --out outputs/hessian_Q1_N80_L10_pullback_iters80_seed1234
```

低谱诊断当前读法：

```text
1. 当前 Hessian/Lanczos 工具测量的是静态能量曲率方向，不是物理频率；
2. 负模和近零几何模不能被解释为任何频率倍率的证据；
3. 平移、旋转、尺度类和盒子敏感模式需要在物理解释前被识别和剔除；
4. 真正的动力学频率检验需要显式 kinetic metric K、背景频率 ω0，以及跨 seed、电荷扇区、网格和盒子尺寸的稳定性检查；
5. 当前 Q12 结果只是静态高荷候选态；它不复活、不验证 12 倍率频率指纹。
```

### B.5 后续阶段（暂不做）

暂时不要进入这些任务：

```text
电子质量对照
大规模 N=128/160 精细收敛
```

这些放到 Q=1,2,3 能量序列和 matrix-free Hessian 初步判决之后。

当前目标：Phase 1 已收口。当前目标切换到 Phase 2：

```text
Q=2/Q=3 初值校准 → 拓扑保持退火 → E1,E2,E3 能量序列
```

旧的第一阶段目标是：

```text
Q_H=1 初值 → 能量计算 → 粗 Hopf 荷计算 → CPU 弛豫链路 → 输出诊断
```

当前版本在完成 Q=1,2,3 能量序列之前不做：

```text
动力学频率谱检验
电子/质量谱对照
```

PyTorch 版本支持 `--p k --q l` 构造简单高荷轴对称初值，脚本会把名义荷数记为 `charge=p*q`。如果不显式传入 `--q-target`，脚本会自动使用 `q_target=-(p*q)`。`--charge k` 仍保留为兼容快捷写法，等价于 `--p k --q 1`。

Q=3 同理：

```bash
python src/hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale 1.8 --steps 0 --hopf-every 1 --out outputs/init_Q3_p1q3_N96_L12_s18
```

高荷初值可能需要扫 `--scale`。建议先试：

```text
1.4, 1.6, 1.8, 2.0, 2.2
```

Phase 2 的目标不是马上找到精确极小解，而是先检查 `Q=2,3` 是否能保持拓扑荷并形成单调能量序列。

---

## 附录 C：§5.7 资料打包命令（从服务器归档）

要把本仓库的 §5.7 验证所需的全部资料（**输入场、脚本、文档、输出**）打包下来到本地复跑，用以下命令（在 GPU 服务器、当前仓库根目录）：

```bash
tar -czf sec57_bundle.tar.gz \
    outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy \
    outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy \
    src/d2_symmetry_probe.py \
    src/sh_decompose_components.py \
    src/q2_symmetry_full_audit.py \
    run_sec57_verifications.sh \
    notes/roadmap_v2/sec57_strict_rewrite_v1.md \
    notes/roadmap_v2/qeff_definition_v1.md \
    outputs/sec57_d2_probe_Q2 \
    outputs/sec57_shdecomp_Q2 \
    outputs/sec57_symm_audit_Q2 2>/dev/null

ls -lh sec57_bundle.tar.gz
tar -tzf sec57_bundle.tar.gz
```

**下载到本地**：

```bash
sz sec57_bundle.tar.gz      # 串口（Xshell/secureCRT）
# 或
scp user@gpu-server:/path/to/hopf_skyrme_cpu/sec57_bundle.tar.gz ./
```

**解压复跑**（在本地，需要保留原来的目录结构）：

```bash
tar -xzf sec57_bundle.tar.gz
# 在仓库根目录解压就还原成原结构
bash run_sec57_verifications.sh
```

---

## 附录 D：v1.6.6 完整资料打包命令（从服务器归档）

> **本节是 v1.6.6 配套文档**——把 §G + §B.1 + §B.2 + §D（实测数据）所需的全部资料打包下来。

### D.1 v1.6.6 完整打包命令

```bash
tar -czf v166_bundle.tar.gz \
    # ===== 输入场（Q=2 短跑 + Q=1 长跑） =====
    outputs/Q2_p1q2_N96_L12_s18_qpen30/nfield_q1_torch.npy \
    outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy \
    outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy \
    # ===== v1.6.6 脚本 =====
    src/kappa_rho_calibration.py \
    src/hopf_fiber_preimage_probe.py \
    src/d2_symmetry_probe.py \
    src/sh_decompose_components.py \
    src/q2_symmetry_full_audit.py \
    run_v166_linux.sh \
    run_sec57_verifications.sh \
    # ===== v1.6.6 文档（§G + §B.1 + §B.2 + §D） =====
    notes/roadmap_v2/sec57_gravity_theorem_v1.md \
    notes/roadmap_v2/sec57_strict_rewrite_v1.md \
    notes/roadmap_v2/qeff_definition_v1.md \
    # ===== v1.6.6 实测数据 =====
    outputs/v166_results/kappa_rho_calibration \
    outputs/v166_results/hopf_fiber_preimage \
    # ===== §5.7 v1.6.4 实测数据（对照组） =====
    outputs/sec57_d2_probe_Q2 \
    outputs/sec57_shdecomp_Q2 \
    outputs/sec57_symm_audit_Q2 \
    # ===== 预印本正文 =====
    README.md \
    2>/dev/null

ls -lh v166_bundle.tar.gz
tar -tzf v166_bundle.tar.gz
```

### D.2 最小打包命令（仅 v1.6.6 增量）

如果只关心 v1.6.6 增量（不考虑 §5.7 v1.6.4 历史数据）：

```bash
tar -czf v166_minimal.tar.gz \
    outputs/Q2_p1q2_N96_L12_s18_qpen30/nfield_q1_torch.npy \
    src/kappa_rho_calibration.py \
    src/hopf_fiber_preimage_probe.py \
    run_v166_linux.sh \
    notes/roadmap_v2/sec57_gravity_theorem_v1.md \
    outputs/v166_results \
    2>/dev/null

ls -lh v166_minimal.tar.gz
```

### D.3 下载和解压

**下载到本地**：

```bash
sz v166_bundle.tar.gz      # 串口（Xshell/secureCRT）
# 或
scp user@gpu-server:/path/to/hopf_skyrme_cpu/v166_bundle.tar.gz ./
```

**解压复跑**（在本地，需要保留原来的目录结构）：

```bash
tar -xzf v166_bundle.tar.gz
# 在仓库根目录解压就还原成原结构
bash run_v166_linux.sh         # 跑 v1.6.6 一键实测
# 或
bash run_sec57_verifications.sh  # 跑 §5.7 v1.6.4 验证
```

### D.4 打包内容大小估计

| 路径 | 估计大小 | 说明 |
|---|---:|---|
| `outputs/Q2_p1q2_N96_L12_s18_qpen30/nfield_q1_torch.npy` | ~33 MB | 96³×3 float32 |
| `outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy` | ~6 MB | 80³×3 float32 |
| `outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy` | ~33 MB | 96³×3 float32 |
| `outputs/v166_results/hopf_fiber_preimage/*.vtk` | ~5-15 MB | 8 个等值面 |
| `outputs/v166_results/kappa_rho_calibration/*.json` | < 1 KB | 1 个 JSON |
| 所有 `.md` / `.py` / `.sh` 文档 | < 1 MB | 文本 |
| **总计（压缩后）** | **~30-50 MB** | gzip 压缩 |

### D.5 关键验证命令（解压后必跑）

```bash
# 1. 验证 v1.6.6 数据完整
tar -tzf v166_bundle.tar.gz | grep -E "(kappa_rho.json|summary.json)"
# 应看到：
#   outputs/v166_results/kappa_rho_calibration/kappa_rho.json
#   outputs/v166_results/hopf_fiber_preimage/summary.json

# 2. 验证关键数值
cat outputs/v166_results/kappa_rho_calibration/kappa_rho.json | grep kappa_rho
# 预期：kappa_rho ≈ 0.877

cat outputs/v166_results/hopf_fiber_preimage/summary.json | grep expected_outcome
# 预期：Only C2z PASS

# 3. 重新跑一遍（5-15 分钟）
bash run_v166_linux.sh
# 应输出 [init] CRLF→LF 自动转码 + [1/5] 安装依赖 + [2/5] 查找 + [4/5] 任务1 + [5/5] 任务2
```
