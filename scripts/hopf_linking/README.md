# Preimage linking number（拓扑交叉检验）

从 sibling 包 `hopf_linking/` 迁入。对单位场 \(n:\mathbb{R}^3\to S^2\) 用 **preimage 链环数** \(Q_{\mathrm{link}}\) 复算 Hopf 荷，与 Whitehead / FFT 的 \(Q_{\mathrm{fft}}\) 对照。

## 与断点 2.5（半经典引力）的关系

| 问题 | 答案 |
|------|------|
| 这是 \(G_c^{(\kappa)}\) / Poisson 判据吗？ | **否** |
| 用在引力流水线哪里？ | 确认输入场拓扑未崩（\(Q_{\mathrm{link}}\approx Q_{\mathrm{fft}}\)） |
| 能否替代 κ 墙扫描？ | **不能** |

引力动力学与 κ 正定墙见 [`../breakpoint_2_5_gravity/`](../breakpoint_2_5_gravity/)。

## 网格约定（必读）

| `grid_mode` | `--length` 含义 | 坐标 |
|-------------|-----------------|------|
| **`half`（本仓默认）** | 半盒长 \(L\)，与 `hopf_skyrme_torch` 一致 | `linspace(-L,L,N)`，\(h=2L/(N-1)\) |
| `side`（legacy） | 全边长 | 胞心网格 \([-L/2+h/2,L/2-h/2]\)，\(h=L/N\) |

对 canonical Q=1 场（N=80, 半盒 L=10）请用：

```bash
python hopf_linking.py \
  --field /path/to/nfield_q1_torch.npy \
  --length 10 --grid-mode half \
  --out /tmp/q1_link/
```

链环数在整体伸缩下不变；但坐标诊断（曲线位置）必须与生成场的网格一致，故对本仓场 **务必 `half`**。

## 依赖

`numpy`, `scipy`, `scikit-image`（已在仓库根 `requirements.txt`）。

## 自检（无外部场）

```bash
cd scripts/hopf_linking
python test_hopf_analytic.py
# 期望：half 与 side 下各对 preimage 均 Q_link = -1
```

## 对真实弛豫场

```bash
python hopf_linking.py \
  --field ../../outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy \
  --length 10 --grid-mode half \
  --out /tmp/q1_link/
```

历史对照（legacy 跑通记录）：Q=1 → \(Q_{\mathrm{link}}=-1\)；Q=2 → \(-2\)（与 \(Q_{\mathrm{fft}}\) 同号、近整数）。

一键与 \(Q_{\mathrm{fft}}\) 对照：[`../breakpoint_2_5_gravity/crosscheck_Q_link.py`](../breakpoint_2_5_gravity/crosscheck_Q_link.py)。
