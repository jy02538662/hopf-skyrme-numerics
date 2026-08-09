# 断点 2.5：半经典引力试水（降级阶段 1 → 阶段 2）

先读 [`PREREQUISITES.md`](PREREQUISITES.md)（字典已冻结，**与 \(k_{\mathrm{ex}}\) 脱钩**）。  
论文表述：[`PHASE2_FRAMING.md`](PHASE2_FRAMING.md)。  
弱场收口：[`reports/weak_field_stability_G1e-3.md`](reports/weak_field_stability_G1e-3.md)。

## 时间线（当前）

| 时段 | 目标 | 状态 |
|------|------|------|
| 当天 | **目标1** 弱场稳定性报告 | ✅ `reports/weak_field_*` |
| 第1–2周 | **目标2** \(G_{\mathrm{eff}}\) 动力学扫描 | ✅ 冷启动收口：\(G_c^{(\kappa)}\!\approx\!1.82\times10^{-2}\)，墙前无分叉 |
| — | ~~绝热爬坡找自洽窗口~~ | **明确不做**（原理性边界，非调参问题） |
| 目标3 | Q=2 的 \(G_c^{(\kappa)}\) 对比 | ✅ \(G_c(2)/G_c(1)\!\approx\!0.29\)（`OPPOSE_BP1_NAIVE`） |
| **下一阶段** | **阶段3 \(M(R)\) 标度律** | ✅ 可收口：\(G_c\!\sim\!R/E\) 方向支持；U 型本协议未建立 |
| 表述 | 荷质比 / 层次问题 | **仅定性**；本阶段零数值验收 |

---

## 阶段 1a（已实现）：纯后处理

```bash
cd /path/to/breakpoint_2_5_gravity
OUT=/tmp/bp25_p1 && mkdir -p "$OUT"

python phase1_single_shot.py --abs-outputs --Q 1 \
  --G-eff 1e-3 --kappa0 1.0 --c2 1.0 \
  --r-fit 4,5,6,7,8 --out-dir "$OUT"
```

Q=1 真场（\(G=10^{-3}\)）：\(\alpha_\Phi\approx1.04\)、弱场 OK、\(\kappa_{\mathrm{eff}}>0\)。

## 路径 A：重加权 G 扫描（固定 n）

```bash
OUT=/tmp/bp25_Gscan && mkdir -p "$OUT"
python phase1_G_scan.py --abs-outputs --Q 1 \
  --G-list 1e-4,1e-3,1e-2,5e-2 --out-dir "$OUT"
```

GREEN≤\(10^{-3}\)；ORANGE自\(10^{-2}\)；RED自\(5\times10^{-2}\)。**≠** 动力学 \(G_c\)。

## 路径 B：阶段 1b 固定 κ_eff 短弛豫

```bash
OUT=/tmp/bp25_1b && mkdir -p "$OUT"
python phase1b_short_relax.py --abs-outputs --Q 1 --G-eff 1e-3 \
  --steps 150 --lr 1e-3 --device cuda --also-baseline --out-dir "$OUT"
```

GREEN/ORANGE：与 baseline **无显著动力学分叉**。

## 阶段2：自洽外环 + 弱场对照

```bash
OUT=/tmp/bp25_p2 && mkdir -p "$OUT"
python phase2_self_consistent_loop.py --abs-outputs --Q 1 --G-eff 1e-3 \
  --outer 80 --n-relax 20 --alpha 0.4 --lr 5e-4 \
  --etol 1e-4 --qtol 5e-4 --device cuda --out-dir "$OUT"

# 对照：冻结初始 κ_eff
python phase2_self_consistent_loop.py --abs-outputs --Q 1 --G-eff 1e-3 \
  --freeze-kappa --outer 80 --n-relax 20 --lr 5e-4 \
  --etol 1e-4 --qtol 5e-4 --device cuda --out-dir "$OUT"
```

弱场结论：SC 与 freeze **曲线重合** → 回写无可分辨贡献；三重判据未过 = 缓降弛豫，非机制失败。

History 现含 `rho_max` / `chi_max`（死结1 核心密度监控）。

## 目标2/4：动力学 G 扫描（SC + freeze）

```bash
OUT=/tmp/bp25_Gscan_dyn && mkdir -p "$OUT"
python phase2_G_scan.py --abs-outputs --Q 1 --device cuda \
  --G-list 3e-3,1e-2,3e-2,5e-2 \
  --outer 40 --n-relax 20 --lr 5e-4 --alpha 0.4 \
  --out-dir "$OUT"
```

### 第一轮 + κ 墙细扫（Q=1）

→ [`reports/G_scan_round1_Q1.md`](reports/G_scan_round1_Q1.md)、[`reports/G_scan_kappa_wall_Q1.md`](reports/G_scan_kappa_wall_Q1.md)

| \(G\) | 结论 |
|-------|------|
| \(\le0.018\) | SC≈freeze，无分叉 |
| \(\in(0.018,0.020)\) | **\(G_c^{(\kappa)}\!\approx\!1.82\times10^{-2}\)**（κ 正定硬墙） |
| \(\ge0.020\) | init \(\kappa_{\mathrm{raw}}<0\) → SC abort |

**目标2冷启动收口**：墙前无动力学自洽窗口；硬墙是 κ≤0，不是发散振荡。

细扫 κ 墙（已完成）：

```bash
python phase2_G_scan.py --abs-outputs --Q 1 --device cuda \
  --G-list 1.2e-2,1.5e-2,1.8e-2,2.0e-2,2.2e-2 \
  --outer 20 --n-relax 20 --lr 5e-4 \
  --out-dir /tmp/bp25_Gscan_kappa_wall
```

启发式 class：`WEAK_OR_NO_BIFURCATION` | `BIFURCATED_FROM_FREEZE` | `CONVERGING_HINT` | `DIVERGING` | `CORE_DENSITY_RISING` | `INIT_KAPPA_WALL` | `ABORT`

**请同步最新** `phase2_self_consistent_loop.py`（init κ≤0 拒跑 freeze+floor）。

## 目标3：Q=2 的 \(G_c^{(\kappa)}\) 对比（性价比最高；不做绝热爬坡）

\(G_c\) 是 **κ 正定墙**，不是动力学收敛边界。墙前回写无感、墙后 κ<0 → **不做绝热爬坡找自洽窗口**。

一次对比 Q=1 与 Q=2（推荐）：

```bash
OUT=/tmp/bp25_Gc_by_Q && mkdir -p "$OUT"
python compare_Gc_kappa_by_Q.py --abs-outputs --Q-list 1,2 \
  --out-dir "$OUT"
```

看终端 `CROSS-CHECK` 标签：
- `SUPPORT_BP1`：\(G_c(Q2)/G_c(Q1)\ge1.2\)
- `NO_SIGNIFICANT_DIFF`：比值 ∈(0.8,1.2)
- `OPPOSE_BP1_NAIVE`：比值 ≤0.8（反对「Q↑⇒\(G_c\)↑」朴素映射，≠否定远场屏蔽）

### 已跑结果 → [`reports/Gc_Q1_Q2.md`](reports/Gc_Q1_Q2.md)

| Q | \(G_c^{(\kappa)}\) |
|---|-------------------|
| 1 | \(1.82\times10^{-2}\) |
| 2 | \(5.28\times10^{-3}\)（比值 ≈0.29） |

标签：`OPPOSE_BP1_NAIVE`。字典内解释：\(\int\rho=E\) + 更紧 χ → 更深 \(\Phi\) → 更低墙。

## 阶段3：\(M(R)\) 标度律（下一刀）

协议：[`reports/MR_scaling_PROTOCOL.md`](reports/MR_scaling_PROTOCOL.md)。  
结果模板：[`reports/MR_scaling_Q1.md`](reports/MR_scaling_Q1.md)。

固定 Q=1、网格 N=80/L=10，扫 ansatz `scale` → 短弛豫 → \(E(R)\)、\(G_c(R)\)；拟合 \(E=aR+b/R\) 与 \(G_c\approx k(R/E)\)。

```bash
OUT=/tmp/bp25_MR_Q1_r2 && mkdir -p "$OUT"
python phase3_MR_scale_scan.py --device cuda --Q 1 \
  --n 80 --length 10 \
  --scale-list 0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4 \
  --steps 500 --lr 1e-3 --out-dir "$OUT"
# → $OUT/phase3_MR_Q1.json
# R_primary=R_rms；可选 --steps 0 做纯 ansatz Derrick
```

Round-1 诊断：`R_chi` 顶死 → U 型失败；见 [`reports/MR_scaling_Q1.md`](reports/MR_scaling_Q1.md)。请同步新脚本后跑 round-2。

### 真 Derrick 空间拉伸（推荐补做）

对 canonical 已弛豫场做 \(n_\lambda(x)=n(x/\lambda)\)，不弛豫，扫 λ∈[0.5,2.0]：

```bash
OUT=/tmp/bp25_MR_dilate_pad && mkdir -p "$OUT"
python phase3_MR_dilate.py --abs-outputs --Q 1 --device cuda \
  --embed-length 20 \
  --lambda-list 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0 \
  --out-dir "$OUT"
```

同盒 L=10 首跑因 **R_rms/L≈0.86 填盒** 失败（假边界墙）；必须 `--embed-length 20` 真空垫高。
## Poisson

`poisson_phi.py`：DST-I 狄利克雷 \(\Phi|_{\partial}=0\)，\(\nabla^2\Phi=4\pi G_{\mathrm{eff}}\rho_{\mathrm{eff}}\)。

## 拓扑交叉检验（迁自 hopf_linking；≠ \(G_c\)）

确认输入场 Hopf 荷未崩：\(Q_{\mathrm{link}}\)（preimage 链环）vs \(Q_{\mathrm{fft}}\)。

```bash
# 无外部场：解析 Hopf
python crosscheck_Q_link.py --synthetic

# canonical Q=1（需 outputs/ 或 --abs-outputs）
python crosscheck_Q_link.py --Q 1 --out /tmp/q1_link_vs_fft.json
```

详见 [`../hopf_linking/README.md`](../hopf_linking/README.md)。  
软单极玩具 \(G_{\mathrm{eff}}\)（**不是**本目录 \(G_c^{(\kappa)}\)）：[`../gravity_crosschecks/`](../gravity_crosschecks/)。

