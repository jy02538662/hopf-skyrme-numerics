# 阶段2 论文表述（冻结）

## 发现

在 \(G_{\mathrm{eff}}=10^{-3}\)（弱场）下，自洽外环中 \(\Phi/\kappa_{\mathrm{eff}}\) 几乎不变；能量持续缓降、\(Q\) 慢漂，主要来自**输入场在带 \(a(x)\) 能量下尚未弛豫到位**，而非引力反馈失败或不存在不动点。

## 能写 / 不能写

| 能写 | 不能写 |
|------|--------|
| 弱场回写循环数值稳定、\(\kappa>0\)、不炸 | 已通过三重收敛得到自洽解 |
| 弱场引力回写强度极弱（定量：\(\|2\Phi/c^2\|\sim0.055\) 且外环不更新） | 「阶段2失败 / 机制被否证」 |
| 阶段1 黄/红灯给出**重加权**边界 | 该边界 = 动力学 \(G_c\)（未必） |

## 交付拆分

1. **弱场回写稳定性报告** ✅ → [`reports/weak_field_stability_G1e-3.md`](reports/weak_field_stability_G1e-3.md)  
2. **\(G\) 扫描边界探索**（目标2）：\(3\times10^{-3}\)–\(5\times10^{-2}\)，每点 SC + freeze（目标4）→ `phase2_G_scan.py`  
3. **Q=2 的 \(G_c^{(\kappa)}\)**（目标3）✅ → [`reports/Gc_Q1_Q2.md`](reports/Gc_Q1_Q2.md)  
4. **阶段3 \(M(R)\) 标度** → `phase3_MR_scale_scan.py` + [`reports/MR_scaling_PROTOCOL.md`](reports/MR_scaling_PROTOCOL.md)  
5. 荷质比 / 层次问题：**仅定性表述**，不进数值验收

## 对照结果（2026-08，Q=1，\(G=10^{-3}\)）

自洽外环 vs `--freeze-kappa`（同初始 \(\kappa_{\mathrm{eff}}\)）：**\(E\)、\(dE/E\)、\(Q\)、\(dQ\) 曲线在数值精度内重合**。

→ 坐实：弱场下瓶颈是「带 \(a(x)\) 未弛豫到位」，**不是**引力自洽失败。

### \(G\) 扫描（冷启动收口）

- 墙前（\(G\le0.018\)）：全程 SC≈freeze  
- **\(G_c^{(\kappa)}\!\approx\!1.82\times10^{-2}\)**（细扫 \(\in(0.018,0.020)\)）  
- 详见 [`reports/G_scan_kappa_wall_Q1.md`](reports/G_scan_kappa_wall_Q1.md)

**明确不做**：绝热 \(G\) 爬坡找动力学分叉。\(G_c\) 是刚度墙；墙前无感、墙后 κ<0 — 无「调参出自洽窗口」的空间。

### Q=1 vs Q=2 \(G_c\)（已跑）

\(G_c(Q=2)/G_c(Q=1)\!\approx\!0.29\) → 朴素「Q↑⇒墙↑」不成立。  
字典内：\(\int\rho=E\) 下更紧 χ 加深 \(\max|\Phi|\)。详见 [`reports/Gc_Q1_Q2.md`](reports/Gc_Q1_Q2.md)。

### 阶段3（可收口）

- Whitehead / 短弛豫族：\(G_c\)–\(R_{\mathrm{rms}}/E\) 正相关（另报）。  
- **Derrick 拉伸**（pad L=20）：参考态 \(E_2\!\gg\!E_4\)，\(\lambda_\ast\!\approx\!0.28\)；窗 [0.5,2] **仅上升支**（与 Derrick 一致），**未见窗内 U 谷**；本族 \(G_c\!\propto\!R/E\) **不成立**。  
- 详见 [`reports/MR_scaling_Q1.md`](reports/MR_scaling_Q1.md)。

交付标签：
- **连续范式 κ 墙 + 弱场无回写** 仍成立  
- canonical Q=1 **非 virial**；完整 U 型需更紧参考场或承认压缩支不可达  
- 勿写「无平衡半径」；勿跨族混用 \(G_c\!\sim\!R/E\)
