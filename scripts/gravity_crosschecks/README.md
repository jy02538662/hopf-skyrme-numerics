# 引力相关交叉检验（legacy，非 BP2.5 \(G_c\)）

从 `hopf_linking/vortex_monopole_3d_interaction/` 迁入的 **软单极远场 \(A/R\)** 标定。  
用于核对「点源极限下相互作用系数」与由 \(A_{\mathrm{fit}}\) 导出的 **legacy** \(G_{\mathrm{eff}},\kappa_t\)。

## 字典对照（勿混用）

| 符号 | 本目录（monopole toy） | 断点 2.5 |
|------|------------------------|----------|
| 源 | 软单极 \(\theta\) 梯度交叉项 | \(\rho=\kappa_\rho\chi^2\) + Poisson \(\Phi\) |
| \(G_{\mathrm{eff}}\) | \(\|A\|/m_Q^2\)（玩具导出） | 扫描参数；墙 \(G_c^{(\kappa)}\) 来自 \(\kappa_{\mathrm{eff}}>0\) |
| 结论能否并入 BP2.5 主文？ | **否**（仅交叉检验 / 历史玩具） | 见 `breakpoint_2_5_gravity/reports/` |

## 复现

```bash
cd scripts/gravity_crosschecks
pip install matplotlib   # 若尚未安装
python monopole_3d_farfield_A.py --skip-diag \
  --core-box 12 --core-step 0.08 \
  --out /tmp/monopole_3d_farfield_A.png
```

通过标准：\(|A_{\mathrm{fit}}-Q^2/(4\pi)|/A_{\mathrm{theory}}<5\%\)，且 \(U\cdot R\) 平台波动 \(<5\%\)。

拓扑整数荷请用 [`../hopf_linking/`](../hopf_linking/)，不要用本脚本。
