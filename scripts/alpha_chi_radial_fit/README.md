# α / χ 径向远场拟合

## 对 Q=1 结果的正确解读

| 量 | 典型 β (L=10) | 含义 |
|----|---------------|------|
| **α** | ≈1.00–1.03 | 物理：1/r 单极 ✓ |
| **χ = sinα** | ≈0.75–0.83 | **几何**：α 在窗内不小，不是数值 bug |
| **arcsin(χ)** | ≈ β_α | 交叉验证：应回到 1/r |

单位模下点点 χ=sinα。若 α≈A/r，则

\[
\frac{d\ln\sin\alpha}{d\ln r}=-\alpha\cot\alpha
\]

故 χ 的表观指数 β_eff=α cot(α)，仅当 α→0 才→1。A≈5、r∈[5,9] 时 α~0.5–1 rad，β_eff~0.7–0.9，与实测一致。

先前“贴边本底”假说被否定：内收窗口后 β_χ **更低**（α 更大），且边壳 χ≈sin(⟨α⟩)。

**远场主 proxy 用 α**（或 arcsin χ）。勿把 β_χ<1 写成物理屏蔽。

## 推荐命令

```bash
python alpha_chi_radial_fit.py \
  --field /path/to/nfield_q1_torch.npy \
  --length 10 \
  --r-min-fit 5.5 --r-max-fit 8.5 \
  --normalize --scan-windows \
  --out alpha_radial_fit_Q1.png \
  --save-csv alpha_chi_radial_Q1.csv
```

期望：β_α≈1；β_χ≈脚本打印的 `predict β from sin(A/r^β)`；β_arcsinχ≈β_α。
