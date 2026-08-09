# 断点2：Berry 曲率响应对 \(k_{\mathrm{ex}}\)（工作假设框架）

## 置信层级（与正文一致）

| 内容 | 层级 |
|------|------|
| \(\mathbf b[\mathbf n]\)、极角倾斜 \(\delta\mathbf n=\alpha\tilde\theta\,\hat{\boldsymbol\theta}\) | 可严格编码 |
| 响应核 \(K,M\)（**已修正**，见下） | 给定 \(\delta\mathbf n\) 下可交叉检验 |
| 构型空间 Berry = 实空间通量 | **工作假设**（验证2/3 不自动证明） |
| 真空 \(\Delta\Theta^{(0)}=4\pi Q_H\) | **不预设**；由数值标定 |

## 公式修正（相对你上一稿）

上一稿 \(M_{ij}=\varepsilon_{ikl}\mathbf n\cdot(\partial_k\hat{\boldsymbol\theta}\times\partial_l\mathbf n)\) **有误**（自由指标 \(j\) 未出现；\(\partial\hat{\boldsymbol\theta}\) 应进 \(K\)）。

正确形式见 `berry_kernels.py`：

\[
\frac{\delta b_i}{\alpha}=K_i\tilde\theta+M_{im}\partial_m\tilde\theta,
\quad
M_{im}=\varepsilon_{imk}\,\mathbf n\cdot(\hat{\boldsymbol\theta}\times\partial_k\mathbf n).
\]

\(K\) 含 \(\hat{\boldsymbol\theta}\cdot(\partial\mathbf n\times\partial\mathbf n)\) 与 \(\partial\hat{\boldsymbol\theta}\) 两项。

## 验证1（优先）

```bash
cd /path/to/k_ex_berry_response
# 需能 import 旁路 screening_progression_3step/common.py，或把整个 scripts 拷上 GPU
OUT=/tmp/k_ex_v1 && mkdir -p "$OUT"

python verify1_kernel_maps.py --abs-outputs --Q 1 \
  --theta-mode soft_monopole --soft 2.0 --alpha 1e-4 --out-dir "$OUT"
```

看 `VERDICT: PASS/WEAK_PASS/FAIL`（核 vs 有限差 \(\delta\mathbf b/\alpha\)）。

本地无场烟测：

```bash
python verify1_kernel_maps.py --synthetic --out-dir ./out_v1
```

## 验证2（通量双通道）

```bash
OUT=/tmp/k_ex_v2 && mkdir -p "$OUT"

python verify2_flux_dual_channel.py --abs-outputs --Q 1 \
  --theta-mode soft_monopole --soft 2.0 --alpha 1e-4 \
  --planes xy,xz,yz --radii 2,3,4,5 \
  --alphas 1e-5,1e-4,1e-3 --rel-floor 0.05 \
  --out-dir "$OUT"
```

Q=1 真场结果（2026-08）：**WEAK_PASS** — PRIMARY xy R=3 rel≈9.6%；α=1e-5 时 rel≈0.7%；外盘 R=5 rel≈25%。xz/yz 为噪声 SKIP。

## 验证3（真空通量 / Stokes）

拆成两命题：**(A) 同边界曲面无关**（主验收）；**(B) 平面盘半径扫描**（诊断，Hopfion 上通常不恒定）。

```bash
OUT=/tmp/k_ex_v3 && mkdir -p "$OUT"

python verify3_vacuum_flux_stability.py --abs-outputs --Q 1 \
  --stokes-R 3.0 --radii 2,3,4,5 --offsets 0,-1,1 \
  --cone-height 3.0 --stokes-tol 0.10 --out-dir "$OUT"
```

需与更新后的 `berry_kernels.py`（含 `hemisphere_flux_xy` / `cone_flux_xy`）一起拷贝。

Q=1 真场（2026-08）：**WEAK_SUPPORT_STOKES** — 同边界 spread≈12.3%（略超 10%）；半径扫描强烈不恒定（spread≈69%）；xz/yz 通量≈0。

## 验证4（\(k_{\mathrm{ex}}\) 路径/曲面）

同验证3拆分：**(A) 同边界 Stokes**（主验收）；**(B) 换半径**（若大变 ⇒ 非拓扑常数）。

```bash
OUT=/tmp/k_ex_v4 && mkdir -p "$OUT"

python verify4_kex_path_independence.py --abs-outputs --Q 1 \
  --theta-mode soft_monopole --soft 2.0 --alpha 1e-5 \
  --stokes-R 3.0 --radii 2,3,4,5 --offsets 0,-1,1 \
  --cone-height 3.0 --channel both --stokes-tol 0.10 --out-dir "$OUT"
```

Q=1 真场（2026-08）：**FAIL_STOKES** — 平底盘 k≈−0.62，半球/锥≈−1.49（spread≈58%）；换半径甚至变号（R=2 正、R≥3 负）。kernel 与 FD 彼此一致 ⇒ 非通道误差。  
**结论：在当前操作定义下，不能把 \(k_{\mathrm{ex}}\) 写成路径无关拓扑常数。**

## 断点2汇总（Q=1）

| 验证 | 结果 | 含义 |
|------|------|------|
| 1 | PASS | \(K,M\) 点态线性响应成立 |
| 2 | WEAK_PASS | 主盘上通量双通道大致成立 |
| 3 | WEAK_SUPPORT_STOKES | 真空 \(\mathbf b\) 同边界弱一致；平面换 R 非恒定 |
| 4 | FAIL_STOKES | \(k_{\mathrm{ex}}\) 同边界已失败；强依赖曲面/半径 |

## 注意

验证1/2 检验线性 \(\delta\mathbf b\)；验证3/4 的 Stokes 只支持通量几何；半径非恒定 ⇒ **不能**宣称 \(k_{\mathrm{ex}}\) 为路径无关拓扑常数。亦不证明构型空间交换 Berry。
