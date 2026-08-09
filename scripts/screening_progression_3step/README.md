# 屏蔽递进律验证（能写/不能写边界）

| 步骤 | 脚本 | 结论状态 |
|------|------|----------|
| 1 | `step1_alpha_fit.py` | ⟨χ⟩/tilt 的 α 递进 — **经验成立** |
| 2 | `step2_c2_parity_ratio.py` | 奇 m ≫ 偶 m — **成立** |
| 3 | `step3_signed_vs_chi.py` | χ 天然有 l=0 — **澄清成立** |
| 4 | `step4_m_spectrum.py` | \|m\|=Q 匹配 — **已否证**（主导均为 \|m\|=1） |
| 5 | `step5_l_spectrum.py` | 固定 \|m\|=1 内 l 随 Q 上移 — **SUPPORT** |
| 6 | `step6_laplace_phase.py` | 远场拉普拉斯相：\|r²∇²dn₊\|/\|dn₊\| ≪ 1 |

网格：`linspace(-L,L,N)`。Fibonacci 球壳 + 三线性。nfield `(N,N,N,3)`。

## GPU：第 6 步（拉普拉斯相，默认 Q=1）

```bash
cd /path/to/screening_progression_3step
OUT=/tmp/screening_3step && mkdir -p "$OUT"

python step6_laplace_phase.py --abs-outputs --Q 1 \
  --r-shells 5,6,7,8,9 --n-points 6000 --eta-tol 0.01 --out-dir "$OUT"
```

主判据无量纲 **eta = |r² ∇² dn₊| / |dn₊|**（有量纲的 \|∇²\|/\|f\| 单位是 1/L²，不能直接与 10⁻² 比）。
看末尾 `SUPPORT` / `WEAK_SUPPORT` / `NOT_IN_LAPLACE_PHASE`。

默认场路径见 `common.py`。
