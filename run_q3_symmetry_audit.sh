#!/usr/bin/env bash
# ============================================================================
# Q=3 对称群诊断（后处理，CPU 跑，5-10 分钟）
# ----------------------------------------------------------------------------
# 复用 q2_symmetry_full_audit.py 的 8 操作扫描，验证 Q=3 整场对称群
# ============================================================================
set -e

cd /

OUT_DIR=/outputs/v166_results/q3_symmetry_audit
mkdir -p "$OUT_DIR"

echo "[init] Q=3 对称群诊断开始 $(date '+%H:%M:%S')"
echo "[init] 扫描 8 操作：C2x, C2y, C2z, Mx, My, Mz, S4x, S4y"
echo ""

# Q=3 phase2 短跑场
Q3_FIELD="/outputs/Q3_L12_phase2/nfield_q3_torch.npy"

if [ ! -f "$Q3_FIELD" ]; then
    echo "[error] $Q3_FIELD 不存在"
    echo "[hint] 可用 Q=3 场："
    ls -la /outputs/Q3_*/nfield_q3_torch.npy 2>/dev/null || echo "  (没找到任何 Q3 场)"
    exit 1
fi

echo "[1/3] 字段元数据："
python3 -c "
import numpy as np
a = np.load('$Q3_FIELD')
print(f'  shape: {a.shape}')
print(f'  dtype: {a.dtype}')
"

echo ""
echo "[2/3] 跑 8 操作扫描（theta 网格 181 点 × 8 操作）..."
python3 /src/q2_symmetry_full_audit.py \
    --field "$Q3_FIELD" \
    --length 12 \
    --tolerance 1e-2 \
    --out "$OUT_DIR"

echo ""
echo "[3/3] 复制 summary 到 stdout："
if [ -f "$OUT_DIR/q2_symmetry_full_audit.json" ]; then
    python3 -c "
import json
with open('$OUT_DIR/q2_symmetry_full_audit.json') as f:
    s = json.load(f)
print('--- Q=3 GROUP VERDICT ---')
print(s['group_verdict'])
print()
print('--- EXACT/NEAR 总结 ---')
for r in s['per_op']:
    print(f\"  {r['op']:5s} L_inf_min={r['L_inf_min_over_sweep']:.3e}  theta_opt={r['theta_opt_deg']:6.1f} deg  {r['verdict']}\")
"
fi

echo ""
echo "[done] Q=3 诊断完成 $(date '+%H:%M:%S')"
echo "[out ] $OUT_DIR/q2_symmetry_full_audit.json"