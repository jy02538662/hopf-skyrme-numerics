#!/usr/bin/env bash
# ============================================================================
# Q=4 对称群诊断（后处理，CPU 跑，5-10 分钟）
# ----------------------------------------------------------------------------
# 复用 q2_symmetry_full_audit.py 的 8 操作扫描，验证 Q=4 整场对称群
# ============================================================================
set -e

cd /

OUT_DIR=/outputs/v166_results/q4_symmetry_audit
mkdir -p "$OUT_DIR"

echo "[init] Q=4 对称群诊断开始 $(date '+%H:%M:%S')"
echo "[init] 扫描 8 操作：C2x, C2y, C2z, Mx, My, Mz, S4x, S4y"
echo ""

# Q=4 phase2 短跑场
Q4_FIELD="/outputs/Q4_L12_phase2/nfield_q4_torch.npy"

if [ ! -f "$Q4_FIELD" ]; then
    echo "[error] $Q4_FIELD 不存在"
    echo "[hint] 可用 Q=4 场："
    ls -la /outputs/Q4_*/nfield_q4_torch.npy 2>/dev/null || echo "  (没找到任何 Q4 场)"
    exit 1
fi

echo "[1/3] 字段元数据："
python3 -c "
import numpy as np
a = np.load('$Q4_FIELD')
print(f'  shape: {a.shape}')
print(f'  dtype: {a.dtype}')
"

echo ""
echo "[2/3] 跑 8 操作扫描（theta 网格 181 点 × 8 操作）..."
python3 /src/q2_symmetry_full_audit.py \
    --field "$Q4_FIELD" \
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
print('--- Q=4 GROUP VERDICT ---')
print(s['group_verdict'])
print()
print('--- EXACT/NEAR 总结 ---')
for r in s['per_op']:
    print(f\"  {r['op']:5s} L_inf_min={r['L_inf_min_over_sweep']:.3e}  theta_opt={r['theta_opt_deg']:6.1f} deg  {r['verdict']}\")
"
fi

echo ""
echo "[done] Q=4 诊断完成 $(date '+%H:%M:%S')"
echo "[out ] $OUT_DIR/q2_symmetry_full_audit.json"