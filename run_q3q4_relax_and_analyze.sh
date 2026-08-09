#!/bin/bash
# ============================================================
# Q=3 / Q=4 松弛 + 远场分析
# 使用前：先运行 run_q3q4_verification.sh 阶段 0 确定最优参数
# 然后修改下面的 BEST_* 变量
# ============================================================

set -e
cd "$(dirname "$0")/src"

OUTBASE="../outputs"
mkdir -p "$OUTBASE"

# ============================================================
# 用户配置区：根据阶段 0 的结果填写
# ============================================================

Q3_P=1
Q3_Q=3
Q3_SCALE=4.0

Q4_P=1
Q4_Q=4
Q4_SCALE=4.5

# ============================================================
# 阶段 1：Q=3 松弛
# ============================================================

echo "========== 阶段 1A：Q=3 松弛（5000步）=========="

python hopf_skyrme_torch.py --n 96 --length 12 \
  --p $Q3_P --q $Q3_Q --scale $Q3_SCALE \
  --optimizer riemannian --lr 2e-3 \
  --q-penalty 300 --q-target -3 \
  --steps 5000 --float64 \
  --print-every 500 --hopf-every 500 \
  --out "$OUTBASE/Q3_L12_phase1"

echo ""
echo "--- Q=3 phase1 完成 ---"
python -c "
import json, sys
s = json.load(open('$OUTBASE/Q3_L12_phase1/summary.json'))
print(f\"Q_fft = {s['Q_fft']:.4f}, E = {s['E']:.4f}\")
if abs(s['Q_fft']) < 2.5:
    print('WARNING: Q leaked below 2.5!')
else:
    print('OK: Q stable.')
"

echo ""
echo "========== 阶段 1B：Q=3 接力松弛（5000步）=========="

Q3_INIT=$(find "$OUTBASE/Q3_L12_phase1/" -name "nfield_*.npy" | head -1)

python hopf_skyrme_torch.py --n 96 --length 12 \
  --p $Q3_P --q $Q3_Q --scale $Q3_SCALE \
  --optimizer riemannian --lr 1e-3 \
  --q-penalty 300 --q-target -3 \
  --init "$Q3_INIT" \
  --steps 5000 --float64 \
  --print-every 500 --hopf-every 500 \
  --out "$OUTBASE/Q3_L12_phase2"

# ============================================================
# 阶段 2：Q=3 远场分析
# ============================================================

echo ""
echo "========== 阶段 2：Q=3 远场 =========="

Q3_FIELD=$(find "$OUTBASE/Q3_L12_phase2/" -name "nfield_*.npy" | head -1)
echo "Using: $Q3_FIELD"

python farfield_multipole.py --field "$Q3_FIELD" \
  --length 12 --proxy transverse --r-min 8 --r-max 10 --n-shells 12 \
  --out "$OUTBASE/Q3_farfield_transverse_r8_10"

python farfield_multipole.py --field "$Q3_FIELD" \
  --length 12 --proxy transverse --r-min 10 --r-max 11.5 --n-shells 12 \
  --out "$OUTBASE/Q3_farfield_transverse_r10_115"

python farfield_multipole.py --field "$Q3_FIELD" \
  --length 12 --proxy dev_norm --r-min 8 --r-max 10 --n-shells 12 \
  --out "$OUTBASE/Q3_farfield_devnorm_r8_10"

# ============================================================
# 阶段 3：Q=4 松弛
# ============================================================

echo ""
echo "========== 阶段 3A：Q=4 松弛（5000步）=========="

python hopf_skyrme_torch.py --n 96 --length 12 \
  --p $Q4_P --q $Q4_Q --scale $Q4_SCALE \
  --optimizer riemannian --lr 1e-3 \
  --q-penalty 400 --q-target -4 \
  --steps 5000 --float64 \
  --print-every 500 --hopf-every 500 \
  --out "$OUTBASE/Q4_L12_phase1"

echo ""
echo "--- Q=4 phase1 完成 ---"
python -c "
import json
s = json.load(open('$OUTBASE/Q4_L12_phase1/summary.json'))
print(f\"Q_fft = {s['Q_fft']:.4f}, E = {s['E']:.4f}\")
if abs(s['Q_fft']) < 3.5:
    print('WARNING: Q leaked below 3.5!')
else:
    print('OK: Q stable.')
"

echo ""
echo "========== 阶段 3B：Q=4 接力松弛 =========="

Q4_INIT=$(find "$OUTBASE/Q4_L12_phase1/" -name "nfield_*.npy" | head -1)

python hopf_skyrme_torch.py --n 96 --length 12 \
  --p $Q4_P --q $Q4_Q --scale $Q4_SCALE \
  --optimizer riemannian --lr 5e-4 \
  --q-penalty 500 --q-target -4 \
  --init "$Q4_INIT" \
  --steps 5000 --float64 \
  --print-every 500 --hopf-every 500 \
  --out "$OUTBASE/Q4_L12_phase2"

# ============================================================
# 阶段 4：Q=4 远场分析
# ============================================================

echo ""
echo "========== 阶段 4：Q=4 远场 =========="

Q4_FIELD=$(find "$OUTBASE/Q4_L12_phase2/" -name "nfield_*.npy" | head -1)
echo "Using: $Q4_FIELD"

python farfield_multipole.py --field "$Q4_FIELD" \
  --length 12 --proxy transverse --r-min 8 --r-max 10 --n-shells 12 \
  --out "$OUTBASE/Q4_farfield_transverse_r8_10"

python farfield_multipole.py --field "$Q4_FIELD" \
  --length 12 --proxy transverse --r-min 10 --r-max 11.5 --n-shells 12 \
  --out "$OUTBASE/Q4_farfield_transverse_r10_115"

python farfield_multipole.py --field "$Q4_FIELD" \
  --length 12 --proxy dev_norm --r-min 8 --r-max 10 --n-shells 12 \
  --out "$OUTBASE/Q4_farfield_devnorm_r8_10"

# ============================================================
# 汇总
# ============================================================

echo ""
echo "============================================"
echo "全部完成！"
echo ""
echo "判决标准："
echo "  alpha ~ 1 -> 未屏蔽（长程尾场）"
echo "  alpha ~ 3 -> 屏蔽态"
echo "  alpha ~ 2 -> 偶极主导"
echo ""
echo "如果 Q=3 alpha~1, Q=4 alpha~3 -> 奇偶效应"
echo "如果 Q=3,4 都 alpha~3 -> 只有Q=1有长程场"
echo "============================================"
