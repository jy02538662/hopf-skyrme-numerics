#!/bin/bash
# ============================================================
# Q=3 / Q=4 远场验证完整流程
# 目标：在 4090 租机上一次跑完，判断屏蔽规律
# 预计耗时：Q=3 约 30-60 分钟，Q=4 约 60-90 分钟
# ============================================================

set -e
cd "$(dirname "$0")/src"

OUTBASE="../outputs"
mkdir -p "$OUTBASE"

# ============================================================
# 阶段 0：初始化校准（找最优 scale）
# 目的：确认哪个 scale 让初始 Q_fft 最接近目标值
# ============================================================

echo "========== 阶段 0：Q=3 初值校准 =========="

# 方案 A：p=3, q=1（W^3 型，标准高荷 Hopf 映射）
for scale in 2.0 2.5 3.0 3.5 4.0; do
  echo "--- Q=3 (p=3,q=1) scale=$scale ---"
  python hopf_skyrme_torch.py --n 96 --length 12 --p 3 --q 1 --scale $scale \
    --steps 0 --hopf-every 1 --float64 \
    --out "$OUTBASE/init_Q3_p3q1_s${scale}" 2>&1 | grep -E "Q_fft|initial"
done

# 方案 B：p=1, q=3（路线图中的方案）
for scale in 2.0 2.5 3.0 3.5 4.0; do
  echo "--- Q=3 (p=1,q=3) scale=$scale ---"
  python hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 3 --scale $scale \
    --steps 0 --hopf-every 1 --float64 \
    --out "$OUTBASE/init_Q3_p1q3_s${scale}" 2>&1 | grep -E "Q_fft|initial"
done

echo ""
echo "========== 阶段 0：Q=4 初值校准 =========="

for scale in 2.5 3.0 3.5 4.0 4.5; do
  echo "--- Q=4 (p=4,q=1) scale=$scale ---"
  python hopf_skyrme_torch.py --n 96 --length 12 --p 4 --q 1 --scale $scale \
    --steps 0 --hopf-every 1 --float64 \
    --out "$OUTBASE/init_Q4_p4q1_s${scale}" 2>&1 | grep -E "Q_fft|initial"
done

for scale in 2.5 3.0 3.5 4.0 4.5; do
  echo "--- Q=4 (p=1,q=4) scale=$scale ---"
  python hopf_skyrme_torch.py --n 96 --length 12 --p 1 --q 4 --scale $scale \
    --steps 0 --hopf-every 1 --float64 \
    --out "$OUTBASE/init_Q4_p1q4_s${scale}" 2>&1 | grep -E "Q_fft|initial"
done

echo ""
echo "============================================"
echo "阶段 0 完成。请检查上面输出，选择 Q_fft 最接近 -3/-4 的配置。"
echo "然后手动运行阶段 1（松弛）和阶段 2（远场分析）。"
echo "============================================"
