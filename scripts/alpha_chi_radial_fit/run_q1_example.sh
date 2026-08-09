#!/usr/bin/env bash
# 改 FIELD 后执行。默认拟合窗已收进边界内侧，并做诊断 + 多窗口扫描。
set -euo pipefail

FIELD="${FIELD:-/path/to/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy}"
LENGTH="${LENGTH:-10}"
# 避开盒子边缘（旧窗 6.5–9.5 易把 χ 的 β 拉低）
RMIN="${RMIN:-5.5}"
RMAX="${RMAX:-8.5}"
OUTDIR="${OUTDIR:-.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/alpha_chi_radial_fit.py" \
  --field "${FIELD}" \
  --length "${LENGTH}" \
  --r-min-fit "${RMIN}" \
  --r-max-fit "${RMAX}" \
  --normalize \
  --scan-windows \
  --out "${OUTDIR}/alpha_radial_fit_Q1.png" \
  --save-csv "${OUTDIR}/alpha_chi_radial_Q1.csv"
