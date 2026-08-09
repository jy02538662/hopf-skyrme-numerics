#!/usr/bin/env bash
# Run on GPU machine where /outputs/... nfields exist.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$DIR/results_gpu}"
mkdir -p "$OUT"

echo "========== STEP 1 =========="
python "$DIR/step1_alpha_fit.py" --abs-outputs --out-dir "$OUT"

echo "========== STEP 2 =========="
python "$DIR/step2_c2_parity_ratio.py" --abs-outputs --r-shell 8 --out-dir "$OUT"

echo "========== STEP 3 =========="
python "$DIR/step3_signed_vs_chi.py" --abs-outputs --Q 2 --r-shell 8 --out-dir "$OUT"

echo "Done. Results in $OUT"
