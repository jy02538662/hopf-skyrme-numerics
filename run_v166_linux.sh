#!/usr/bin/env bash
# ============================================================================
# v1.6.6 GPU Linux 执行手册（ASCII-safe）
# ----------------------------------------------------------------------------
# 用途：从 GPU 服务器上一键完成
#       1. kappa_rho 标定（CPU，5 秒）
#       2. Hopf 纤维 preimage 探针（CPU，5-15 分钟）
#
# 准备：
#   1. 把这份脚本 scp 到 GPU 服务器的任意目录
#   2. cd 到脚本所在目录（或任意目录均可——脚本自定位）
#   3. bash run_v166_linux.sh
#
# 不会破坏任何已有文件。所有输出到 outputs/v166_results/
# ============================================================================

set -e  # 任一步失败立即终止

# ----------------------------------------------------------------------------
# -1. 自适应 CRLF→LF（防止 Windows 上写的脚本到 Linux 报错）
# ----------------------------------------------------------------------------
SELF_PATH="${BASH_SOURCE[0]:-$0}"
if [ -f "$SELF_PATH" ]; then
    if file "$SELF_PATH" 2>/dev/null | grep -q "CRLF\|with CR"; then
        echo "[init] 检测到 CRLF，自动转 LF..."
        sed -i 's/\r$//' "$SELF_PATH"
        # 也对 src/ 下的 Python 文件做同样转码（它们同样可能是 CRLF）
        if [ -d "src" ]; then
            for f in src/*.py; do
                if [ -f "$f" ] && file "$f" 2>/dev/null | grep -q "CRLF\|with CR"; then
                    sed -i 's/\r$//' "$f"
                fi
            done
        fi
        echo "[init] CRLF 转 LF 完成，重新执行..."
        exec bash "$SELF_PATH" "$@"
    fi
fi

echo "============================================================"
echo "  v1.6.6 GPU Linux 执行手册"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  主机: $(hostname)"
echo "  Python: $(which python3)"
echo "============================================================"

# ----------------------------------------------------------------------------
# 0. 进入项目根目录（自适应：从脚本所在目录开始）
# ----------------------------------------------------------------------------
# 自动定位脚本所在目录，并将其设为工作目录
# 这样无论您把脚本放在 /、/root、/tmp 还是任何位置都能跑
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
cd "$SCRIPT_DIR"
echo "[0/5] 工作目录：$SCRIPT_DIR"
echo "[0/5] 父目录树预览："
ls -1 "$SCRIPT_DIR" 2>/dev/null | head -10 | sed 's/^/    /'

# 如果父目录里没 src/ 但有上一级目录含 src/，自动向上找一次
if [ ! -d "$SCRIPT_DIR/src" ]; then
    PARENT_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"
    if [ -d "$PARENT_DIR/src" ]; then
        echo "[0/5] 自动跳转到上一层：$PARENT_DIR"
        cd "$PARENT_DIR"
        SCRIPT_DIR="$PARENT_DIR"
    fi
fi

if [ ! -d "$SCRIPT_DIR/src" ]; then
    echo "[ERROR] 找不到 src/ 目录。请把脚本放到含 src/ 的项目根目录下。"
    echo "  当前目录：$SCRIPT_DIR"
    echo "  父目录内容："
    ls -la "$SCRIPT_DIR/.." 2>/dev/null | head -10
    exit 1
fi

# ----------------------------------------------------------------------------
# 1. 安装依赖
# ----------------------------------------------------------------------------
echo
echo "[1/5] 安装依赖..."
pip install --quiet --upgrade numpy scipy scikit-image >/dev/null 2>&1 \
    || pip3 install --quiet --upgrade numpy scipy scikit-image >/dev/null 2>&1 \
    || {
        echo "  [WARN] 静默安装失败，尝试 verbose 模式"
        python3 -m pip install --user numpy scipy scikit-image
    }

python3 -c "import numpy, scipy, skimage; print('  numpy =', numpy.__version__)"
python3 -c "import scipy; print('  scipy =', scipy.__version__)"
python3 -c "import skimage; print('  scikit-image =', skimage.__version__)"

# pyvista 可选（仅当需要 .vtk 输出）
echo "[1/5] (可选) 安装 pyvista（用于 .vtk 输出）..."
python3 -c "import pyvista" 2>/dev/null \
    && echo "  pyvista 已安装" \
    || {
        python3 -m pip install --user --quiet pyvista 2>/dev/null \
            && echo "  pyvista 已安装" \
            || echo "  pyvista 未安装（可选，可不装）"
    }

# ----------------------------------------------------------------------------
# 2. 找到 Q=1 长跑结果
# ----------------------------------------------------------------------------
echo
echo "[2/5] 查找 Q=1 主场文件..."
# 限定到当前目录树（避免 / 根目录下的杂项文件干扰）
Q1_FILES=$(find . -type f -name "nfield_q1_torch.npy" -not -path "./.git/*" 2>/dev/null)
echo "$Q1_FILES" | sed 's/^/  /'

if [ -z "$Q1_FILES" ]; then
    echo
    echo "[ERROR] 没找到任何 nfield_q1_torch.npy"
    echo "  请确认两件事："
    echo "  1. outputs/ 目录是否存在并包含 Q=1 结果？"
    echo "  2. 文件名是否不同？请用 find 查找 .npy："
    echo "       find . -name '*.npy' -not -path './.git/*' | head -20"
    exit 1
fi

# 取第一个（按文件名字典序）
Q1_FILE=$(echo "$Q1_FILES" | head -n 1)
echo
echo "[2/5] 将处理：$Q1_FILE"

# 推断盒子大小（从路径名启发式解析；若失败则用默认）
case "$Q1_FILE" in
    *N80_L10*) Q1_L=10.0 ;;
    *N96_L12*) Q1_L=12.0 ;;
    *N96_L10*) Q1_L=10.0 ;;
    *)         Q1_L=12.0 ;;
esac
echo "[2/5] 推断盒子 L = $Q1_L"

# 推断 Q=1 能量（按版号）；用户可在下一节覆盖
case "$Q1_FILE" in
    *q1_torch_N80_L10_s18_qpen0_long*) Q1_E=434.415 ;;
    *) Q1_E=400.13 ;;  # v1.6.5 §7.3 的 96^3 默认值
esac
echo "[2/5] 推断能量 E = $Q1_E（可在下面命令覆盖）"

# ----------------------------------------------------------------------------
# 3. 找到 Q=2 主场文件
# ----------------------------------------------------------------------------
echo
echo "[3/5] 查找 Q=2 主场文件..."
Q2_FILES=$(find . -type f -name "nfield_q1_torch.npy" -path "*Q2*" -not -path "./.git/*" 2>/dev/null)
# 注意：Q=2 的文件命名也常是 nfield_q1_torch.npy（每个 Q 单独跑）
# 这里靠路径里含 Q2 区分
if [ -z "$Q2_FILES" ]; then
    # 退化：找包含 p1q2 的目录
    Q2_FILES=$(find . -type d -name "*p1q2*" -not -path "./.git/*" 2>/dev/null | head -1)
    if [ -n "$Q2_FILES" ]; then
        Q2_FILE="$Q2_FILES/nfield_q1_torch.npy"
    else
        Q2_FILE=""
    fi
else
    Q2_FILE=$(echo "$Q2_FILES" | head -n 1)
fi

echo "$Q2_FILE" | sed 's/^/  /'

if [ -z "$Q2_FILE" ] || [ ! -f "$Q2_FILE" ]; then
    echo
    echo "[WARN] 没找到 Q=2 主场文件，跳过 Hopf 纤维 preimage 任务 2"
    echo "  完成任务 1 后退出；任务 2 用户自行处理"
    SKIP_TASK2=1
else
    echo "[3/5] 将处理：$Q2_FILE"
    # 推断 L
    case "$Q2_FILE" in
        *N96_L12*) Q2_L=12.0 ;;
        *)         Q2_L=12.0 ;;
    esac
    echo "[3/5] 推断盒子 L = $Q2_L"
    SKIP_TASK2=0
fi

# ----------------------------------------------------------------------------
# 4. 任务 1：kappa_rho 标定
# ----------------------------------------------------------------------------
echo
echo "[4/5] 任务 1：kappa_rho 标定（CPU 5 秒）"
echo "------------------------------------------------------------"

# 输出到当前位置的 outputs/v166_results/（绝对路径会显示）
OUT_BASE="outputs/v166_results"
OUT_BASE_ABS="$(cd "$SCRIPT_DIR" && pwd)/$OUT_BASE"
mkdir -p "$OUT_BASE/kappa_rho_calibration"
mkdir -p "$OUT_BASE/hopf_fiber_preimage_Q2"
echo "[3.5/5] 输出绝对路径：$OUT_BASE_ABS"

# 跑标定（用项目目录下的脚本；不复制进 GPU 服务器的话请手动调整路径）
python3 src/kappa_rho_calibration.py \
    --field "$Q1_FILE" \
    --length "$Q1_L" \
    --energy "$Q1_E" \
    --out "$OUT_BASE/kappa_rho_calibration"

echo
echo "[4/5] 任务 1 完成。结果："
echo "  $OUT_BASE/kappa_rho_calibration/kappa_rho.json"

if [ "$SKIP_TASK2" = "1" ]; then
    echo
    echo "[5/5] 跳过任务 2。"
    echo "============================================================"
    echo "  全部完成"
    echo "============================================================"
    exit 0
fi

# ----------------------------------------------------------------------------
# 5. 任务 2：Hopf 纤维 preimage 探针
# ----------------------------------------------------------------------------
echo
echo "[5/5] 任务 2：Hopf 纤维 preimage 探针（CPU 5-15 分钟）"
echo "------------------------------------------------------------"

python3 src/hopf_fiber_preimage_probe.py \
    --field "$Q2_FILE" \
    --length "$Q2_L" \
    --n-phases 8 \
    --tolerance-frac 0.05 \
    --out "$OUT_BASE/hopf_fiber_preimage_Q2"

echo
echo "[5/5] 任务 2 完成。结果："
echo "  $OUT_BASE/hopf_fiber_preimage_Q2/"
echo "    preimage_*.vtk          (ParaView 可视化)"
echo "    summary.json             (对称判定结果)"

# ----------------------------------------------------------------------------
# 6. 显示结论
# ----------------------------------------------------------------------------
echo
echo "============================================================"
echo "  全部完成！"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo
echo "下一步建议："
echo "  1. cat outputs/v166_results/kappa_rho_calibration/kappa_rho.json"
echo "  2. cat outputs/v166_results/hopf_fiber_preimage_Q2/summary.json"
echo "  3. 安装 ParaView（桌面应用，免费），打开 .vtk 文件看双环几何"
echo
