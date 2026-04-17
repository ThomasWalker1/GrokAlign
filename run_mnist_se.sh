#!/usr/bin/env bash
# Run the full grokking regularization experiment for MNIST:
#   4 conditions × NUM_SEEDS seeds × 2 loss functions squared error.
#
# Environment overrides:
#   DEVICE    — torch device string (default: cuda:0)
#   NUM_SEEDS — number of random seeds per condition (default: 25)
#   PYTHON    — python interpreter (default: python)
#   DATA_DIR  — MNIST download directory (default: ./data)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cuda}"
NUM_SEEDS="${NUM_SEEDS:-10}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"

echo "=== GrokAlign MNIST Experiments ==="
echo "  Device   : ${DEVICE}"
echo "  Seeds    : ${NUM_SEEDS}"
echo "  Output   : ${SCRIPT_DIR}/outputs/mnist_se"
echo "  Data     : ${DATA_DIR}"
echo "  Python   : $(${PYTHON} --version 2>&1)"
echo ""

mkdir -p "${SCRIPT_DIR}/outputs/mnist_se"

run_condition() {
    local name="$1"
    local loss_fn="$2"
    shift 2
    local out_dir="${SCRIPT_DIR}/outputs/mnist_${loss_fn}"
    echo "--- ${name} ---"
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "  seed ${seed}"
        "${PYTHON}" "${SCRIPT_DIR}/mnist.py" \
            --loss_fn "${loss_fn}" \
            --device "${DEVICE}" \
            --seed "${seed}" \
            --dir "${DATA_DIR}" \
            --output_dir "${out_dir}" \
            "$@"
    done
    echo ""
}

echo "=== Square-Error Loss ==="
run_condition "Baseline (SE)"          se
run_condition "GrokFast (SE)"          se --grokfast
run_condition "GrokAlign λ=0.01 (SE)"  se --lambda_reg 0.01
run_condition "OrthoGrad (SE)"         se --orthogonal_gradients

echo "=== MNIST-SE experiments complete ==="
echo ""
echo "Summarize results with:"
echo "  ${PYTHON} ${SCRIPT_DIR}/summarize_results.py --dataset mnist_se"
