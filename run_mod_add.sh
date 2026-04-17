#!/usr/bin/env bash
# Run the full grokking regularization experiment for modular addition:
#   3 conditions × NUM_SEEDS seeds.
#
# Environment overrides:
#   DEVICE    — torch device string (default: cuda:0)
#   NUM_SEEDS — number of random seeds per condition (default: 25)
#   PYTHON    — python interpreter (default: python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs/mod_add"
DEVICE="${DEVICE:-cuda}"
NUM_SEEDS="${NUM_SEEDS:-10}"
PYTHON="${PYTHON:-python}"

echo "=== GrokAlign Modular Addition Experiments ==="
echo "  Device   : ${DEVICE}"
echo "  Seeds    : ${NUM_SEEDS}"
echo "  Output   : ${OUTPUTS_DIR}"
echo "  Python   : $(${PYTHON} --version 2>&1)"
echo ""

mkdir -p "${OUTPUTS_DIR}"

run_condition() {
    local name="$1"; shift
    echo "--- ${name} ---"
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "  seed ${seed}"
        "${PYTHON}" "${SCRIPT_DIR}/mod_add.py" \
            --device "${DEVICE}" \
            --seed "${seed}" \
            --output_dir "${OUTPUTS_DIR}" \
            "$@"
    done
    echo ""
}

run_condition "Baseline"
run_condition "GrokFast"           --grokfast
run_condition "GrokAlign (λ=0.01)" --lambda_reg 0.01
run_condition "OrthoGrad"         --orthogonal_gradients

echo "=== Modular addition experiments complete ==="
echo ""
echo "Summarize results with:"
echo "  ${PYTHON} ${SCRIPT_DIR}/summarize_results.py --dataset mod_add"
