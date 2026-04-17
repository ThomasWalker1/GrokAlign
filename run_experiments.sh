#!/usr/bin/env bash
# Run the complete grokking regularization experiment suite:
#   sparse parity (4 conditions) + modular addition (3 conditions) + XOR (4 conditions)
#   + MNIST CE (4 conditions) + MNIST SE (4 conditions), each over NUM_SEEDS seeds.
#
# To run a single setting instead:
#   bash run_sparse_parity.sh
#   bash run_mod_add.sh
#   bash run_xor.sh
#   bash run_mnist.sh
#
# Environment overrides:
#   DEVICE    — torch device string (default: cuda:0)
#   NUM_SEEDS — number of random seeds per condition (default: 25)
#   PYTHON    — python interpreter (default: python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEVICE="${DEVICE:-cuda}"
export NUM_SEEDS="${NUM_SEEDS:-10}"
export PYTHON="${PYTHON:-python}"

echo "=== GrokAlign Full Experiment Suite ==="
echo "  Device   : ${DEVICE}"
echo "  Seeds    : ${NUM_SEEDS}"
echo "  Python   : $(${PYTHON} --version 2>&1)"
echo ""

bash "${SCRIPT_DIR}/run_sparse_parity.sh"
bash "${SCRIPT_DIR}/run_mod_add.sh"
bash "${SCRIPT_DIR}/run_xor.sh"
bash "${SCRIPT_DIR}/run_mnist_ce.sh"
bash "${SCRIPT_DIR}/run_mnist_se.sh"

echo "=== All experiments complete ==="
echo ""
echo "Summarize results with:"
echo "  ${PYTHON} ${SCRIPT_DIR}/summarize_results.py --dataset sparse_parity"
echo "  ${PYTHON} ${SCRIPT_DIR}/summarize_results.py --dataset mod_add"
echo "  ${PYTHON} ${SCRIPT_DIR}/summarize_results.py --dataset xor"
echo "  ${PYTHON} ${SCRIPT_DIR}/summarize_results.py --dataset mnist_ce"
echo "  ${PYTHON} ${SCRIPT_DIR}/summarize_results.py --dataset mnist_se"
