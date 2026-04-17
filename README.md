# GrokAlign

Demonstrating the effectiveness of the GrokAlign regularization strategy proposed in "Normal Alignment: The Geometric Structure of Models Learning Sparse Data" for accelerating grokking.

## Tasks

### Sparse Parity

A two-hidden-layer MLP (200 × 200, ReLU) trained on the sparse parity task:
- Input: `num_parity_features + num_noise_features` binary features (default: 3 + 40)
- Target: XOR parity of the first `num_parity_features` features only
- 2000 samples, 50/50 train/test split, full-batch AdamW + weight decay
- Grokking criterion: train accuracy > 99% **and** test accuracy > 90%

### Modular Addition

A single-hidden-layer FCN (256 hidden units, quadratic activation) trained on `x + y (mod p)`:
- Input: one-hot encoding of `(x, y)` for `x, y ∈ {0, …, p−1}` (default: `p=61`)
- 30% training split, mini-batch AdamW + weight decay
- Grokking criterion: train accuracy > 99% **and** test accuracy > 99%

### XOR

A single-hidden-layer MLP (2048 hidden units, ReLU, no bias) trained on a high-dimensional XOR task:
- Input: `p=40,000` features — 2 signal features `x, y ∈ {±1}`, rest are noise `∈ {±ε}` (default: `ε=0.05`)
- Target: `sign(x · y)` (XOR of the sign of the two signal features)
- `n=400` training samples, full-batch SGD + weight decay
- Grokking criterion: train accuracy > 99% **and** test accuracy > 95% **and** adversarial accuracy > 95% (adversarial: noise amplitude 0.2 added to test features at eval)

### MNIST

A three-hidden-layer MLP (200 × 200 × 200, ReLU) trained on a small MNIST subset with two loss variants:
- Input: flattened, normalised 28×28 grayscale images (784 features)
- `num_samples=1024` examples drawn from the front of the train and test splits
- Mini-batch AdamW + weight decay; model weights are scaled by `α=8` at initialisation
- **CE** (`--loss_fn ce`): standard cross-entropy loss
- **SE** (`--loss_fn se`): mean squared error against one-hot targets
- Grokking criterion: train accuracy > 99% **and** test accuracy > 80%

Results for each loss variant are saved to separate output directories (`outputs/mnist_ce/`, `outputs/mnist_se/`).

## Regularization Conditions

| Condition | Sparse Parity | Modular Addition | XOR | MNIST |
|-----------|:---:|:---:|:---:|:---:|
| Baseline (AdamW/SGD + weight decay) | ✓ | ✓ | ✓ | ✓ |
| GrokFast — EMA gradient filter ([Lim et al., 2024](https://arxiv.org/abs/2405.20233)) | ✓ | ✓ | ✓ | ✓ |
| GrokAlign — joint penalty `λ(‖J_x‖_F + ‖b_x‖²₂)` | λ=0.1 | λ=0.01 | λ=1.0 | λ=0.01 |
| OrthoGrad ([Prieto et al., 2025](https://arxiv.org/abs/2501.04697)) — Gram-Schmidt gradient projection | ✓ | — | ✓ | ✓ |

## Setup

```bash
pip install torch numpy tqdm
```

## Running Experiments

```bash
# All settings: sparse parity + modular addition + XOR + MNIST CE + MNIST SE
bash run_experiments.sh

# Individual settings
bash run_sparse_parity.sh
bash run_mod_add.sh
bash run_xor.sh
bash run_mnist_ce.sh
bash run_mnist_se.sh

# Override device and seed count
DEVICE=cuda NUM_SEEDS=10 bash run_experiments.sh

# Single run (sparse parity)
python sparse_parity.py --seed 0 --grokfast
python sparse_parity.py --seed 0 --lambda_reg 0.1
python sparse_parity.py --seed 0 --orthogonal_gradients

# Single run (modular addition)
python mod_add.py --seed 0 --grokfast
python mod_add.py --seed 0 --lambda_reg 0.01

# Single run (XOR)
python xor.py --seed 0 --grokfast
python xor.py --seed 0 --lambda_reg 1.0
python xor.py --seed 0 --orthogonal_gradients

# Single run (MNIST)
python mnist.py --seed 0 --loss_fn ce --grokfast
python mnist.py --seed 0 --loss_fn se --lambda_reg 0.01
```

## Summarizing Results

```bash
# Per setting
python summarize_results.py --dataset sparse_parity
python summarize_results.py --dataset mod_add
python summarize_results.py --dataset xor
python summarize_results.py --dataset mnist_ce
python summarize_results.py --dataset mnist_se

# Custom directory (dataset auto-detected from run names)
python summarize_results.py outputs/xor/
```

## File Structure

```
GrokAlign/
├── sparse_parity.py        # Training script for sparse parity (CLI, saves JSON per run)
├── mod_add.py              # Training script for modular addition (CLI, saves JSON per run)
├── xor.py                  # Training script for XOR task (CLI, saves JSON per run)
├── utils.py                # Dataset, model, and optimizer utilities
├── summarize_results.py    # Load outputs/ and print comparison table
├── run_experiments.sh      # Run all conditions × N seeds for all settings
├── run_sparse_parity.sh    # Run sparse parity experiments only
├── run_mod_add.sh          # Run modular addition experiments only
├── run_xor.sh              # Run XOR experiments only
├── run_mnist_ce.sh            # Run MNIST-CE experiments only
├── run_mnist_se.sh            # Run MNIST-SE experiments only
├── outputs/
│   ├── sparse_parity/      # Per-run JSON results for sparse parity
│   ├── mod_add/            # Per-run JSON results for modular addition
│   ├── xor/                # Per-run JSON results for XOR
│   ├── mnist_ce/           # Per-run JSON results for MNIST (cross-entropy)
│   └── mnist_se/           # Per-run JSON results for MNIST (squared error)
└── regularization/
    ├── grokalign.py        # Jacobian norm regularization (GrokAlign)
    ├── grokfast.py         # EMA gradient filtering (GrokFast)
    └── orthograd.py        # Orthogonal gradient projection (OrthoGrad)
```

## Output Format

Each run writes a JSON file to the appropriate `outputs/<setting>/` subdirectory:

```json
{
  "run_name": "sparse_parity-GF-wd0.1-seed3",
  "condition": "GF",
  "config": { "seed": 3, "grokfast": true, "weight_decay": 0.1, ... },
  "grokked": true,
  "grokking_epoch": 11400,
  "metrics": [
    { "epoch": 0, "train_acc": 50.0, "train_loss": 0.693, "test_acc": 49.8, "test_loss": 0.694, ... },
    ...
  ]
}
```

## Key Arguments

### Shared flags (both scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `0` | Random seed |
| `--device` | `cuda:0` | Torch device |
| `--num_epochs` | varies | Max training epochs |
| `--weight_decay` | varies | L2 regularization coefficient |
| `--lambda_reg` | `0.0` | GrokAlign joint penalty coefficient |
| `--grokfast` | off | Enable GrokFast EMA filter |
| `--orthogonal_gradients` | off | Enable OrthoGrad |
| `--output_dir` | setting-specific | Where to save result JSON files |

### Sparse parity (`sparse_parity.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--num_parity_features` | `3` | Number of signal features |
| `--num_noise_features` | `40` | Number of noise features |
| `--num_samples` | `2000` | Total dataset size |
| `--train_fraction` | `0.5` | Fraction used for training |

### Modular addition (`mod_add.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--prime` / `-p` | `61` | Modulus `p` |
| `--operation` / `-op` | `x+y` | Arithmetic operation |
| `--training_fraction` | `0.3` | Fraction used for training |
| `--hidden_width` | `256` | Hidden layer width |
| `--act_fn` | `quadratic` | Activation function |
| `--batch_size` | `32` | Mini-batch size |

### XOR (`xor.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--p` | `40000` | Total number of input features |
| `--n` | `400` | Number of training samples |
| `--epsilon` | `0.05` | Noise feature magnitude |
| `--hidden_sizes` | `[2048]` | Hidden layer widths |
| `--train_dtype` | `float32` | Training precision |

### MNIST (`mnist.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--loss_fn` | `ce` | Loss function: `ce` (cross-entropy) or `se` (squared error) |
| `--num_samples` | `1024` | Samples drawn from each split |
| `--batch_size` | `196` | Mini-batch size |
| `--hidden_sizes` | `[200,200,200]` | Hidden layer widths |
| `--alpha` | `8.0` | Weight scale applied at initialisation |
| `--train_dtype` | `float64` | Training precision |
| `--dir` | `./data` | MNIST download cache directory |
