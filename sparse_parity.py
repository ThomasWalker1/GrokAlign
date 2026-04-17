import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regularization'))

from grokfast import gradfilter_ema
from grokalign import GrokAlign
from utils import (
    evaluate,
    get_dataset,
    get_model,
    get_optimizer,
    softmax_cross_entropy,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'sparse_parity')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Study grokking under different regularization strategies.")

    # Model
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[200, 200])
    parser.add_argument('--activation_function', type=str, default='ReLU')

    # Training
    parser.add_argument('--num_epochs', type=int, default=20_000)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--adam_epsilon', type=float, default=1e-25)

    # Regularization
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--orthogonal_gradients', action='store_true', default=False)
    parser.add_argument('--grokfast', action='store_true', default=False)

    # Dataset
    parser.add_argument('--dataset', type=str, default='sparse_parity')
    parser.add_argument('--train_fraction', type=float, default=0.5)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--num_parity_features', type=int, default=3)
    parser.add_argument('--num_noise_features', type=int, default=40)

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_dtype', type=str, default='float32')
    parser.add_argument('--cross_entropy_dtype', type=str, default='float32')
    parser.add_argument('--output_dir', type=str, default=OUTPUTS_DIR)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Build a human-readable condition tag and run name
    reg_parts = []
    if args.orthogonal_gradients:
        reg_parts.append('OG')
    if args.grokfast:
        reg_parts.append('GF')
    if args.lambda_reg > 0:
        reg_parts.append(f'GA{args.lambda_reg}')
    condition = '-'.join(reg_parts) if reg_parts else 'baseline'
    run_name = f"{args.dataset}-{condition}-wd{args.weight_decay}-seed{args.seed}"

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    train_dtype = getattr(torch, args.train_dtype)
    ce_dtype = getattr(torch, args.cross_entropy_dtype)

    # Data — full-batch training, so batch_size == dataset size
    train_dataset, test_dataset = get_dataset(args)
    args.batch_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Model & optimizer
    args.lr /= (args.alpha ** 2)
    model = get_model(args).to(device).to(train_dtype)
    optimizer = get_optimizer(model, args)

    # Load full training set onto device once for efficient full-batch updates
    grads = None
    all_data = train_dataset.dataset.data[train_dataset.indices].to(device, dtype=train_dtype)
    all_targets = train_dataset.dataset.targets[train_dataset.indices].to(device, dtype=torch.long)

    grokked = False
    grokking_epoch = None
    metrics = []
    opt_time = 0

    pbar = tqdm(range(args.num_epochs + 1), desc=f"[{run_name}]")
    for epoch in pbar:
        log_this_epoch = (
            (epoch < 100) or
            (epoch < 1_000 and epoch % 10 == 0) or
            (epoch < 10_000 and epoch % 100 == 0) or
            (epoch % 1_000 == 0)
        )

        if log_this_epoch:
            train_loss, train_acc = evaluate(model, train_loader, dtype=train_dtype)
            test_loss, test_acc = evaluate(model, test_loader, dtype=train_dtype)

            if not grokked and train_acc > 99 and test_acc > 90:
                grokked = True
                grokking_epoch = epoch

            metrics.append({
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'opt_time': opt_time,
                'grokked': grokked,
            })
            pbar.set_description(f"[{run_name}] train={train_acc:.1f}% test={test_acc:.1f}%")

        if grokked:
            break

        # Full-batch gradient step with per-epoch shuffle
        model.train()
        t0 = time.time()
        perm = torch.randperm(all_data.size(0))
        shuffled_data = all_data[perm]
        shuffled_targets = all_targets[perm]

        optimizer.zero_grad()

        if args.lambda_reg > 0:
            grok_align = GrokAlign(
                model,
                criterion=lambda x: softmax_cross_entropy(x * args.alpha, shuffled_targets, dtype=ce_dtype),
                lambda_reg=args.lambda_reg,
                device=device,
            )
            loss = grok_align(shuffled_data)
        else:
            output = model(shuffled_data) * args.alpha
            loss = softmax_cross_entropy(output, shuffled_targets, dtype=ce_dtype)

        loss.backward()

        if args.grokfast:
            grads = gradfilter_ema(model, grads=grads, alpha=0.8, lamb=0.1)

        optimizer.step()
        opt_time += time.time() - t0

    # Save results to outputs directory
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'run_name': run_name,
        'condition': condition,
        'config': vars(args),
        'grokked': grokked,
        'grokking_epoch': grokking_epoch,
        'metrics': metrics,
    }
    out_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Grokked: {grokked} | Epoch: {grokking_epoch}")


if __name__ == '__main__':
    main()
