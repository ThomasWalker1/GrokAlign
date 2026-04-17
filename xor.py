import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regularization'))

from grokfast import gradfilter_ema
from grokalign import GrokAlign
from utils import (
    evaluate_xor,
    get_dataset,
    get_model,
    get_optimizer,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'xor')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Study grokking under different regularization strategies for the XOR task."
    )

    # Model architecture
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[2048])
    parser.add_argument('--activation_function', type=str, default='ReLU')

    # Training
    parser.add_argument('--num_epochs', type=int, default=1_000)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--adam_epsilon', type=float, default=1e-25)

    # Regularization
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lambda_reg', type=float, default=0.0,
                        help="GrokAlign Jacobian regularization coefficient.")
    parser.add_argument('--orthogonal_gradients', action='store_true', default=False)
    parser.add_argument('--grokfast', action='store_true', default=False)

    # Dataset
    parser.add_argument('--dataset', type=str, default='xor')
    parser.add_argument('--p', type=int, default=40_000,
                        help="Total number of input features (2 signal + p-2 noise).")
    parser.add_argument('--n', type=int, default=400, help="Number of training samples.")
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help="Noise feature magnitude.")

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_dtype', type=str, default='float32')
    parser.add_argument('--output_dir', type=str, default=OUTPUTS_DIR)

    return parser.parse_args()


def main():
    args = parse_arguments()

    reg_parts = []
    if args.orthogonal_gradients:
        reg_parts.append('OG')
    if args.grokfast:
        reg_parts.append('GF')
    if args.lambda_reg > 0:
        reg_parts.append(f'GA{args.lambda_reg}')
    condition = '-'.join(reg_parts) if reg_parts else 'baseline'
    run_name = f"xor-{condition}-wd{args.weight_decay}-seed{args.seed}"

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    train_dtype = getattr(torch, args.train_dtype)

    train_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset),
        shuffle=True, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=len(test_dataset),
        shuffle=False, num_workers=4, pin_memory=True,
    )

    model = get_model(args).to(device).to(train_dtype)
    optimizer = get_optimizer(model, args)
    grads = None

    grokked = False
    grokking_epoch = None
    metrics = []
    opt_time = 0.0

    pbar = tqdm(range(args.num_epochs + 1), desc=f"[{run_name}]")
    for epoch in pbar:
        log_this_epoch = (
            (epoch < 100) or
            (epoch < 1_000 and epoch % 10 == 0) or
            (epoch < 10_000 and epoch % 100 == 0) or
            (epoch % 1_000 == 0)
        )

        if log_this_epoch:
            train_acc = evaluate_xor(model, train_loader)
            test_acc  = evaluate_xor(model, test_loader)
            adv_acc   = evaluate_xor(model, test_loader, amp=0.2)

            if not grokked and train_acc > 99 and test_acc > 95 and adv_acc > 95:
                grokked = True
                grokking_epoch = epoch

            metrics.append({
                'epoch': epoch,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'adv_acc': adv_acc,
                'opt_time': opt_time,
                'grokked': grokked,
            })
            pbar.set_description(
                f"[{run_name}] train={train_acc:.1f}% test={test_acc:.1f}% adv={adv_acc:.1f}%"
            )

        if grokked:
            break

        model.train()
        t0 = time.time()
        for X, y in train_loader:
            X, y = X.to(device).to(train_dtype), y.to(device).to(train_dtype)
            optimizer.zero_grad()

            if args.lambda_reg > 0:
                ga = GrokAlign(
                    model,
                    criterion=lambda x, _y=y: F.mse_loss(x.squeeze(), _y),
                    lambda_reg=args.lambda_reg,
                    device=device,
                )
                loss = ga(X)
            else:
                output = model(X).squeeze()
                loss = F.mse_loss(output, y)

            loss.backward()

            if args.grokfast:
                grads = gradfilter_ema(model, grads=grads, alpha=0.8, lamb=0.1)

            optimizer.step()

        opt_time += time.time() - t0

    # Save results
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
