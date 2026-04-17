import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regularization'))

from grokfast import gradfilter_ema
from grokalign import GrokAlign
from utils import evaluate, get_dataset, get_model, get_optimizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Study grokking under different regularization strategies for MNIST."
    )

    # Model architecture
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[200, 200, 200])
    parser.add_argument('--activation_function', type=str, default='ReLU')

    # Training
    parser.add_argument('--num_epochs', type=int, default=20_000)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'se'],
                        help="Loss function: 'ce' (cross-entropy) or 'se' (squared error).")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=8.0,
                        help="Scale applied to model weights at initialisation.")
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--adam_epsilon', type=float, default=1e-25)

    # Regularization
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=0.0,
                        help="GrokAlign Jacobian regularization coefficient.")
    parser.add_argument('--orthogonal_gradients', action='store_true', default=False)
    parser.add_argument('--grokfast', action='store_true', default=False)

    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help="Number of samples drawn from the front of the train/test split.")
    parser.add_argument('--batch_size', type=int, default=196)
    parser.add_argument('--dir', type=str, default='./data',
                        help="Directory for MNIST download cache.")

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_dtype', type=str, default='float64')
    parser.add_argument('--cross_entropy_dtype', type=str, default='float64')
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Where to save results. Defaults to outputs/mnist_{loss_fn}/.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Resolve output directory: separate subdirectory per loss function
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'outputs', f'mnist_{args.loss_fn}'
        )

    # Build condition tag and run name (mirrors sparse_parity.py convention)
    reg_parts = []
    if args.orthogonal_gradients:
        reg_parts.append('OG')
    if args.grokfast:
        reg_parts.append('GF')
    if args.lambda_reg > 0:
        reg_parts.append(f'GA{args.lambda_reg}')
    condition = '-'.join(reg_parts) if reg_parts else 'baseline'
    # Embed loss_fn in the dataset tag so run names are unambiguous
    dataset_tag = f'mnist_{args.loss_fn}'
    run_name = f"{dataset_tag}-{condition}-wd{args.weight_decay}-seed{args.seed}"

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    train_dtype = getattr(torch, args.train_dtype)

    train_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    model = get_model(args).to(device).to(train_dtype)
    # Scale weights by alpha at init (equivalent to output scaling but baked in)
    with torch.no_grad():
        for p in model.parameters():
            p.data = args.alpha * p.data

    optimizer = get_optimizer(model, args)

    # One-hot targets pre-built for squared-error loss
    one_hots = torch.eye(10, dtype=train_dtype, device=device)

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
            train_loss, train_acc = evaluate(model, train_loader, dtype=train_dtype)
            test_loss, test_acc  = evaluate(model, test_loader,  dtype=train_dtype)

            if not grokked and train_acc > 99 and test_acc > 80:
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

        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device).to(train_dtype)
            y = y.to(device)

            optimizer.zero_grad()

            if args.lambda_reg > 0:
                # GrokAlign.forward only accepts inputs; labels are captured via
                # default-argument so each batch's y is bound at closure creation.
                if args.loss_fn == 'ce':
                    ga = GrokAlign(
                        model,
                        criterion=lambda logits, _y=y: nn.CrossEntropyLoss()(logits, _y),
                        lambda_reg=args.lambda_reg,
                        device=device,
                    )
                else:
                    ga = GrokAlign(
                        model,
                        criterion=lambda logits, _y=y: F.mse_loss(logits, one_hots[_y]),
                        lambda_reg=args.lambda_reg,
                        device=device,
                    )
                loss = ga(x)
            else:
                output = model(x)
                if args.loss_fn == 'ce':
                    loss = nn.CrossEntropyLoss()(output, y)
                else:
                    loss = F.mse_loss(output, one_hots[y])

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
