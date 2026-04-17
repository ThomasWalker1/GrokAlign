## Inspired by https://github.com/nmallinar/rfm-grokking

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regularization'))

from grokfast import gradfilter_ema
from grokalign import GrokAlign
from utils import (
    OneLayerFCN,
    evaluate_mod_add,
    get_optimizer,
    make_data_splits,
    make_dataloader,
    operation_mod_p_data,
)

torch.set_default_dtype(torch.float64)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'mod_add')


class _ActWrapper(torch.nn.Module):
    """Wraps OneLayerFCN to fix the activation function for callers that invoke
    the model without an ``act`` keyword argument (e.g. GrokAlign internals)."""

    def __init__(self, model: OneLayerFCN, act: str):
        super().__init__()
        self._inner = model
        self._act = act

    def forward(self, x, **kwargs):
        return self._inner(x, act=self._act)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Study grokking under different regularization strategies for modular addition."
    )

    # Model architecture
    parser.add_argument('--hidden_width', type=int, default=256)
    parser.add_argument('--act_fn', type=str, default='quadratic',
                        help="Activation function: relu, swish, quadratic, softplus, linear.")
    parser.add_argument('--init_scale', type=float, default=1.0)

    # Training
    parser.add_argument('--num_epochs', type=int, default=1_000)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--adam_epsilon', type=float, default=1e-25)

    # Regularization
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0,
                        help="GrokAlign Jacobian regularization coefficient.")
    parser.add_argument('--orthogonal_gradients', action='store_true', default=False)
    parser.add_argument('--grokfast', action='store_true', default=False)

    # Dataset
    parser.add_argument('--prime', '-p', type=int, default=61)
    parser.add_argument('--operation', '-op', type=str, default='x+y')
    parser.add_argument('--training_fraction', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)

    # System
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
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
    run_name = f"mod_add-{condition}-wd{args.weight_decay}-seed{args.seed}"

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Data
    all_inputs, all_labels = operation_mod_p_data(args.operation, args.prime)
    X_tr, y_tr, X_te, y_te = make_data_splits(all_inputs, all_labels, args.training_fraction)

    X_tr = F.one_hot(X_tr, args.prime).view(-1, 2 * args.prime).double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    X_te = F.one_hot(X_te, args.prime).view(-1, 2 * args.prime).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    train_loader = make_dataloader(X_tr, y_tr_onehot, args.batch_size, shuffle=True, drop_last=False)
    test_loader = make_dataloader(X_te, y_te_onehot, args.batch_size, shuffle=False, drop_last=False)

    base_model = OneLayerFCN(
        num_tokens=args.prime,
        hidden_width=args.hidden_width,
        context_len=2,
        init_scale=args.init_scale,
        n_classes=args.prime,
    ).to(device)
    model = _ActWrapper(base_model, args.act_fn)

    optimizer = get_optimizer(model, args)
    ce_loss = torch.nn.CrossEntropyLoss()
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
            train_acc = evaluate_mod_add(model, train_loader, args)
            test_acc = evaluate_mod_add(model, test_loader, args)

            if not grokked and train_acc > 99 and test_acc > 99:
                grokked = True
                grokking_epoch = epoch

            metrics.append({
                'epoch': epoch,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'opt_time': opt_time,
                'grokked': grokked,
            })
            pbar.set_description(f"[{run_name}] train={train_acc:.1f}% test={test_acc:.1f}%")

        if grokked:
            break

        model.train()
        t0 = time.time()
        for batch in train_loader:
            inputs, labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            if args.lambda_reg > 0:
                ga = GrokAlign(
                    model,
                    criterion=lambda x, _lbl=labels: ce_loss(x, torch.argmax(_lbl, dim=1)),
                    lambda_reg=args.lambda_reg,
                    device=device,
                )
                loss = ga(inputs)
            else:
                output = model(inputs)
                loss = ce_loss(output, torch.argmax(labels, dim=1))

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
