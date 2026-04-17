"""
Load all per-run JSON files from an outputs subdirectory and print a summary
table comparing regularization strategies.

Usage:
    python summarize_results.py --dataset sparse_parity
    python summarize_results.py --dataset mod_add
    python summarize_results.py [output_dir]         # auto-detects dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from math import erf, sqrt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    'sparse_parity': {
        'output_dir': os.path.join(SCRIPT_DIR, 'outputs', 'sparse_parity'),
        'condition_order': ['baseline', 'GF', 'GA0.1', 'OG'],
        'title': 'Sparse Parity',
    },
    'mod_add': {
        'output_dir': os.path.join(SCRIPT_DIR, 'outputs', 'mod_add'),
        'condition_order': ['baseline', 'GF', 'GA0.01'],
        'title': 'Modular Addition',
    },
    'xor': {
        'output_dir': os.path.join(SCRIPT_DIR, 'outputs', 'xor'),
        'condition_order': ['baseline', 'GF', 'GA1.0', 'OG'],
        'title': 'XOR',
    },
    'mnist_ce': {
        'output_dir': os.path.join(SCRIPT_DIR, 'outputs', 'mnist_ce'),
        'condition_order': ['baseline', 'GF', 'GA0.01', 'OG'],
        'title': 'MNIST (Cross-Entropy)',
    },
    'mnist_se': {
        'output_dir': os.path.join(SCRIPT_DIR, 'outputs', 'mnist_se'),
        'condition_order': ['baseline', 'GF', 'GA0.01', 'OG'],
        'title': 'MNIST (Squared Error)',
    },
}

CONDITION_LABELS = {
    'baseline': 'Baseline',
    'GF':       'GrokFast (EMA)',
    'GA0.1':    'GrokAlign (λ=0.1)',
    'GA0.01':   'GrokAlign (λ=0.01)',
    'GA1.0':    'GrokAlign (λ=1.0)',
    'OG':       'OrthoGrad',
}


def detect_dataset(results):
    """Infer the dataset from run name prefixes in the loaded results."""
    for r in results:
        name = r.get('run_name', '')
        if name.startswith('mod_add'):
            return 'mod_add'
        if name.startswith('xor'):
            return 'xor'
        if name.startswith('mnist_ce'):
            return 'mnist_ce'
        if name.startswith('mnist_se'):
            return 'mnist_se'
        if name.startswith('sparse_parity'):
            return 'sparse_parity'
    return 'sparse_parity'


def load_results(output_dir):
    results = []
    path = Path(output_dir)
    if not path.exists():
        print(f"Output directory not found: {output_dir}")
        return results
    for p in sorted(path.glob('*.json')):
        with open(p) as f:
            try:
                results.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"  Warning: could not parse {p.name}: {e}")
    return results


def summarize(results, title, condition_order):
    if not results:
        print("No results found.")
        return

    # Group runs by condition
    by_condition: dict[str, list] = {}
    for r in results:
        cond = r.get('condition', 'unknown')
        by_condition.setdefault(cond, []).append(r)

    all_conditions = condition_order + sorted(
        c for c in by_condition if c not in condition_order
    )

    # ---- Summary table ----
    print()
    print(f"Regularization Strategy Comparison — {title}")
    print("=" * 74)

    col_w = max((len(CONDITION_LABELS.get(c, c)) for c in by_condition), default=12)
    col_w = max(col_w, 18)

    header = (
        f"{'Condition':<{col_w}} | {'Runs':>4} | {'Grokked':>9} | "
        f"{'Mean Epoch':>11} | {'Std':>8} | {'Median':>8}"
    )
    print(header)
    print("-" * len(header))

    for cond in all_conditions:
        if cond not in by_condition:
            continue
        runs = by_condition[cond]
        grokked_runs = [r for r in runs if r.get('grokked')]
        epochs = [r['grokking_epoch'] for r in grokked_runs if r.get('grokking_epoch') is not None]

        label = CONDITION_LABELS.get(cond, cond)
        n = len(runs)
        g = len(grokked_runs)

        if epochs:
            mean_e  = f"{np.mean(epochs):>11.0f}"
            std_e   = f"{np.std(epochs):>8.0f}"
            med_e   = f"{np.median(epochs):>8.0f}"
        else:
            mean_e = std_e = med_e = f"{'N/A':>8}"

        print(f"{label:<{col_w}} | {n:>4} | {g:>4}/{n:<4} | {mean_e} | {std_e} | {med_e}")

    print()

    # ---- Statistical summaries and comparisons vs baseline ----
    summaries = {}
    for cond, runs in by_condition.items():
        grokked_runs = [r for r in runs if r.get('grokked')]
        epochs = [r['grokking_epoch'] for r in grokked_runs if r.get('grokking_epoch') is not None]
        times = []
        for r in grokked_runs:
            ge = r.get('grokking_epoch')
            metrics = r.get('metrics', []) or []
            if ge is None:
                continue
            total = 0.0
            for m in metrics:
                e = m.get('epoch')
                if e is None:
                    continue
                if e <= ge:
                    total += float(m.get('opt_time', 0.0) or 0.0)
            times.append(total)

        summaries[cond] = {
            'n_runs': len(runs),
            'n_grokked': len(grokked_runs),
            'epochs': np.array(epochs) if epochs else np.array([], dtype=float),
            'times': np.array(times) if times else np.array([], dtype=float),
        }

    # Helper: Welch t-test with scipy fallback to normal approx
    def welch_ttest(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.size < 2 or b.size < 2:
            return float('nan'), float('nan')
        try:
            from scipy.stats import ttest_ind
            res = ttest_ind(a, b, equal_var=False, nan_policy='omit')
            return float(res.statistic), float(res.pvalue)
        except Exception:
            na, nb = a.size, b.size
            ma, mb = float(np.mean(a)), float(np.mean(b))
            sa, sb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
            denom = sqrt(sa/na + sb/nb)
            if denom == 0:
                return float('nan'), float('nan')
            t = (ma - mb) / denom
            num = (sa/na + sb/nb) ** 2
            den = 0.0
            if na > 1:
                den += (sa**2) / ((na**2) * (na - 1))
            if nb > 1:
                den += (sb**2) / ((nb**2) * (nb - 1))
            df = num / den if den != 0 else max(1.0, na + nb - 2)
            z = t
            p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
            return float(t), float(p)

    baseline = 'baseline'
    base_summary = summaries.get(baseline)
    print("Comparison vs baseline (Welch t-test; p-values approx if scipy unavailable)")
    print("=" * 74)
    hdr = f"{'Condition':<{col_w}} | {'Mean Ep':>8} | {'Ratio Ep':>8} | {'p(ep)':>8} | {'Mean Time(s)':>12} | {'Ratio Time':>11} | {'p(time)':>8}"
    print(hdr)
    print('-' * len(hdr))

    for cond in all_conditions:
        if cond not in summaries:
            continue
        s = summaries[cond]
        mean_ep = float(np.mean(s['epochs'])) if s['epochs'].size else float('nan')
        mean_time = float(np.mean(s['times'])) if s['times'].size else float('nan')

        if base_summary and base_summary['epochs'].size and s['epochs'].size and cond != baseline:
            t_ep, p_ep = welch_ttest(s['epochs'], base_summary['epochs'])
        else:
            t_ep, p_ep = float('nan'), float('nan')

        if base_summary and base_summary['times'].size and s['times'].size and cond != baseline:
            t_time, p_time = welch_ttest(s['times'], base_summary['times'])
        else:
            t_time, p_time = float('nan'), float('nan')

        if base_summary and base_summary['epochs'].size and not np.isnan(mean_ep):
            base_mean_ep = float(np.mean(base_summary['epochs']))
            if base_mean_ep != 0 and mean_ep > 0:
                raw = mean_ep / base_mean_ep
                ratio_ep = raw if raw >= 1.0 else 1.0 / raw
            else:
                ratio_ep = float('nan')
        else:
            ratio_ep = float('nan')

        if base_summary and base_summary['times'].size and not np.isnan(mean_time):
            base_mean_time = float(np.mean(base_summary['times']))
            if base_mean_time != 0 and mean_time >= 0:
                rawt = mean_time / base_mean_time
                ratio_time = rawt if rawt >= 1.0 else 1.0 / rawt
            else:
                ratio_time = float('nan')
        else:
            ratio_time = float('nan')

        mean_ep_s = f"{mean_ep:8.1f}" if not np.isnan(mean_ep) else f"{'N/A':>8}"
        ratio_ep_s = f"{ratio_ep:8.2f}" if not np.isnan(ratio_ep) else f"{'N/A':>8}"
        p_ep_s = f"{p_ep:8.3g}" if not np.isnan(p_ep) else f"{'N/A':>8}"

        mean_time_s = f"{mean_time:12.2f}" if not np.isnan(mean_time) else f"{'N/A':>12}"
        ratio_time_s = f"{ratio_time:11.2f}" if not np.isnan(ratio_time) else f"{'N/A':>11}"
        p_time_s = f"{p_time:8.3g}" if not np.isnan(p_time) else f"{'N/A':>8}"

        label = CONDITION_LABELS.get(cond, cond)
        print(f"{label:<{col_w}} | {mean_ep_s} | {ratio_ep_s} | {p_ep_s} | {mean_time_s} | {ratio_time_s} | {p_time_s}")

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', nargs='?', default=None,
                        help="Directory containing result JSON files. Defaults to the dataset-specific output dir.")
    parser.add_argument('--dataset', choices=list(DATASET_CONFIG.keys()), default=None,
                        help="Which dataset's results to summarize. Auto-detected from run names if omitted.")
    args = parser.parse_args()

    # Resolve output directory and load results
    if args.output_dir:
        output_dir = args.output_dir
        results = load_results(output_dir)
        dataset = args.dataset or detect_dataset(results)
    elif args.dataset:
        output_dir = DATASET_CONFIG[args.dataset]['output_dir']
        results = load_results(output_dir)
        dataset = args.dataset
    else:
        dataset = 'sparse_parity'
        output_dir = DATASET_CONFIG[dataset]['output_dir']
        results = load_results(output_dir)

    cfg = DATASET_CONFIG[dataset]
    print(f"Loaded {len(results)} result(s) from {output_dir}")
    summarize(results, title=cfg['title'], condition_order=cfg['condition_order'])
