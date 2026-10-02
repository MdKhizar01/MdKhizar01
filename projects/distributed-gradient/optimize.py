"""Illustrative ring-distributed constrained quadratic optimization."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def project(vector, k, equality=True):
    """Project onto 0 <= x <= 1 and sum(x) = k (or <= k)."""
    if not 0 <= k <= vector.size:
        raise ValueError('k must be between zero and the number of items')
    clipped = np.clip(vector, 0, 1)
    if not equality and clipped.sum() <= k:
        return clipped
    if k == 0:
        return np.zeros_like(vector)
    if k == vector.size:
        return np.ones_like(vector)
    low, high = float(vector.min() - 1), float(vector.max())
    for _ in range(70):
        midpoint = (low + high) / 2
        if np.clip(vector - midpoint, 0, 1).sum() > k:
            low = midpoint
        else:
            high = midpoint
    return np.clip(vector - (low + high) / 2, 0, 1)


def solve(scores, k, steps=500, equality=True):
    agents, items = scores.shape
    if agents < 3 or steps < 1:
        raise ValueError('Need at least three agents and one iteration')
    x = np.stack([project(np.zeros(items), k, equality) for _ in range(agents)])
    reference = project(scores.mean(axis=0), k, equality)
    history = []
    for iteration in range(steps):
        mixed = (x + np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0)) / 3
        rate = .4 / ((iteration + 1) ** .6)
        x = np.stack([project(v, k, equality) for v in mixed - rate * (x - scores)])
        average = x.mean(axis=0)
        history.append([iteration + 1, rate, float(np.linalg.norm(average - reference)),
            float(np.linalg.norm(x - average)), float(average.sum())])
    return x, reference, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', type=int, default=1000)
    parser.add_argument('--agents', type=int, default=8)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--constraint', choices=['equality', 'inequality'], default='equality')
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    if args.items < 1 or args.agents < 3 or args.steps < 1 or not 0 <= args.k <= args.items:
        parser.error('Invalid items, agents, steps, or k')
    rng = np.random.default_rng(42)
    shared = rng.normal(0, 1, args.items)
    scores = shared + rng.normal(0, .2, (args.agents, args.items))
    x, ref, history = solve(scores, args.k, args.steps, args.constraint == 'equality')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'convergence.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'step_size', 'reference_error', 'consensus_error', 'mass'])
        writer.writerows(history)
    average = x.mean(axis=0)
    np.save(args.output / 'activations.npy', average)
    summary = {'items': args.items, 'agents': args.agents, 'constraint': args.constraint,
        'mass': float(average.sum()), 'reference_error': history[-1][2],
        'consensus_error': history[-1][3],
        'top_k_indices': np.argsort(-average, kind='stable')[:args.k].tolist()}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
