"""Reconstructed sorting benchmark with deterministic inputs."""
import argparse
import csv
import random
import time
from pathlib import Path


def insertion(values):
    a = list(values)
    for i in range(1, len(a)):
        value, j = a[i], i - 1
        while j >= 0 and a[j] > value:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = value
    return a


def selection(values):
    a = list(values)
    for i in range(len(a)):
        smallest = min(range(i, len(a)), key=a.__getitem__)
        a[i], a[smallest] = a[smallest], a[i]
    return a


def merge(values):
    a = list(values)
    if len(a) < 2:
        return a
    middle = len(a) // 2
    left, right = merge(a[:middle]), merge(a[middle:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]


def heap(values):
    a = list(values)
    def sift(root, end):
        while 2 * root + 1 < end:
            child = 2 * root + 1
            if child + 1 < end and a[child] < a[child + 1]:
                child += 1
            if a[root] >= a[child]:
                break
            a[root], a[child] = a[child], a[root]
            root = child
    for start in range(len(a) // 2 - 1, -1, -1):
        sift(start, len(a))
    for end in range(len(a) - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift(0, end)
    return a


ALGORITHMS = {'insertion': insertion, 'selection': selection, 'merge': merge, 'heap': heap}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[100, 1000, 3000])
    parser.add_argument('--output', type=Path, default=Path('artifacts/benchmark.csv'))
    args = parser.parse_args()
    if any(n < 0 for n in args.sizes):
        parser.error('sizes must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    with args.output.open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['algorithm', 'size', 'order', 'seconds'])
        for n in args.sizes:
            sample = [rng.randrange(max(1, n)) for _ in range(n)]
            for order, data in [('random', sample), ('sorted', sorted(sample)), ('reversed', sorted(sample, reverse=True))]:
                for name, algorithm in ALGORITHMS.items():
                    start = time.perf_counter(); actual = algorithm(data); elapsed = time.perf_counter() - start
                    assert actual == sorted(data), f'{name} returned an incorrect ordering'
                    writer.writerow([name, n, order, elapsed])
    print(args.output)
