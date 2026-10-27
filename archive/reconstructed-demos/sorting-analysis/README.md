# Sorting Algorithm Analysis — reconstructed

Python implementations of insertion sort, selection sort, merge sort, and heap sort, with CSV timing results for random, sorted, and reversed inputs.

```sh
python3 benchmark.py --sizes 100 1000 3000
```

No external dependencies. Every timed output is checked against Python's `sorted`. Quadratic algorithms become expensive on large inputs. Timings are machine-dependent; this code does not recover the original assignment measurements or language.
