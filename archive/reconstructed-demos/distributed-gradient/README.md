# Distributed Projected Gradient — reconstructed illustration

Eight agents communicate on a ring and minimize the average quadratic objective `0.5 * ||x - scores_i||²`, subject to `0 <= x <= 1` and either `sum(x) = k` or `sum(x) <= k`. Each iteration mixes neighbors, takes a diminishing local gradient step, and projects onto the feasible set. Bisection computes the capped-simplex projection.

```sh
python3 -m pip install -r requirements.txt
python3 optimize.py --items 1000 --k 20 --constraint equality
python3 optimize.py --items 1000 --k 20 --constraint inequality --output artifacts/inequality
```

Outputs include activation values, convergence CSV, reference/consensus errors, and top-k indices. A centralized projection of the average score supplies an exact reference for this quadratic objective. The returned activations can be fractional; the top-k indices are a ranking derived from them.

This newly written educational example preserves the remembered ring, equality/inequality, and 1000-item themes. It is **not** a reproduction of the original constrained neural k-WTA paper or proof of matching its simulations. No original experiment results are claimed.
