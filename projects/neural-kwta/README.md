# Distributed k-Winners-Take-All Simulations

Original CSCE 5215 team project notebook recovered from the uploaded submissions. It explores centralized and distributed gradient-style updates, equality and inequality constraint experiments, and ring-network communication.

## Run

```bash
python -m pip install -r requirements.txt
jupyter notebook kwta_simulation.ipynb
```

Run cells in order. NumPy generates the experimental input, and Matplotlib displays constraint-error, dual-variable, and final-state plots. Random input is not seeded in the original notebook, so exact plots vary between runs.

This is an educational simulation. Importing the original notebook does not establish an exact reproduction of the research paper or independently validate its convergence claims. It is separate from the reconstructed projected-gradient example in `../distributed-gradient`.

## Team

Mohammed Momin Khizar Uddin, Mohammed Ilyas Ahmed, Mohammed Abdul Muqtadeer Ali, and Shafaatulla Bandar Syed.

Notebook outputs and environment metadata were cleared for a compact source-focused repository; computational code was retained.
