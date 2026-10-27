# Diabetes Prediction Extension — reconstructed demo

Random-forest training, a Flask form and JSON prediction API, input validation, and local SQLite output history. The original group project used an existing diabetes application; this is a newly written standalone reconstruction, not a copy of that repository.

```sh
python3 -m pip install -r requirements.txt
python3 train.py --demo
python3 app.py
```

Open `http://127.0.0.1:5000`. Supply all eight named features. `POST /api/predict` accepts a JSON object; `GET /api/history` returns the 50 most recent outputs. To use your own dataset: `python3 train.py --csv /path/to/diabetes.csv`, with the eight standard feature columns in `train.py` and binary `Outcome`.

Synthetic labels come from an explicitly artificial formula. No historical accuracy, clinical validation, calibrated risk, or personalized medical recommendation is claimed. History stores outputs and timestamps, not submitted feature values. The app has no authentication and is intended only for local use; do not expose it as a shared clinical service. Only load model files you trust.
