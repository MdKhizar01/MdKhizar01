"""Reconstructed CSV cleaning, classification, and anomaly pipeline."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

FEATURES = ['src_port', 'dest_port', 'packet_length', 'packets_per_time']


def demo(seed=42, rows=1000):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, rows)
    return pd.DataFrame({
        'src_port': rng.integers(1024, 65536, rows),
        'dest_port': rng.choice([53, 80, 443, 8080], rows),
        'packet_length': rng.integers(64, 1500, rows),
        'packets_per_time': rng.gamma(2, 15, rows) + target * 70,
        'target': target,
    })


def clean(frame):
    missing = set(FEATURES + ['target']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    frame = frame.copy().drop_duplicates()
    for col in FEATURES + ['target']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame[frame.target.isin([0, 1])].copy()
    for col in FEATURES:
        frame.loc[frame[col] < 0, col] = np.nan
        if frame[col].notna().sum() == 0:
            raise ValueError(f'No valid values for {col}')
    for col in ['src_port', 'dest_port']:
        frame.loc[frame[col] > 65535, col] = np.nan
    return frame


def run(frame, output):
    frame = clean(frame)
    if frame.target.nunique() != 2 or frame.target.value_counts().min() < 4:
        raise ValueError('Need at least four rows in each target class')
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame.target.astype(int), test_size=.25,
        stratify=frame.target, random_state=42)
    classifier = make_pipeline(SimpleImputer(strategy='median'),
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    anomaly = make_pipeline(SimpleImputer(strategy='median'),
        IsolationForest(n_estimators=100, contamination='auto', random_state=42))
    anomaly.fit(x_train)
    results = x_test.copy()
    results['actual'] = y_test
    results['predicted'] = predictions
    results['anomaly'] = anomaly.predict(x_test) == -1
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / 'predictions.csv', index=False)
    joblib.dump({'features': FEATURES, 'classifier': classifier, 'anomaly': anomaly}, output / 'models.joblib')
    report = {'rows_after_cleaning': len(frame), 'train_rows': len(x_train),
        'test_rows': len(x_test), 'classification': classification_report(y_test, predictions, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist()}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    if bool(args.csv) == args.demo:
        parser.error('Choose exactly one of --csv or --demo')
    print(json.dumps(run(demo() if args.demo else pd.read_csv(args.csv), args.output), indent=2))
