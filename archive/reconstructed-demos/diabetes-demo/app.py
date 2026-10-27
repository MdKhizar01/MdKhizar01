"""Local Flask demonstration with validated inputs and SQLite history."""
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template
from train import FEATURES


def create_app(model_path=None, database=None):
    app=Flask(__name__)
    bundle=joblib.load(model_path or os.environ.get('MODEL_PATH','artifacts/model.joblib'))
    db=Path(database or os.environ.get('HISTORY_DB','artifacts/history.sqlite3'))
    db.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, timestamp TEXT, prediction INTEGER, score REAL)')

    @app.get('/')
    def home():return render_template('index.html',features=FEATURES,synthetic=bundle['synthetic'])

    @app.post('/api/predict')
    def predict():
        data=request.get_json(silent=True)
        try:
            if not isinstance(data,dict):raise ValueError('Expected a JSON object')
            row={name:float(data[name]) for name in FEATURES}
            if any(not math.isfinite(v) or v<0 for v in row.values()):raise ValueError('Values must be finite and nonnegative')
        except (KeyError,ValueError,TypeError) as exc:
            return jsonify(error=f'Invalid inputs: {exc}'),400
        x=pd.DataFrame([row],columns=FEATURES)
        prediction=int(bundle['model'].predict(x)[0])
        score=float(bundle['model'].predict_proba(x)[0,1])
        with sqlite3.connect(db) as conn:
            conn.execute('INSERT INTO history(timestamp,prediction,score) VALUES(?,?,?)',
                (datetime.now(timezone.utc).isoformat(),prediction,score))
        return jsonify(prediction=prediction,model_score=score,synthetic=bundle['synthetic'],
            note='Educational model output; not a diagnosis or calibrated clinical risk.')

    @app.get('/api/history')
    def history():
        with sqlite3.connect(db) as conn:
            rows=conn.execute('SELECT id,timestamp,prediction,score FROM history ORDER BY id DESC LIMIT 50').fetchall()
        return jsonify([dict(zip(['id','timestamp','prediction','model_score'],row)) for row in rows])
    return app


if __name__=='__main__':create_app().run(host='127.0.0.1',port=5000,debug=False)
