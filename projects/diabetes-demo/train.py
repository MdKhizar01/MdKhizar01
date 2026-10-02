"""Train an educational diabetes feature classifier; demo labels are synthetic."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


def demo():
    rng = np.random.default_rng(42); n=600
    values = np.column_stack([rng.integers(0,10,n),rng.uniform(70,200,n),rng.uniform(50,110,n),
        rng.uniform(10,50,n),rng.uniform(20,250,n),rng.uniform(18,45,n),rng.uniform(.1,2,n),rng.integers(18,80,n)])
    frame=pd.DataFrame(values,columns=FEATURES)
    # Explicit toy decision rule: no empirical clinical interpretation.
    frame['Outcome']=(values[:,1]+2*values[:,5]+rng.normal(0,20,n)>200).astype(int)
    return frame


def train(frame, output, synthetic):
    missing = set(FEATURES+['Outcome'])-set(frame.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    x=frame[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    y=pd.to_numeric(frame.Outcome,errors='raise')
    if set(y.unique()) != {0,1}: raise ValueError('Outcome must contain both 0 and 1')
    if x.notna().sum().min()==0: raise ValueError('Every feature needs numeric values')
    a,b,c,d=train_test_split(x,y,stratify=y,test_size=.2,random_state=42)
    model=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=100,random_state=42))
    model.fit(a,c)
    report=classification_report(d,model.predict(b),output_dict=True,zero_division=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'features':FEATURES,'synthetic':synthetic},output)
    output.with_suffix('.json').write_text(json.dumps({'synthetic':synthetic,'report':report},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path);p.add_argument('--demo',action='store_true')
    p.add_argument('--output',type=Path,default=Path('artifacts/model.joblib'));a=p.parse_args()
    if bool(a.csv)==a.demo:p.error('Choose exactly one of --csv or --demo')
    print(json.dumps(train(demo() if a.demo else pd.read_csv(a.csv),a.output,a.demo),indent=2))
