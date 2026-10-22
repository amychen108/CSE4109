from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports' / 'evaluations'
LABELS = [1, 2, 3, 4]

def load_test():
    path = Path(os.environ.get('TRIAGE_DATA_PATH',str(ROOT / 'dataset' / 'triage_level.csv'))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found at {path}. Set TRIAGE_DATA_PATH to the authorized triage_level.csv path.')
    df = pd.read_csv(path).dropna(subset=['HPI', 'triage']).copy()
    if not df['triage'].isin(LABELS).all():
        raise ValueError('Unexpected ESI levels. Expected 1, 2, 3, 4.')
    positions = np.arange(len(df))
    _, temporary = train_test_split(positions, test_size=.30, random_state=42, stratify=df['triage'].to_numpy())
    _, test_positions = train_test_split(temporary, test_size=.50, random_state=42, stratify=df['triage'].to_numpy()[temporary])
    return df.iloc[test_positions].reset_index(drop=True)

def predictions():
    df = load_test()
    model_path = ROOT / 'models' / 'baseline_model.pkl'
    if not model_path.is_file():
        raise FileNotFoundError(f'Model not found: {model_path}')
    model = joblib.load(model_path)
    y = df['triage'].to_numpy(dtype=int)
    pred = np.asarray(model.predict(df['HPI'].tolist()), dtype=int)
    proba = np.asarray(model.predict_proba(df['HPI'].tolist()), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if proba.shape != (len(y), len(classes)):
        raise ValueError('Unexpected predict_proba shape')
    REPORTS.mkdir(parents=True, exist_ok=True)
    return df, model, y, pred, proba, classes
