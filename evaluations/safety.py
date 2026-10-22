"Measures how often the model predicts the wrong urgency level"
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from common import predictions, LABELS, REPORTS

def safety_metrics(y, pred):
    d = pred - y
    return {'n':len(y), 'accuracy':float(np.mean(d == 0)),
            'undertriage_rate':float(np.mean(d > 0)),
            'overtriage_rate':float(np.mean(d < 0)),
            'severe_undertriage_rate':float(np.mean(d >= 2)),
            'esi1_undertriage_rate':float(np.mean(d[y == 1] > 0)) if np.any(y == 1) else np.nan,
            'esi1_to_esi3_or_4_rate':float(np.mean(d[y == 1] >= 2)) if np.any(y == 1) else np.nan}

def main():
    _, _, y, pred, _, _ = predictions()
    summary = pd.Series(safety_metrics(y, pred), name='value')
    summary.to_csv(REPORTS/'safety_summary.csv', header=True)
    print(summary.to_string())
    print('\nPer-class report:\n', classification_report(y, pred, labels=LABELS, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=[f'actual_{i}' for i in LABELS], columns=[f'pred_{i}' for i in LABELS])
    cm.to_csv(REPORTS/'safety_confusion_matrix.csv')
    print('\nConfusion matrix:\n', cm)

if __name__ == '__main__': main()
