"Tests whether the model's predicted confidence matches its actual accuracy."
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import predictions, REPORTS

def calibration(y, pred, proba, classes, bins=10):
    confidence = proba.max(axis=1)
    correct = (y == pred).astype(float)
    boundaries = np.linspace(0, 1, bins+1)
    indices = np.minimum(np.searchsorted(boundaries, confidence, side='right')-1, bins-1)
    rows=[]
    for k in range(bins):
        mask=indices==k
        if mask.any():
            rows.append({'bin':k+1, 'n':int(mask.sum()), 'confidence':float(confidence[mask].mean()), 'accuracy':float(correct[mask].mean())})
    table=pd.DataFrame(rows)
    ece=float(np.sum(table['n']*abs(table['accuracy']-table['confidence']))/len(y))
    one_hot=(y[:,None] == classes[None,:]).astype(float)
    brier=float(np.mean(np.sum((proba-one_hot)**2, axis=1)))
    return table, ece, brier

def main():
    _, _, y, pred, proba, classes = predictions()
    table, ece, brier = calibration(y,pred,proba,classes)
    table.to_csv(REPORTS/'calibration_bins.csv',index=False)
    print(f'Expected calibration error: {ece:.4f}\nMulticlass Brier score (sum across classes): {brier:.4f}')
    print(table.to_string(index=False))
    fig, ax=plt.subplots(figsize=(6,6))
    ax.plot([0,1],[0,1], '--', color='gray', label='Perfect calibration')
    ax.plot(table.confidence, table.accuracy, 'o-', label='Baseline model')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted confidence',ylabel='Observed accuracy',title='Top-label reliability')
    ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS/'calibration.png',dpi=160); plt.close(fig)

if __name__=='__main__': main()
