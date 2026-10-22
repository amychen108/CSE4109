"Uses repeated resampling of the test data to calculate 95% confidence intervals for performance metrics."
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common import predictions, LABELS, REPORTS

def main():
    _, _, y, pred, _, _ = predictions()
    rng = np.random.default_rng(42)
    scores = {
        'accuracy': [],
        'macro_f1': [],
        'undertriage_rate': [],
        'esi1_recall': [],            
        'esi1_undertriage_rate': []    
    }
    
    for _ in range(2000):
        ix = rng.integers(0, len(y), size=len(y))
        yt, yp = y[ix], pred[ix]

        scores['accuracy'].append(float(np.mean(yt == yp)))
        scores['macro_f1'].append(float(
            f1_score(yt, yp, labels=LABELS,
                     average='macro', zero_division=0)
        ))
        scores['undertriage_rate'].append(
            float(np.mean(yp > yt))
        )
        esi1_mask = yt == 1

        if esi1_mask.sum() > 0:
            scores['esi1_recall'].append(
                float(np.mean(yp[esi1_mask] == 1))
            )
            scores['esi1_undertriage_rate'].append(
                float(np.mean(yp[esi1_mask] > 1))
            )
    esi1_mask = y == 1 
    point = {
        'accuracy': float(np.mean(y == pred)),
        'macro_f1': float(
            f1_score(y, pred, labels=LABELS,
                     average='macro', zero_division=0)
        ),
        'undertriage_rate': float(np.mean(pred > y)),
        'esi1_recall': float(
            np.mean(pred[esi1_mask] == 1)
        ),
        'esi1_undertriage_rate': float(
            np.mean(pred[esi1_mask] > 1)
        )
    }

    result = pd.DataFrame([
        {
            'metric': key,
            'estimate': point[key],
            'ci_2.5': np.quantile(v, .025),
            'ci_97.5': np.quantile(v, .975)
        }
        for key, v in scores.items()
    ])

    REPORTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS / 'bootstrap_ci.csv', index=False)

    print(result.to_string(index=False))
    print(
        'Note: visit-level bootstrap assumes observations '
        'are independent; multiple visits by the same patient '
        'may violate this assumption.'
    )

if __name__ == '__main__':
    main()

