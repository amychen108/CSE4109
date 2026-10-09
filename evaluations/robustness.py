"Tests whether the model gives consistent predictions when the same symptoms are described using clinical terminology versus everyday language."
import joblib
import numpy as np
import pandas as pd
from common import ROOT, REPORTS

PAIRS=[
 ('Patient reports dyspnea and chest pain','I have chest pain and trouble breathing'),
 ('Acute unilateral arm weakness','Suddenly one of my arms became weak'),
 ('Patient presents with emesis and abdominal pain','My stomach hurts and I keep throwing up'),
 ('Persistent headache with nausea','I have had a headache and feel sick to my stomach'),
 ('Reports dizziness on standing','I feel lightheaded when I stand up'),
]

def main():
    model=joblib.load(ROOT/'models'/'baseline_model.pkl')
    rows=[]
    for a,b in PAIRS:
        labels=model.predict([a,b]); confidence=model.predict_proba([a,b]).max(axis=1)
        rows.append({'clinical_text':a,'everyday_text':b,'clinical_esi':int(labels[0]),
                     'everyday_esi':int(labels[1]),'same_prediction':bool(labels[0]==labels[1]),
                     'clinical_confidence':float(confidence[0]),'everyday_confidence':float(confidence[1])})
    result=pd.DataFrame(rows)
    result.to_csv(REPORTS/'synthetic_robustness.csv',index=False)
    print(f'Prediction agreement: {result.same_prediction.mean():.1%} across {len(result)} synthetic pairs')
    print(result[['clinical_esi','everyday_esi','same_prediction']].to_string(index=False))
    print('Agreement measures wording sensitivity, not clinical validity. These examples are not gold-labeled.')

if __name__=='__main__': main()
