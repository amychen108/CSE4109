# CSE4109
Introduction to AI for Health Project

## Symphony DX — Baseline Model

Symphony DX predicts an Emergency Severity Index (ESI) triage level from a description of a patient's symptoms. This README covers the **baseline model**: a simple, symptoms-only model that serves as the reference point for the project.

> **Research use only.** This is a course project. It is not a medical device and is not a substitute for professional medical advice.

## Dataset

[MIMIC-IV-Ext Clinical Decision Support for Referral, Triage and Diagnosis](https://physionet.org/) (v1.0.2), a PhysioNet dataset of about 9,150 emergency department visits. Access requires PhysioNet credentialing and a signed data use agreement.

The data is **not** in this repository and must not be committed. The baseline needs only one file, `triage_level.csv`. `baseline_demo1.ipynb` reads it from your Desktop:

```python
DATA_PATH = Path.home() / "Desktop" / "triage_level.csv"
```

If your copy is somewhere else, change that line in the first code cell.

## Setup

```bash
pip install -r requirements.txt
```

`scikit-learn` is pinned to 1.6.1, the version the model was saved with. Other versions load it with a warning or may fail to load it.

In VS Code, pick the Anaconda Python kernel (top-right of the notebook) so the installed packages are found.

## Baseline model

- **Input:** history of present illness (HPI) text only. No vitals or demographics.
- **Model:** TF-IDF + logistic regression.
  - TF-IDF: up to 5,000 words and two-word phrases, English stop words removed, letters only (numbers and `___` placeholders ignored).
  - Logistic regression: balanced class weights so rare levels are not ignored.
- **Output:** ESI level 1–4 and a probability for each level.
- **Saved model:** `models/baseline_model.pkl`.

## Notebooks

| Notebook | What it does |
|---|---|
| `src/backend/train_baseline.ipynb` | Trains the model on Colab and saves `models/baseline_model.pkl`. Reports accuracy, F1, classification report, confusion matrix (as numbers) and sample predictions. |
| `src/backend/baseline_demo1.ipynb` | Evaluates and explains the model for Demo 1. Retrains with identical settings and the same split, checks it matches the saved `.pkl` (never overwrites it), and adds the comparison, confusion-matrix plot, confidence check and reasoning. |

Both use the same 70/15/15 train/validation/test split with the same random seed, so the test rows are identical. `baseline_demo1.ipynb` prints only aggregate numbers, plots and made-up examples, never patient rows or note text.

## Demo 1 checklist

| Item | Where |
|---|---|
| Working form + results display | `streamlit run app.py` |
| Trained on real MIMIC data | `train_baseline.ipynb`; retrained and checked in `baseline_demo1.ipynb` section 2 |
| Evaluated with metrics (accuracy / F1) | `baseline_demo1.ipynb` section 3 |
| Real model output | The app loads `models/baseline_model.pkl` |
| Confidence scores | App shows confidence; `baseline_demo1.ipynb` section 6 checks whether it can be trusted |
| Reasoning | `baseline_demo1.ipynb` section 7: words that pushed each prediction, top words per ESI level |
| Comparison | `baseline_demo1.ipynb` section 4: vs "always ESI 3" and a random guesser |
| Confusion matrix | `baseline_demo1.ipynb` section 5, saved to `reports/baseline_confusion_matrix.png` |

## Results

Test set: 1,373 visits the model never saw during training.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Always predict ESI 3 | 0.536 | _from `baseline_demo1.ipynb`_ | _from `baseline_demo1.ipynb`_ |
| Random guess (class frequencies) | _from `baseline_demo1.ipynb`_ | | |
| **Baseline model** | **0.609** | **0.403** | **0.619** |

Validation set: accuracy 0.609, macro F1 0.407, weighted F1 0.618, so the model does not overfit to the training data.

Recall per level (test): ESI 1 0.42, ESI 2 0.54, ESI 3 0.70, ESI 4 0.00.

Confusion matrix: `reports/baseline_confusion_matrix.png` (created by `baseline_demo1.ipynb`).

## Using the model

Run the app from the repo folder:

```bash
streamlit run app.py
```

Or load the model in Python, also from the repo folder:

```python
import joblib

model = joblib.load("models/baseline_model.pkl")
model.predict(["Severe chest pain and difficulty breathing"])        # ESI level
model.predict_proba(["Severe chest pain and difficulty breathing"])  # probability per level
```

Using the model does not need the dataset; running the notebooks does.

## Limitations

- **Text only.** The model ignores vitals, age and other patient information, which matter a lot for triage. This is intentional: the baseline is the reference point.
- **Rare levels.** There are only 64 ESI 4 cases and no ESI 5, and the model never predicts ESI 4 correctly. It cannot recognise problems suited to a routine appointment or self-care.
- **Balanced class weights.** They help the model catch ESI 1 patients, at the cost of overall accuracy.
- **Clinician notes vs. everyday language.** The model was trained on HPI text written by clinicians. The app asks for symptoms in the user's own words, so app predictions are less reliable than the test scores suggest.
- **Confidence.** The confidence shown is the model's probability for its chosen level. `baseline_demo1.ipynb` section 6 shows how closely it matches actual accuracy.

## Project structure

```
app.py                          Streamlit app
requirements.txt                Python packages
models/
  baseline_model.pkl            Saved baseline model
reports/
  baseline_confusion_matrix.png Confusion matrix plot (from baseline_demo1.ipynb)
src/backend/
  train_baseline.ipynb          Trains and saves the baseline
  baseline_demo1.ipynb          Demo 1 evaluation and explanations
src/frontend/
  layout.py                     Page header and sidebar
  components.py                 Form, model loading, prediction and results
  styles.py                     Custom CSS
```
