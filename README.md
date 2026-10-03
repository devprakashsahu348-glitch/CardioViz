# CardioViz: cardiovascular risk prediction with a 3D coronary map

Multimodal AI Hackathon 2026, Track A (Cardiovascular Risk Visualization & Prediction).

CardioViz predicts overall coronary artery disease (CAD) and stenosis of the three main coronary vessels
(LAD, LCX, RCA) from clinical data, colours each artery on an interactive 3D heart by its predicted
probability, and explains every prediction with exact SHAP values.

> **Clinical safety notice:** predictions are for decision support and education only. They are not a
> diagnosis and do not replace formal diagnostic imaging or a clinician's judgement.

**Live demo:** https://claude.ai/artifact/CRa58YPWaQR7MKJVPZ46wY
**Demo video:** _add your YouTube link here_

## Features

| Requirement | Where |
|---|---|
| Predict overall CAD and LAD / LCX / RCA stenosis | `web/index.html` (in-browser), `pipeline/train.py` (Python) |
| No target leakage (Cath, LAD, LCX, RCA never used as inputs) | enforced in both code paths |
| Accuracy, precision, recall, F1, ROC-AUC | metrics table in the app, `model/metrics.json` |
| Interactive 3D heart (Three.js), rotate / zoom / select | `web/index.html` |
| Artery colours update from predicted probabilities | green (low) to red (high), live |
| Interpretable breakdown and per-feature contribution | SHAP bars per target, with measurement values |
| Safety disclaimer | banner at the top of the page |
| Runs without a dedicated GPU | low-poly meshes, pixel ratio capped at 1.5 |

## Quick start (web app, no install)

The app is a single static file.

```bash
git clone <your-repo-url> && cd cardioviz
python -m http.server 8000 --directory web      # or just double-click web/index.html
# open http://localhost:8000
```

An internet connection is needed once per load, because Three.js comes from cdnjs. To work fully offline,
download `three.min.js` (r128) into `web/` and change the `<script src>` tag.

### Using the real dataset

1. Download the **Z-Alizadeh Sani** extension dataset from the UCI Machine Learning Repository
   (also on Kaggle) as CSV. It needs the columns `Cath`, `LAD`, `LCX`, `RCA`.
2. In the app, click **Retrain on Z-Alizadeh Sani CSV** and choose the file. All four models are retrained in the
   browser and the metrics table updates. Nothing is uploaded anywhere.

Without a CSV, the app starts on a built-in **synthetic** demo cohort so it works offline. Its metrics are not
clinically meaningful. Replace it with the real data before quoting any numbers.

## Python training pipeline

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r pipeline/requirements.txt
mkdir data && cp /path/to/z_alizadeh_sani.csv data/
python pipeline/train.py --csv data/z_alizadeh_sani.csv                # the 16 inputs used by the web app
python pipeline/train.py --csv data/z_alizadeh_sani.csv --features all # every clinical column except targets
```

Outputs go to `model/`: `weights.json` (coefficients, intercepts, scaler mean and scale per target) and
`metrics.json` (stratified 80/20 held-out metrics and 5-fold cross-validated metrics).
`pipeline/make_demo_csv.py` writes a small synthetic CSV for smoke-testing the script only.

## How it works

- **Inputs (16):** age, sex, blood pressure, pulse rate, typical chest pain, diabetes, hypertension, LDL, HDL,
  triglycerides, fasting glucose, ST elevation, T inversion, ejection fraction (EF-TTE), region with RWMA, BMI.
- **Models:** one L2-regularised logistic regression per target (CAD, LAD, LCX, RCA) on standardised features.
  A linear model gives exact SHAP values: `phi_j = w_j * (x_j - mean_j) / sd_j`, in log-odds.
- **Validation:** stratified 80/20 split plus 5-fold cross-validation in Python; 80/20 split in the browser.
- **3D viewer:** Three.js r128. A translucent torso with lungs, a heart mesh and three tube meshes for the coronary
  arteries. Probability maps to hue (120 deg green to 0 deg red). Raycasting handles click selection.
  Selecting a vessel in the 3D view or on a probability card updates the explanation panel.
- **Region with RWMA** is a clinical input, not an anatomical lesion map. The dataset has no pixel-level or 3D
  lesion locations, so the arteries show per-vessel probabilities only.

## Repository layout

```
web/index.html            single-file app: UI, model training/inference, SHAP, 3D viewer
pipeline/train.py         scikit-learn training and evaluation, exports weights and metrics
pipeline/make_demo_csv.py synthetic CSV for smoke tests
pipeline/requirements.txt
docs/DOCUMENTATION.md     project documentation (max 6 pages)
model/                    weights.json and metrics.json written by train.py
```

## Extending

- **New feature:** add a row to the `F` array in `web/index.html` (and an alias in `AL` for CSV import), and its
  column name to `WEB` in `pipeline/train.py`.
- **New model:** replace `fit()` in the web app or the pipeline in `train.py`. Non-linear models need a SHAP
  library (`pip install shap`) instead of the closed-form values.
- **New anatomy:** add a path to `paths` (a list of 3D points on the heart surface) and a target in `TK` / `TL`.
- **Better mesh:** load a `.glb` from BodyParts3D or Sketchfab with `GLTFLoader` and attach the artery tubes to it.

## Limitations

Small dataset (303 patients), a single centre, no external validation, and linear models only. The heart and
arteries are schematic, not patient-specific anatomy. Research prototype only.

## Data and credits

Z-Alizadeh Sani dataset, UCI Machine Learning Repository (Alizadehsani et al.). Three.js (MIT). Licensed under MIT.
