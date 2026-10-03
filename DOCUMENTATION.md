# CardioViz: project documentation

Multimodal AI Hackathon 2026, Track A. _Fill the bracketed items with your real results before submitting. Keep
the final PDF to 6 pages._

## 1. Overview
Predicts overall CAD and LAD / LCX / RCA stenosis from clinical data and maps each vessel's probability onto an
interactive 3D heart, with SHAP explanations. Decision support and education only.

## 2. Dataset and preprocessing
- Z-Alizadeh Sani extension, 303 patients. Targets: `Cath` (CAD), `LAD`, `LCX`, `RCA` (stenosis >= 50%).
- Leakage control: the four target columns are removed from the inputs in every code path.
- Cleaning: column names normalised (lower case, alphanumerics only); text values mapped (`Male`=1, `Fmale`=0,
  `Cad`/`Stenotic`=1, `Normal`=0); missing values filled with the column median; features standardised.
- Features: 16 inputs in the app. `--features all` uses every clinical column. [State which you report.]

## 3. Model architecture
One L2-regularised logistic regression per target (C = 1.0). Chosen for exact, fast SHAP values and because it runs
in the browser with no backend. [Add any comparison you run, e.g. random forest or gradient boosting.]

## 4. Evaluation
Stratified 80/20 held-out split and 5-fold cross-validation. Report accuracy, precision, recall, F1 and ROC-AUC
for all four targets from `model/metrics.json`.

| Target | Acc | Prec | Recall | F1 | AUC |
|---|---|---|---|---|---|
| CAD | [ ] | [ ] | [ ] | [ ] | [ ] |
| LAD | [ ] | [ ] | [ ] | [ ] | [ ] |
| LCX | [ ] | [ ] | [ ] | [ ] | [ ] |
| RCA | [ ] | [ ] | [ ] | [ ] | [ ] |

## 5. 3D pipeline
Three.js r128 via cdnjs. Torso and lungs are translucent shells; the heart is a deformed sphere; each artery is a
`TubeGeometry` along a `CatmullRomCurve3` on the heart surface. Probability to colour: `hsl((1 - p) * 120, 85%, 50%)`.
Rotation uses pointer drag, zoom uses wheel or pinch, selection uses a raycaster. Rendering is
lightweight: low polygon counts, pixel ratio capped at 1.5.

## 6. Clinical dashboard
Four probability cards (CAD, LAD, LCX, RCA), a SHAP explanation for the selected target showing each measurement's
value and share of total attribution, and the metrics table. Visible safety disclaimer.

## 7. Usage
See the README: run `web/index.html`, adjust patient data or load an example, rotate and click vessels, optionally
retrain on the real CSV.

## 8. Limitations and future work
Small single-centre dataset; no external validation; linear models; schematic anatomy. Next: calibrated non-linear
models with TreeSHAP, a BodyParts3D mesh, uncertainty intervals, and more clinical features.
