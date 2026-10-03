"""Train CAD + LAD/LCX/RCA stenosis classifiers on the Z-Alizadeh Sani dataset.

Usage:
    python pipeline/train.py --csv data/z_alizadeh_sani.csv [--features web|all]

Targets : Cath (overall CAD), LAD, LCX, RCA
Leakage : the four target columns are always removed from the inputs.
Outputs : model/weights.json (coefficients + scaler) and model/metrics.json
          (held-out 20% and 5-fold cross-validated metrics).
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

TARGETS = {"cad": "cath", "lad": "lad", "lcx": "lcx", "rca": "rca"}
WEB = ["age", "sex", "bp", "pr", "typicalchestpain", "dm", "htn", "ldl", "hdl", "tg",
       "fbs", "stelevation", "tinversion", "eftte", "regionrwma", "bmi"]  # the 16 web-app inputs
POS = {"cad", "stenotic", "yes", "y", "positive", "male"}
NEG = {"fmale", "female", "normal", "n", "no", "negative"}

def norm(c): return re.sub(r"[^a-z0-9]", "", str(c).lower())

def to_num(s):
    def f(v):
        t = str(v).strip().lower()
        if t in POS: return 1.0
        if t in NEG: return 0.0
        try: return float(t)
        except ValueError: return np.nan
    return s.map(f)

def metrics(y, p, s):
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p), f1=f1_score(y, p), roc_auc=roc_auc_score(y, s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", choices=["web", "all"], default="web")
    ap.add_argument("--out", default="model")
    a = ap.parse_args()

    df = pd.read_csv(a.csv); df.columns = [norm(c) for c in df.columns]
    missing = [t for t in TARGETS.values() if t not in df.columns]
    if missing: raise SystemExit(f"Missing target columns: {missing}")
    Y = {k: (to_num(df[c]) == 1).astype(int) for k, c in TARGETS.items()}
    drop = set(TARGETS.values())                      # target-leakage guard
    cols = [c for c in (WEB if a.features == "web" else df.columns) if c in df.columns and c not in drop]
    X = df[cols].apply(to_num); X = X.fillna(X.median())
    print(f"{len(df)} patients, {len(cols)} features ({a.features})")

    out, weights = {}, {"features": cols}
    for k in TARGETS:
        y = Y[k]
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
        pipe.fit(Xtr, ytr)
        held = metrics(yte, pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1])
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        sc = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        cvm = metrics(y, (sc >= .5).astype(int), sc)
        out[k] = {"held_out": held, "cv5": cvm, "positives": int(y.sum())}
        pipe.fit(X, y)                                 # final model on all data
        sc_, lr = pipe[0], pipe[1]
        weights[k] = dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                          mean=sc_.mean_.tolist(), scale=sc_.scale_.tolist())
        print(f"{k.upper():4s} held-out AUC {held['roc_auc']:.3f} F1 {held['f1']:.3f} | 5-fold AUC {cvm['roc_auc']:.3f}")

    Path(a.out).mkdir(exist_ok=True)
    json.dump(weights, open(Path(a.out) / "weights.json", "w"), indent=1)
    json.dump(out, open(Path(a.out) / "metrics.json", "w"), indent=1)
    print("saved", a.out)

if __name__ == "__main__": main()
