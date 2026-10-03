"""Writes a small SYNTHETIC CSV with the Z-Alizadeh Sani column names, only for smoke-testing train.py."""
import numpy as np, pandas as pd
r = np.random.default_rng(0); n = 303
d = pd.DataFrame({"Age": r.normal(58, 10, n).round(), "Sex": r.choice(["Male", "Fmale"], n), "BP": r.normal(130, 18, n).round(),
  "PR": r.normal(75, 11, n).round(), "Typical Chest Pain": r.integers(0, 2, n), "DM": r.integers(0, 2, n), "HTN": r.integers(0, 2, n),
  "LDL": r.normal(110, 35, n).round(), "HDL": r.normal(45, 10, n).round(), "TG": r.normal(150, 70, n).round(), "FBS": r.normal(100, 35, n).round(),
  "St Elevation": r.integers(0, 2, n), "Tinversion": r.integers(0, 2, n), "EF-TTE": r.normal(55, 7, n).round(),
  "Region RWMA": r.integers(0, 5, n), "BMI": r.normal(27, 4, n).round(1)})
z = .05 * (d.Age - 58) + .8 * d["Typical Chest Pain"] + .6 * d.DM + .7 * d["St Elevation"] + .6 * d.Tinversion - .05 * (d["EF-TTE"] - 55)
for t, s in [("Cath", .8), ("LAD", .2), ("LCX", -.7), ("RCA", -.5)]:
    d[t] = (r.random(n) < 1 / (1 + np.exp(-(z + s + r.normal(0, .7, n))))).astype(int)
d["Cath"] = d["Cath"].map({1: "Cad", 0: "Normal"}); d.to_csv("demo.csv", index=False)
