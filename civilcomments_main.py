import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np, os



CSV = "data/all_data_with_identities.csv"
df = pd.read_csv(CSV)


TEXT_CANDIDATES  = ["comment_text","text","Text"]
LABEL_CANDIDATES = ["toxicity","target","label"]
IDENTITY_CANDIDATES = [
    "male","female",
    "black","white","asian","hispanic","other_race_or_ethnicity",
    "christian","jewish","muslim","atheist","buddhist","hindu"
]

TEXT  = next(c for c in TEXT_CANDIDATES  if c in df.columns)
LABEL = next(c for c in LABEL_CANDIDATES if c in df.columns)
ID_COLS = [c for c in IDENTITY_CANDIDATES if c in df.columns]

print("Using TEXT:",TEXT,"LABEL:",LABEL,"ID_COLS:",ID_COLS[:8],"...")

df = df.dropna(subset=[TEXT, LABEL]).copy()
if df[LABEL].dtype.kind in "fc":
    df[LABEL] = (df[LABEL] >= 0.5).astype(int)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL])

vec = TfidfVectorizer(min_df=5, ngram_range=(1,2), max_features=200_000)
Xtr = vec.fit_transform(train_df[TEXT]); ytr = train_df[LABEL].values
clf = LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)

Xv = vec.transform(val_df[TEXT]); yv = val_df[LABEL].values
probs = clf.predict_proba(Xv)[:,1]; preds = (probs >= 0.5).astype(int)

print("Overall AUC:", round(roc_auc_score(yv, probs), 4))
print("Overall F1 :", round(f1_score(yv, preds), 4))

os.makedirs("out", exist_ok=True)

THRESH = 0.6
preds = (probs >= THRESH).astype(int)

def subgroup_mask(frame, col, thr=0.5):
    return (frame[col] >= thr).fillna(False) if col in frame.columns else pd.Series(False, index=frame.index)

rows=[]
for c in ID_COLS:
    m = subgroup_mask(val_df, c)
    if m.sum() < 150:  # skip tiny groups for stability
        continue
    y = yv; p = probs; pr = preds

    try:
        auc_sub = roc_auc_score(y[m], p[m]) if len(np.unique(y[m]))>1 else np.nan
    except:
        auc_sub = np.nan

    bpsn = ((~m) & (y==1)) | (m & (y==0))
    try:
        auc_bpsn = roc_auc_score(y[bpsn], p[bpsn]) if len(np.unique(y[bpsn]))>1 else np.nan
    except:
        auc_bpsn = np.nan

    bnsp = ((~m) & (y==0)) | (m & (y==1))
    try:
        auc_bnsp = roc_auc_score(y[bnsp], p[bnsp]) if len(np.unique(y[bnsp]))>1 else np.nan
    except:
        auc_bnsp = np.nan

    def rate(mask, cond):
        den = max(1,(cond & mask).sum())
        return ((pr==1) & cond & mask).sum()/den
    tpr_g = rate(m, (y==1)); tpr_bg = rate(~m, (y==1))
    fpr_g = rate(m, (y==0)); fpr_bg = rate(~m, (y==0))

    rows.append({
        "group": c,
        "support": int(m.sum()),
        "Subgroup AUC": round(float(auc_sub), 4) if pd.notnull(auc_sub) else np.nan,
        "BPSN AUC": round(float(auc_bpsn), 4) if pd.notnull(auc_bpsn) else np.nan,
        "BNSP AUC": round(float(auc_bnsp), 4) if pd.notnull(auc_bnsp) else np.nan,
        "TPR gap": round(tpr_g - tpr_bg, 4),
        "FPR gap": round(fpr_g - fpr_bg, 4),
    })

report = pd.DataFrame(rows).sort_values("Subgroup AUC", ascending=True)
print("\n=== Bias by group (validation) @ threshold", THRESH, "===")
print(report.to_string(index=False))
report.to_csv("out/baseline_bias_report.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("out/baseline_bias_report.csv")

race_cols = ["black","white","asian","hispanic","other_race_or_ethnicity"]
gender_cols = ["male","female"]
religion_cols = ["christian","jewish","muslim","atheist","buddhist","hindu"]

def make_axis_charts(df_axis, axis_name, out_prefix):
    # 1) Extra False Alarms (FPR gap)
    d1 = df_axis.sort_values("FPR gap", ascending=False)
    plt.figure(figsize=(8,4.5))
    plt.bar(d1["group"], d1["FPR gap"])
    plt.title(f"False Positive Alarms — {axis_name}")
    plt.xlabel("Group")
    plt.ylabel("FPR")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_extra_false_alarms.png", dpi=160, bbox_inches="tight")
    plt.show()

    d2 = df_axis.sort_values("Subgroup AUC", ascending=True)
    plt.figure(figsize=(8,4.5))
    plt.bar(d2["group"], d2["Subgroup AUC"])
    plt.title(f"How well the model ranks within each group (higher is better) — {axis_name}")
    plt.xlabel("Group")
    plt.ylabel("ROC-AUC (within group)")
    plt.ylim(0.6, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_subgroup_auc.png", dpi=160, bbox_inches="tight")
    plt.show()

race_df = df[df["group"].isin(race_cols)]
gender_df = df[df["group"].isin(gender_cols)]
religion_df = df[df["group"].isin(religion_cols)]

make_axis_charts(race_df, "Race/Ethnicity", "race")
make_axis_charts(gender_df, "Gender", "gender")
make_axis_charts(religion_df, "Religion", "religion")


import joblib, os
os.makedirs("out", exist_ok=True)
joblib.dump(vec, "out/vec.joblib"); joblib.dump(clf, "out/clf.joblib")

