# retrain_after_cda.py
import pandas as pd, numpy as np, os, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

TRAIN_CSV = "out/civilcomments_train_with_cda.csv"
VAL_CSV   = "civil_comments_validation.csv"

TEXT_CANDS  = ["comment_text","text","Text"]
LABEL_CANDS = ["toxicity","target","label"]
IDENTITY_CANDS = [
    "male","female","black","white","asian","hispanic","other_race_or_ethnicity",
    "christian","jewish","muslim","atheist","buddhist","hindu"
]
THRESH = 0.6

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

TEXT  = next(c for c in TEXT_CANDS  if c in train_df.columns and c in val_df.columns)
LABEL = next(c for c in LABEL_CANDS if c in train_df.columns and c in val_df.columns)
ID_COLS = [c for c in IDENTITY_CANDS if c in val_df.columns]

for df in (train_df, val_df):
    df.dropna(subset=[TEXT, LABEL], inplace=True)
    if df[LABEL].dtype.kind in "fc":
        df[LABEL] = (df[LABEL] >= 0.5).astype(int)

vec = TfidfVectorizer(min_df=5, ngram_range=(1,2), max_features=200_000)
Xtr = vec.fit_transform(train_df[TEXT]); ytr = train_df[LABEL].values
clf = LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)

os.makedirs("out", exist_ok=True)
joblib.dump(vec, "out/vec_after_cda.joblib")
joblib.dump(clf, "out/clf_after_cda.joblib")

Xv = vec.transform(val_df[TEXT]); yv = val_df[LABEL].values
probs = clf.predict_proba(Xv)[:,1]; preds = (probs >= THRESH).astype(int)
print("After-CDA AUC:", round(roc_auc_score(yv, probs), 4))
print("After-CDA F1 :", round(f1_score(yv, preds), 4))

def subgroup_mask(frame, col, thr=0.5):
    return (frame[col] >= thr).fillna(False) if col in frame.columns else pd.Series(False, index=frame.index)

rows=[]
for c in ID_COLS:
    m = subgroup_mask(val_df, c)
    if m.sum() < 150:
        continue
    y, p, pr = yv, probs, preds

    def auc_safe(mask):
        u = np.unique(y[mask])
        if len(u) < 2:
            return np.nan
        return roc_auc_score(y[mask], p[mask])

    auc_sub  = auc_safe(m)
    bpsn = ((~m) & (y==1)) | (m & (y==0))
    auc_bpsn = auc_safe(bpsn)
    bnsp = ((~m) & (y==0)) | (m & (y==1))
    auc_bnsp = auc_safe(bnsp)

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

rep = pd.DataFrame(rows).sort_values("Subgroup AUC", ascending=True)
print("\n=== After-CDA Bias by group (validation) @ threshold", THRESH, "===")
print(rep.to_string(index=False))
rep.to_csv("out/baseline_bias_report_after_cda.csv", index=False)
print("Wrote: out/baseline_bias_report_after_cda.csv")
