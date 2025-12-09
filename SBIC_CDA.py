import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split


trn = pd.read_csv("SBIC.v2.trn.csv")
dev = pd.read_csv("SBIC.v2.dev.csv")
tst = pd.read_csv("SBIC.v2.tst.csv")

df = pd.concat([trn, dev, tst], ignore_index=True)



texts = df["post"].astype(str).str.lower()

labels = df["offensiveYN"].apply(lambda x: 1 if x >= 0.5 else 0)



GROUP_KEYWORDS = {
    "women": ["women", "woman", "female", "girl", "girls", "she", "her"],
    "men": ["men", "man", "male", "boy", "boys", "he", "him"],
    "black": ["black", "african american", "blacks"],
    "white": ["white", "caucasian"],
    "asian": ["asian", "chinese", "japanese", "korean", "thai"],
    "latino": ["latino", "hispanic", "mexican"],
    "immigrant": ["immigrant", "immigrants"],
    "muslim": ["muslim", "islam", "arab"],
    "jewish": ["jew", "jewish", "hebrew"],
    "lgbt": ["lgbt", "gay", "lesbian", "trans", "queer"],
    "disability": ["disabled", "disability", "handicapped"]
}

def detect_groups(text):
    flags = {}
    for group, keywords in GROUP_KEYWORDS.items():
        flags[group] = int(any(k in text for k in keywords))
    return flags

group_df = texts.apply(detect_groups)
group_df = pd.DataFrame(list(group_df))



X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    texts, labels, group_df, test_size=0.2, random_state=42, stratify=labels
)



tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,
    min_df=3
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)



clf = LogisticRegression(
    max_iter=2000,
    C=2.0,
    class_weight="balanced"
)

clf.fit(X_train_tfidf, y_train)

y_prob = clf.predict_proba(X_test_tfidf)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)



def compute_group_metrics(y_true, y_pred, y_prob, mask):
    idx = mask == 1

    if idx.sum() < 30:
        return None

    y_g = y_true[idx]
    y_hat_g = y_pred[idx]
    prob_g = y_prob[idx]

    tn, fp, fn, tp = confusion_matrix(y_g, y_hat_g, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn + 1e-6)
    tpr = tp / (tp + fn + 1e-6)
    fnr = 1 - tpr

    try:
        auc = roc_auc_score(y_g, prob_g)
    except:
        auc = np.nan

    return {
        "samples": int(idx.sum()),
        "FPR": round(fpr, 4),
        "TPR": round(tpr, 4),
        "FNR": round(fnr, 4),
        "Subgroup AUC": round(auc, 4)
    }

results = {}
for group in group_df.columns:
    metrics = compute_group_metrics(
        y_test.values,
        y_pred,
        y_prob,
        g_test[group].values
    )
    if metrics:
        results[group] = metrics

results_df = pd.DataFrame(results).T
print("\n=== FINAL SUBGROUP FAIRNESS METRICS (KEYWORD MATCHED) ===\n")
print(results_df)
