# plot_fairness_before_after.py
import pandas as pd
import matplotlib.pyplot as plt
import os

BEFORE = "out/baseline_bias_report.csv"
AFTER  = "out/baseline_bias_report_after_cda.csv"

b = pd.read_csv(BEFORE)
a = pd.read_csv(AFTER)

common = set(b["group"]).intersection(set(a["group"]))
b = b[b["group"].isin(common)].copy()
a = a[a["group"].isin(common)].copy()

race_groups    = ["black","white","asian","hispanic","other_race_or_ethnicity"]
gender_groups  = ["male","female"]
relig_groups   = ["muslim","christian","jewish","atheist","buddhist","hindu"]

axes = [
    ("Race/Ethnicity", race_groups, "race"),
    ("Gender",         gender_groups, "gender"),
    ("Religion",       relig_groups, "religion"),
]

os.makedirs("out", exist_ok=True)

def prep_axis(df, groups):
    df2 = df[df["group"].isin(groups)].copy()
    df2["order"] = df2["group"].apply(lambda g: groups.index(g) if g in groups else 999)
    df2 = df2.sort_values("order")
    return df2

def bar_pair(groups, metric, title, fname):
    bb = prep_axis(b, groups)
    aa = prep_axis(a, groups)
    if bb.empty or aa.empty:
        print(f"[skip] No overlap for {title} – check group names in CSVs.")
        return
    labels = bb["group"].tolist()
    x = range(len(labels))

    w = 0.38
    plt.figure(figsize=(9,4.8))
    plt.bar([i - w/2 for i in x], bb[metric], width=w, label="Before")
    plt.bar([i + w/2 for i in x], aa[metric], width=w, label="After CDA")
    plt.title(f"{title} — {metric}")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"out/{fname}_{metric.replace(' ','_').lower()}.png", dpi=160)
    plt.close()

def print_delta(groups, metric, title):
    bb = prep_axis(b, groups)[["group", metric]].rename(columns={metric: "before"})
    aa = prep_axis(a, groups)[["group", metric]].rename(columns={metric: "after"})
    m  = bb.merge(aa, on="group")
    m["delta"] = m["after"] - m["before"]
    print(f"\n=== {title}: Δ(after - before) for {metric} ===")
    print(m[["group","before","after","delta"]].to_string(index=False))

for title, groups, tag in axes:
    bar_pair(groups, "FPR gap", f"{title} — Extra false flags", f"{tag}_extra_false_flags")
    print_delta(groups, "FPR gap", f"{title} — Extra false flags")

    bar_pair(groups, "Subgroup AUC", f"{title} — Ranking quality (within group)", f"{tag}_ranking_quality")
    print_delta(groups, "Subgroup AUC", f"{title} — Ranking quality (within group)")

    bar_pair(groups, "BPSN AUC", f"{title} — Robust to non-toxic mentions", f"{tag}_robust_nontoxic")
    print_delta(groups, "BPSN AUC", f"{title} — Robust to non-toxic mentions")

    bar_pair(groups, "BNSP AUC", f"{title} — Robust to toxic mentions", f"{tag}_robust_toxic")
    print_delta(groups, "BNSP AUC", f"{title} — Robust to toxic mentions")

print("\nSaved charts to the out/ folder:")
print(" - *_fpr_gap.png  (extra false alarms)")
print(" - *_subgroup_auc.png")
print(" - *_bpsn_auc.png")
print(" - *_bnsp_auc.png")
