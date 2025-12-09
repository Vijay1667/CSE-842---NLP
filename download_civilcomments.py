import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "data/all_data_with_identities.csv"

TEXT_CANDS  = ["comment_text","text","Text"]
LABEL_CANDS = ["toxicity","target","label"]

df = pd.read_csv(SRC)

TEXT  = next(c for c in TEXT_CANDS  if c in df.columns)
LABEL = next(c for c in LABEL_CANDS if c in df.columns)

df = df.dropna(subset=[TEXT, LABEL]).copy()
if df[LABEL].dtype.kind in "fc":
    df[LABEL] = (df[LABEL] >= 0.5).astype(int)

train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df[LABEL], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df[LABEL], random_state=42
)

train_df.to_csv("civil_comments_train.csv", index=False)
val_df.to_csv("civil_comments_validation.csv", index=False)
test_df.to_csv("civil_comments_test.csv", index=False)

print(
    "Wrote civil_comments_train.csv / civil_comments_validation.csv / civil_comments_test.csv "
    f"({len(train_df)} train, {len(val_df)} val, {len(test_df)} test)"
)