import pandas as pd, re, os

TRAIN_CSV = "civil_comments_train.csv"

TEXT_CANDS  = ["comment_text","text","Text"]
LABEL_CANDS = ["toxicity","target","label"]

df = pd.read_csv(TRAIN_CSV)
TEXT  = next(c for c in TEXT_CANDS  if c in df.columns)
LABEL = next(c for c in LABEL_CANDS if c in df.columns)

df = df.dropna(subset=[TEXT, LABEL]).copy()
if df[LABEL].dtype.kind in "fc":
    df[LABEL] = (df[LABEL] >= 0.5).astype(int)

SWAPS = [
    ("black","white"), ("white","black"),
    ("muslim","christian"), ("christian","muslim"),
    ("male","female"),("female","male")
]

BAD_TOKENS = {}
def safe(txt: str) -> bool:
    t = str(txt).lower()
    return not any(tok in t for tok in BAD_TOKENS)

def swap_once(text: str, a: str, b: str):
    pat = re.compile(rf"\b{re.escape(a)}\b", flags=re.IGNORECASE)
    if not pat.search(text):
        return None
    def _repl(m):
        return b if m.group(0).islower() else b.capitalize()
    return pat.sub(_repl, text)

aug_rows = []
for _, r in df.iterrows():
    t = r[TEXT]
    if not isinstance(t, str) or not safe(t):
        continue
    made = set()
    for a,b in SWAPS:
        out = swap_once(t, a, b)
        if out and out != t:
            made.add(out)
    for out_text in made:
        nr = r.copy()
        nr[TEXT] = out_text
        aug_rows.append(nr)

aug = pd.DataFrame(aug_rows)
os.makedirs("out", exist_ok=True)
train_cda = pd.concat([df, aug], ignore_index=True).sample(frac=1.0, random_state=42)
print("here")
train_cda.to_csv("out/civilcomments_train_with_cda.csv", index=False)

print(f"Original: {len(df):,}  Augmented: {len(aug):,}  Total: {len(train_cda):,}")
print("Wrote: out/civilcomments_train_with_cda.csv")
