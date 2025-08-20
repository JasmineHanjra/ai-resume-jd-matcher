# scripts/build_jd_index.py
import pathlib, re, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

jd_path = DATA / "jobs_clean.csv"
out_path = DATA / "jd_index.csv"

PER_CATEGORY = 200       # target per category
MIN_LEN = 200            # primary length threshold
FALLBACK_MIN = 80        # relaxed threshold if too few

def clean_len(s: str) -> int:
    return len(str(s))

df = pd.read_csv(jd_path)

need = {"jd_text","jd_category"}
if not need.issubset(df.columns):
    raise ValueError(f"jobs_clean.csv must have {need}; found {df.columns}")

# make stable id
if "Job Id" in df.columns:
    df["jd_id"] = df["Job Id"].astype(str)
else:
    df["jd_id"] = pd.util.hash_pandas_object(df["jd_text"], index=False).astype(str)

# basic cleanup
df = df.dropna(subset=["jd_text"]).drop_duplicates(subset=["jd_text"])
df["len"] = df["jd_text"].map(clean_len)

blocks = []
for cat, dcat in df.groupby("jd_category"):
    # try strict length
    cand = dcat[dcat["len"] >= MIN_LEN]
    if len(cand) < min(40, PER_CATEGORY//5):
        # relax if we have too few
        cand = dcat[dcat["len"] >= FALLBACK_MIN]
    if len(cand) == 0:
        # take top-N longest as last resort
        cand = dcat.sort_values("len", ascending=False).head(min(PER_CATEGORY, len(dcat)))
    # balanced sample
    take = cand.sample(n=min(PER_CATEGORY, len(cand)), random_state=42)
    blocks.append(take)

small = pd.concat(blocks, ignore_index=True)
keep_cols = ["jd_id","jd_text","jd_title","jd_role","jd_category"]
small = small[keep_cols]

small.to_csv(out_path, index=False, encoding="utf-8")
print("[OK] Wrote", out_path, "rows=", len(small), "categories=", small["jd_category"].nunique())
print("Per-category counts:")
print(small["jd_category"].value_counts().to_string())
