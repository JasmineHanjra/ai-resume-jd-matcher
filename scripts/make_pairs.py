# scripts/make_pairs.py
import pathlib, re
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
rng = np.random.default_rng(42)

# Fallback title/role→category mapping if jd_category is missing
ROLE2CAT = [
    (r"\b(data|ml|machine learning|ai|software|developer|engineer|it|programmer|analyst)\b", "Information-Technology"),
    (r"\b(finance|financial analyst|accountant|accounting|bank|banking|investment|treasury|audit|fp&a)\b", "Finance"),
    (r"\b(engineer|mechanical|electrical|civil|manufacturing|process|quality)\b", "Engineering"),
    (r"\b(hr|human resources|recruiter|talent acquisition|people operations)\b", "HR"),
    (r"\b(sales|business development|bd|account executive|account manager|partnerships)\b", "Business-Development"),
    (r"\b(teacher|instructor|professor|education|educator)\b", "Teacher"),
    (r"\b(healthcare|nurse|medical|clinic|therapist|physician|pharma)\b", "Healthcare"),
    (r"\b(design|designer|ui|ux|graphic|product designer|interaction)\b", "Designer"),
    (r"\b(consultant|consulting|advisory|strategy)\b", "Consultant"),
    (r"\b(digital marketing|social media|content|seo|sem|ppc|campaign)\b", "Digital-Media"),
]

def role_to_category(title: str, role: str) -> str:
    text = f"{title or ''} {role or ''}".lower()
    for patt, cat in ROLE2CAT:
        if re.search(patt, text):
            return cat
    return "Other"

# ---- Load cleaned data ----
resumes = pd.read_csv(DATA / "resumes_clean.csv")  # resume_text, Category
jobs    = pd.read_csv(DATA / "jobs_clean.csv")     # jd_title, jd_role, jd_text, (optional) Job Id, jd_category

# Prefer provided jd_category; else infer from title/role
if "jd_category" not in jobs.columns:
    jobs["jd_category"] = jobs.apply(lambda r: role_to_category(r.get("jd_title",""), r.get("jd_role","")), axis=1)

# Drop unknown category & empty jd; deduplicate JDs to improve diversity
jobs = jobs.dropna(subset=["jd_text"])
jobs = jobs[jobs["jd_category"] != "Other"].copy()
if "Job Id" in jobs.columns:
    jobs["jd_id"] = jobs["Job Id"].astype(str)
else:
    jobs["jd_id"] = pd.util.hash_pandas_object(jobs["jd_text"], index=False).astype(str)
jobs = jobs.drop_duplicates(subset=["jd_text"]).reset_index(drop=True)

# Build per-category pools (indices shuffled once)
pools = {}
for cat, dfc in jobs.groupby("jd_category"):
    idx = dfc.index.to_numpy()
    rng.shuffle(idx)
    pools[cat] = list(idx)

valid_cats = set(pools.keys())
# Keep only resumes whose category is covered by the JD pools
resumes = resumes[resumes["Category"].isin(valid_cats)].reset_index(drop=True)

def draw_from_pool(cat, k):
    if cat not in pools or not pools[cat]:
        return []
    picked = []
    for _ in range(k):
        i = pools[cat].pop(0)
        picked.append(i)
        pools[cat].append(i)
    return picked

# knobs: per-resume samples (ensures at least 1 pos and 1 neg)
k_pos, k_neg = 3, 3

rows = []
for _, rr in resumes.iterrows():
    r_txt = rr["resume_text"]
    r_cat = rr["Category"]

    # positives
    pos_idx = draw_from_pool(r_cat, min(k_pos, len(pools.get(r_cat, [])) or 0))

    # negatives (pick categories != r_cat)
    other_cats = [c for c in valid_cats if c != r_cat and pools.get(c)]
    if len(other_cats) == 0:
        continue
    choose_k = min(k_neg, len(other_cats))
    neg_cats = rng.choice(other_cats, size=choose_k, replace=False)
    neg_idx = []
    for c in neg_cats:
        neg_idx.extend(draw_from_pool(c, 1))

    # REQUIRE at least 1 positive and 1 negative to avoid label collapse
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        continue

    for j in pos_idx:
        jrow = jobs.loc[j]
        rows.append({"label": 1, "resume_text": r_txt, "jd_text": jrow["jd_text"], "jd_id": jrow["jd_id"]})
    for j in neg_idx:
        jrow = jobs.loc[j]
        rows.append({"label": 0, "resume_text": r_txt, "jd_text": jrow["jd_text"], "jd_id": jrow["jd_id"]})

pairs = pd.DataFrame(rows).drop_duplicates()

# ---- Balance pos/neg to ~1:1 ----
pos = pairs[pairs["label"] == 1]
neg = pairs[pairs["label"] == 0]

if len(pos) == 0 or len(neg) == 0:
    raise ValueError(f"After filtering, pos={len(pos)}, neg={len(neg)}; not enough diversity to train.")

if len(pos) < len(neg):
    # upsample positives
    pos_up = pos.sample(n=len(neg), replace=True, random_state=42)
    pairs_bal = pd.concat([neg, pos_up], ignore_index=True)
else:
    # downsample negatives
    neg_dn = neg.sample(n=len(pos), replace=False, random_state=42)
    pairs_bal = pd.concat([pos, neg_dn], ignore_index=True)

# Shuffle for good measure
pairs_bal = pairs_bal.sample(frac=1.0, random_state=42).reset_index(drop=True)

out = DATA / "pairs.csv"
pairs_bal.to_csv(out, index=False, encoding="utf-8")

print(f"[OK] Wrote {out} rows={len(pairs_bal)} | pos={ (pairs_bal['label']==1).sum() } | neg={ (pairs_bal['label']==0).sum() } | unique JDs={ pairs_bal['jd_id'].nunique() } | resumes={ pairs_bal['resume_text'].nunique() }")
