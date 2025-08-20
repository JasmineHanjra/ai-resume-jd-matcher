# scripts/clean_data.py
import pandas as pd, pathlib, re, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"<[^>]+>", " ", s)                 # strip HTML
    s = re.sub(r"[^A-Za-z0-9 +#\-_/]", " ", s)     # keep tech chars
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# ---------- resumes: Resume.csv -> resumes_clean.csv ----------
resumes_path = DATA / "Resume.csv"  # adjust if your file is named differently
resumes = pd.read_csv(resumes_path, encoding="utf-8", engine="python")

resumes.columns = [c.strip() for c in resumes.columns]
needed = [c for c in ["ID", "Resume_str", "Category"] if c in resumes.columns]
assert {"Resume_str", "Category"}.issubset(set(needed)), \
    f"Resume.csv must have 'Resume_str' and 'Category'. Found: {resumes.columns}"

resumes = resumes[needed].dropna(subset=["Resume_str", "Category"])
resumes["resume_text"] = resumes["Resume_str"].apply(clean_text)
resumes_clean = resumes[["resume_text", "Category"]].copy()
resumes_clean.to_csv(DATA / "resumes_clean.csv", index=False, encoding="utf-8")
print(f"[OK] Wrote {DATA/'resumes_clean.csv'}  rows={len(resumes_clean)}")

# -------- jobs.csv -> jobs_clean.csv (handles big file) --------
import re

jobs_source = DATA / ("jobs_small.csv" if (DATA / "jobs_small.csv").exists() else "job_descriptions.csv")

def role_to_category(title: str, role: str, text: str) -> str:
    s = f"{title or ''} {role or ''} {text or ''}".lower()
    pats = [
        (r"\b(data\s*science|data\s*scientist|machine\s*learning|ml|deep\s*learning|pytorch|tensorflow|sklearn|statistics|analytics)\b", "Information-Technology"),
        (r"\b(software\s*engineer|backend|frontend|full\s*stack|java(script)?|spring|react|node|c\+\+|c#)\b", "Information-Technology"),
        (r"\b(aws|azure|gcp|cloud|devops|docker|kubernetes|terraform|ci/cd|cicd|sre)\b", "Information-Technology"),
        (r"\b(finance|financial|accountant|accounting|bank|treasury|audit|fp&a)\b", "Finance"),
        (r"\b(engineer|mechanical|electrical|civil|manufacturing|process|quality)\b", "Engineering"),
        (r"\b(hr|human resources|recruiter|talent acquisition|people operations)\b", "HR"),
        (r"\b(sales|business development|account executive|partnerships)\b", "Sales"),
        (r"\b(healthcare|nurse|medical|clinic|therapist|physician|pharma)\b", "Healthcare"),
        (r"\b(design|designer|ui|ux|graphic)\b", "Designer"),
        (r"\b(teacher|instructor|education|educator)\b", "Teacher"),
        (r"\b(consultant|consulting|advisory|strategy)\b", "Consultant"),
        (r"\b(digital marketing|social media|content|seo|sem|ppc|campaign)\b", "Digital-Media"),
    ]
    for p, cat in pats:
        if re.search(p, s):
            return cat
    return "Information-Technology"

def clean_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"[^A-Za-z0-9 +#\\-_/]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip().lower()
    return s

def clean_jobs_chunk(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # normalize columns if present
    keep = [c for c in ["Job Id","Job Title","Role","Job Description","jd_text","jd_title","jd_role","jd_category"] if c in df.columns]
    df = df[keep].copy()

    # unify text columns
    if "jd_text" not in df.columns:
        if "Job Description" in df.columns:
            df["jd_text"] = df["Job Description"].astype(str)
        else:
            raise ValueError("Need 'jd_text' or 'Job Description'.")

    df["jd_text"]  = df["jd_text"].astype(str).map(clean_text)
    df["jd_title"] = df.get("jd_title", df.get("Job Title", "")).astype(str).str.lower()
    df["jd_role"]  = df.get("jd_role",  df.get("Role", "")).astype(str).str.lower()

    # category: preserve if present, else infer
    if "jd_category" not in df.columns or df["jd_category"].isna().all():
        df["jd_category"] = df.apply(lambda r: role_to_category(r.get("jd_title",""), r.get("jd_role",""), r.get("jd_text","")), axis=1)

    # drop empties
    df = df[df["jd_text"].str.len() > 50]
    # output shape
    out_cols = ["Job Id","jd_title","jd_role","jd_text","jd_category"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""
    return df[out_cols]

out_path = DATA / "jobs_clean.csv"
first = True
for chunk in pd.read_csv(jobs_source, chunksize=20000, encoding="utf-8", engine="python"):
    cleaned = clean_jobs_chunk(chunk)
    cleaned.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
    first = False

print(f"[OK] Wrote {out_path} (source: {jobs_source.name})")
