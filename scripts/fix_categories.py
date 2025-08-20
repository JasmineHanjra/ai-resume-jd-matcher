# ─────────────────────────────────────────────────────────────────────────────
# Project: AI Resume ↔ JD Matcher
# File: scripts/fix_categories.py
# Author: Jasmine Kaur Hanjra
# Created: Aug 2025
# Purpose:
#   Heuristically relabel jd_category from jd_title/jd_role/jd_text using regex rules.
#   Overwrites data/jobs_clean.csv in-place.
# Caveats:
#   - Rule-based; imperfect. Prefer ML classifier if you later curate labels.
# Future work:
#   - Collect labeled titles → train a lightweight title→category classifier.
#   - Track coverage/precision per rule.
# ─────────────────────────────────────────────────────────────────────────────

import re, pathlib, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
src = DATA / "jobs_clean.csv"
dst = DATA / "jobs_clean.csv"  # overwrite in place

df = pd.read_csv(src)

def textbag(r):
    return " ".join(str(r.get(c, "")) for c in ["jd_title","jd_role","jd_text"]).lower()

RULES = [
    ("Healthcare", r"\b(healthcare|medical|clinic|hospital|patient|vital|ehr|epic|cpt|icd[- ]?10|billing|coding|triage|phlebotomy|rma|cmaa|ccma|medication|provider)\b"),
    ("Finance", r"\b(finance|financial|account(ant|ing)?|audit|treasury|budget|payroll|tax|cpa)\b"),
    ("Information-Technology", r"\b(software|developer|engineer|qa|tester|it|network|systems?|admin(istrator)?|cloud|devops|aws|azure|gcp|kubernetes|docker|sql|python|java(script)?|react|node|spring|linux)\b"),
    ("Engineering", r"\b(mechanical|electrical|civil|manufacturing|process|quality engineer|cad|autocad|solidworks)\b"),
    ("Sales", r"\b(sales|account executive|account manager|inside sales|business development)\b"),
    ("Business-Development", r"\b(business development|partnerships)\b"),
    ("HR", r"\b(human resources|hr|recruiter|talent acquisition|people operations)\b"),
    ("Designer", r"\b(ux|ui|product design|graphic design|figma|sketch|adobe|illustrator|photoshop)\b"),
    ("Teacher", r"\b(teacher|instructor|tutor|educator|curriculum)\b"),
    ("Consultant", r"\b(consultant|consulting|advisory|strategy)\b"),
    ("Digital-Media", r"\b(digital marketing|social media|content|seo|sem|ppc|campaign|copywriter)\b"),
]

def assign_cat(row):
    s = textbag(row)
    for cat, patt in RULES:
        if re.search(patt, s):
            return cat
    return "Other"

df["jd_category"] = df.apply(assign_cat, axis=1)

df.to_csv(dst, index=False)
print(f"[OK] Relabeled categories and wrote {dst}")
print("Counts:\n", df["jd_category"].value_counts().to_string())
