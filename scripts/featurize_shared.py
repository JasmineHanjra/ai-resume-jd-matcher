# ─────────────────────────────────────────────────────────────────────────────
# Project: AI Resume ↔ JD Matcher
# File: scripts/featurize_shared.py
# Author: Jasmine Kaur Hanjra
# Created: Aug 2025
# Purpose:
#   Shared feature builders used by training and the Streamlit app:
#   - clean_text: HTML/whitespace/charset normalization
#   - extract_skills: lexicon-based skill hits
#   - tfidf_cos: lexical similarity
#   - sbert_cos: semantic cosine (SBERT, CPU-only; safe fallback to None)
#   - build_features: assembles model features + metadata
# Contracts:
#   return (X, resume_skills, jd_skills, meta) where:
#     X -> pandas.DataFrame with columns [sim_tfidf, sim_sbert, overlap, len_ratio, rs_cnt, js_cnt]
#     meta -> dict mirrors the numeric features for UI/debug
# Future work:
#   - Add domain-adaptive embeddings per JD category.
#   - Plug in phrase-level overlap (e.g., noun chunks) and weighted skill sets.
# ─────────────────────────────────────────────────────────────────────────────
import json, pathlib, re
from typing import Optional, Tuple, Dict, Set
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------- utils ----------
def clean_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"[^A-Za-z0-9 +#\\-_/]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip().lower()
    return s

def load_skills() -> Set[str]:
    skills = json.loads((ROOT / "skills.json").read_text(encoding="utf-8"))
    return {s.lower() for s in skills}

SKILLS = load_skills()

def extract_skills(t: str, extra: Optional[Set[str]] = None) -> Set[str]:
    vocab = set(SKILLS)
    if extra: vocab |= {e.lower() for e in extra}
    t = " " + (t or "").lower() + " "
    return {s for s in vocab if (" " + s + " ") in t}

# ---------- features ----------
def tfidf_cos(a: str, b: str) -> float:
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1, max_df=1.0)
    X = v.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])

_SBERT_MODEL = None
def sbert_cos(a: str, b: str):
    """Return SBERT cosine similarity, or None if SBERT isn't available.
    We force CPU + disable low-memory init to avoid meta-tensor errors on Windows."""
    global _SBERT_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        if _SBERT_MODEL is None:
            # Force CPU and disable low_cpu_mem_usage to avoid meta tensors
            _SBERT_MODEL = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False, "device_map": None},
            )
        emb = _SBERT_MODEL.encode([a, b], normalize_embeddings=True, convert_to_numpy=True)
        # cosine since they’re L2-normalized
        return float(np.dot(emb[0], emb[1]))
    except Exception:
        # Any issue -> gracefully disable semantic feature
        return None

def build_features(resume_text: str, jd_text: str, profile_mode: bool = False,
                   extra_jd_tokens: Optional[Set[str]] = None
) -> Tuple[pd.DataFrame, Set[str], Set[str], Dict[str, float]]:
    a, b = clean_text(resume_text), clean_text(jd_text)
    sim_tfidf = tfidf_cos(a, b)
    sim_sbert = sbert_cos(a, b)
    rs = extract_skills(a, extra_jd_tokens)
    js = extract_skills(b, extra_jd_tokens)
    overlap = len(rs & js) / (len(js) or 1)
    # avoid weird extremes
    len_ratio = 1.0 if profile_mode else (min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) else 1.0)
    len_ratio = max(0.3, min(1.0, len_ratio))

    cols = ["sim_tfidf", "sim_sbert", "overlap", "len_ratio", "rs_cnt", "js_cnt"]
    row = [
        sim_tfidf,
        (sim_sbert if sim_sbert is not None else 0.0),
        overlap,
        len_ratio,
        len(rs),
        len(js),
    ]
    X = pd.DataFrame([row], columns=cols)
    meta = {"sim_tfidf": sim_tfidf, "sim_sbert": (sim_sbert if sim_sbert is not None else -1.0),
            "overlap": overlap, "len_ratio": len_ratio, "rs_cnt": len(rs), "js_cnt": len(js)}
    return X, rs, js, meta
