# app_simple.py
# Upload a resume, filter job descriptions from your dataset, and see TOP matches.
# Uses a trained model + TF-IDF & SBERT similarity features (via scripts/featurize_shared.py).
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str((ROOT / "scripts").resolve()))

import io
import json
import pathlib
import re
import base64
from typing import Optional

import pandas as pd
import streamlit as st
from joblib import load

# ---- Our shared featurizer (clean_text + build_features: tfidf, sbert, overlaps) ----
# Make sure scripts/featurize_shared.py exists (I provided it earlier).
from featurize_shared import build_features, clean_text

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

# -------------------- Cached loaders --------------------
@st.cache_resource
def load_model():
    return load(DATA / "matcher_model.joblib")

@st.cache_data
def load_threshold():
    return json.loads((DATA / "threshold.json").read_text())["threshold"]

@st.cache_data
def load_jd_index():
    p = DATA / "jd_index.csv"
    if not p.exists():
        st.error("data/jd_index.csv not found. Run: python scripts/build_jd_index.py")
        return pd.DataFrame(columns=["jd_id","jd_text","jd_title","jd_role","jd_category"])
    df = pd.read_csv(p)
    # Normalize a bit
    for c in ["jd_text","jd_title","jd_role","jd_category"]:
        if c in df.columns: df[c] = df[c].fillna("").astype(str)
    return df

clf = load_model()
THRESH = load_threshold()
JDS = load_jd_index()
# --- JD keyword scoring (importance) ---
def jd_keyword_scores(jd_text: str) -> dict:
    """Return TF-IDF score per keyword for a JD (uses the same VEC)."""
    t = clean_text(jd_text)
    X = VEC.transform([t])
    feats = VEC.get_feature_names_out()
    arr = X.toarray()[0]
    return {feats[i]: float(arr[i]) for i in range(len(feats)) if arr[i] > 0}

# --- simple HTML highlighter ---
def highlight_terms(text: str, terms: list[str]) -> str:
    s = text
    for t in sorted(set(terms), key=len, reverse=True):
        if not t or len(t) < 3: continue
        # wrap whole-word-ish matches
        s = re.sub(rf"(?i)(?<!\w)({re.escape(t)})", r"<mark>\1</mark>", s)
    return s

# --- bullet templates for suggestions (edit/extend freely) ---
TEMPLATES = {
    # IT / Data
    "python":      "Developed data pipelines in Python with logging and unit tests.",
    "sql":         "Optimized SQL queries and indexes, reducing report latency by 40%.",
    "pandas":      "Built reproducible analytics in Pandas; validated edge cases and null handling.",
    "sklearn":     "Trained and evaluated ML models in scikit-learn with cross-validation and calibration.",
    "pytorch":     "Implemented PyTorch models; tracked experiments and metrics for reproducibility.",
    "tensorflow":  "Prototyped TensorFlow CNNs; improved validation AUC by 6%.",
    "docker":      "Containerized apps with Docker for consistent local and CI/CD environments.",
    "aws":         "Built data workflows on AWS (S3/EC2/Lambda), improving reliability and scale.",
    "gcp":         "Automated GCP data pipelines with Cloud Functions and BigQuery.",
    "azure":       "Deployed services on Azure; integrated monitoring and alerts.",
    "git":         "Maintained clean Git workflows (code review, feature branches, CI).",
    "kubernetes":  "Orchestrated services on Kubernetes; wrote Helm charts and readiness probes.",
    "airflow":     "Scheduled ETL in Apache Airflow with SLAs, retries, and alerting.",
    "javascript":  "Built small UI tools in JavaScript to visualize model results for stakeholders.",
    # Healthcare
    "ehr":         "Recorded patient intake and vitals in EHR (Epic/Cerner) with 100% HIPAA compliance.",
    "icd-10":      "Applied ICD-10 codes for diagnoses and CPT for procedures during patient encounters.",
    "cpt":         "Performed CPT coding for procedures; collaborated with billing to resolve denials.",
    "triage":      "Triaged patients, prioritized urgent cases, and documented symptoms accurately.",
    "vital signs": "Measured and documented vitals (BP, HR, SpO2, Temp) and reported abnormalities.",
    "phlebotomy":  "Collected and labeled specimens; ensured chain-of-custody and safe handling.",
}

def bullets_for_missing(missing: list[str], category: str, top_n: int = 5) -> list[str]:
    """Return up to top_n concise bullets tailored by missing keywords."""
    out = []
    for k in missing:
        k_l = k.lower().strip()
        if k_l in TEMPLATES:
            out.append(f"• {TEMPLATES[k_l]}")
        else:
            # generic but useful
            if category.lower().startswith("health"):
                out.append(f"• Demonstrated proficiency with {k} in daily clinical workflows; documented accurately in the EHR.")
            else:
                out.append(f"• Gained hands-on experience with {k} through a scoped project; documented results and impact.")
        if len(out) >= top_n: break
    return out

def simulate_with_inserts(resume_text: str, inserts: list[str], jd_text: str, clf):
    """What happens to the score if we add these bullets to the resume?"""
    augmented = resume_text.strip() + "\n" + "\n".join(inserts)
    X, *_ = build_features(augmented, jd_text)
    return float(clf.predict_proba(X)[0, 1])

# -------------------- JD keyword vectorizer (for data-driven 'missing skills') --------------------
@st.cache_resource
def build_jd_vectorizer(jd_df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = jd_df["jd_text"].astype(str).str.lower().tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=5, max_df=0.6)
    vec.fit(corpus)
    return vec

VEC = build_jd_vectorizer(JDS)

COMMON = set("""
responsibilities requirement requirements responsibilities include ability excellent strong
communication skills ability work team environment experience years using knowledge familiarity preferred
preferred qualifications required duties job candidate successful ideal including
""".split())

def top_keywords_for_jd(jd_text: str, topn: int = 12):
    t = clean_text(jd_text)
    X = VEC.transform([t])
    feats = VEC.get_feature_names_out()
    arr = X.toarray()[0]
    idxs = arr.argsort()[::-1]
    kws = []
    for i in idxs:
        if arr[i] <= 0: break
        tok = feats[i]
        if len(tok) < 3: continue
        if tok.isdigit(): continue
        if tok in COMMON: continue
        kws.append(tok)
        if len(kws) == topn: break
    return kws

# -------------------- File reading --------------------
def read_upload(file) -> str:
    name = file.name.lower()
    raw = file.read()
    if name.endswith(".txt"):
        try:    return raw.decode("utf-8", errors="ignore")
        except: return raw.decode("latin1", errors="ignore")
    if name.endswith(".docx"):
        try:
            from docx import Document
        except Exception:
            st.error("python-docx not installed. Run: pip install python-docx")
            return ""
        with io.BytesIO(raw) as buf:
            doc = Document(buf)
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".pdf"):
        try:
            from pdfminer.high_level import extract_text
        except Exception:
            st.error("pdfminer.six not installed. Run: pip install pdfminer.six")
            return ""
        with io.BytesIO(raw) as buf:
            return extract_text(buf) or ""
    return raw.decode("utf-8", errors="ignore")

# -------------------- Scoring --------------------
def score_resume_against_jds(resume_text: str, jd_df: pd.DataFrame, top_k: int = 10):
    rows = []
    # Use raw resume text; build_features will clean internally.
    resume_text_raw = resume_text
    resume_text_clean = clean_text(resume_text_raw)
    resume_text_with_spaces = " " + resume_text_clean + " "

    for _, r in jd_df.iterrows():
        jd_txt = r["jd_text"]

        # Build features (includes sim_tfidf + sim_sbert when available)
        X, rs, js, meta = build_features(resume_text_raw, jd_txt)

        proba = clf.predict_proba(X)[0, 1].item()
        sim_tfidf = meta["sim_tfidf"]
        sim_sbert = meta["sim_sbert"]  # -1 if SBERT unavailable
        overlap   = meta["overlap"]

        # HARD-NEGATIVE GATE: if all signals are tiny, cap probability
        if sim_sbert >= 0:  # SBERT available
            if sim_tfidf < 0.03 and sim_sbert < 0.30 and overlap < 0.05:
                proba = min(proba, 0.20)
        else:               # fallback when SBERT isn't installed
            if sim_tfidf < 0.03 and overlap < 0.05:
                proba = min(proba, 0.25)

        # Data-driven keywords from this JD to populate "missing skills"
        jd_kws = top_keywords_for_jd(jd_txt, topn=12)
        jd_skillset = set(js) | set(jd_kws)
        missing = [k for k in sorted(jd_skillset) if f" {k} " not in resume_text_with_spaces]

        rows.append({
            "proba": proba,
            "sim_sbert": round(sim_sbert, 4) if sim_sbert >= 0 else -1,
            "sim_tfidf": round(sim_tfidf, 4),
            "overlap": round(overlap, 4),
            "len_ratio": round(meta["len_ratio"], 3),
            "jd_id": r["jd_id"],
            "jd_title": r["jd_title"] or "(no title)",
            "jd_role": r["jd_role"] or "",
            "jd_category": r["jd_category"] or "",
            "missing_skills": ", ".join(missing[:15]) if missing else "",
            "jd_text": jd_txt[:800] + ("…" if len(jd_txt) > 800 else "")
        })

    out = pd.DataFrame(rows).sort_values("proba", ascending=False).head(top_k).reset_index(drop=True)
    return out

# -------------------- UI --------------------
st.set_page_config(page_title="AI Resume ↔ JD Matcher (Dataset-Backed)", layout="wide")
st.title("AI Resume ↔ JD Matcher (Dataset-Backed)")

colL, colR = st.columns([2,1], gap="large")

with colL:
    up = st.file_uploader("Upload resume (PDF/DOCX/TXT) or paste below", type=["pdf","docx","txt"])
    resume_text = ""
    if up:
        resume_text = read_upload(up)
    resume_text = st.text_area("Or paste resume text", resume_text, height=260, placeholder="Paste your resume here…")

with colR:
    cats = ["(All)"] + sorted([c for c in JDS["jd_category"].unique() if isinstance(c, str)])
    cat = st.selectbox("Filter by JD category", cats, index=0)
    kw = st.text_input("Filter JDs by keyword (optional)", "")
    top_k = st.slider("Show top N matches", min_value=5, max_value=30, value=10, step=1)
    strict = st.slider("Decision strictness", -10, 10, 0, help="Offset from learned threshold; positive = stricter.")
    st.caption(f"Learned threshold = {THRESH:.2f}")

# Filter JD index
jd_view = JDS
if cat != "(All)":
    jd_view = jd_view[jd_view["jd_category"] == cat]
if kw.strip():
    k = kw.strip().lower()
    m = (
        jd_view["jd_title"].str.lower().str.contains(k, na=False) |
        jd_view["jd_role"].str.lower().str.contains(k, na=False)  |
        jd_view["jd_text"].str.lower().str.contains(k, na=False)
    )
    jd_view = jd_view[m]
st.caption(f"{len(jd_view)} JDs in search space")

# Compute matches
if st.button("Compute Matches", type="primary"):
    if not resume_text.strip():
        st.warning("Please upload or paste a resume.")
    elif len(jd_view) == 0:
        st.warning("No JDs match the current filters. Clear filters or rebuild jd_index.csv with more samples.")
    else:
        results = score_resume_against_jds(resume_text, jd_view, top_k=top_k)
        if results.empty:
            st.warning("No matches computed.")
        else:
            thr = max(0.0, min(1.0, THRESH + strict/100))
            results_disp = results.copy()
            results_disp.insert(0, "Decision", (results_disp["proba"] >= thr).map({True:"PASS", False:"REVIEW"}))
            results_disp["Match %"] = (results_disp["proba"]*100).round(1)
            results_disp = results_disp[[
                "Decision","Match %","sim_sbert","sim_tfidf","overlap","len_ratio",
                "jd_category","jd_title","jd_role","missing_skills","jd_text","jd_id"
            ]]

            st.caption("PASS = probability ≥ threshold. Move **Decision strictness** right to be stricter.") 
            st.subheader("Top Matches")
            st.dataframe(results_disp, use_container_width=True, hide_index=True)

            # Download CSV
            csv = results_disp.to_csv(index=False).encode("utf-8")
            st.download_button("Download results (CSV)", data=csv, file_name="matches.csv", mime="text/csv")

            # Detailed view
            st.markdown("---")
            st.markdown("### Details")
            for i, row in results.iterrows():
                with st.expander(f"[{i+1}] {row['jd_title']} • {row['jd_category']} • Match={row['proba']*100:.1f}%"):
                    st.markdown(f"**Signals** — SBERT: `{row['sim_sbert']}`  •  TF-IDF: `{row['sim_tfidf']}`  •  Overlap: `{row['overlap']}`  •  LenRatio: `{row['len_ratio']}`")
                    # PRIORITIZE missing items by JD importance
                    jd_txt_full = jd_view.loc[jd_view["jd_id"] == row["jd_id"], "jd_text"].iloc[0]
                    kw_scores = jd_keyword_scores(jd_txt_full)
                    missing_list = [m.strip() for m in (row["missing_skills"].split(",") if row["missing_skills"] else []) if m.strip()]
                    missing_ranked = sorted(missing_list, key=lambda k: kw_scores.get(k, 0.0), reverse=True)

                    if missing_ranked:
                        st.write("**Missing (prioritized)**:", ", ".join(missing_ranked))
                    else:
                        st.write("**Missing (prioritized):** None")

                    # Highlight a JD snippet with top matches (for context)
                    show_terms = missing_ranked[:6]  # show the top few
                    html = highlight_terms(jd_txt_full[:1200], show_terms)
                    st.markdown("**JD excerpt (highlighted keywords):**", help="Top missing items highlighted")
                    st.markdown(f"<div style='border:1px solid #ddd;padding:10px;border-radius:8px'>{html}</div>", unsafe_allow_html=True)

                    # Bullet suggestions the user can copy
                    if missing_ranked:
                        st.markdown("**Bullet suggestions to add to your resume:**")
                        bullets = bullets_for_missing(missing_ranked, row["jd_category"], top_n=5)
                        st.code("\n".join(bullets), language="text")

                        # 'What-if' simulator: if user adds these bullets, how would the score change?
                        try:
                            new_p = simulate_with_inserts(resume_text, bullets, jd_txt_full, clf)
                            st.caption(f"🔮 If you add those bullets, predicted Match ≈ **{new_p*100:.1f}%** (was {row['proba']*100:.1f}%).")
                        except Exception:
                            pass  # never block the UI

                        # Download suggestions
                        s = f"# Resume suggestions for JD: {row['jd_title']} ({row['jd_category']})\n\n" + "\n".join(bullets)
                        st.download_button("Download these bullet suggestions (.txt)", s.encode("utf-8"), file_name=f"suggestions_{row['jd_id']}.txt")
                    else:
                        st.info("No obvious gaps detected for this JD. Consider tightening your summary to mirror JD terminology.")

