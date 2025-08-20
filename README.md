
# AI Resume ↔ JD Matcher (Dataset-Backed, Streamlit)

> Upload a resume (PDF/DOCX/TXT), pick a JD category, and get a **match probability**, **missing skills/keywords**, **copy-paste bullet suggestions**, and a **what-if** score lift if you add them.

<img width="1808" height="854" alt="image" src="https://github.com/user-attachments/assets/5b86dae5-1991-4a47-982a-9a3fc097511f" />
<img width="1786" height="765" alt="image" src="https://github.com/user-attachments/assets/9bae36fc-4333-4565-9770-4b29e0f7386f" />
<img width="1764" height="698" alt="image" src="https://github.com/user-attachments/assets/90417ac8-9acf-4383-9506-23dd74e038aa" />
<img width="1776" height="776" alt="image" src="https://github.com/user-attachments/assets/42a0efaa-011e-43ff-a5f8-341c2df0795f" />

---

### Highlights

- **Hybrid signals:** TF-IDF (lexical) + **SBERT** (semantic) + engineered features (overlap, length ratio).
- **Leak-proof evaluation:** Grouped CV by resume and by JD (prevents data leakage).
- **Hard-negative gate:** Suppresses cross-domain false positives (e.g., medical resume vs IT JD).
- **Coaching UI:** Prioritized **missing keywords**, **bullet suggestions**, and **what-if simulator** (expected score if you add bullets).
- **Runs locally & free:** CPU-only, dataset-backed, no paid APIs.

---

## Results (Grouped Cross-Validation)

| Split Type          | AUC (mean) | Acc (mean) | F1 (mean) | Notes                          |
|---------------------|------------|------------|-----------|--------------------------------|
| By Resume           | ~**0.988** | ~0.959     | ~0.960    | Unseen resumes per fold        |
| By Job Description  | ~**0.985** | ~0.961     | ~0.964    | Unseen JDs per fold            |

- Final decision threshold: **0.50**
- JD index used in the app: **~263 JDs across 11 categories**

> Numbers come from `scripts/train_semantic.py` on balanced pairs (`data/pairs.csv`) built from educational datasets.

---

## Quickstart (Windows / PowerShell)

### 1) Create venv & install
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

### 2) Ensure required artifacts exist (already committed)

```
data/jd_index.csv
data/matcher_model.joblib
data/threshold.json
data/model_metrics.json
data/skills.json
```

### 3) Run the app

```powershell
streamlit run app_simple.py
```

Open the URL Streamlit prints (e.g., `http://localhost:8501`).

---

## 🧭 How to Use

1. **Upload** your resume (PDF/DOCX/TXT) or **paste** the text.
2. **Filter** JDs by category and/or keyword (optional).
3. Click **Compute Matches**.
4. In **Top Matches**, see:

   * Match % (model probability)
   * Semantic similarity (SBERT), TF-IDF similarity, overlap, length ratio
   * Missing skills/keywords
5. Expand any row for:

   * **Prioritized missing keywords** (ranked by JD TF-IDF importance)
   * **Bullet suggestions** to paste into your resume
   * **What-if**: predicted match if you add those bullets

> Tip: Move the **Decision strictness** slider right to be stricter (higher threshold).

---

## Project Structure

```
.
├── app_simple.py                  # Streamlit app (UI + coaching)
├── requirements.txt
├── skills.json                    # small skill lexicon (extensible)
├── data/
│   ├── jd_index.csv               # compact JD index for the app
│   ├── matcher_model.joblib       # trained model (TF-IDF + SBERT features)
│   ├── threshold.json             # decision threshold
│   └── model_metrics.json         # CV metrics summary
└── scripts/
    ├── featurize_shared.py        # clean_text, TF-IDF cosine, SBERT cosine, build_features
    ├── clean_data.py              # raw → cleaned CSVs (chunked JDs)
    ├── fix_categories.py          # heuristic relabeling of jd_category
    ├── build_jd_index.py          # balanced index per category with length fallbacks
    ├── make_pairs.py              # balanced positive/negative pairs
    └── train_semantic.py          # training with grouped CV + SBERT feature
```

---

## 🧠 Model & Features (At a Glance)

* **TF-IDF cosine** between resume and JD text
* **SBERT cosine**: `sentence-transformers/all-MiniLM-L6-v2` (CPU)
* **Overlap**: fraction of JD skills/keywords present in resume
* **Length ratio**: guardrail vs very short/long texts
* **Hard-negative gate**: if SBERT/TF-IDF/overlap are all tiny → cap probability

**Coaching signals**

* JD-specific **keywords** extracted by TF-IDF (data-driven, domain-agnostic)
* **Missing keywords** prioritized by per-JD TF-IDF weight
* **Bullet suggestions** mapped from common keywords (domain-aware templates)
* **What-if simulator** recomputes score after adding those bullets

---

## 🔁 Reproducible Pipeline (from raw CSVs)

If you also have the raw datasets locally (not included to keep repo small):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts\clean_data.py        # → data/resumes_clean.csv, data/jobs_clean.csv
python scripts\fix_categories.py    # heuristic jd_category relabel
python scripts\build_jd_index.py    # → data/jd_index.csv
python scripts\make_pairs.py        # → data/pairs.csv (balanced pos/neg)
python scripts\train_semantic.py    # → matcher_model.joblib, threshold.json, model_metrics.json
```

Then:

```powershell
streamlit run app_simple.py
```

> Raw CSVs (GB-scale) are intentionally **not committed**. See `.gitignore`.

---

## 🧩 Troubleshooting

* **SBERT “meta tensor” / device errors on Windows**
  `featurize_shared.py` forces **CPU** and disables low-mem init; if SBERT still fails, the app gracefully **falls back** to TF-IDF (you’ll see `sim_sbert = -1`).

* **Only one category appears in the app**
  Rebuild the index:

  ```powershell
  python scripts\fix_categories.py
  python scripts\build_jd_index.py
  ```

* **Cross-domain pairs score too high**
  Ensure you trained with `scripts/train_semantic.py` (uses SBERT + hard-negative gate).

---

## 🧯 Limitations & Ethics

* Educational/synthetic datasets; real-world drift is possible.
* Matching is a **signal**, not a hiring decision — use responsibly.
* Files processed locally; no PII stored.

---

## 🗺️ Roadmap

* Reliability curves & probability calibration
* Domain-adaptive embeddings per category
* Phrase-level overlap (noun chunks, dependency patterns)
* Export a tailored resume draft (DOCX) with selected bullets
* Dockerfile & one-click deploy

---


## 👩‍💻 Author

**Jasmine Kaur Hanjra**
GitHub: [https://github.com/](https://github.com/)<JasmineHanjra>
LinkedIn: [https://www.linkedin.com/in/jasminehanjra](https://www.linkedin.com/in/jasminehanjra)




