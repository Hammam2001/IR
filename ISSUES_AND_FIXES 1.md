# 🛠️ IR Project — Issues & Fixes (for the team)

> هذا الملف يشرح كل مشكلة في المشروع مقارنةً بمتطلبات الـ PDF، مع **الحل الكامل والكود الجاهز** لكل مشكلة. طُبّق ما تستطيع قبل المقابلة. (E-drive و«تقسيم العمل» مستثنيان حسب الطلب.)

This document lists every issue found when auditing the project against `IR Project 2026.pdf`, each with a **detailed, drop-in solution**. Apply them in the order of the priority table.

---

## ⚠️ Do we need to rebuild the cache / "re-train"? (READ FIRST)

**Short answer: you NEVER need to re-encode BERT for any of these fixes** — and BERT encoding is the only slow part (hours on CPU for ~500K docs). Everything else (BM25, TF-IDF, inverted index, tokenization) rebuilds in **minutes**.

**Why:** BERT (`embedding_service.build_embedding_index`) encodes the **raw document text**, *not* the pre-processed text. So changing preprocessing / TF-IDF / BM25 / the inverted index does **not** change the BERT embeddings. Your existing embeddings in the `.pkl` stay valid.

| Fix | Cache change? | Re-encode BERT? |
|---|---|---|
| 1 TF-IDF caching | add TF-IDF to cache | ❌ No |
| 4 FAISS | rebuilt from existing embeddings at startup | ❌ No |
| 5 real `doc_ids` | add `doc_ids` | ❌ No |
| 7 URL removal | rebuild BM25/TF-IDF/inverted | ❌ No |
| 8 BM25 tokens cache | add tokens | ❌ No |
| 10 inverted index (in-memory) | none (built fresh at startup) | ❌ No |
| 2, 3, 6, 9 | none | ❌ No |

### The smart way to update the cache (keeps BERT, takes minutes not hours)
Don't delete the whole `.pkl` (that throws away the embeddings and forces an hours-long re-encode). Instead, run this migration **once** after applying Issues 1/4/5/7/8 — it reuses the BERT embeddings and rebuilds only the cheap indexes:

```python
# migrate_cache.py  — keep the expensive BERT embeddings, rebuild only the cheap indexes
import os, pickle, ir_datasets
from bm25_service import build_bm25_model
from retrieval_service import build_tfidf_index          # from Issue 1

IR_NAME = {"quora": "beir/quora", "lotte": "lotte/lifestyle/dev/search"}

for name, ir_name in IR_NAME.items():
    path = f"{name}_cache_full.pkl"
    if not os.path.exists(path):
        print("no cache yet, skip:", path); continue
    print("migrating", path, "...")
    with open(path, "rb") as f:
        data = pickle.load(f)

    docs = data["docs"]                      # text
    # data["embeddings"] (BERT) is kept untouched — the slow part is reused

    data["bm25"] = build_bm25_model(docs)                                  # fast
    data["tfidf_vectorizer"], data["tfidf_matrix"] = build_tfidf_index(docs)  # fast

    # (Issue 5) recover the real doc IDs by re-reading the dataset (no encoding):
    ds = ir_datasets.load(ir_name)
    data["doc_ids"] = [d.doc_id for d in ds.docs_iter()]

    with open(path, "wb") as f:
        pickle.dump(data, f)
    print("done:", path)

print("Migration complete. BERT embeddings were reused (no re-encoding).")
```
Run once: `python migrate_cache.py`, then start the server normally. FAISS (Issue 4) and the inverted index (Issue 10) are rebuilt in memory at startup, so they don't need to be in the cache.

> The blunt alternative — delete `quora_cache_full.pkl` + `lotte_cache_full.pkl` and let the server rebuild — also works, but it re-encodes BERT (slow). Use the migration above instead.

---

## Priority table

| # | Issue | Severity | Effort | Breaks demo? |
|---|---|---|---|---|
| 1 | TF-IDF rebuilds the whole index on **every query** | 🔴 Critical | Low | **Yes** (TF-IDF freezes on big data) |
| 2 | Evaluation covers only 2 models on 1 dataset | 🟠 High | Medium | No (rubric gap) |
| 3 | No code **README** (required deliverable) | 🟠 High | Low | No |
| 4 | FAISS vector store **not connected** to the API | 🟡 Medium | Low | No (your only bonus feature is unused) |
| 5 | `doc_id` returned is a list position, not the real ID | 🟡 Medium | Low | No |
| 6 | Server crashes on an unknown `algorithm` value | 🟢 Low | Tiny | No |
| 7 | Report says preprocessing removes URLs, code doesn't | 🟢 Low | Tiny | No |
| 8 | BM25 rebuilds the whole model when `k1`/`b` change | 🟢 Low | Low | No (slow slider) |
| 9 | Hybrid-serial `initial_k` mismatch (50 vs 100) | 🟢 Trivial | Tiny | No |
| 10 | Inverted index is built but never used | 🟢 Low | Medium | No |

---

# 🔴 Issue 1 — TF-IDF rebuilds the entire index on every query (CRITICAL)

**File:** `retrieval_service.py` → `process_and_match_query()`, used by `api_retrieval_service.py`.

### What's wrong
Every time a user searches with **TF-IDF**, this runs:
```python
processed_docs = [preprocess_text(doc) for doc in documents]   # re-clean ALL docs
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs)         # re-build the WHOLE index
```
On Quora that means **re-preprocessing and re-vectorizing ~522,000 documents on every single keystroke/search.** BM25 and BERT are built once at startup and cached — TF-IDF is the only model that was left rebuilding per query.

### Why it matters
- In the live demo, selecting **TF-IDF** on Quora or Lotte will **freeze for minutes or run out of memory**. This is the #1 thing that can blow up the interview.
- It also violates requirement **§3 (Indexing)**: the index must be built once for fast retrieval.

### The fix — build the TF-IDF index once at startup, like BM25/BERT
**Step A.** In `retrieval_service.py`, add a build function and a fast search function (you can keep the old `process_and_match_query` for the standalone demo):

```python
# retrieval_service.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_service import preprocess_text


def build_tfidf_index(documents):
    """OFFLINE: fit the TF-IDF vectorizer ONCE over all documents and keep the matrix."""
    print("Building TF-IDF index (once)...")
    processed_docs = [preprocess_text(doc) for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)   # sparse, stays in memory/cache
    return vectorizer, tfidf_matrix


def search_tfidf(query, vectorizer, tfidf_matrix, documents, top_k=100):
    """ONLINE: only transform the query and compare against the prebuilt matrix."""
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    results = [(doc_id, float(score), documents[doc_id])
               for doc_id, score in enumerate(similarities) if score > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

**Step B.** In `api_retrieval_service.py`, build & cache the TF-IDF index alongside BM25/BERT:

```python
# api_retrieval_service.py  (imports)
from retrieval_service import build_tfidf_index, search_tfidf
```
```python
# inside load_and_index_dataset(), where it currently builds bm25 + embeddings:
    bm25_model = build_bm25_model(docs)
    embeddings = build_embedding_index(docs)
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(docs)        # NEW

    datasets_store[dataset_name] = {
        "docs": docs,
        "bm25": bm25_model,
        "embeddings": embeddings,
        "tfidf_vectorizer": tfidf_vectorizer,   # NEW
        "tfidf_matrix": tfidf_matrix,           # NEW
    }
```

**Step C.** In `search_api()`, replace the slow TF-IDF branch:

```python
    # BEFORE:
    # if request.algorithm == "tfidf":
    #     raw_results = process_and_match_query(final_query_text, real_docs)

    # AFTER:
    if request.algorithm == "tfidf":
        raw_results = search_tfidf(
            final_query_text,
            selected_data["tfidf_vectorizer"],
            selected_data["tfidf_matrix"],
            real_docs,
        )
```

### After applying
- **Delete the old `.pkl` caches and rebuild** (see the cache rule at the top) — the new cache will include the TF-IDF index.
- Verify: pick TF-IDF on Quora in the UI → results should now return in well under a second.

---

# 🟠 Issue 2 — Evaluation only covers 2 models on 1 dataset

**Files:** `evaluate_system.py`, `evaluation_notebook.ipynb` (both only test `bm25` + `hybrid_serial` on Quora).

### What's wrong
Requirement **§8** says: compute **MAP, Recall, Precision@10, nDCG** for **every representation model** *and* **every dataset**, and **before vs after** the additional features. You currently have only **BM25 + Hybrid-Serial on Quora**, no Lotte, no TF-IDF/BERT/Hybrid-Parallel, and no before/after comparison.

### Why it matters
This is the biggest rubric gap. Even a *sampled* evaluation of all models on both datasets, before/after refinement, satisfies the requirement far better.

### The fix — one script that evaluates everything
Create **`evaluate_all.py`** in the project root. It talks to the running API (start the server first), loops over **both datasets × all 5 models × refinement off/on**, and prints + saves a results table. Use a sample size so it finishes in reasonable time (raise it if you have time).

> ⚠️ Run this **after** Issue 1 (TF-IDF fix) and ideally Issue 4 (FAISS) are applied, otherwise TF-IDF/BERT/Parallel will be very slow. Keep `SAMPLE_SIZE` small (50–200) for the slow models.

```python
# evaluate_all.py
import math
import ir_datasets
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8009/search"
SAMPLE_SIZE = 100      # raise if you have time; lower if too slow
TOP_K = 10

# API dataset name  ->  how to load its docs (full) and its queries+qrels (test)
DATASETS = {
    "quora": {"full": "beir/quora",                 "test": "beir/quora/test"},
    "lotte": {"full": "lotte/lifestyle/dev/search", "test": "lotte/lifestyle/dev/search"},
}
ALGORITHMS = ["tfidf", "bm25", "bert", "hybrid_parallel", "hybrid_serial"]


def get_dcg(rels):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))


def metrics(retrieved_texts, relevant_texts, k=10):
    rels = [1 if t in relevant_texts else 0 for t in retrieved_texts[:k]]
    precision = sum(rels) / k
    recall = sum(rels) / len(relevant_texts) if relevant_texts else 0
    hits, sum_prec = 0, 0.0
    for i, rel in enumerate(rels):
        if rel:
            hits += 1
            sum_prec += hits / (i + 1)
    ap = sum_prec / len(relevant_texts) if relevant_texts else 0
    idcg = get_dcg(sorted(rels, reverse=True)) or get_dcg([1] * min(len(relevant_texts), k))
    ndcg = get_dcg(rels) / idcg if idcg > 0 else 0
    return precision, recall, ap, ndcg


def load_dataset(api_name, cfg):
    print(f"\nLoading qrels for '{api_name}' ...", flush=True)
    full = ir_datasets.load(cfg["full"])
    doc_text = {d.doc_id: d.text.strip().lower() for d in full.docs_iter()}
    test = ir_datasets.load(cfg["test"])
    qrels = {}
    for qr in test.qrels_iter():
        if qr.relevance > 0 and qr.doc_id in doc_text:
            qrels.setdefault(qr.query_id, set()).add(doc_text[qr.doc_id])
    queries = [q for q in test.queries_iter() if q.query_id in qrels][:SAMPLE_SIZE]
    return queries, qrels


def evaluate(api_name, algo, use_refinement, queries, qrels):
    acc = {"P": 0, "R": 0, "MAP": 0, "nDCG": 0}
    for q in queries:
        payload = {"query": q.text, "dataset": api_name, "algorithm": algo,
                   "use_refinement": use_refinement, "limit": TOP_K}
        try:
            r = requests.post(API_URL, json=payload, timeout=120)
            data = r.json()
            retrieved = [res.get("text", "").strip().lower() for res in data.get("results", [])]
            p, rec, ap, ndcg = metrics(retrieved, qrels[q.query_id], k=TOP_K)
            acc["P"] += p; acc["R"] += rec; acc["MAP"] += ap; acc["nDCG"] += ndcg
        except Exception as e:
            print("   request error:", e)
    n = len(queries)
    return {k: v / n for k, v in acc.items()}


rows = []
for api_name, cfg in DATASETS.items():
    queries, qrels = load_dataset(api_name, cfg)
    for algo in ALGORITHMS:
        for refine in (False, True):            # before vs after additional features
            tag = "with refinement" if refine else "baseline"
            print(f"Evaluating {api_name} | {algo} | {tag} on {len(queries)} queries ...", flush=True)
            m = evaluate(api_name, algo, refine, queries, qrels)
            rows.append({
                "Dataset": api_name, "Algorithm": algo, "Mode": tag,
                "Precision@10": round(m["P"] * 100, 2),
                "Recall": round(m["R"] * 100, 2),
                "MAP": round(m["MAP"] * 100, 2),
                "nDCG@10": round(m["nDCG"] * 100, 2),
            })

df = pd.DataFrame(rows)
print("\n================ FINAL RESULTS ================")
print(df.to_string(index=False))
df.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")
print("\nSaved -> evaluation_results.csv")
```

### Notes
- This produces the table for **§8** (all models, both datasets, baseline vs +refinement). Drop it into the report.
- If the full run is too slow, run the slow models (`tfidf`, `bert`, `hybrid_parallel`) with a smaller `SAMPLE_SIZE` and the fast ones (`bm25`, `hybrid_serial`) with a bigger one, then merge — just say so in the report.
- Matching is by **text** because of Issue 5. If you apply Issue 5 (real doc IDs), switch the matching to IDs for accuracy.

---

# 🟠 Issue 3 — No code README (required deliverable)

**Missing file:** `README.md` in the repo root. Requirement (Deliverables) asks for a **GitHub readme that clearly explains the code structure.**

### The fix — drop this in as `README.md`
```markdown
# Smart Search Engine — Information Retrieval 2026 (Damascus University)

A custom search engine over two large datasets (Quora, Lotte) supporting five
retrieval models — TF-IDF, BM25, BERT embeddings, and two hybrids — exposed via a
FastAPI service with a Streamlit UI, built on a Service-Oriented Architecture (SOA).

## Architecture (SOA)
UI (Streamlit) → REST API (FastAPI) → retrieval services → ranked results.
Indexes (BM25, BERT, TF-IDF) are built once at startup and cached to `*.pkl`.

## Datasets
- `beir/quora` — ~522K Q&A documents (+ `beir/quora/test` qrels).
- `lotte/lifestyle/dev/search` — >200K documents (+ qrels).

## Requirements
Python 3.10+. Install:
```
pip install fastapi uvicorn streamlit ir_datasets rank_bm25 sentence-transformers \
            scikit-learn nltk faiss-cpu pyspellchecker pandas requests
```
First run downloads NLTK data automatically (see `data_service.py`).

## How to run
1. Start the backend (builds/loads indexes, may take a while the first time):
   ```
   uvicorn api_retrieval_service:app --port 8009
   ```
2. Start the UI in another terminal:
   ```
   streamlit run app.py
   ```
3. Open the browser, pick a dataset + algorithm, and search.

## Evaluation
With the server running:
```
python evaluate_all.py        # all models, both datasets, baseline vs +refinement
python prove_metrics.py       # single-query live metric demo
```

## Code structure (services)
| File | Responsibility |
|------|----------------|
| `data_service.py` | Text pre-processing (lowercase, punctuation, stopwords, lemmatization) |
| `indexing_service.py` | Inverted index (term → doc ids) |
| `representation_service.py` | TF-IDF representation demo |
| `retrieval_service.py` | TF-IDF index build + cosine search |
| `bm25_service.py` | BM25 (Okapi) build + search |
| `embedding_service.py` | BERT (all-MiniLM-L6-v2) embeddings + cosine search |
| `vector_store_service.py` | FAISS vector store (fast semantic search) |
| `hybrid_service.py` | Parallel (score fusion) + serial (re-rank) hybrids |
| `query_refinement_service.py` | Spell-correction + synonym (WordNet) expansion |
| `evaluation_service.py` | Metric functions (P@10, Recall, MAP, nDCG) |
| `evaluate_all.py` / `evaluate_system.py` / `prove_metrics.py` | Evaluation runners |
| `api_retrieval_service.py` | FastAPI gateway: loads/caches indexes, `/search` endpoint |
| `app.py` | Streamlit UI |

## Team
- [names]
```

Also add a **`requirements.txt`** with the same packages so the project is reproducible.

---

# 🟡 Issue 4 — FAISS vector store is built but never connected

**Files:** `vector_store_service.py` (built, has a working demo) but `api_retrieval_service.py` uses brute-force cosine in `embedding_service.search_embeddings` instead.

### Why it matters
- FAISS is your **bonus feature** (requirement #11). Right now it's not used by the live system, so it may not be counted as "achieved."
- It also makes BERT search **much faster** (the brute-force version compares the query to all ~522K vectors every search).

### The fix — index the existing embeddings in FAISS and use it for the `bert` path
**Step A.** Add helpers (e.g. in `vector_store_service.py`) that build FAISS **from the embeddings you already computed** (don't re-encode all docs):

```python
# vector_store_service.py
import faiss
import numpy as np
from embedding_service import model


def build_faiss_from_embeddings(embeddings):
    """Build a cosine FAISS index from already-computed BERT embeddings."""
    emb = embeddings.cpu().numpy() if hasattr(embeddings, "cpu") else np.asarray(embeddings)
    emb = np.ascontiguousarray(emb, dtype="float32")
    faiss.normalize_L2(emb)                       # normalize -> inner product == cosine
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def search_faiss(query, index, documents, top_k=100):
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, idx = index.search(q, top_k)
    results = []
    for i in range(len(idx[0])):
        doc_id = int(idx[0][i])
        if doc_id != -1:
            results.append((doc_id, float(scores[0][i]), documents[doc_id]))
    return results
```

**Step B.** Build the FAISS index at startup in `api_retrieval_service.py`:
```python
from vector_store_service import build_faiss_from_embeddings, search_faiss
```
```python
    embeddings = build_embedding_index(docs)
    faiss_index = build_faiss_from_embeddings(embeddings)          # NEW
    datasets_store[dataset_name] = {
        "docs": docs, "bm25": bm25_model, "embeddings": embeddings,
        "faiss": faiss_index,                                     # NEW
        # ... (plus the tfidf entries from Issue 1)
    }
```
> ⚠️ `IndexFlatIP` is not picklable the same way as numpy. Either (a) rebuild it from `embeddings` after loading the cache, or (b) save it with `faiss.write_index(...)`. Simplest: after `pickle.load(...)`, rebuild: `datasets_store[name]["faiss"] = build_faiss_from_embeddings(datasets_store[name]["embeddings"])`.

**Step C.** Use FAISS in the `bert` branch of `search_api()`:
```python
    elif request.algorithm == "bert":
        raw_results = search_faiss(final_query_text, selected_data["faiss"], real_docs)
```

This makes BERT fast **and** turns your bonus feature into something the system actually uses (great talking point in the interview).

---

# 🟡 Issue 5 — `doc_id` is a list position, not the real document ID

**File:** `api_retrieval_service.py` (`docs = [doc.text for doc in ...]` throws away `doc.doc_id`).

### Why it matters
- The UI shows a fake "Document ID" (just the row number).
- It's the reason evaluation has to match by **text** instead of by ID (fragile).

### The fix — keep a parallel list of real IDs
**Step A.** In `load_and_index_dataset()`:
```python
    docs, doc_ids = [], []
    for doc in dataset.docs_iter():
        docs.append(doc.text)
        doc_ids.append(doc.doc_id)          # keep the real ID

    datasets_store[dataset_name] = {
        "docs": docs,
        "doc_ids": doc_ids,                 # NEW
        # ... bm25 / embeddings / tfidf / faiss ...
    }
```
**Step B.** When formatting results in `search_api()`, map the internal index → real ID:
```python
    real_ids = selected_data["doc_ids"]
    for internal_id, score, text in raw_results[:request.limit]:
        display_score = f"{round(score * 100, 1)}%" if request.algorithm == "tfidf" else str(round(score, 3))
        formatted_results.append({
            "doc_id": real_ids[internal_id],     # real dataset ID now
            "score": display_score,
            "text": text,
        })
```
The internal search functions still use the positional index for `documents[doc_id]`, so nothing else needs to change. After this, evaluation can match by real ID.

> ⚠️ Changes the cache shape → delete and rebuild the `.pkl` files.

---

# 🟢 Issue 6 — Server crashes on an unknown algorithm

**File:** `api_retrieval_service.py`, `search_api()`.

### What's wrong
The dispatch is `if/elif` with no `else`. If `request.algorithm` is anything unexpected, `raw_results` is never assigned → `NameError` → 500 error.

### The fix
```python
    raw_results = []                       # safe default at the top of the dispatch
    if request.algorithm == "tfidf":
        ...
    elif request.algorithm == "hybrid_serial":
        ...
    else:
        return {"error": f"Unknown algorithm: {request.algorithm}"}
```

---

# 🟢 Issue 7 — Report says URLs are removed, code doesn't

**File:** `data_service.py` (`preprocess_text`). The report (§4) claims "Regex removes URLs," but the code only lowercases, strips punctuation, removes stopwords, and lemmatizes.

### The fix (make the code match the report)
```python
import re   # add at the top

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)     # remove URLs (matches the report)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)
```
> ⚠️ Changes preprocessing → **delete and rebuild the `.pkl` caches**. (If you'd rather not rebuild, instead fix the *report* to remove the URL claim.)

---

# 🟢 Issue 8 — BM25 rebuilds the whole model when `k1`/`b` change

**File:** `api_retrieval_service.py` — when the user moves the BM25 sliders off the defaults, it calls `build_bm25_model(real_docs, k1=..., b=...)` **inside the request**, re-tokenizing all docs every time.

### The fix — cache the tokenized docs once, rebuild only the cheap scorer
**Step A.** In `bm25_service.py`, expose tokenization and a build-from-tokens:
```python
from rank_bm25 import BM25Okapi
from data_service import preprocess_text

def tokenize_docs(documents):
    return [preprocess_text(doc).split() for doc in documents]

def build_bm25_from_tokens(tokenized_docs, k1=1.5, b=0.75):
    return BM25Okapi(tokenized_docs, k1=k1, b=b)
```
**Step B.** Cache `tokenized_docs` at startup in `load_and_index_dataset()`:
```python
    tokenized = tokenize_docs(docs)
    bm25_model = build_bm25_from_tokens(tokenized)
    datasets_store[dataset_name] = { ..., "bm25_tokens": tokenized, "bm25": bm25_model }
```
**Step C.** In the `bm25` branch, rebuild only the scorer from cached tokens:
```python
    elif request.algorithm == "bm25":
        if request.k1 != 1.5 or request.b != 0.75:
            custom = build_bm25_from_tokens(selected_data["bm25_tokens"], k1=request.k1, b=request.b)
            raw_results = search_bm25(final_query_text, custom, real_docs)
        else:
            raw_results = search_bm25(final_query_text, selected_data["bm25"], real_docs)
```
This avoids re-tokenizing ~500K docs every time a slider moves.

---

# 🟢 Issue 9 — Hybrid-serial `initial_k` mismatch

**Files:** `hybrid_service.py` (`serial_hybrid_search(..., initial_k=100)`) vs `api_retrieval_service.py` which calls it with `initial_k=50`.

Not a bug, just inconsistent. Pick one value and document it (50 is fine and faster). Make the default match what you actually use:
```python
def serial_hybrid_search(query, bm25_model, document_embeddings, documents, initial_k=50):
    ...
```

---

# 🟢 Issue 10 — Inverted index is built but never used → EASIEST WIRING

**File:** `indexing_service.py` — `build_inverted_index()` exists but nothing calls it.

### The easiest way to actually "use" it (no cache rebuild, ~10 lines)
Build it in memory at startup and expose it as its own API endpoint — this literally turns it into the **"Indexing Service"** your SOA diagram promises, and you can demo it live. **No `.pkl` change, so no cache rebuild needed.**

**Step A.** In `api_retrieval_service.py`, **after** the two `load_and_index_dataset(...)` calls, build the index in memory for each dataset:
```python
from indexing_service import build_inverted_index

for _name in datasets_store:
    print(f"Building inverted index for {_name} ...")
    datasets_store[_name]["inverted_index"] = build_inverted_index(datasets_store[_name]["docs"])
```
> This runs once per server start and is **not** saved to the `.pkl`, so it needs no cache rebuild. It adds a few minutes to startup (it pre-processes the docs once) — a one-time cost while the server boots.

**Step B.** Add a tiny endpoint so the inverted index is a real, callable service:
```python
from data_service import preprocess_text

@app.get("/inverted_lookup")
def inverted_lookup(dataset: str, term: str):
    data = datasets_store.get(dataset)
    if not data:
        return {"error": "dataset not found"}
    key = preprocess_text(term).strip()
    posting = data["inverted_index"].get(key, [])
    return {"term": term, "normalized_term": key,
            "document_frequency": len(posting), "doc_ids": posting[:50]}
```
Now you can show it live in the browser:
```
http://127.0.0.1:8009/inverted_lookup?dataset=quora&term=python
```
→ returns the **posting list** (term → document ids) instantly. That *is* the inverted index doing its job, and it's a clean talking point ("our Indexing Service answers term→documents lookups in O(1)").

### Optional upgrade (a bit more code) — use it to pre-filter TF-IDF candidates
If you want it inside retrieval too, narrow the search to docs that share a query term before scoring:
```python
def candidate_ids(query, inverted_index):
    ids = set()
    for term in preprocess_text(query).split():
        ids.update(inverted_index.get(term, []))
    return ids
```
…then score only those candidate ids in `search_tfidf`. The endpoint in Step B is enough for the requirement and the demo; this is just a bonus.

---

## ✅ Suggested order to apply tonight
1. **Issue 1** (TF-IDF cache) — protects the demo. *(rebuild caches)*
2. **Issue 6** (unknown-algorithm guard) — 1 minute.
3. **Issue 4** (FAISS wired in) — makes your bonus feature real + speeds BERT. *(rebuild/repair caches)*
4. **Issue 5** (real doc IDs). *(rebuild caches)*
5. **Issue 3** (README) + **Issue 9** (initial_k) — quick.
6. **Issue 2** (`evaluate_all.py`) — run it once the above are in, capture the table for the report.
7. **Issues 7, 8, 10** — if time remains.

> Remember the **cache rule**: after Issues 1, 4, 5, or 7, delete `quora_cache_full.pkl` and `lotte_cache_full.pkl` and let the server rebuild once.
