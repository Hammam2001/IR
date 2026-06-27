import math
import ir_datasets
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8009/search"
SAMPLE_SIZE = 80
TOP_K = 10

DATASETS = {
    "quora": {"full": "beir/quora", "test": "beir/quora/test"},
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
        payload = {
            "query": q.text,
            "dataset": api_name,
            "algorithm": algo,
            "use_refinement": use_refinement,
            "limit": TOP_K,
        }
        try:
            r = requests.post(API_URL, json=payload, timeout=120)
            data = r.json()
            retrieved = [res.get("text", "").strip().lower() for res in data.get("results", [])]
            p, rec, ap, ndcg = metrics(retrieved, qrels[q.query_id], k=TOP_K)
            acc["P"] += p
            acc["R"] += rec
            acc["MAP"] += ap
            acc["nDCG"] += ndcg
        except Exception as exc:
            print("   request error:", exc)
    n = len(queries)
    return {k: v / n for k, v in acc.items()}


rows = []
for api_name, cfg in DATASETS.items():
    queries, qrels = load_dataset(api_name, cfg)
    for algo in ALGORITHMS:
        for refine in (False, True):
            tag = "with refinement" if refine else "baseline"
            print(f"Evaluating {api_name} | {algo} | {tag} on {len(queries)} queries ...", flush=True)
            m = evaluate(api_name, algo, refine, queries, qrels)
            rows.append({
                "Dataset": api_name,
                "Algorithm": algo,
                "Mode": tag,
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
