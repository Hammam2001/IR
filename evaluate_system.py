import ir_datasets
import requests
import math

# إعدادات الاتصال والبيانات
API_URL = "http://127.0.0.1:8009/search"
DATASET_API_NAME = "quora"
TEST_DATASET_NAME = "beir/quora/test"
FULL_DATASET_NAME = "beir/quora"
SAMPLE_SIZE = 10  
TOP_K = 10        

def get_dcg(rels):
    return sum([rel / math.log2(idx + 2) for idx, rel in enumerate(rels)])

def calculate_metrics(retrieved_texts, relevant_texts, k=10):
    """حساب المقاييس بمطابقة النصوص بدلاً من الأرقام"""
    retrieved_k = retrieved_texts[:k]
    
    # 1 إذا كان النص المسترجع موجوداً ضمن نصوص الإجابات الصحيحة، 0 إذا لم يكن
    rels = [1 if text in relevant_texts else 0 for text in retrieved_k]
    
    precision = sum(rels) / k
    recall = sum(rels) / len(relevant_texts) if relevant_texts else 0
    
    hits = 0
    sum_precs = 0
    for i, rel in enumerate(rels):
        if rel:
            hits += 1
            sum_precs += hits / (i + 1.0)
    ap = sum_precs / len(relevant_texts) if relevant_texts else 0
    
    dcg = get_dcg(rels)
    ideal_rels = sorted([1] * min(len(relevant_texts), k) + [0] * (k - min(len(relevant_texts), k)), reverse=True)
    idcg = get_dcg(ideal_rels)
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return precision, recall, ap, ndcg

def run_evaluation():
    print("📥 Loading the database to map model answers to their corresponding texts...", flush=True)
    
    # 1. تحميل النصوص الأصلية
    full_dataset = ir_datasets.load(FULL_DATASET_NAME)
    doc_texts = {}
    for doc in full_dataset.docs_iter():
        # تنظيف النص لضمان المطابقة
        doc_texts[doc.doc_id] = doc.text.strip().lower()

    # 2. تحميل ملفات الاختبار
    test_dataset = ir_datasets.load(TEST_DATASET_NAME)
    qrels_dict = {}
    for qrel in test_dataset.qrels_iter():
        if qrel.relevance > 0 and qrel.doc_id in doc_texts:
            if qrel.query_id not in qrels_dict: 
                qrels_dict[qrel.query_id] = set()
            # حفظ "النص" كإجابة صحيحة بدلاً من الرقم
            qrels_dict[qrel.query_id].add(doc_texts[qrel.doc_id])

    queries = []
    for q in test_dataset.queries_iter():
        if q.query_id in qrels_dict:
            queries.append(q)
            if len(queries) >= SAMPLE_SIZE:
                break

    algorithms = ["bm25", "hybrid_serial"]
    results_summary = {algo: {"P": 0, "R": 0, "MAP": 0, "nDCG": 0} for algo in algorithms}

    print(f"\n🚀 Starting automatic evaluation for {SAMPLE_SIZE} queries...", flush=True)
    print("-" * 60)

    for algo in algorithms:
        print(f"⏳ Testing algorithm: {algo}", flush=True)
        for i, query in enumerate(queries):
            print(f"  - Sending query {i+1}/{SAMPLE_SIZE} ...", end=" ", flush=True)
            payload = {"query": query.text, "dataset": DATASET_API_NAME, "algorithm": algo, "limit": TOP_K}
            
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    # استخراج "نص" المستند القادم من الخادم
                    retrieved_texts = []
                    for res in data["results"]:
                        text = res.get("text", res.get("content", ""))
                        retrieved_texts.append(text.strip().lower())
                    
                    p, r, ap, ndcg = calculate_metrics(retrieved_texts, qrels_dict[query.query_id], k=TOP_K)
                    
                    results_summary[algo]["P"] += p
                    results_summary[algo]["R"] += r
                    results_summary[algo]["MAP"] += ap
                    results_summary[algo]["nDCG"] += ndcg
                    
                    if p > 0:
                        print("✅ (Found a matching document!)", flush=True)
                    else:
                        print("✔️ (No match found)", flush=True)
                else:
                    print(f"⚠️ Server Error", flush=True)
            except Exception as e:
                print(f"⚠️ Error: {e}", flush=True)

        for metric in results_summary[algo]:
            results_summary[algo][metric] /= SAMPLE_SIZE

    print("\n📊 النتائج النهائية للتقييم الأكاديمي:")
    print("=" * 65)
    print(f"{'الخوارزمية':<15} | {'Precision@10':<12} | {'Recall':<10} | {'MAP':<10} | {'nDCG@10'}")
    print("-" * 65)
    for algo in algorithms:
        p = results_summary[algo]['P'] * 100
        r = results_summary[algo]['R'] * 100
        map_score = results_summary[algo]['MAP'] * 100
        ndcg = results_summary[algo]['nDCG'] * 100
        print(f"{algo:<15} | {p:>9.2f}% | {r:>8.2f}% | {map_score:>8.2f}% | {ndcg:>8.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()